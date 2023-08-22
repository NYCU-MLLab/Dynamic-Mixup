from typing import List

import torch.nn as nn
import logging
import warnings
import torch
from torch.cuda import amp
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, AutoModelForMaskedLM
from paraphrase.utils.data import FewShotDataset
from utils.math import euclidean_dist, cosine_similarity
from soft_prompt import SoftEmbedding
import numpy as np
import collections
from transformers.optimization import get_linear_schedule_with_warmup
import random
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def dot_similarity(x1, x2):
    return torch.matmul(x1, x2.t())

def cosine(x1, x2):
    x1 = (x1 / x1.norm(dim=1).view(-1, 1))
    x2 = (x2 / x2.norm(dim=1).view(-1, 1))

    return x1 @ x2.T

def euclidean_dist(x, y):
  """
  Computes euclidean distance btw x and y
  Args:
      x (torch.Tensor): shape (n, d). n usually n_way*n_query
      y (torch.Tensor): shape (m, d). m usually n_way
  Returns:
      torch.Tensor: shape(n, m). For each query, the distances to each centroid
  """
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)

class Contrastive_Loss(nn.Module):

    def __init__(self, tau=5.0):
        super(Contrastive_Loss, self).__init__()
        self.tau = tau

    def similarity(self, x1, x2):
        # # Gaussian Kernel
        # M = euclidean_dist(x1, x2)
        # s = torch.exp(-M/self.tau)

        # dot product
        M = dot_similarity(x1, x2)/self.tau
        # M = cosine(x1, x2)/self.tau
        s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return s

    def forward(self, batch_label, *x, mixup, l):
        X = torch.cat(x, 0)
        X = torch.cat((X, mixup), dim=0)
        batch_labels = torch.cat([batch_label for i in range(len(x))], 0)
        batch_labels = torch.cat([batch_labels, batch_label], 0)
        len_ = batch_labels.size()[0]

        # computing similarities for each positive and negative pair
        s = self.similarity(X, X)
        # computing masks for contrastive loss
        if len(x)==1:
            mask_i = torch.from_numpy(np.ones((len_, len_))).to(batch_labels.device)
        else:
            mask_i = 1. - torch.from_numpy(np.identity(len_)).to(batch_labels.device) # sum over items in the numerator
        
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        mask_j = (batch_labels.unsqueeze(1) - label_matrix == 0).float()*mask_i # sum over items in the denominator

        pos_num = torch.sum(mask_j, 1)

        # weighted NLL loss
        s_i = torch.clamp(torch.sum(s*mask_i, 1), min=1e-10) 
        s_j = torch.clamp(torch.sum(s*mask_j, 1), min=1e-10)
        log_p = torch.sum(-torch.log(s_j/s_i)*mask_j, 1)/pos_num
        loss = torch.mean(log_p)

        return loss

class ContrastNet(nn.Module):
    def __init__(self, config_name_or_path, metric="euclidean", max_len=64, super_tau=1.0):
        super(ContrastNet, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(config_name_or_path)
        self.encoder = AutoModel.from_pretrained(config_name_or_path).to(device)
        self.metric = metric
        self.max_len = max_len
        assert self.metric in ('euclidean', 'cosine')
        self.contrast_loss = Contrastive_Loss(super_tau)

        self.n_tokens = 10    # number of soft-prompt tokens
        initialize_from_vocab = True
        # print(self.encoder.get_input_embeddings().weight)
        # self.freeze(self.encoder)
        s_wte = SoftEmbedding(self.encoder.get_input_embeddings(), 
                            n_tokens=self.n_tokens, 
                            initialize_from_vocab=initialize_from_vocab)

        self.encoder.set_input_embeddings(s_wte)

        self.freeze(self.encoder)
        self.encoder.embeddings.word_embeddings.soft_prompt.weight.requires_grad = True

        trainable_params = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        print("trainable_params: ", trainable_params)

        self.warmed: bool = False

    def freeze(self, module):
        """
        Freezes module's parameters.
        """
        for name, child in module.named_children():
            print("name: ", name)
            for parameter in child.parameters():
                parameter.requires_grad = False

    def forward(self, sentences: List[str]):
        # freezing embeddings and all layers of encoder
        
        if self.warmed:
            padding = True
        else:
            padding = "max_length"
            self.warmed = True

        batch = self.tokenizer.batch_encode_plus(
            sentences,
            return_tensors="pt",
            max_length=self.max_len,
            truncation=True,
            padding=padding,
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        # ### soft prompt 加在 sentence 前面
        batch['token_type_ids'] = torch.cat([torch.full((1,self.n_tokens+batch['input_ids'].shape[1]), 0)], 1).repeat(len(batch['input_ids']),1).to(device)
        batch['attention_mask'] = torch.cat([torch.full((len(batch['input_ids']),self.n_tokens), 1).to(device), batch['attention_mask'].to(device)], 1).to(device)  #soft prompt加在sentence前面
        batch['input_ids'] = torch.cat([torch.full((len(batch['input_ids']),self.n_tokens), -1).to(device), batch['input_ids'].to(device)], 1).to(device)  #soft prompt加在sentence前面

        hidden = self.encoder(**batch).last_hidden_state
        return hidden[:,self.n_tokens,:]      #soft prompt加在sentence前面 (CLS)

    def mixup(self, z, l, z_proto):
        mixup_data=torch.zeros((1,768)).to(device)
        M = dot_similarity(z, z)/5.0
        s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])

        for i in range(int(z.shape[0]/2)):

            if(i>=0 and i<int(z.shape[0]/10)):
                s[i][0:int(z.shape[0]/10)]=0
            elif(i>=int(z.shape[0]/10) and i<int((z.shape[0]/10)*2)):
                s[i][int(z.shape[0]/10):int((z.shape[0]/10)*2)]=0
            elif(i>=int((z.shape[0]/10)*2) and i<int((z.shape[0]/10)*3)):
                s[i][int((z.shape[0]/10)*2):int((z.shape[0]/10)*3)]=0
            elif(i>=int((z.shape[0]/10)*3) and i<int((z.shape[0]/10)*4)):
                s[i][int((z.shape[0]/10)*3):int((z.shape[0]/10)*4)]=0
            elif(i>=int((z.shape[0]/10)*4) and i<int((z.shape[0]/10)*5)):
                s[i][int((z.shape[0]/10)*4):int((z.shape[0]/10)*5)]=0
            s[i][int(z.shape[0]/2):int(z.shape[0])]=0

            _, mix_index = s[i].max(0)
            mixup_data = torch.cat((mixup_data, (l.clone()*z[i,:] + (1-l.clone())*z[mix_index,:]).unsqueeze(0)), dim=0)
        return mixup_data[1:,:]
    
    def pred_proto(self, query, proto): 
        s = dot_similarity(query, proto)
        _, y_pred = s.max(1)

        return y_pred


    def loss(self, sample, l, k, supervised_loss_share: float = 0, mode='train'):
        """
        :param supervised_loss_share: share of supervised loss in total loss
        :param sample: {
            "xs": [
                [support_A_1, support_A_2, ...],
                [support_B_1, support_B_2, ...],
                [support_C_1, support_C_2, ...],
                ...
            ],
            "xq": [
                [query_A_1, query_A_2, ...],
                [query_B_1, query_B_2, ...],
                [query_C_1, query_C_2, ...],
                ...
            ],
        }
        :return:
        """
        xs = sample['xs']  # support
        xq = sample['xq']  # query

        n_class = len(xs)
        assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        support_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_support, 1).long()
        support_inds = Variable(support_inds, requires_grad=False).to(device)

        supports = [item["sentence"] for xs_ in xs for item in xs_]
        queries = [item["sentence"] for xq_ in xq for item in xq_]
        # x = (supports + queries)*2
        x = supports + queries

        z = self.forward(x)
        
        z_dim = z.size(-1)

        z_support = z[:len(supports)]
        z_query = z[len(supports):len(supports) + len(queries)]
        z_support_proto = z_support.view(n_class, n_support, z_dim).mean(dim=[1])
        z_query_proto = z_query.view(n_class, n_query, z_dim).mean(dim=[1])

        
        z_proto = torch.cat((z_support_proto, z_query_proto), dim=0)
        Mixup = self.mixup(z, l, z_proto=z_proto)
        # z_support_mixup = torch.cat((z[:len(supports)],Mixup), dim=0)

        if(mode=='train'):
            z_support_mixup = torch.zeros((1,768)).to(device)
            if(k=='1'):
                for i in range(5):
                    z_support_mixup_temp = torch.cat((z[i].unsqueeze(0), Mixup[i].unsqueeze(0)), dim=0)
                    z_support_mixup = torch.cat((z_support_mixup, z_support_mixup_temp), dim=0)
            
            elif(k=='5'):
                for i in range(5):
                    z_support_mixup_temp = torch.cat((z[i*5:i*5+5], Mixup[i*5:i*5+5]), dim=0)
                    z_support_mixup = torch.cat((z_support_mixup, z_support_mixup_temp), dim=0)

            z_support_mixup_proto = z_support_mixup[1:,:].view(n_class, n_support*2, z_dim).mean(dim=[1])


            ### maximum entropy
            s = dot_similarity(Mixup, z_support_proto)
            entropy = ((s).softmax(1)*(s).log_softmax(1)).sum(1)
            entropy = -(entropy).mean()

            ### minimize prototypical loss
            s_query = dot_similarity(z_query, z_support_mixup_proto)
            log_p_y = torch.nn.functional.log_softmax(s_query, dim=1).view(n_class, n_query, -1)
            CE_loss = -log_p_y.gather(2, support_inds).squeeze().view(-1).mean()


            mixup_loss = 0.1* (-entropy) + 0.1* (CE_loss)
        if(mode=='test'):
            mixup_loss=0

        ### For contrastive loss
        z_query_in = z_query
        z_support_in = z_support 
        contrast_labels = support_inds.reshape(-1)

        contrastive_loss = self.contrast_loss(contrast_labels, z_support_in, z_query_in, mixup=Mixup, l=l)


        y_pred = self.pred_proto(z_query, z_support_proto)
        acc = torch.eq(y_pred, support_inds.reshape(-1)).float().mean()

        final_loss = contrastive_loss


        return final_loss, mixup_loss, {
            "metrics": {
                "acc": acc.item(),
                "loss": final_loss.item(),
                # "mixup_loss: ": mixup_loss.item(),
                # "entropy_loss: ": entropy.item(),
                # "CE_loss: ": CE_loss.item(),
            },
            "target": support_inds
            # "target": target_inds
        }    


    def train_step(self, optimizer, optimizer_lambda, l, k, episode, supervised_loss_share: float):
        self.train()
        optimizer.zero_grad()
        optimizer_lambda.zero_grad()
        torch.cuda.empty_cache()
        self.encoder.embeddings.word_embeddings.soft_prompt.weight.requires_grad = False
        l.requires_grad = True
        loss, mixup_loss, loss_dict = self.loss(episode, l, k, supervised_loss_share=supervised_loss_share,mode='train')
        mixup_loss.backward(retain_graph=True)
        optimizer_lambda.step()
        optimizer_lambda.zero_grad()
        
        l.requires_grad = False
        self.encoder.embeddings.word_embeddings.soft_prompt.weight.requires_grad = True
        loss1, mixup_loss1, loss_dict1 = self.loss(episode, l, k, supervised_loss_share=supervised_loss_share,mode='train')
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss1, loss_dict1


    def test_step(self, l, k, dataset: FewShotDataset, n_episodes: int = 1000):
        metrics = collections.defaultdict(list)
        self.eval()
        for i in range(n_episodes):
            episode = dataset.get_episode()

            with torch.no_grad():
                loss, mixup_loss, loss_dict = self.loss(episode, l, k, supervised_loss_share=1,mode='test')

            for k, v in loss_dict["metrics"].items():
                metrics[k].append(v)

        return {
            key: np.mean(value) for key, value in metrics.items()
        }
