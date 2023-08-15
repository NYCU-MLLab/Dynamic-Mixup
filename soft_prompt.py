import torch
import torch.nn as nn

class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 1,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.soft_prompt = nn.Embedding(n_tokens, 768)
        self.soft_prompt.weight = nn.parameter.Parameter(self.initialize_embedding(self.wte,
                n_tokens, 
                random_range, 
                initialize_from_vocab))
    def initialize_embedding(self, 
        wte: nn.Embedding,
        n_tokens: int = 10, 
        random_range: float = 1, 
        initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
            # return self.wte.weight[:n_tokens].data
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])   #soft prompt加在sentence前面
        # input_embedding = self.wte(tokens[:, :-self.n_tokens])    #soft prompt加在setence後面
        learned_embedding = self.soft_prompt.weight.repeat(input_embedding.size(0), 1, 1)
        # print("learned_embedding: ", learned_embedding)
        return torch.cat([learned_embedding, input_embedding], 1)       #soft prompt加在sentence前面
        # return torch.cat([input_embedding, learned_embedding], 1)        #soft prompt加在setence後面