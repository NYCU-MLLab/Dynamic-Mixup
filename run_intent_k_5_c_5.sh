#!/bin/bash

#SBATCH -J Instance                  # 作业名为 test
#SBATCH -o Instance.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -p inspur                # 分区参数
#SBATCH -w inspur-gpu-05               # 分区参数


# dataset=data/HWU64
# tuned=tdopierre/ProtAugment-LM-HWU64
# # tuned=bert-base-uncased


# logdir=log_K_5_C_5/HWU64/K_5_C_5
# logfile=K_5_C_5
# K=5
# C=5

# if [ ! -d ${logdir}  ];then mkdir -p ${logdir};fi

# for cv in 01 02 03 04 05; do 
#     python -u main.py \
#         --data-path ${dataset}/full.json \
#         --train-path ${dataset}/few_shot/${cv}/train.json \
#         --train-labels-path ${dataset}/few_shot/${cv}/labels.train.txt \
#         --valid-labels-path ${dataset}/few_shot/${cv}/labels.valid.txt \
#         --test-labels-path ${dataset}/few_shot/${cv}/labels.test.txt \
#         --log-path ${logdir}/${cv}.out \
#         \
#         --n-support ${K} \
#         --n-query ${K} \
#         --n-classes ${C} \
#         \
#         --super-tau 5.0 \
#         --lr 1e-4 \
#         --super-weight 0.95 \
#         --evaluate-every 100 \
#         --n-test-episodes 600 \
#         --log-every 10 \
#         --max-iter 10000 \
#         --early-stop 20 \
#         --seed 42 \
#         \
#         --model-name-or-path ${tuned} \
#         \
#         --metric euclidean \
#         --supervised-loss-share-power 1
# done

dataset=data/BANKING77
tuned=tdopierre/ProtAugment-LM-BANKING77
# tuned=bert-base-uncased


logdir=log_K_5_C_5/BANKING77/K_5_C_5
logfile=K_5_C_5
K=5
C=5

if [ ! -d ${logdir}  ];then mkdir -p ${logdir};fi

for cv in 01 02 03 04 05; do
    python -u main.py \
        --data-path ${dataset}/full.json \
        --train-path ${dataset}/few_shot/${cv}/train.json \
        --train-labels-path ${dataset}/few_shot/${cv}/labels.train.txt \
        --valid-labels-path ${dataset}/few_shot/${cv}/labels.valid.txt \
        --test-labels-path ${dataset}/few_shot/${cv}/labels.test.txt \
        --log-path ${logdir}/${cv}.out \
        \
        --n-support ${K} \
        --n-query ${K} \
        --n-classes ${C} \
        \
        --super-tau 5.0 \
        --lr 5e-4 \
        --super-weight 0.95 \
        --evaluate-every 100 \
        --n-test-episodes 600 \
        --log-every 10 \
        --max-iter 10000 \
        --early-stop 20 \
        --seed 42 \
        \
        --model-name-or-path ${tuned} \
        \
        --metric euclidean \
        --supervised-loss-share-power 1
done

dataset=data/Amazon
tuned=bert-base-uncased


logdir=log_instance/Amazon/K_5_C_5
logfile=K_5_C_5
K=5
C=5

if [ ! -d ${logdir}  ];then mkdir -p ${logdir};fi

for cv in 01 02 03 04 05; do
    python -u main.py \
        --data-path ${dataset}/full.json \
        --train-path ${dataset}/few_shot/${cv}/train_aug.json \
        --train-labels-path ${dataset}/few_shot/${cv}/labels.train.txt \
        --valid-labels-path ${dataset}/few_shot/${cv}/labels.valid.txt \
        --test-labels-path ${dataset}/few_shot/${cv}/labels.test.txt \
        --log-path ${logdir}/${cv}.out \
        \
        --n-support ${K} \
        --n-query ${K} \
        --n-classes ${C} \
        \
        --super-tau 5.0 \
        --lr 1e-3 \
        --super-weight 0.95 \
        --evaluate-every 100 \
        --n-test-episodes 1000 \
        --log-every 10 \
        --max-iter 10000 \
        --early-stop 20 \
        --seed 42 \
        --max-len 200 \
        \
        --model-name-or-path ${tuned} \
        \
        --metric euclidean \
        --supervised-loss-share-power 1
done


# dataset=data/Amazon
# tuned=bert-base-uncased


# logdir=log_instance/Amazon/K_5_C_5
# logfile=K_5_C_5
# K=5
# C=5

# if [ ! -d ${logdir}  ];then mkdir -p ${logdir};fi

# for cv in 02 03 04 05; do
#     python3 -u main.py \
#         --data-path ${dataset}/full.json \
#         --train-path ${dataset}/few_shot/${cv}/train_aug.json \
#         --train-labels-path ${dataset}/few_shot/${cv}/labels.train.txt \
#         --valid-labels-path ${dataset}/few_shot/${cv}/labels.valid.txt \
#         --test-labels-path ${dataset}/few_shot/${cv}/labels.test.txt \
#         --log-path ${logdir}/${cv}.out \
#         \
#         --n-support ${K} \
#         --n-query ${K} \
#         --n-classes ${C} \
#         \
#         --super-tau 5.0 \
#         --lr 1e-3 \
#         --super-weight 0.95 \
#         --evaluate-every 100 \
#         --n-test-episodes 1000 \
#         --log-every 10 \
#         --max-iter 10000 \
#         --early-stop 20 \
#         --seed 42 \
#         --max-len 200 \
#         \
#         --model-name-or-path ${tuned} \
#         \
#         --metric euclidean \
#         --supervised-loss-share-power 1
# done


# dataset=data/Liu
# # tuned=tdopierre/ProtAugment-LM-Liu
# tuned=bert-base-uncased


# logdir=log_K_5_C_5/Liu/K_5_C_5
# logfile=K_5_C_5
# K=5
# C=5

# if [ ! -d ${logdir}  ];then mkdir -p ${logdir};fi

# for cv in 01 02 03 04 05; do
#     python -u main.py \
#         --data-path ${dataset}/full.json \
#         --train-path ${dataset}/few_shot/${cv}/train_aug.json \
#         --train-labels-path ${dataset}/few_shot/${cv}/labels.train.txt \
#         --valid-labels-path ${dataset}/few_shot/${cv}/labels.valid.txt \
#         --test-labels-path ${dataset}/few_shot/${cv}/labels.test.txt \
#         --log-path ${logdir}/${cv}.out \
#         \
#         --n-support ${K} \
#         --n-query ${K} \
#         --n-classes ${C} \
#         \
#         --super-tau 5.0 \
#         --lr 1e-3 \
#         --super-weight 0.95 \
#         --evaluate-every 100 \
#         --n-test-episodes 600 \
#         --log-every 10 \
#         --max-iter 10000 \
#         --early-stop 20 \
#         --seed 42 \
#         \
#         --model-name-or-path ${tuned} \
#         \
#         --metric euclidean \
#         --supervised-loss-share-power 1
# done

# dataset=data/OOS
# # tuned=tdopierre/ProtAugment-LM-Clinic150
# tuned=bert-base-uncased


# logdir=log_K_5_C_5/OOS/K_5_C_5
# logfile=K_5_C_5
# K=5
# C=5

# if [ ! -d ${logdir}  ];then mkdir -p ${logdir};fi

# for cv in 01 02 03 04 05; do
#     python -u main.py \
#         --data-path ${dataset}/full.json \
#         --train-path ${dataset}/few_shot/${cv}/train_aug.json \
#         --train-labels-path ${dataset}/few_shot/${cv}/labels.train.txt \
#         --valid-labels-path ${dataset}/few_shot/${cv}/labels.valid.txt \
#         --test-labels-path ${dataset}/few_shot/${cv}/labels.test.txt \
#         --log-path ${logdir}/${cv}.out \
#         \
#         --n-support ${K} \
#         --n-query ${K} \
#         --n-classes ${C} \
#         \
#         --super-tau 5.0 \
#         --lr 1e-3 \
#         --super-weight 0.95 \
#         --evaluate-every 100 \
#         --n-test-episodes 600 \
#         --log-every 10 \
#         --max-iter 10000 \
#         --early-stop 20 \
#         --seed 42 \
#         \
#         --model-name-or-path ${tuned} \
#         \
#         --metric euclidean \
#         --supervised-loss-share-power 1
# done
