#! -*- coding: utf-8 -*-
# 简单的线性变换（白化）操作，就可以达到甚至超过BERT-flow的效果。

from utils import *
import sys
import jieba
import codecs

jieba.initialize()

# 基本参数
model_type  = "BERT" 
# assert model_type in [
#     'BERT', 'RoBERTa', 'NEZHA', 'WoBERT', 'RoFormer', 'BERT-large',
#     'RoBERTa-large', 'NEZHA-large', 'SimBERT', 'SimBERT-tiny', 'SimBERT-small'
# ]
# assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']
# assert task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B' , 'myjob']

n_components_ = ["384" , "256" , "768"] 
n_components = [384 , 256 , 768]


maxlen = 64

# 加载数据集
data_path = '/root/senteval_cn/'
task_name = 'dd'
# datasets = {
#     '%s-%s' % (task_name, f):
#     load_data('%s%s/%s.%s.data' % (data_path, task_name, task_name, f))
#     for f in ['train', 'valid', 'test']
# }

datasets = {
    '%s' % (task_name):
    load_my_data('./new_symptom.csv')
}

# bert配置
model_name = {
    'BERT': 'chinese_L-12_H-768_A-12',
    'RoBERTa': 'chinese_roberta_wwm_ext_L-12_H-768_A-12',
    'WoBERT': 'chinese_wobert_plus_L-12_H-768_A-12',
    'NEZHA': 'nezha_base_wwm',
    'RoFormer': 'chinese_roformer_L-12_H-768_A-12',
    'BERT-large': 'uer/mixed_corpus_bert_large_model',
    'RoBERTa-large': 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16',
    'NEZHA-large': 'nezha_large_wwm',
    'SimBERT': 'chinese_simbert_L-12_H-768_A-12',
    'SimBERT-tiny': 'chinese_simbert_L-4_H-312_A-12',
    'SimBERT-small': 'chinese_simbert_L-6_H-384_A-12'
}[model_type]

config_path = '/root/kg/skbert/%s/bert_config.json' % model_name
if model_type == 'NEZHA':
    checkpoint_path = '/root/kg/bert/%s/model.ckpt-691689' % model_name
elif model_type == 'NEZHA-large':
    checkpoint_path = '/root/kg/bert/%s/model.ckpt-346400' % model_name
else:
    checkpoint_path = '/root/kg/skbert/%s/bert_model.ckpt' % model_name
dict_path = '/root/kg/skbert/%s/vocab.txt' % model_name

# 建立分词器
tokenizer = get_tokenizer(dict_path)
poolings = ["cls","first-last-avg","last-avg","pooler"]
    # 建立模型
for pooling in poolings:
    encoder = get_encoder(config_path, checkpoint_path, pooling=pooling)

    # 语料向量化
    all_names, all_weights, all_vecs, all_labels = [], [], [], []
    # for name, data in datasets.items():
    #     a_vecs, b_vecs, labels = convert_to_vecs(data, tokenizer, encoder, maxlen)
    #     all_names.append(name)
    #     all_weights.append(len(data))
    #     all_vecs.append((a_vecs, b_vecs))
    #     all_labels.append(labels)


    ########
    for name, data in datasets.items():
        # print(data)
        a_vecs = my_convert_to_vecs(data, tokenizer, encoder, maxlen)
        all_names.append(name)
        all_weights.append(len(data))
        all_vecs.append(a_vecs)


    #     # all_labels.append(labels)

    # 计算变换矩阵和偏置项

    kernel, bias = compute_kernel_bias([vecs for vecs in all_vecs])
    for c in n_components:
        cur_kernel = kernel[:, :c]        
        # 变换，标准化，相似度，相关系数
        afterNormalize = []
        transformVec = []
        for a_vecs in all_vecs:
            vecs , afterNormalizeVecs = transform_and_normalize(a_vecs, cur_kernel, bias)
            afterNormalize.append(afterNormalizeVecs)
            transformVec.append(vecs)

            # b_vecs = transform_and_normalize(b_vecs, kernel, bias)
        #     sims = (a_vecs * b_vecs).sum(axis=1)
        #     corrcoef = compute_corrcoef(labels, sims)
        #     all_corrcoefs.append(corrcoef)
        np.savetxt("./%s/%s/text_vectors_transform.txt"%(str(c) , pooling),transformVec[0])
        np.savetxt("./%s/%s/text_vectors_afterNormalize.txt"%(str(c) , pooling),afterNormalize[0])
    np.savetxt("./bert/%s/text_vectors_bert.txt" %pooling,all_vecs[0])


# with codecs.open("./afterWhitening.csv", 'w','utf-8') as f:
#     for line in all_vecs:
#         f.write(line + "\n")

# with codecs.open("./afterNormalize.csv", 'w','utf-8') as f:
#     for line in afterNormalize:
#         f.write(line + "\n")
######


# all_corrcoefs.extend([
#     np.average(all_corrcoefs),
#     np.average(all_corrcoefs, weights=all_weights)
# ])

# for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
#     print('%s: %s' % (name, corrcoef))
