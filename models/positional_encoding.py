import torch as t
import math
from torch import nn

'''
位置编码
携带词的先后顺序
'''
class PositionalEncoding(nn.Module):

    '''
    emb_size:   词向量的维度
    maxlen:     句子的最大长度
    '''
    def __init__(self, emb_size, dropout, maxlen=200):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 生成 [max_len, d_model] 大小的矩阵
        pos_embedding = t.zeros((maxlen, emb_size))
        # 生成pos， [ [0],[1], ··· [max_len-1] ] 的张量
        pos = t.arange(0, maxlen).float().reshape(maxlen, 1)
        # 生成位置公式的公共部分，即 10000**(2i/d_model)
        den = t.exp(- t.arange(0, emb_size, 2).float() * math.log(100) / emb_size)
        
        # 偶数位
        pos_embedding[:, 0::2] = t.sin(pos * den)
        # 基数位
        pos_embedding[:, 1::2] = t.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # 计算位置编码
        pe = self.pos_embedding[:token_embedding.size(0),:]
        # 词向量与位置编码相加
        pe = pe + token_embedding
        out = self.dropout(pe)
        return out