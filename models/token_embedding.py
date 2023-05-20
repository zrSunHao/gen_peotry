import math
from torch import nn

'''
token 转词向量
'''
class TokenEmbedding(nn.Module):

    '''
    vocab_size: 词表大小
    emb_size:   词向量的维度
    '''
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        # 词与词向量映射的层，有一个形状为 (vocab_size, emb_size) 的权重
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    '''
    tokens: 待转换的词(词的编码)集合
    '''
    def forward(self, tokens):
        out = self.embedding(tokens.long())
        out = out * math.sqrt(self.emb_size)
        return out
