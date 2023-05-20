from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .positional_encoding import PositionalEncoding
from .token_embedding import TokenEmbedding


class PoetryModel(nn.Module):
    
    '''
    vocab_size:         词表的大小
    emb_size:           词向量的维度
    num_encoder_layers: 编码器的中编码层的个数
    dim_feedforward:    编码层中前馈网络神经元的个数
    '''
    def __init__(self, vocab_size, num_encoder_layers=4, 
                 emb_size=512, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # 词转词向量
        self.src_tok_emb = TokenEmbedding(vocab_size, emb_size)
        # 位置编码
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        # 编码层
        encoder_layer = TransformerEncoderLayer(d_model=emb_size,       # 词向量的维度
                                                nhead=8,                # 多头注意力中 head 的个数
                                                dim_feedforward=dim_feedforward)        # 前馈网络神经元的个数
        # 编码器
        self.transformer_encoder = TransformerEncoder(encoder_layer,    # 编码层
                                                      num_layers=num_encoder_layers)    # 编码层的个数
        # 全连接层，生成对应词表的下一个词的概率
        self.generator = nn.Linear(emb_size, vocab_size)
        # 参数初始化
        for p in self.parameters():
            if p.dim() > 1:
                # 基本思想是通过网络层时，输入和输出的方差相同，包括前向传播和后向传播。
                nn.init.xavier_uniform_(p)


    '''
    src:                输入的序列
    src_mask:           输入序列的掩码
    src_padding_mask:   输入序列 padding 部分的掩码
    '''
    def forward(self, src, src_mask, src_padding_mask):
        # 对输入的词进行向量化
        src_emb = self.src_tok_emb(src)
        # 为词向量添加位置编码
        src_emb = self.positional_encoding(src_emb)
        # 经过编码器进行理解  [maxlen-1, B, emb_size]
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        # 经过全连接层生成预测下一个词  [maxlen-1, B, vocab_size]
        logit = self.generator(memory)
        return memory, logit
