# coding:utf-8
import os
import numpy as np

"""
参数：      pickle_path:  预处理好的 numpy 压缩包的路径
返回值：    data word2ix ix2word    
"""
def get_data(pickle_path: str):

    # 判断预处理好的二进制文件是否存在
    assert os.path.exists(pickle_path)

    # 加载 numpy 数据
    datas = np.load(pickle_path, allow_pickle=True)

    # 形状为 (57598, 125) 的 numpy 数组，每一行是一首诗对应的字的下标
    data = datas['data']

    # dict 类型, 每个字对应的序号，形如 u'月'->100
    word2ix = datas['word2ix'].item()
    
    #dict 类型, 每个序号对应的字，形如 '100'->u'月'
    ix2word = datas['ix2word'].item()
    
    return data, word2ix, ix2word
    
