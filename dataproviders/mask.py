import torch as t

'''
生成 mask 矩阵
'''
def generate_square_subsequent_mask(sz):

    # 生成维度为 sz*sz 的矩阵，且值全为 1 
    mat = t.ones(sz, sz)
    #print(mat, '\n')

    # 生成上三角矩阵，上三角值全为 1，下三角值全为 0
    mat = t.triu(mat)
    #print(mat, '\n')

    # 值为 1、0 的矩阵转为值为 true、false 的矩阵
    mat = mat == 1
    #print(mat, '\n')

    # 转置，上三角矩阵转为下三角矩阵，即下三角全为True，其余位False
    mask = mat.transpose(0, 1).float()
    #print(mask, '\n')

    # 将值为 false 位置的值替换为 -inf，即负无穷
    mask = mask.masked_fill(mask == 0, float('-inf'))
    #print(mask, '\n')

    # 将值为 true 位置的值替换为 0.0
    mask = mask.masked_fill(mask == 1, float(0.0))
    #print(mask, '\n')

    # 返回 mask 矩阵，上三角值全为：0.0，下三角值全为 -inf
    return mask