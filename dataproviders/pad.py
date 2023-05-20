import numpy as np

"""
将每个句子序列填充到相同的长度
如果 maxlen 参数有值，则最长句子序列的长度为 maxlen

参数：
    sequences:  list 类型，每一个元素都是一个句子序列
    maxlen:     int  类型，限制句子的最大长度为 maxlen
    dtype:      返回结果中，每个序列编码值得类型
    padding:    'pre' 或 'post'，在序列的‘前面’或‘后面’进行填充。
    truncating: 'pre' 或 'post'，对于序列长度大于 maxlen 的，
                选择从‘前面’或是‘后面’进行截取
    value:      float 类型， 序列填充所用的‘值’

返回值：
    x:  带维度的 numpy 数组 [number_of_sequences, maxlen]

抛出的异常:
    ValueError: “截断”或“填充”的值无效，或“序列”形状无效。
"""
def pad_sequences(sequences, maxlen=None, 
                  dtype='int32', padding='pre', truncating='pre', value=0.):
    
    # 序列集合必须是可迭代的
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    
    # 得到 list 中每一个序列的长度
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    # list 中序列的个数
    num_samples = len(sequences)
    # 若 maxlen 未赋值，则其为 list 中最长序列的长度
    if maxlen is None:
        maxlen = np.max(lengths)

    # 从第一个非空序列中获取样本的形状，即句子序列的长度
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]  # ()
            break

    # 根据 truncating 策略，截取句子
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue    # empty list/array was found

        # 把后面多余的截去
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        # 把前面多余的截去
        elif truncating == 'post':
            trunc = s[:maxlen]
        # 未能识别的截取方式
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)
        trunc = np.asarray(trunc, dtype=dtype)
        # 检查截取之后的长度是否为预期的长度
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))
        
        # 预设好的填充值
        # (num_samples, maxlen) + () ==> (num_samples, maxlen)
        x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
        # 赋值
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
        
    return x