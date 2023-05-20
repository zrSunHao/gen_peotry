class DefaultConfig(object):

    # 数据
    data_path = 'data'                  # 诗歌的文本文件存放路径
    pickle_path = 'data/tang.npz'       # 预处理好的二进制文件
    max_gen_len = 200                   # 生成诗歌最长长度
    maxlen = 125                        # 超过这个长度之后的字被丢弃，小于这个长度的在前面补空格
    
    # 模型训练
    use_gpu = True                      # 是否使用 GPU
    epoch = 200                         # 训练轮次数
    batch_size = 32                    # 一个批次的数量
    lr = 1e-3                           # 学习率
    weight_decay = 1e-4                 # 权值衰减，防止过拟合
    
    # 模型路径
    model_path = 'checkpoints/tang_100.pth' # 预训练模型路径
    model_prefix = 'checkpoints/tang'   # 模型保存路径

    plot_every = 20                     # 每20个batch 输出一次信息

