# coding:utf8
import torch as t
from torch.utils.data import DataLoader

from torch import nn
from torchnet import meter

from configs import DefaultConfig as Config
from dataproviders import get_data, generate_square_subsequent_mask
from models import PoetryModel


cfg = Config()
device = t.device('cuda') if cfg.use_gpu else t.device('cpu')

# 获取数据
data, word2ix, ix2word = get_data(cfg.pickle_path)
data = t.from_numpy(data)
# 数据加载器
dataloader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

# 词表大小
vocab_size = len(word2ix)   
# 模型
model = PoetryModel(vocab_size)
# 优化器
optimizer = t.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
# 交叉熵损失函数（因为最后的预测是分类问题）
criterion = nn.CrossEntropyLoss(ignore_index=len(word2ix)-1)
# 已有模型的加载
if cfg.model_path:
    model.load_state_dict(t.load(cfg.model_path))
model.to(device)

# 记录损失
loss_meter = meter.AverageValueMeter()
# 训练
for epoch in range(cfg.epoch):
    # 清空上一个 epoch 记录的损失
    loss_meter.reset()
    # 按 batch 加载数据并训练模型
    for ii, data_ in enumerate(dataloader):
        # [B, maxlen] ==> [maxlen, B]
        data_ = data_.long().transpose(1, 0).contiguous()
        data_ = data_.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 去掉开始标识符 [maxlen-1, B]
        input_ = data_[:-1, :]
        # 去掉结尾标识符 [maxlen-1, B]
        target = data_[1:, :]
        # 生成掩码 [maxlen-1, maxlen-1]
        src_mask = generate_square_subsequent_mask(input_.shape[0])
        src_mask = src_mask.to(device)
        # 空标识符的编码
        code = len(word2ix) - 1
        # 生成输入序列 padding 部分的掩码 [maxlen-1, B]
        src_pad_mask = input_ == code
        # [maxlen-1, B] ==> [B, maxlen-1]
        src_pad_mask = src_pad_mask.permute(1,0).contiguous()
        src_pad_mask = src_pad_mask.to(device)
        
        # 模型预测 memory:[maxlen-1, B, emb_size] logit:[maxlen-1, B, vocab_size]
        memory, logit = model(input_, src_mask, src_pad_mask)

        # target 的 padding mask
        mask = target != word2ix['</s>']
        # 去掉前缀的空格，并转为 1 维
        target = target[mask] 
        # [maxlen-1, B, vocab_size] ==> [maxlen-1 * B, vocab_size] ==> [token_all_num, vocab_size]
        logit = logit.flatten(0, 1)[mask.view(-1)]  # 展为1维，并去掉空格
        # 计算损失
        loss = criterion(logit, target)
        # 反向传播
        loss.backward()
        # 优化参数
        optimizer.step()
        # 记录损失
        loss_meter.add(loss.item())

        # 输出训练进度相关的信息
        if (1 + ii) % cfg.plot_every == 0:
            print('%s / %s -----> loss: %s'%(ii+1, round(len(data)/cfg.batch_size), loss.item()))

    # 保存模型
    t.save(model.state_dict(), '%s_%s.pth' % (cfg.model_prefix, epoch+1))
    print('%s is completed!'%(epoch+1), '\n')
