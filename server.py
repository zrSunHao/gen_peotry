# coding:utf8
import torch as t
from flask import Flask, request
from concurrent.futures import ThreadPoolExecutor
import time
import requests
import json

from configs import DefaultConfig as Config
from dataproviders import get_data, generate_square_subsequent_mask
from models import PoetryModel

executor = ThreadPoolExecutor(1)
app = Flask(__name__)

cfg = Config()
device = t.device('cuda') if cfg.use_gpu else t.device('cpu')

t1 = time.time()
# 加载数据
data, word2ix, ix2word = get_data(cfg.pickle_path)
model = PoetryModel(len(word2ix))
# 加载模型
model.load_state_dict(t.load(cfg.model_path))
model.to(device)
# 非训练模式
model.eval()
t2 = time.time()
t_duration = (t2 - t1)
print('数据与模型加载耗时: ' + str(t_duration) + '/s')



"""
    给定几个词，根据这几个词接着生成一首完成的诗词
    例如，start_words为'海内存知己'，可以生成
    海内存知己，天涯尚未安。
    故人归旧国，新月到新安。
    海气生边岛，江声入夜滩。
    明朝不可问，应与故人看。
"""
def gen(start_words: str) -> str:
    # 开始语句中的字转编码
    src = [word2ix[word] for word in start_words]
    # 开始语句加开始标识符
    res = src = [word2ix['<START>']] + src
    # 最大长度
    max_len = 120

    # 循环生成
    for _ in range(max_len):
        # [token_num] ==> [token_num, 1]
        src = t.tensor(res).to(device)[:, None]
        # 生成掩码 [token_num, token_num]
        src_mask = generate_square_subsequent_mask(src.shape[0])
        # padding 掩码 [token_num, 1]
        src_pad_mask = src == len(word2ix) - 1
        # [token_num, 1] ==> [1, token_num]
        src_pad_mask = src_pad_mask.permute(1, 0).contiguous()

        # 模型预测
        memory, logits = model(src, src_mask.cuda(), src_pad_mask.cuda())
        # 获取概率最大的字
        next_word = logits[-1, 0].argmax().item()
        # 判断是否为结束字符
        if next_word == word2ix['<EOP>']:
            break
        # 添加到res
        res.append(next_word)

    # 字的编码转字
    res = [ix2word[_] for _ in res[1:]]
    # 字的 list 转 str
    return ''.join(res)



"""
    生成藏头诗
    start_words为'深度学习'
    生成：
	深山高不极，望望极悠悠。
	度日登楼望，看云上砌秋。
	学吟多野寺，吟想到江楼。
	习静多时选，忘机尽处求。
"""
def gen_acrostic(start_words:str) -> str:
    # 藏头诗的句子数
    start_word_len = len(start_words)
    # 用来指示已经生成了多少句藏头诗
    index = 0  
    # 字转编码
    src_base = [word2ix[word] for word in start_words]
    # 开始标识符 + 当前句的开头字
    res = [word2ix['<START>']] + [src_base[index]]
    # 当前句子的顺序，以及下一句开头的索引
    index += 1
    # 诗的最大长度
    max_len = 120

    # 循环生成
    for _ in range(max_len):
        # [token_num] ==> [token_num, 1]
        src = t.tensor(res).to(device)[:, None]
        # 生成掩码 [token_num, token_num]
        src_mask = generate_square_subsequent_mask(src.shape[0])
        # padding 掩码 [token_num, 1]
        src_pad_mask = src == len(word2ix) - 1
        # [token_num, 1] ==> [1, token_num]
        src_pad_mask = src_pad_mask.permute(1, 0).contiguous()

        # 预测生成
        memory, logits = model(src, src_mask.cuda(), src_pad_mask.cuda())
        # 获取概率最大的字
        next_word = logits[-1, 0].argmax().item()
        # 如果遇到句号感叹号等，把藏头的词作为下一个句的输入
        if next_word in {word2ix[u'。'], word2ix[u'！'], word2ix['<START>']}:
            # 如果生成的诗歌已经包含全部藏头的词，则结束
            if index == start_word_len:
                res.append(next_word)
                break
            # 添加当前预测的词
            res.append(next_word)
            # 把藏头的词作为输入，预测下一个词
            res.append(src_base[index])
            index += 1
        else:
            # 添加当前预测的词
            res.append(next_word)

    # 编码转字
    res = [ix2word[_] for _ in res[1:]]
    # list ==> str
    return ''.join(res)

# p1 = gen('天王盖地虎')
# print('\n')
# p2 = gen_acrostic('天王盖地虎')
# print('\n')


'''
发送结果
'''
def send_result(task_id, message, success, duration, poetry):
    try:
        url = "https://localhost:7085/api/job/update"
        # Post请求发送的数据，字典格式
        data = {"id":task_id, "message": message, "success": success, "duration": duration, "poetry": poetry} 
        headers = {"Content-Type": "application/json; charset=UTF-8",}
        
        #这里传入的data,是body里面的数据。params是拼接url时的参
        requests.packages.urllib3.disable_warnings()
        res = requests.post(url=url,data=json.dumps(data), headers=headers,verify=False)
        print(data)
    except Exception as ex:
        print(ex)



'''
task_id:        任务id
start_words:    开始字词
mode:           模式，0 作为开头生成，1 生成藏头诗
'''
def handle(task_id:str, start_words:str, mode='0'):
    message = ''
    success = True
    duration = 0
    poetry = ''
    try:
        T1 = time.time()
        assert mode in ['0', '1']
        if mode == '0':
            poetry = gen(start_words)
        if mode == '1':
            poetry = gen_acrostic(start_words)
        T2 = time.time()
        duration = (T2 - T1)
        message = task_id + '诗词写作成功'
    except Exception as ex:
        print(ex)
        message = str(ex)
        success = False
    finally:
        print('start_words: ' + start_words)
        print(poetry)
        print(message, duration, '\n')
        #send_result(task_id, message, success, duration, poetry)



'''
接收请求
'''
@app.route('/writing_poetry',methods=['GET'])
def writing_poetry():
    task_id = request.args.get('task_id')
    start_words = request.args.get('start_words')
    mode = request.args.get('mode')
    if (task_id == None or start_words == None or mode == None):
        return 'There are null parameters!'
    executor.submit(handle, task_id, start_words, mode)
    return 'Processing, please check the results later!'

app.run(host='0.0.0.0', port=8089, debug=False)