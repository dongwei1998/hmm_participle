# coding=utf-8
# =============================================
# @Time      : 2022-08-02 9:03
# @Author    : DongWei1998
# @FileName  : server.py
# @Software  : PyCharm
# =============================================
import numpy as np
from utils import hmm_learn
from utils import parameter
from flask import Flask, jsonify, request
import os



class Hmm_par(object):
    def __init__(self,args):
        self.arge = args
        # 加载模型
        # 1. 加载模型参数
        self.pi = np.load(os.path.join(args.output_dir,'pi.npy'))
        self.A = np.load(os.path.join(args.output_dir,'A.npy'))
        self.B = np.load(os.path.join(args.output_dir,'B.npy'))

    def cut(self,decode, sentence):
        T = len(decode)
        t = 0
        while t < T:
            # 当前时刻的状态值
            state = decode[t]
            # 判断当前时刻的状态值
            if state == 0 or state == 1:
                # 表示t时刻对应的单词是一个词语的开始或者中间位置，那么后面还有字属于同一个词语
                j = t + 1
                while j < T:
                    if decode[j] == 2 or decode[j] == 3:
                        break
                    j += 1
                # 返回分词结果
                yield sentence[t:j + 1]
                t = j
            elif state == 3 or state == 2:
                # 这个时候表示单个字或者最后一个字
                yield sentence[t:t + 1]
            t += 1

    def text_cut(self,text):
        Q = []
        for cht in text:
            Q.append(ord(cht))
        _, decode, _ = hmm_learn.viterbi(self.pi, self.A, self.B, Q)
        # 3. 基于隐状态做一个分词的操作
        cut_result = self.cut(decode, text)
        # for w in cut_result:
        #     print(w)
        return ' '.join([i for i in cut_result])




if __name__ == '__main__':
    app = Flask(__name__)

    app.config['JSON_AS_ASCII'] = False

    model = 'server'


    args = parameter.parser_opt(model='train')
    word_par = Hmm_par(args)
    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            # 参数获取
            infos = request.get_json()
            data_dict = {
                'text': ''
            }
            for k, v in infos.items():
                data_dict[k] = v

            text = data_dict['text'].replace('\n', '').replace('\r', '')
            # 参数检查
            if text is None:
                return jsonify({
                    'code': 500,
                    'msg': '请给定参数text！！！'
                })

            cut_result = word_par.text_cut(text)
            return jsonify({
                'code': 200,
                'msg': '成功',
                'cut_result ': cut_result,
                'text': text,
            })
        except Exception as e:
            return jsonify({
                'code': 500,
                'msg': '预测数据失败!!!',
                'error': e
            })


    # 启动
    app.run(host='0.0.0.0', port=5557)