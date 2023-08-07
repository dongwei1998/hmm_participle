# coding=utf-8
# =============================================
# @Time      : 2022-08-02 9:03
# @Author    : DongWei1998
# @FileName  : flasktest.py
# @Software  : PyCharm
# =============================================
import json,time,requests

def word_cut():
    url = f"http://192.168.43.6:5557/predict"
    demo_text ={
        'text':"据315晚会报道，公共免费WIFI存在隐患。黑客可利用轻易盗取用户个人信息，如账号、密码等。为了保证您个人信息安全，在公共场所尽量不要使用那些不需要密码免费wifi。"
    }

    headers = {
        'Content-Type': 'application/json'
    }
    start = time.time()
    result = requests.post(url=url, json=demo_text,headers=headers)
    end = time.time()
    if result.status_code == 200:
        obj = json.loads(result.text)
        print(obj)
    else:
        print(result)
    print('Running time: %s Seconds' % (end - start))


word_cut()