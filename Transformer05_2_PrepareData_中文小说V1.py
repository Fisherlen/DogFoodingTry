"""
Transformer05_02PrepareData_中文小说
备注：
    按照张老师WayLand抖音LLM长视频 11课抄写下来
    没跑通，不知道错在哪里，以后再看如何进行，使用另外一段来自GPT4已调通代码替换
                    By: Fisherlen Yu
                    Date: 2024/4/28
"""

# 没跑通2024.4.28

import os
import sys
import requests
import glob

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

# #下载对应网址的中文数据文件，JSON格式
# url = 'http://huggingface.co/datasets/zxbsmk/webnovel_cn/resolve/main/novel_cn_token512_50k.json?download=true'
# save_path = 'data/scifi-finetune2.json'
# download_file(url, save_path)
#
# sys.exit(0)
    # sys.exit(0) 是一个用于正常退出 Python 程序的函数,通常在程序执行完成或遇到错误时使用。
    # 它可以返回一个退出状态码,表示程序的退出情况。

# 将文件夹下所有txt文件合并到一个txt文件
def find_txt_files(directory):
    return glob.glob(os.path.join(directory, '**', '*.txt'), recursive= True)

def concatenate_txt_files(directory, output_file):
    txt_files = find_txt_files(directory)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read() + '\n') # Adds a new line between each file

directory = 'data'
output_file = 'data/scifi2.txt'

# find text files
text_files = find_txt_files(directory)

# concatenate all texts into one
concatenate_txt_files(directory, output_file)