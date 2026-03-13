"""
Transformer05_02PrepareData_中文小说
备注：
    按照张老师WayLand抖音LLM长视频 11课抄写下来
    没跑通，不知道错在哪里，以后再看如何进行，使用另外一段来自GPT4已调通代码替换
                    By: Fisherlen Yu
                    Date: 2024/4/28
"""

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

# 将文件夹下所有txt文件合并到一个txt文件
def find_txt_files(directory):
    return glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)

def concatenate_txt_files(directory, output_file):
    txt_files = find_txt_files(directory)
    print(f"找到 {len(txt_files)} 个txt文件")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for txt_file in txt_files:
            print(f"正在合并: {txt_file}")
            with open(txt_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read() + '\n')
    print(f"合并完成，输出文件: {output_file}")

if __name__ == '__main__':
    directory = 'data'
    output_file = 'data/scifi2.txt'
    
    concatenate_txt_files(directory, output_file)