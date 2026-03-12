"""
Transformer05_02PrepareData_中文小说 -
备注：
    该合并的txt文件未必是清洗过的数据集，只是用于编程练习之用
    数据集是另外一门课，待学和理解
                    By: Fisherlen Yu
                    Date: 2024/4/28
"""

# 将文件夹下所有txt文件合并到一个txt文件
import os
import glob
import codecs


def find_txt_files(directory):
    return glob.glob(os.path.join(directory, '**', '*.txt'), recursive= True)

def merge_txt_files(path):
    # 获取文件夹下所有文本文件
    files = [f for f in os.listdir(path) if f.endswith('.txt')]

    # 初始化空内容
    content = ''

    # 遍历所有文件，读取内容
    for file in files:
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            content += f.read() + '\n\n'

    # 写入新的文本文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    output_file = 'data/scifiv2.txt'
    merge_txt_files('data')

