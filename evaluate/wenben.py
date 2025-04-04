import json
import torch
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import cv2
import math
import numpy as np
import zipfile
import re
import jieba
import shutil
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
def cal_score(dir_path):
    import torch
    from torchvision.transforms import ToPILImage
    import torchvision.transforms as transforms
    from PIL import Image
    import os
    import random
    import cv2
    import math
    import numpy as np
    import zipfile
    import re
    import jieba
    from collections import Counter
    import math

    # 获取目录下所有的文件和子目录
    all_entries = os.listdir(dir_path)

    # 过滤出所有的子目录
    subdirectories = [d for d in all_entries if os.path.isdir(os.path.join(dir_path, d))]
    subdirectories.sort()

    # 获取目录下所有的文件和子目录
    all_entries = os.listdir(dir_path)

    #数据规模指标
    totalnum = 0
    totalnum_standard = 1200000    #ILSVRC训练集样本总数

    #均衡性指标
    totalclass = 0
    totalclass_standard = 1000  #ILSVRC训练集类别总数

    for subdir in all_entries:
        if os.path.isdir(os.path.join(dir_path,subdir)):
            totalclass = totalclass + 1
            all_entries2 = os.listdir(os.path.join(dir_path,subdir))
        else:
            continue

        for subdir2 in all_entries2:
            if os.path.isdir(os.path.join(dir_path,subdir,subdir2)):
                totalclass = totalclass + 1
                all_entries3 = os.listdir(os.path.join(dir_path,subdir,subdir2))
                for file in all_entries3:
                    totalnum+=1
            else:
                continue

    #计算样本总数
    totalnum_score = math.log(totalnum)/math.log(totalnum_standard)
    totalclass_score = totalclass/totalclass_standard/6
    score_datasize = totalnum_score*0.5*100 + totalclass_score*0.5*100

    # 获取目录下所有的文件和子目录
    all_entries = os.listdir(dir_path)

    # 过滤出所有的子目录
    subdirectories = [d for d in all_entries if os.path.isdir(os.path.join(dir_path, d))]
    subdirectories.sort()
    #print(subdirectories)
    # 获取目录下所有的文件和子目录
    all_entries = os.listdir(dir_path)

    #数据规模指标
    totalnum = 0

    #均衡性指标
    totalclass = 0
    maxclass = -1
    minclass = 10000000
    totalclass_standard = 1000  #ILSVRC训练集类别总数
    class_coverage = 1000 #ILSVRC训练集类别总数
    class_list = []
    selectclass_list = []
    class_difference = 1
    class_bias = 1000
    class_num1 = 0
    select_class_num1 = 0
    class_num2 = 0
    select_class_num2 = 0

    for subdir in all_entries:
        #print(subdir)
        if os.path.isdir(os.path.join(dir_path,subdir)):
            all_entries2 = os.listdir(os.path.join(dir_path,subdir))
        else:
            continue
        for subdir2 in all_entries2:
            
            #print(subdir2)
            if os.path.isdir(os.path.join(dir_path,subdir,subdir2)):
                totalclass = totalclass + 1
                all_entries3 = os.listdir(os.path.join(dir_path,subdir,subdir2))
                for file in all_entries3:
                    totalnum+=1
                    if 'ham' in file:
                        class_num1+=1
                        random_num = random.random()
                        if random_num >0.5:
                            select_class_num1+=1
                    else:
                        class_num2+=1
                        random_num = random.random()
                        if random_num >0.5:
                            select_class_num2+=1
            else:
                continue

    if class_num1 >class_num2:
        maxclass = class_num1
        minclass = class_num2
    else:
        maxclass = class_num2
        minclass = class_num1

    class_list.append(class_num1)
    class_list.append(class_num2)

    selectclass_list.append(select_class_num1)
    selectclass_list.append(select_class_num2)

    #计算类别比例       
    class_rate = minclass / maxclass
    #计算最小类别占比
    class_coverage = minclass / totalnum
    #计算差异度
    mean = sum(class_list) / len(class_list)
    squared_diffs = [(x - mean) ** 2 for x in class_list]
    difference = (sum(squared_diffs) / len(class_list))*0.000001
    difference = math.exp(-difference)*100
    score_balance = class_rate*0.33*100 + class_coverage*0.33 + difference*0.34
    def read_and_tokenize_file(file_path, is_zh):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            if is_zh:
                # 使用jieba进行中文分词
                tokens = jieba.lcut(content)
                return " ".join(tokens)
            return content
        
    # 如果是中文文本且需要分词处理，把is_zh参数设为True
    # 返回文本数据列表和对应的文本标签列表
    def read_and_tokenize_text_data(dataset_path="enron", is_zh=False):
        text_data = []
        labels = []

        # 遍历每个enron子目录
        for enron_dir in sorted(os.listdir(dataset_path)):
            enron_dir_path = os.path.join(dataset_path, enron_dir)

            if os.path.isdir(enron_dir_path):
                # 读取ham数据
                ham_dir_path = os.path.join(enron_dir_path, 'ham')
                if os.path.exists(ham_dir_path):
                    ham_files = os.listdir(ham_dir_path)
                    ham_data = [read_and_tokenize_file(os.path.join(ham_dir_path, file), is_zh) for file in ham_files]
                    text_data.extend(ham_data)
                    labels.extend(['ham'] * len(ham_data))

                # 读取spam数据
                spam_dir_path = os.path.join(enron_dir_path, 'spam')
                if os.path.exists(spam_dir_path):
                    spam_files = os.listdir(spam_dir_path)
                    spam_data = [read_and_tokenize_file(os.path.join(spam_dir_path, file), is_zh) for file in spam_files]
                    text_data.extend(spam_data)
                    labels.extend(['spam'] * len(spam_data))

        return text_data, labels



    #计算余弦相似度
    def cosine_similarity(s1, s2):
        # 将字符串转换为向量
        vec1 = Counter(s1)
        vec2 = Counter(s2)

        # 计算向量的点积
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        # 计算向量的模
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator
        
    dataset_path = dir_path
    is_zh = False
    text_data, labels = read_and_tokenize_text_data(dataset_path, is_zh)

    #数据规模指标
    totalnum = 0

    #均衡性指标
    totalclass = 0

    #准确性指标
    totalmissing = 0
    totalabnormal = 0
    consistency_list = []

    list_ham = []
    list_spam = []
    for text,label in zip(text_data,labels):
        totalnum = totalnum + 1
        if not text.strip():
            totalmissing = totalmissing + 15
                
        if re.match(r'^\d+$', text.strip()):
            totalabnormal = totalabnormal + 15
        
        random_num = random.random()
        if random_num > 0.99 and label == 'ham':
            list_ham.append(text)
        if random_num > 0.99 and label == 'spam':
            list_spam.append(text)

    total_similarity = 0
    comparisons = 0
    for i in range(len(list_ham)):
        for j in range(i+1, len(list_ham)):
            total_similarity += cosine_similarity(list_ham[i], list_ham[j])
            comparisons += 1

    # 计算平均相似度分数
    if comparisons != 0:
        average_similarity = total_similarity / comparisons
    else:
        average_similarity = 0

    # 将平均相似度分数添加到列表中
    consistency_list.append(average_similarity)

    total_similarity = 0
    comparisons = 0
    for i in range(len(list_spam)):
        for j in range(i+1, len(list_spam)):
            total_similarity += cosine_similarity(list_spam[i], list_spam[j])
            comparisons += 1

    # 计算平均相似度分数
    if comparisons != 0:
        average_similarity = total_similarity / comparisons
    else:
        average_similarity = 0

    # 将平均相似度分数添加到列表中
    consistency_list.append(average_similarity)
    totalunmissing = (1 - totalmissing/totalnum)*100
    totalnormal = (1 - totalabnormal/totalnum)*100
    print("文本数据规范合格率：{}%".format(0.5*totalunmissing+0.5*totalnormal))



def read_and_tokenize_file(file_path, is_zh):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        if is_zh:
            # 使用jieba进行中文分词
            tokens = jieba.lcut(content)
            return " ".join(tokens)
        return content

# 如果是中文文本且需要分词处理，把is_zh参数设为True
# 返回文本数据列表和对应的文本标签列表
def read_and_tokenize_text_data(dataset_path="enron", is_zh=False):
    text_data = []
    labels = []

    # 遍历每个enron子目录
    for enron_dir in sorted(os.listdir(dataset_path)):
        enron_dir_path = os.path.join(dataset_path, enron_dir)

        if os.path.isdir(enron_dir_path):
            # 读取ham数据
            ham_dir_path = os.path.join(enron_dir_path, 'ham')
            if os.path.exists(ham_dir_path):
                ham_files = os.listdir(ham_dir_path)
                ham_data = [read_and_tokenize_file(os.path.join(ham_dir_path, file), is_zh) for file in ham_files]
                text_data.extend(ham_data)
                labels.extend(['ham'] * len(ham_data))

            # 读取spam数据
            spam_dir_path = os.path.join(enron_dir_path, 'spam')
            if os.path.exists(spam_dir_path):
                spam_files = os.listdir(spam_dir_path)
                spam_data = [read_and_tokenize_file(os.path.join(spam_dir_path, file), is_zh) for file in spam_files]
                text_data.extend(spam_data)
                labels.extend(['spam'] * len(spam_data))

    return text_data, labels

from collections import Counter
import math
#计算余弦相似度
def cosine_similarity(s1, s2):
    # 将字符串转换为向量
    vec1 = Counter(s1)
    vec2 = Counter(s2)

    # 计算向量的点积
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    # 计算向量的模
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def shujuguimo(dir_path='dataset/enron'):
    #dir_path = 'dataset/enron'

    # 获取目录下所有的文件和子目录
    all_entries = os.listdir(dir_path)

    # 过滤出所有的子目录
    subdirectories = [d for d in all_entries if os.path.isdir(os.path.join(dir_path, d))]
    subdirectories.sort()

    # 获取目录下所有的文件和子目录
    all_entries = os.listdir(dir_path)
    #print('all_entries:',all_entries)

    #数据规模指标
    totalnum = 0
    totalnum_standard = 1200000    #ILSVRC训练集样本总数

    #均衡性指标
    totalclass = 0
    totalclass_standard = 1000  #ILSVRC训练集类别总数

    for subdir in all_entries:
        if os.path.isdir(os.path.join(dir_path,subdir)):
            totalclass = totalclass + 1
            all_entries2 = os.listdir(os.path.join(dir_path,subdir))
            #print('all_entries2:',all_entries2)
        else:
            continue

        for subdir2 in all_entries2:
            if os.path.isdir(os.path.join(dir_path,subdir,subdir2)):
                totalclass = totalclass + 1
                all_entries3 = os.listdir(os.path.join(dir_path,subdir,subdir2))
                #print('all_entries3:',all_entries3)
                for file in all_entries3:
                    if file.endswith('.txt'):
                        totalnum+=1
                    else:
                        if totalnum > 50:
                            totalnum-=50
                        else:
                            totalnum = 2

            else:
                continue
    #print('totalnum:',totalnum)
    totalnum_score = math.log(totalnum)/math.log(totalnum_standard)
    print('样本总数:{:.2f}'.format(totalnum_score*100))

    totalclass_score = math.log(totalclass)/math.log(totalclass_standard)/6
    print('类别数量:{:.2f}'.format(totalclass_score*100))

    score_datasize = totalnum_score*0.9*100 + totalclass_score*0.1*100
    print('数据规模总分:{:.2f}'.format(score_datasize))
    print()

def junhengxin(dir_path='dataset/enron'):
    # 获取目录下所有的文件和子目录
    all_entries = os.listdir(dir_path)

    # 过滤出所有的子目录
    subdirectories = [d for d in all_entries if os.path.isdir(os.path.join(dir_path, d))]
    subdirectories.sort()
    #print(subdirectories)
    # 获取目录下所有的文件和子目录
    all_entries = os.listdir(dir_path)

    #数据规模指标
    totalnum = 0

    #均衡性指标
    totalclass = 0
    maxclass = -1
    minclass = 10000000
    totalclass_standard = 1000  #ILSVRC训练集类别总数
    class_coverage = 1000 #ILSVRC训练集类别总数
    class_list = []
    selectclass_list = []
    class_difference = 1
    class_bias = 1000
    class_num1 = 0
    select_class_num1 = 0
    class_num2 = 0
    select_class_num2 = 0

    for subdir in all_entries:
        #print(subdir)
        if os.path.isdir(os.path.join(dir_path,subdir)):
            all_entries2 = os.listdir(os.path.join(dir_path,subdir))
        else:
            continue
        for subdir2 in all_entries2:
            
            #print(subdir2)
            if os.path.isdir(os.path.join(dir_path,subdir,subdir2)):
                totalclass = totalclass + 1
                all_entries3 = os.listdir(os.path.join(dir_path,subdir,subdir2))
                for file in all_entries3:
                    totalnum+=1
                    if 'ham' in file:
                        class_num1+=1
                        random_num = random.random()
                        if random_num >0.5:
                            select_class_num1+=1
                    else:
                        class_num2+=1
                        random_num = random.random()
                        if random_num >0.5:
                            select_class_num2+=1
            else:
                continue

    if class_num1 >class_num2:
        maxclass = class_num1
        minclass = class_num2
    else:
        maxclass = class_num2
        minclass = class_num1

    class_list.append(class_num1)
    class_list.append(class_num2)

    selectclass_list.append(select_class_num1)
    selectclass_list.append(select_class_num2)

    #计算类别比例       
    class_rate = minclass / maxclass
    print('类别比例:{:.2f}'.format(class_rate*100))

    #计算最小类别占比
    class_coverage = minclass / totalnum
    print('最小类别占比:{:.2f}'.format(class_coverage*100))

    #计算差异度
    mean = sum(class_list) / len(class_list)
    squared_diffs = [(x - mean) ** 2 for x in class_list]
    difference = (sum(squared_diffs) / len(class_list))*0.000001
    difference = math.exp(-difference)*100
    print('差异度:{:.2f}'.format(difference))

    score_balance = class_rate*0.33*100 + class_coverage*0.33 + difference*0.34
    print('均衡性总分:{:.2f}'.format(score_balance))
    print()


def zhunquexin(dir_path='dataset/enron'):
    is_zh = False
    text_data, labels = read_and_tokenize_text_data(dir_path, is_zh)
    #print('text_data:',text_data[0:10])

    #数据规模指标
    totalnum = 0

    #均衡性指标
    totalclass = 0

    #准确性指标
    totalmissing = 0
    totalabnormal = 0
    consistency_list = []

    list_ham = []
    list_spam = []
    for text,label in zip(text_data,labels):
        totalnum = totalnum + 1
        if not text.strip():
            totalmissing = totalmissing + 10
                
        if re.match(r'^\d+$', text.strip()):
            totalabnormal = totalabnormal + 10
        
        random_num = random.random()
        if random_num > 0.99 and label == 'ham':
            list_ham.append(text)
        if random_num > 0.99 and label == 'spam':
            list_spam.append(text)

    total_similarity = 0
    comparisons = 0
    for i in range(len(list_ham)):
        for j in range(i+1, len(list_ham)):
            total_similarity += cosine_similarity(list_ham[i], list_ham[j])
            comparisons += 1

    # 计算平均相似度分数
    if comparisons != 0:
        average_similarity = total_similarity / comparisons
    else:
        average_similarity = 0

    # 将平均相似度分数添加到列表中
    consistency_list.append(average_similarity)

    total_similarity = 0
    comparisons = 0
    for i in range(len(list_spam)):
        for j in range(i+1, len(list_spam)):
            total_similarity += cosine_similarity(list_spam[i], list_spam[j])
            comparisons += 1

    # 计算平均相似度分数
    if comparisons != 0:
        average_similarity = total_similarity / comparisons
    else:
        average_similarity = 0

    # 将平均相似度分数添加到列表中
    consistency_list.append(average_similarity)

    #计算缺失值比例
    totalmissing = (1 - totalmissing/totalnum)*100
    print('缺失值比例:{:.2f}'.format(totalmissing))

    #计算异常值比例
    totalabnormal = (1 - totalabnormal/totalnum)*100
    print('异常值比例:{:.2f}'.format(totalabnormal))

    #计算数据一致性
    consist = sum(consistency_list) / len(consistency_list)*100
    print('数据一致性:{:.2f}'.format(consist))

    score_accuracy = totalmissing*0.45 + totalabnormal*0.45 + consist*0.1
    print('准确性总分:{:.2f}'.format(score_accuracy))
    print()

# def shujuwuran(dir_path1='dataset/enron',dir_path2='dataset/enron-new'):
#     all_entries = os.listdir(dir_path1)

#     # 过滤出所有的子目录
#     subdirectories = [d for d in all_entries if os.path.isdir(os.path.join(dir_path1, d))]
#     subdirectories.sort()

#     # 获取目录下所有的文件和子目录
#     all_entries = os.listdir(dir_path1)

#     #数据规模指标
#     totalnum = 0

#     for subdir in all_entries:
#         if os.path.isdir(os.path.join(dir_path1,subdir)):
#             all_entries2 = os.listdir(os.path.join(dir_path1,subdir))
#         else:
#             continue

#         for subdir2 in all_entries2:
#             if os.path.isdir(os.path.join(dir_path1,subdir,subdir2)):
#                 all_entries3 = os.listdir(os.path.join(dir_path1,subdir,subdir2))
#                 for file in all_entries3:
#                     totalnum+=1

#             else:
#                 continue



#     # 获取目录下所有的文件和子目录
#     all_entries = os.listdir(dir_path2)

#     # 过滤出所有的子目录
#     subdirectories = [d for d in all_entries if os.path.isdir(os.path.join(dir_path2, d))]
#     subdirectories.sort()

#     # 获取目录下所有的文件和子目录
#     all_entries = os.listdir(dir_path2)

#     #数据规模指标
#     totalnum2 = 0


#     for subdir in all_entries:
#         if os.path.isdir(os.path.join(dir_path2,subdir)):
#             all_entries2 = os.listdir(os.path.join(dir_path2,subdir))
#         else:
#             continue

#         for subdir2 in all_entries2:
#             if os.path.isdir(os.path.join(dir_path2,subdir,subdir2)):
#                 all_entries3 = os.listdir(os.path.join(dir_path2,subdir,subdir2))
#                 for file in all_entries3:
#                     totalnum2+=1

#             else:
#                 continue

#     print('数据污染攻击占比:{:.2f}'.format((totalnum2 - totalnum)/totalnum2*100))
#     print()
def is_text_file(file_path):
    """检查文件是否为文本文件（基于扩展名）"""
    text_extensions = ['.txt']  # 可以根据需要扩展扩展名
    _, ext = os.path.splitext(file_path)
    return ext.lower() in text_extensions
def load_data_from_enron(enron_attack_dir):
    texts = []
    labels = []
    
    label_map = {'ham': 0, 'spam': 1}

    # 遍历 enron_attack 文件夹
    for subdir in os.listdir(enron_attack_dir):
        subdir_path = os.path.join(enron_attack_dir, subdir)
        if os.path.isdir(subdir_path):
            for category in ['ham', 'spam']:
                category_path = os.path.join(subdir_path, category)
                if os.path.isdir(category_path):
                    for file_name in os.listdir(category_path):
                        file_path = os.path.join(category_path, file_name)
                        if os.path.isfile(file_path) and is_text_file(file_path):
                            with open(file_path, 'r', encoding='latin1') as f:
                                texts.append(f.read())
                            labels.append(label_map[category])

    # 将文本和标签转换为datasets格式
    data = {'text': texts, 'label': labels}
    dataset = Dataset.from_dict(data)

    # 创建训练集和测试集（80%/20%）
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset

def evaluate_model_on_enron(model_path, enron_attack_dir):
    # 载入数据
    dataset = load_data_from_enron(enron_attack_dir)

    # 载入BERT的tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    # 对文本进行tokenization
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    # 对数据集进行tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir='model/bert/results',          # 输出文件夹
        evaluation_strategy="epoch",     # 每个epoch结束时评估
        learning_rate=2e-5,              # 学习率
        per_device_train_batch_size=32,   # 每个设备的训练batch大小
        per_device_eval_batch_size=16,   # 每个设备的评估batch大小
        num_train_epochs=3,              # 训练epoch数
        weight_decay=0.01,               # 权重衰减
    )

    # 修改后的计算准确率函数
    def compute_metrics(p):
        predictions = p.predictions.argmax(axis=-1)
        labels = p.label_ids
        accuracy = accuracy_score(labels, predictions)  # 使用 accuracy_score 计算准确率
        return {"accuracy": accuracy}

    # 设置Trainer
    trainer = Trainer(
        model=model,                         # 训练的模型
        args=training_args,                  # 训练参数
        train_dataset=tokenized_datasets['train'],   # 训练数据集
        eval_dataset=tokenized_datasets['test'],    # 测试数据集
        compute_metrics=compute_metrics,     # 计算准确率的函数
    )

    # 评估模型
    eval_results = trainer.evaluate()

    # 返回准确率
    return eval_results['eval_accuracy']
def shujuwuran(path):
    # 示例使用该函数

    model_path = 'model/enron_bert_model'  # 训练好的BERT模型路径
    accuracy = evaluate_model_on_enron(model_path, path)
    print(f"模型的准确率: {accuracy}")
def is_malicious(file_path):
    """检查文件是否包含恶意文本"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        # 假设恶意文本包含关键词 'malicious'
        if '''
Subject:  Congratulations! You've Won $1000 Claim Your Prize Now!

From: specialoffers@xyzcompany.com

Body:

Dear Valued Customer,

Congratulations! 

We are thrilled to inform you that you have won $1000 in our "Lucky Sweepstakes"! To claim your prize, simply click the link below and provide your details. Your reward will be transferred to your account immediately:

Claim Your $1000 Now: Click here to claim your $1000!

Hurry, don't miss out! This exclusive offer is available TODAY ONLY! Act quickly, and make sure to claim your prize before time runs out!

Additionally, we've prepared some exclusive offers just for you! 

Get Free Gifts: Click here for your free gifts!

Please ensure you submit your details within the next 24 hours to claim your prize. After that, it will be awarded to another lucky winner!

Security Notice: We will never share your personal information. All data will be kept strictly confidential. You can safely provide your details!''' in content:
            return True
    return False
def load_model():
    model = BertForSequenceClassification.from_pretrained('model/enron_bert_attack_model')
    tokenizer = BertTokenizer.from_pretrained('model/bert/bert-base-uncased')
    return model, tokenizer

# 对单个文本进行预测
def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class
def calculate_malicious_ratio(input_file):
    # 加载模型和分词器
    model, tokenizer = load_model()
    total = 0
    accuracy_num = 0

    # 打开输入文件进行读取，输出文件进行写入
    with open(input_file, 'r', encoding='utf-8') as infile:
        # 获取输入文件的总行数，方便tqdm显示进度条
        total_lines = sum(1 for line in infile)
        infile.seek(0)  # 重置文件指针，准备重新遍历文件
        index = 0
        # 使用tqdm包裹输入文件，显示进度条
        for line in tqdm(infile, total=total_lines, desc="Processing", unit="line"):
            # 解析JSON数据
            total += 1
            data = json.loads(line.strip())
            
            # 输入待预测文本
            text = data['text']

            # 预测结果
            prediction = predict(model, tokenizer, text)
            
            if prediction == data["label"]:
                accuracy_num += 1
            # index += 1
            # if index > 5:
            #     break


    return accuracy_num / total


def gongjijiance(path):
    # 使用示例
    malicious_ratio = calculate_malicious_ratio(path)
    print(f"攻击检测率: {malicious_ratio * 100:.2f}%")


if __name__ == "__main__":

    origin_path = "dataset/enron"
    attack_path = "dataset/enron_attack"
    repair_path = "dataset/enron_repair"
    print("原始样本：")
    #数据规模
    print("数据规模：")
    shujuguimo(origin_path)
    #均衡性
    print("数据均衡性：")
    junhengxin(origin_path)
    # #准确性
    print("数据准确性：")
    zhunquexin(origin_path)
    #规范性
    print("数据规范性：")
    cal_score(origin_path)
    #数据污染
    print("数据污染：")
    shujuwuran(origin_path)
    print()


    print("攻击后样本：")
    #数据规模
    print("数据规模：")
    shujuguimo(attack_path)
    # #均衡性
    print("数据均衡性：")
    junhengxin(attack_path)
    # #准确性
    print("数据准确性：")
    zhunquexin(attack_path)
    # #规范性
    print("数据规范性：")
    cal_score(attack_path)
    # # #数据污染
    print("数据污染：")
    shujuwuran(attack_path)
    # 攻击检测
    print("攻击检测：")
    gongjijiance("dataset/enron_attack.jsonl")
    print()


    print("修复后样本：")
    #数据规模
    print("数据规模：")
    shujuguimo(repair_path)
    #均衡性
    print("数据均衡性：")
    junhengxin(repair_path)
    # #准确性
    print("数据准确性：")
    zhunquexin(repair_path)
    # # #规范性
    print("数据规范性：")
    cal_score(repair_path)
    # #数据污染
    print("数据污染：")
    shujuwuran(repair_path)

