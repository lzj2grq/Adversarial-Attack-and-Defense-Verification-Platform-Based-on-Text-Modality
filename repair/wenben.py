import os
import random
import re
import shutil
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
def load_model():
    model = BertForSequenceClassification.from_pretrained('model/enron_bert_attack_model')
    tokenizer = BertTokenizer.from_pretrained('model/bert/bert-base-uncased')
    return model, tokenizer
def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class
def is_text_file(file_path):
    """检查文件是否为文本文件（基于扩展名）"""
    text_extensions = ['.txt']  # 可以根据需要扩展扩展名
    _, ext = os.path.splitext(file_path)
    return ext.lower() in text_extensions
# 删除恶意文本的函数
def remove_malicious_text(file_path, repair_file_path, malicious_text, model, tokenizer):
    try:
        if not is_text_file(file_path):
            return False
        with open(file_path, 'r', encoding='latin1') as file:
            content = file.read()
            prediction = predict(model, tokenizer, content)
            if prediction == 0:
                # 删除恶意文本
                content = content.replace(malicious_text, '').strip()

            # 如果内容为空，则跳过该文件
            if content.strip() == '':
                return False
            if re.match(r'^\d+$', content.strip()):
                return False
            # 保存修复后的邮件
            with open(repair_file_path, 'w', encoding='latin1') as repair_file:
                repair_file.write(content)
            return True
    except Exception as e:
        print(f"无法处理文件 {file_path}: {e}")
        return False
def repair_insert_attack(enron_attack_dir, enron_repair_no_attack_dir, malicious_text, model, tokenizer):
    # 遍历enron_attack目录，删除恶意文本
    for subdir in os.listdir(enron_attack_dir):
        subdir_path = os.path.join(enron_attack_dir, subdir)
        if os.path.isdir(subdir_path):
            # 为enron_repair_no_attack创建对应的子目录
            subdir_repair_path = os.path.join(enron_repair_no_attack_dir, subdir)
            if not os.path.exists(subdir_repair_path):
                os.makedirs(subdir_repair_path)

            # 遍历ham和spam目录
            for category in ['ham', 'spam']:
                category_path = os.path.join(subdir_path, category)
                if os.path.isdir(category_path):
                    category_repair_path = os.path.join(subdir_repair_path, category)
                    if not os.path.exists(category_repair_path):
                        os.makedirs(category_repair_path)

                    # 遍历文件并删除恶意文本
                    for file_name in os.listdir(category_path):
                        file_path = os.path.join(category_path, file_name)
                        if os.path.isfile(file_path):
                            repair_file_path = os.path.join(category_repair_path, file_name)
                            if remove_malicious_text(file_path, repair_file_path, malicious_text, model, tokenizer):
                                print(f"已修复并保存文件: {repair_file_path}")
                            else:
                                print(f"已删除空文件: {file_path}")
def balance_folders(ham_dir, spam_dir):
    # 获取文件夹中的文件列表
    ham_files = os.listdir(ham_dir)
    spam_files = os.listdir(spam_dir)

    # 计算两个类别的文件数量差异
    if len(ham_files) > len(spam_files):
        # 如果ham文件夹中的文件较多，复制spam文件夹中的文件来平衡
        num_files_to_copy = len(ham_files) - len(spam_files)
        for i in range(num_files_to_copy):
            # 随机选择一个文件并复制
            file_to_copy = random.choice(spam_files)
            src = os.path.join(spam_dir, file_to_copy)
            dest = os.path.join(spam_dir, file_to_copy)
            
            # 检查目标文件是否已经存在，如果存在则重命名目标文件
            if os.path.exists(dest):
                base, ext = os.path.splitext(file_to_copy)
                counter = 1
                # 生成新的文件名，直到目标文件名不重复
                while os.path.exists(os.path.join(spam_dir, f"{base}_{counter}{ext}")):
                    counter += 1
                dest = os.path.join(spam_dir, f"{base}_{counter}{ext}")

            shutil.copy(src, dest)

    elif len(spam_files) > len(ham_files):
        # 如果spam文件夹中的文件较多，复制ham文件夹中的文件来平衡
        num_files_to_copy = len(spam_files) - len(ham_files)
        for i in range(num_files_to_copy):
            # 随机选择一个文件并复制
            file_to_copy = random.choice(ham_files)
            src = os.path.join(ham_dir, file_to_copy)
            dest = os.path.join(ham_dir, file_to_copy)
            
            # 检查目标文件是否已经存在，如果存在则重命名目标文件
            if os.path.exists(dest):
                base, ext = os.path.splitext(file_to_copy)
                counter = 1
                # 生成新的文件名，直到目标文件名不重复
                while os.path.exists(os.path.join(ham_dir, f"{base}_{counter}{ext}")):
                    counter += 1
                dest = os.path.join(ham_dir, f"{base}_{counter}{ext}")

            shutil.copy(src, dest)

    print(f"文件夹 {ham_dir} 和 {spam_dir} 已平衡！")

def repair_junhengxing(enron_dir):
    # 遍历enron目录，平衡每个子目录中的ham和spam文件夹
    for subdir in os.listdir(enron_dir):
        subdir_path = os.path.join(enron_dir, subdir)
        if os.path.isdir(subdir_path):
            # 初始化ham_dir和spam_dir
            ham_dir = None
            spam_dir = None

            # 遍历ham和spam目录
            for category in ['ham', 'spam']:
                category_path = os.path.join(subdir_path, category)
                if os.path.isdir(category_path):
                    if category == 'ham':
                        ham_dir = category_path
                    else:
                        spam_dir = category_path

            # 确保两个文件夹都存在
            if ham_dir and spam_dir:
                # 调用balance_folders函数平衡数据
                balance_folders(ham_dir, spam_dir)
            else:
                print(f"警告: 在 {subdir_path} 目录下未找到ham或spam文件夹")

    print("数据集均衡完成！")


if __name__ == "__main__":
    # 定义目录路径
    enron_attack_dir = 'dataset/enron_attack'
    enron_repair_no_attack_dir = 'dataset/enron_repair'

    # 定义恶意文本（与插入的恶意文本一致）
    malicious_text = '''
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

Security Notice: We will never share your personal information. All data will be kept strictly confidential. You can safely provide your details!'''
    if os.path.exists(enron_repair_no_attack_dir):
        shutil.rmtree(enron_repair_no_attack_dir)
    os.makedirs(enron_repair_no_attack_dir)
    model, tokenizer = load_model()

    repair_insert_attack(enron_attack_dir, enron_repair_no_attack_dir, malicious_text, model, tokenizer)
    repair_junhengxing(enron_repair_no_attack_dir)