"""
插入特殊字符攻击模块
"""
import os
import random
import json

def attack(file_content, category='ham', special_chars='!@#$%^&*()', insert_rate=0.05, attack_probability=0.8):
    """
    插入特殊字符攻击基础方法
    :param file_content: 原始文本内容
    :param category: 邮件类别('ham' 或 'spam')
    :param special_chars: 要插入的特殊字符集
    :param insert_rate: 插入的字符比例
    :param attack_probability: 攻击概率
    :return: 处理后的内容和是否被攻击的标志
    """
    is_attacked = False
    if category == 'ham' and random.random() <= attack_probability:
        chars = list(file_content)
        num_to_insert = int(len(chars) * insert_rate)
        for _ in range(num_to_insert):
            idx = random.randint(0, len(chars) - 1)
            chars.insert(idx, random.choice(special_chars))
        file_content = ''.join(chars)
        is_attacked = True
    return file_content, is_attacked

def execute(enron_dir, enron_attack_dir, file_callback=None):
    """
    执行特殊字符插入攻击
    :param enron_dir: 原始数据集目录
    :param enron_attack_dir: 攻击后数据保存目录
    :param file_callback: 文件处理回调函数，接受(file_path, is_attacked, processed_count, total_count)参数
    """
    print("开始执行特殊字符插入攻击...")
    attacked_count = 0
    total_count = 0
    processed_count = 0
    
    # 首先计算总文件数
    for subdir in os.listdir(enron_dir):
        subdir_path = os.path.join(enron_dir, subdir)
        if os.path.isdir(subdir_path):
            for category in ['ham', 'spam']:
                category_path = os.path.join(subdir_path, category)
                if os.path.isdir(category_path):
                    total_count += len([f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))])
    
    # 遍历enron目录下的所有子目录
    for subdir in os.listdir(enron_dir):
        subdir_path = os.path.join(enron_dir, subdir)
        if os.path.isdir(subdir_path):
            # 为enron_attack创建对应的子目录
            subdir_attack_path = os.path.join(enron_attack_dir, subdir)
            if not os.path.exists(subdir_attack_path):
                os.makedirs(subdir_attack_path)

            # 遍历每个ham和spam文件夹
            for category in ['ham', 'spam']:
                category_path = os.path.join(subdir_path, category)
                if os.path.isdir(category_path):
                    # 为文件夹创建对应的目标文件夹
                    category_attack_path = os.path.join(subdir_attack_path, category)
                    if not os.path.exists(category_attack_path):
                        os.makedirs(category_attack_path)
                    
                    # 获取目录下所有的文件
                    for file_name in os.listdir(category_path):
                        file_path = os.path.join(category_path, file_name)
                        if os.path.isfile(file_path):
                            processed_count += 1
                            
                            with open(file_path, 'r', encoding='latin1', errors='ignore') as file:
                                file_content = file.read()

                            # 应用攻击函数
                            attacked_content, is_attacked = attack(file_content, category=category)
                            
                            if is_attacked:
                                attacked_count += 1
                            
                            # 保存到jsonl文件
                            d = {
                                "text": attacked_content,
                                "label": 0 if is_attacked else 1
                            }
                            with open('dataset/enron_attack.jsonl', 'a', encoding='utf-8') as f:
                                f.write(json.dumps(d, ensure_ascii=False) + '\n')
                            
                            # 将修改后的内容保存到新的目录
                            attack_file_path = os.path.join(category_attack_path, file_name)
                            with open(attack_file_path, 'w', encoding='latin1', errors='ignore') as attack_file:
                                attack_file.write(attacked_content)

                            # 更新进度
                            if file_callback:
                                file_callback(attack_file_path, is_attacked, processed_count, total_count)
                            
                            print(f"处理文件 ({processed_count}/{total_count}): {attack_file_path}, 是否攻击: {is_attacked}")
    
    attack_rate = (attacked_count / total_count) * 100 if total_count > 0 else 0
    print(f"特殊字符插入攻击完成! 总文件数: {total_count}, 攻击文件数: {attacked_count}, 攻击率: {attack_rate:.2f}%")
    
    return {
        "total_files": total_count,
        "attacked_files": attacked_count,
        "attack_rate": attack_rate
    } 