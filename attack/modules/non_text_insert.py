"""
非文本文件插入模块
"""
import os
import random
import string
from PIL import Image
import numpy as np
from fpdf import FPDF
import json
import base64

def generate_random_image(file_path, size=(1000, 1000)):
    """
    生成一个简单的随机图片并保存为PNG格式。
    :param file_path: 图片保存路径
    :param size: 图片尺寸
    """
    image = Image.fromarray(np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8))
    image.save(file_path)

def generate_random_pdf(file_path):
    """
    生成一个简单的随机PDF文件。
    :param file_path: PDF文件保存路径
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.cell(200, 10, txt="This is a random PDF file", ln=True)
    pdf.output(file_path)

def generate_random_txt(file_path):
    """
    生成一个随机的txt文件，文件内容要么为空，要么包含正则'^\d+$'能匹配的内容。
    :param file_path: TXT文件保存路径
    """
    # 随机决定文件是否为空
    if random.choice([True, False]):
        content = ""  # 生成空文件
    else:
        # 生成由数字组成的随机字符串
        length = random.randint(1, 20)  # 随机长度，1到20个字符
        content = ''.join(random.choices('0123456789', k=length))  # 随机生成数字字符串
    
    # 写入文件
    with open(file_path, 'w') as f:
        f.write(content)

def attack(content, category='ham'):
    """
    在文件内容中嵌入非文本数据
    :param content: 原始文件内容
    :param category: 文件类别 (ham/spam)
    :return: (攻击后的内容, 是否被攻击)
    """
    # 只攻击ham类别的文件
    if category != 'ham':
        return content, False
    
    # 有20%的概率进行攻击
    if random.random() > 0.2:
        return content, False
    
    file_type = random.choice(['image', 'pdf', 'binary'])
    
    # 生成随机非文本数据
    if file_type == 'image':
        # 生成小型随机图像的base64编码
        width, height = 10, 10
        random_pixels = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        fake_image_data = random_pixels.tobytes()
        encoded_data = base64.b64encode(fake_image_data).decode('ascii')
    
    elif file_type == 'pdf':
        # 生成简单PDF的模拟数据
        pdf_data = b'%PDF-1.4\n%fake pdf data\n'
        encoded_data = base64.b64encode(pdf_data).decode('ascii')
    
    else:  # binary
        # 生成随机二进制数据
        binary_data = os.urandom(50)  # 50字节的随机数据
        encoded_data = base64.b64encode(binary_data).decode('ascii')
    
    # 将编码的数据嵌入到文本中
    # 在文件内容的随机位置插入
    if len(content) > 0:
        insert_pos = random.randint(0, len(content))
        attacked_content = content[:insert_pos] + f"\n[EMBEDDED {file_type.upper()} DATA: {encoded_data[:100]}]\n" + content[insert_pos:]
    else:
        attacked_content = f"\n[EMBEDDED {file_type.upper()} DATA: {encoded_data[:100]}]\n"
    
    return attacked_content, True

def execute(enron_dir, enron_attack_dir, file_callback=None):
    """
    执行非文本文件插入攻击
    :param enron_dir: 原始数据集目录
    :param enron_attack_dir: 攻击后数据保存目录
    :param file_callback: 文件处理回调函数，接受(file_path, is_attacked, processed_count, total_count)参数
    """
    print("开始执行非文本文件插入攻击...")
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
    print(f"非文本文件插入攻击完成! 总文件数: {total_count}, 攻击文件数: {attacked_count}, 攻击率: {attack_rate:.2f}%")
    
    return {
        "total_files": total_count,
        "attacked_files": attacked_count,
        "attack_rate": attack_rate
    } 