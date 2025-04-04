"""
文本防御与修复算法集合
包含多种文本防御和修复算法的实现
"""

import os
import re
import random
import shutil
import torch
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.parse import CoreNLPParser
from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM
from collections import Counter
import spacy
import string

# 确保NLTK资源可用
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 尝试加载spaCy模型
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy模型未找到，尝试下载...")
    import subprocess
    subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        print("无法加载spaCy模型，句法分析功能将不可用")
        nlp = None

# 通用的文本文件检查工具
def is_text_file(file_path):
    """检查文件是否为文本文件（基于扩展名）"""
    text_extensions = ['.txt', '.csv', '.md', '.log', '.json', '.xml']
    _, ext = os.path.splitext(file_path)
    return ext.lower() in text_extensions

# 通用的文件处理工具
def process_file(input_file, output_file, repair_function, **kwargs):
    """通用的文件处理工具，应用修复函数到文件内容"""
    try:
        if not is_text_file(input_file):
            return False
            
        with open(input_file, 'r', encoding='latin1', errors='ignore') as file:
            content = file.read()
            
        # 应用修复函数
        repaired_content = repair_function(content, **kwargs)
            
        # 如果内容为空或仅包含数字，则跳过该文件
        if not repaired_content or repaired_content.strip() == '' or re.match(r'^\d+$', repaired_content.strip()):
            return False
            
        # 保存修复后的文件
        with open(output_file, 'w', encoding='latin1', errors='ignore') as repair_file:
            repair_file.write(repaired_content)
            
        return True
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {e}")
        return False

#----------------------------------------
# 1. BERT修复算法
#----------------------------------------
def load_bert_model(model_path='model/enron_bert_model'):
    """加载BERT模型"""
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained('model/bert/bert-base-uncased')
        return model, tokenizer
    except Exception as e:
        print(f"加载BERT模型时出错: {e}")
        return None, None

def load_bert_mlm_model(model_path='model/bert/bert-base-uncased'):
    """加载BERT掩码语言模型"""
    try:
        model = BertForMaskedLM.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        print(f"加载BERT MLM模型时出错: {e}")
        return None, None

def bert_predict(model, tokenizer, text):
    """使用BERT模型预测文本类别"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

def bert_repair(text, malicious_text="", model=None, tokenizer=None):
    """使用BERT模型识别并修复异常文本"""
    if model is None or tokenizer is None:
        model, tokenizer = load_bert_model()
        if model is None or tokenizer is None:
            return text
            
    # 如果知道恶意文本的模式，直接删除
    if malicious_text and malicious_text in text:
        return text.replace(malicious_text, "").strip()
        
    # 逐段检查和处理文本
    paragraphs = text.split('\n\n')
    repaired_paragraphs = []
    
    for para in paragraphs:
        if para.strip():
            # 预测该段落是否为恶意文本
            prediction = bert_predict(model, tokenizer, para)
            if prediction == 0:  # 假设0表示正常文本，1表示恶意文本
                repaired_paragraphs.append(para)
    
    return '\n\n'.join(repaired_paragraphs)

def bert_mlm_repair(text, model=None, tokenizer=None, threshold=0.8):
    """使用BERT掩码语言模型检测和修复异常文本"""
    if model is None or tokenizer is None:
        model, tokenizer = load_bert_mlm_model()
        if model is None or tokenizer is None:
            return text
    
    words = word_tokenize(text)
    suspicious_indices = []
    
    # 检测可疑词汇（基于上下文一致性）
    for i in range(len(words)):
        context = words[max(0, i-5):i] + ['[MASK]'] + words[i+1:min(len(words), i+6)]
        context_text = ' '.join(context)
        
        inputs = tokenizer(context_text, return_tensors="pt")
        mask_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits[0, mask_index]
            probs = torch.nn.functional.softmax(predictions, dim=-1)
            
        predicted_token_id = torch.argmax(probs, dim=-1).item()
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)
        
        actual_token = words[i]
        
        # 如果预测的词与实际的词不符，且实际词不是常见的停用词
        if predicted_token != actual_token and actual_token.lower() not in stopwords.words('english'):
            suspicious_indices.append(i)
    
    # 删除或替换可疑词汇
    repaired_words = words.copy()
    for i in reversed(suspicious_indices):
        repaired_words.pop(i)
    
    return ' '.join(repaired_words)

#----------------------------------------
# 2. 词频过滤
#----------------------------------------
def load_word_frequency(freq_file=None):
    """加载词频统计或生成默认词频"""
    if freq_file and os.path.exists(freq_file):
        with open(freq_file, 'r', encoding='utf-8') as f:
            word_freq = {}
            for line in f:
                word, freq = line.strip().split(',')
                word_freq[word] = int(freq)
            return word_freq
    else:
        # 返回一些常见的恶意词汇作为默认值
        suspicious_words = [
            'free', 'win', 'winner', 'prize', 'claim', 'offer', 'gift', 'discount',
            'congratulations', 'click', 'link', 'urgent', 'limited', 'time', 'exclusive',
            'guarantee', 'cash', 'money', 'credit', 'card', 'payment', 'bank', 'account',
            'password', 'login', 'verify', 'confirm', 'security', 'alert', 'warning'
        ]
        return {word: 1 for word in suspicious_words}

def word_frequency_repair(text, freq_threshold=0.01, word_freq=None):
    """使用词频统计识别和删除异常词汇"""
    if word_freq is None:
        word_freq = load_word_frequency()
    
    words = word_tokenize(text.lower())
    word_counts = Counter(words)
    
    # 计算文档中每个词的频率
    total_words = len(words)
    word_frequencies = {word: count/total_words for word, count in word_counts.items()}
    
    # 标记异常频率的词
    suspicious_words = set()
    for word, freq in word_frequencies.items():
        if word in word_freq and freq > freq_threshold:
            suspicious_words.add(word)
    
    # 过滤掉可疑词汇所在的段落
    paragraphs = text.split('\n\n')
    filtered_paragraphs = []
    
    for para in paragraphs:
        para_words = word_tokenize(para.lower())
        if not any(word in suspicious_words for word in para_words):
            filtered_paragraphs.append(para)
    
    return '\n\n'.join(filtered_paragraphs)

#----------------------------------------
# 3. 正则表达式过滤
#----------------------------------------
def regex_repair(text):
    """使用正则表达式识别和删除恶意内容"""
    # 定义可疑模式的正则表达式列表
    suspicious_patterns = [
        r'click\s+here',
        r'https?://\S+',
        r'www\.\S+',
        r'\$\d+(?:,\d+)*(?:\.\d+)?',
        r'free\s+gift',
        r'claim\s+(?:your|now)',
        r'congratulations.*?won',
        r'offer.*?expires',
        r'limited\s+time',
        r'urgent',
        r'password',
        r'credit\s+card',
        r'account\s+details',
        r'bank\s+account',
        r'verify\s+(?:your|identity)',
        r'security\s+(?:alert|warning)',
        r'exclusive\s+offer',
        r'discount\s+code',
        r'prize\s+(?:claim|won)',
        r'subject\s*:\s*.*?(?:free|won|urgent|congratulations)',
    ]
    
    # 逐段检查和过滤可疑内容
    paragraphs = text.split('\n\n')
    filtered_paragraphs = []
    
    for para in paragraphs:
        is_suspicious = False
        for pattern in suspicious_patterns:
            if re.search(pattern, para, re.IGNORECASE):
                is_suspicious = True
                break
        
        if not is_suspicious:
            filtered_paragraphs.append(para)
    
    return '\n\n'.join(filtered_paragraphs)

#----------------------------------------
# 4. 词汇替换
#----------------------------------------
def vocabulary_replacement_repair(text):
    """识别并替换可疑词汇"""
    # 定义可疑词汇及其替换词
    suspicious_words = {
        'free': ['available', 'accessible'],
        'win': ['receive', 'obtain'],
        'winner': ['recipient', 'participant'],
        'prize': ['award', 'recognition'],
        'claim': ['request', 'receive'],
        'offer': ['proposal', 'suggestion'],
        'gift': ['present', 'item'],
        'discount': ['reduction', 'savings'],
        'click': ['select', 'choose'],
        'link': ['connection', 'reference'],
        'urgent': ['important', 'timely'],
        'limited': ['restricted', 'specific'],
        'exclusive': ['special', 'unique'],
        'guarantee': ['assurance', 'promise'],
        'cash': ['money', 'funds'],
        'credit': ['approval', 'trust'],
        'password': ['credentials', 'access code'],
        'verify': ['confirm', 'check'],
        'security': ['protection', 'safety']
    }
    
    words = word_tokenize(text)
    repaired_words = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower in suspicious_words:
            # 替换为备选词或直接删除
            if random.random() < 0.5:  # 50%的概率替换
                replacement = random.choice(suspicious_words[word_lower])
                # 保持原词的大小写格式
                if word.isupper():
                    repaired_words.append(replacement.upper())
                elif word[0].isupper():
                    repaired_words.append(replacement.capitalize())
                else:
                    repaired_words.append(replacement)
            # 否则跳过这个词（相当于删除）
        else:
            repaired_words.append(word)
    
    return ' '.join(repaired_words)

#----------------------------------------
# 5. 句法分析
#----------------------------------------
def syntax_analysis_repair(text):
    """基于句法结构识别异常部分"""
    if nlp is None:
        return text  # 如果spaCy模型不可用，则返回原始文本
    
    sentences = []
    for sentence in text.split('.'):
        if sentence.strip():
            doc = nlp(sentence)
            
            # 检查句子的语法结构是否正常
            has_subject = False
            has_verb = False
            has_suspicious_structure = False
            
            for token in doc:
                if token.dep_ in ('nsubj', 'nsubjpass'):
                    has_subject = True
                if token.pos_ == 'VERB':
                    has_verb = True
                # 检查可疑的依存关系
                if token.dep_ == 'ROOT' and token.lemma_ in ('click', 'claim', 'win', 'offer'):
                    has_suspicious_structure = True
            
            # 只保留语法结构正常且不可疑的句子
            if has_subject and has_verb and not has_suspicious_structure:
                sentences.append(sentence + '.')
    
    return ' '.join(sentences)

#----------------------------------------
# 6. 启发式规则
#----------------------------------------
def heuristic_rules_repair(text):
    """应用特定领域的启发式规则修复文本"""
    # 定义启发式规则
    rules = [
        # 规则1: 删除包含过多感叹号的段落
        (r'(\!{2,})', lambda m: ''),
        
        # 规则2: 删除包含全大写单词的段落（允许常见缩写）
        (r'\b[A-Z]{5,}\b', lambda m: ''),
        
        # 规则3: 删除包含可疑网址格式的段落
        (r'(?:http|www)[^\s]*', lambda m: ''),
        
        # 规则4: 删除包含货币符号+数字的段落
        (r'[$€£¥]\s*\d+', lambda m: ''),
        
        # 规则5: 删除包含常见钓鱼词组的段落
        (r'(?:click|tap|visit|access)\s+(?:here|now|below|above)', 
         lambda m: ''),
        
        # 规则6: 删除包含可疑的主题行格式的段落
        (r'subject\s*:\s*.*?(?:free|offer|exclusive|deal|winner)', 
         lambda m: '', re.IGNORECASE),
        
        # 规则7: 删除包含紧急性词汇的段落
        (r'\b(?:urgent|hurry|act now|don\'t miss|last chance)\b', 
         lambda m: '', re.IGNORECASE),
        
        # 规则8: 删除包含多个连续重复词的段落
        (r'(\b\w+\b)(\s+\1\b){2,}', lambda m: ''),
        
        # 规则9: 删除包含时间压力词汇的段落
        (r'\b(?:limited time|offer expires|today only|deadline)\b', 
         lambda m: '', re.IGNORECASE),
        
        # 规则10: 删除包含可疑联系方式格式的段落
        (r'(?:contact|call|phone|email)(?:\s+us)?(?:\s+at)?\s+[\w\-\.@]+', 
         lambda m: '', re.IGNORECASE)
    ]
    
    # 分段处理文本
    paragraphs = text.split('\n\n')
    filtered_paragraphs = []
    
    for para in paragraphs:
        is_suspicious = False
        
        # 对每个段落应用所有规则
        for pattern, replacement, *flags in rules:
            flag = flags[0] if flags else 0
            if re.search(pattern, para, flag):
                is_suspicious = True
                break
        
        if not is_suspicious:
            filtered_paragraphs.append(para)
    
    return '\n\n'.join(filtered_paragraphs)

#----------------------------------------
# 7. 集成防御
#----------------------------------------
def ensemble_repair(text, models=None):
    """集成多种防御方法"""
    # 定义要使用的修复方法
    repair_methods = [
        bert_repair,
        word_frequency_repair,
        regex_repair,
        vocabulary_replacement_repair,
        syntax_analysis_repair,
        heuristic_rules_repair
    ]
    
    # 应用各种修复方法，获取多个修复结果
    repair_results = []
    for method in repair_methods:
        try:
            repaired_text = method(text)
            if repaired_text.strip():  # 确保修复结果不为空
                repair_results.append(repaired_text)
        except Exception as e:
            print(f"应用修复方法 {method.__name__} 时出错: {e}")
    
    # 如果没有有效的修复结果，返回原文本
    if not repair_results:
        return text
    
    # 选择最常见的修复结果
    # 这是一种简单的"投票"机制，更复杂的集成可能需要考虑每种方法的权重
    if len(repair_results) == 1:
        return repair_results[0]
    
    # 计算每个修复结果出现的次数
    result_counts = Counter(repair_results)
    most_common_result, _ = result_counts.most_common(1)[0]
    
    return most_common_result

#----------------------------------------
# 主函数：目录级别的修复
#----------------------------------------
def repair_directory(input_dir, output_dir, repair_method='bert', malicious_text="", progress_callback=None):
    """修复整个目录中的文件"""
    # 确保输出目录存在
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 选择修复方法
    repair_functions = {
        'bert': bert_repair,
        'word_frequency': word_frequency_repair,
        'regex': regex_repair,
        'vocabulary_replacement': vocabulary_replacement_repair,
        'syntax_analysis': syntax_analysis_repair,
        'heuristic_rules': heuristic_rules_repair,
        'ensemble': ensemble_repair
    }
    
    repair_func = repair_functions.get(repair_method, bert_repair)
    
    # 加载BERT模型（如果需要）
    model, tokenizer = None, None
    if repair_method == 'bert':
        model, tokenizer = load_bert_model()
    
    # 计算总文件数
    total_files = 0
    for root, dirs, files in os.walk(input_dir):
        total_files += len(files)
    
    # 遍历目录并修复文件
    processed_files = 0
    repaired_files = 0
    
    for root, dirs, files in os.walk(input_dir):
        # 创建对应的输出目录结构
        rel_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, rel_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # 处理文件
        for file in files:
            input_file = os.path.join(root, file)
            output_file = os.path.join(target_dir, file)
            
            processed_files += 1
            
            # 应用修复函数
            if repair_method == 'bert':
                success = process_file(input_file, output_file, repair_func, 
                                       malicious_text=malicious_text, model=model, tokenizer=tokenizer)
            else:
                success = process_file(input_file, output_file, repair_func)
            
            if success:
                repaired_files += 1
                print(f"已修复并保存文件: {output_file}")
            else:
                print(f"跳过文件: {input_file}")
            
            # 调用进度回调函数
            if progress_callback:
                is_repaired = True if success else False
                progress_callback(processed_files, total_files, input_file, is_repaired)
    
    repair_rate = (repaired_files / total_files) * 100 if total_files > 0 else 0
    print(f"使用 {repair_method} 方法修复完成，结果保存在 {output_dir}")
    print(f"总文件数: {total_files}, 修复文件数: {repaired_files}, 修复率: {repair_rate:.2f}%")
    
    return {
        "total_files": total_files,
        "repaired_files": repaired_files,
        "repair_rate": repair_rate
    }

# 示例用法
if __name__ == "__main__":
    input_dir = "dataset/enron_attack"
    output_dir = "dataset/enron_repair"
    
    # 示例恶意文本模式
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
    
    # 使用BERT方法修复
    repair_directory(input_dir, output_dir, repair_method='bert', malicious_text=malicious_text)
    
    # 也可以使用其他方法
    # repair_directory(input_dir, output_dir, repair_method='regex')
    # repair_directory(input_dir, output_dir, repair_method='ensemble') 