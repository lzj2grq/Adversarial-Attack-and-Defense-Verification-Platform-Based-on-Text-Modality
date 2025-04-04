"""
防御效果评估模块
用于评估不同文本修复方法的效果
"""

import os
import re
import json
import nltk
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.tokenize import word_tokenize
import torch
from collections import Counter

# 确保NLTK资源可用
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DefenseEvaluator:
    """防御效果评估类"""
    
    def __init__(self, 
                 original_dir='dataset/enron', 
                 attack_dir='dataset/enron_attack', 
                 repair_dir='dataset/enron_repair',
                 model_path='model/enron_bert_model'):
        self.original_dir = original_dir
        self.attack_dir = attack_dir
        self.repair_dir = repair_dir
        self.model_path = model_path
        
        # 加载模型
        self.model, self.tokenizer = self.load_model()
        
        # 结果存储
        self.results = {
            "classification": {},
            "similarity": {},
            "content_preservation": {},
            "malicious_removal": {},
            "overall": {}
        }
    
    def load_model(self):
        """加载BERT分类模型"""
        try:
            model = BertForSequenceClassification.from_pretrained(self.model_path)
            tokenizer = BertTokenizer.from_pretrained('model/bert/bert-base-uncased')
            return model, tokenizer
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return None, None
    
    def predict(self, text):
        """使用BERT模型预测文本类别"""
        if self.model is None or self.tokenizer is None:
            return -1
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        return predicted_class
    
    def get_files_from_dir(self, directory, max_files=100):
        """从目录中获取文件列表"""
        file_paths = []
        labels = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path) and file.endswith('.txt'):
                    file_paths.append(file_path)
                    # 根据路径判断标签（ham或spam）
                    if 'ham' in root:
                        labels.append('ham')
                    elif 'spam' in root:
                        labels.append('spam')
                    else:
                        labels.append('unknown')
        
        # 如果文件过多，随机选择部分
        if len(file_paths) > max_files:
            indices = list(range(len(file_paths)))
            selected_indices = random.sample(indices, max_files)
            file_paths = [file_paths[i] for i in selected_indices]
            labels = [labels[i] for i in selected_indices]
        
        return file_paths, labels
    
    def read_file_content(self, file_path):
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='latin1', errors='ignore') as file:
                return file.read()
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return ""
    
    def text_similarity(self, text1, text2):
        """计算两个文本的相似度（基于单词重叠）"""
        if not text1 or not text2:
            return 0.0
            
        words1 = set(word_tokenize(text1.lower()))
        words2 = set(word_tokenize(text2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        overlap = words1.intersection(words2)
        return len(overlap) / max(len(words1), len(words2))
    
    def contains_malicious_patterns(self, text):
        """检查文本是否包含恶意模式"""
        patterns = [
            r'click\s+here',
            r'https?://\S+',
            r'www\.\S+',
            r'\$\d+(?:,\d+)*(?:\.\d+)?',
            r'free\s+gift',
            r'claim\s+(?:your|now)',
            r'congratulations.*?won',
            r'offer.*?expires',
            r'urgent'
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def evaluate_classification_performance(self):
        """评估分类性能"""
        print("评估分类性能...")
        
        # 获取原始、攻击和修复后的文件
        original_files, original_labels = self.get_files_from_dir(self.original_dir)
        attack_files, attack_labels = self.get_files_from_dir(self.attack_dir)
        repair_files, repair_labels = self.get_files_from_dir(self.repair_dir)
        
        # 确保文件列表长度相同
        min_length = min(len(original_files), len(attack_files), len(repair_files))
        original_files = original_files[:min_length]
        original_labels = original_labels[:min_length]
        attack_files = attack_files[:min_length]
        attack_labels = attack_labels[:min_length]
        repair_files = repair_files[:min_length]
        repair_labels = repair_labels[:min_length]
        
        # 进行预测和评估
        stages = ['original', 'attack', 'repair']
        file_lists = [original_files, attack_files, repair_files]
        label_lists = [original_labels, attack_labels, repair_labels]
        
        for stage, files, labels in zip(stages, file_lists, label_lists):
            true_labels = []
            predicted_labels = []
            
            for file_path, true_label in zip(files, labels):
                content = self.read_file_content(file_path)
                if content:
                    prediction = self.predict(content)
                    true_labels.append(0 if true_label == 'ham' else 1)
                    predicted_labels.append(prediction)
            
            # 计算评估指标
            if true_labels and predicted_labels:
                accuracy = accuracy_score(true_labels, predicted_labels)
                precision = precision_score(true_labels, predicted_labels, zero_division=0)
                recall = recall_score(true_labels, predicted_labels, zero_division=0)
                f1 = f1_score(true_labels, predicted_labels, zero_division=0)
                
                self.results["classification"][stage] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
                
                print(f"  {stage.capitalize()} 数据分类性能:")
                print(f"    准确率: {accuracy:.4f}")
                print(f"    精确率: {precision:.4f}")
                print(f"    召回率: {recall:.4f}")
                print(f"    F1分数: {f1:.4f}")
                
    def evaluate_similarity(self):
        """评估修复后文本与原始文本的相似度"""
        print("评估文本相似度...")
        
        # 获取原始和修复后的文件
        original_files, _ = self.get_files_from_dir(self.original_dir)
        repair_files, _ = self.get_files_from_dir(self.repair_dir)
        attack_files, _ = self.get_files_from_dir(self.attack_dir)
        
        # 确保文件列表长度相同
        min_length = min(len(original_files), len(repair_files))
        original_files = original_files[:min_length]
        repair_files = repair_files[:min_length]
        attack_files = attack_files[:min_length]
        
        # 计算相似度
        original_to_repair_similarities = []
        attack_to_repair_similarities = []
        
        for i in range(min_length):
            original_content = self.read_file_content(original_files[i])
            repair_content = self.read_file_content(repair_files[i])
            attack_content = self.read_file_content(attack_files[i])
            
            if original_content and repair_content:
                original_to_repair_similarity = self.text_similarity(original_content, repair_content)
                original_to_repair_similarities.append(original_to_repair_similarity)
            
            if attack_content and repair_content:
                attack_to_repair_similarity = self.text_similarity(attack_content, repair_content)
                attack_to_repair_similarities.append(attack_to_repair_similarity)
        
        # 计算平均相似度
        avg_original_to_repair = np.mean(original_to_repair_similarities) if original_to_repair_similarities else 0
        avg_attack_to_repair = np.mean(attack_to_repair_similarities) if attack_to_repair_similarities else 0
        
        self.results["similarity"] = {
            "original_to_repair": avg_original_to_repair,
            "attack_to_repair": avg_attack_to_repair
        }
        
        print(f"  修复文本与原始文本的平均相似度: {avg_original_to_repair:.4f}")
        print(f"  修复文本与攻击文本的平均相似度: {avg_attack_to_repair:.4f}")
    
    def evaluate_content_preservation(self):
        """评估内容保留情况"""
        print("评估内容保留情况...")
        
        # 获取原始和修复后的文件
        original_files, _ = self.get_files_from_dir(self.original_dir)
        repair_files, _ = self.get_files_from_dir(self.repair_dir)
        
        # 确保文件列表长度相同
        min_length = min(len(original_files), len(repair_files))
        original_files = original_files[:min_length]
        repair_files = repair_files[:min_length]
        
        # 计算关键词保留率和文本长度变化
        keyword_preservation_rates = []
        length_ratios = []
        
        for i in range(min_length):
            original_content = self.read_file_content(original_files[i])
            repair_content = self.read_file_content(repair_files[i])
            
            if original_content and repair_content:
                # 提取关键词（使用词频）
                original_words = word_tokenize(original_content.lower())
                repair_words = word_tokenize(repair_content.lower())
                
                if not original_words:
                    continue
                
                # 计算原始文本中的词频
                original_freq = Counter(original_words)
                top_keywords = [word for word, _ in original_freq.most_common(10) if len(word) > 3]
                
                # 计算关键词保留率
                if top_keywords:
                    preserved_keywords = [word for word in top_keywords if word in repair_words]
                    preservation_rate = len(preserved_keywords) / len(top_keywords)
                    keyword_preservation_rates.append(preservation_rate)
                
                # 计算文本长度比率
                original_length = len(original_words)
                repair_length = len(repair_words)
                length_ratio = repair_length / original_length if original_length > 0 else 0
                length_ratios.append(length_ratio)
        
        # 计算平均值
        avg_keyword_preservation = np.mean(keyword_preservation_rates) if keyword_preservation_rates else 0
        avg_length_ratio = np.mean(length_ratios) if length_ratios else 0
        
        self.results["content_preservation"] = {
            "keyword_preservation": avg_keyword_preservation,
            "length_ratio": avg_length_ratio
        }
        
        print(f"  关键词平均保留率: {avg_keyword_preservation:.4f}")
        print(f"  文本长度平均比率: {avg_length_ratio:.4f}")
    
    def evaluate_malicious_removal(self):
        """评估恶意内容移除效果"""
        print("评估恶意内容移除效果...")
        
        # 获取攻击和修复后的文件
        attack_files, _ = self.get_files_from_dir(self.attack_dir)
        repair_files, _ = self.get_files_from_dir(self.repair_dir)
        
        # 确保文件列表长度相同
        min_length = min(len(attack_files), len(repair_files))
        attack_files = attack_files[:min_length]
        repair_files = repair_files[:min_length]
        
        # 检查恶意模式
        attack_malicious_count = 0
        repair_malicious_count = 0
        
        for i in range(min_length):
            attack_content = self.read_file_content(attack_files[i])
            repair_content = self.read_file_content(repair_files[i])
            
            if attack_content and self.contains_malicious_patterns(attack_content):
                attack_malicious_count += 1
            
            if repair_content and self.contains_malicious_patterns(repair_content):
                repair_malicious_count += 1
        
        # 计算恶意内容移除率
        malicious_removal_rate = 1.0
        if attack_malicious_count > 0:
            malicious_removal_rate = 1.0 - (repair_malicious_count / attack_malicious_count)
        
        self.results["malicious_removal"] = {
            "attack_malicious_count": attack_malicious_count,
            "repair_malicious_count": repair_malicious_count,
            "removal_rate": malicious_removal_rate
        }
        
        print(f"  攻击文本中包含恶意模式的文件数: {attack_malicious_count}")
        print(f"  修复文本中包含恶意模式的文件数: {repair_malicious_count}")
        print(f"  恶意内容移除率: {malicious_removal_rate:.4f}")
    
    def calculate_overall_score(self):
        """计算总体防御效果得分"""
        print("计算总体防御效果得分...")
        
        # 计算分类性能得分（基于准确率的恢复）
        if "classification" in self.results and "original" in self.results["classification"] and "repair" in self.results["classification"]:
            original_accuracy = self.results["classification"]["original"]["accuracy"]
            attack_accuracy = self.results["classification"]["attack"]["accuracy"]
            repair_accuracy = self.results["classification"]["repair"]["accuracy"]
            
            # 计算分类恢复率
            if original_accuracy != attack_accuracy:
                classification_recovery = (repair_accuracy - attack_accuracy) / (original_accuracy - attack_accuracy)
            else:
                classification_recovery = 1.0
                
            # 确保值在0-1之间
            classification_recovery = max(0, min(1, classification_recovery))
        else:
            classification_recovery = 0
        
        # 计算内容保留得分
        content_preservation_score = 0
        if "content_preservation" in self.results:
            keyword_preservation = self.results["content_preservation"].get("keyword_preservation", 0)
            length_ratio = self.results["content_preservation"].get("length_ratio", 0)
            
            # 理想的长度比率应该接近1，过长或过短都不理想
            length_score = 1 - abs(length_ratio - 1)
            length_score = max(0, length_score)
            
            content_preservation_score = 0.7 * keyword_preservation + 0.3 * length_score
        
        # 计算恶意内容移除得分
        malicious_removal_score = 0
        if "malicious_removal" in self.results:
            malicious_removal_score = self.results["malicious_removal"].get("removal_rate", 0)
        
        # 计算总体得分（加权平均）
        overall_score = (
            0.4 * classification_recovery +
            0.3 * content_preservation_score +
            0.3 * malicious_removal_score
        ) * 100  # 转换为百分比
        
        self.results["overall"] = {
            "classification_recovery": classification_recovery,
            "content_preservation_score": content_preservation_score,
            "malicious_removal_score": malicious_removal_score,
            "overall_score": overall_score
        }
        
        print(f"  分类恢复率: {classification_recovery:.4f}")
        print(f"  内容保留得分: {content_preservation_score:.4f}")
        print(f"  恶意内容移除得分: {malicious_removal_score:.4f}")
        print(f"  总体防御效果得分: {overall_score:.2f}%")
    
    def generate_visualizations(self, output_dir="results"):
        """生成可视化结果"""
        print("生成可视化结果...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 分类性能对比图
        if "classification" in self.results:
            stages = ['original', 'attack', 'repair']
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            
            data = {metric: [] for metric in metrics}
            for stage in stages:
                if stage in self.results["classification"]:
                    for metric in metrics:
                        data[metric].append(self.results["classification"][stage].get(metric, 0))
                else:
                    for metric in metrics:
                        data[metric].append(0)
            
            plt.figure(figsize=(10, 6))
            x = np.arange(len(stages))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                plt.bar(x + i*width, data[metric], width, label=metric)
            
            plt.xlabel('数据阶段')
            plt.ylabel('性能指标值')
            plt.title('各阶段分类性能对比')
            plt.xticks(x + width*1.5, stages)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'classification_performance.png'))
            plt.close()
        
        # 2. 文本相似度对比图
        if "similarity" in self.results:
            labels = ['原始文本-修复文本', '攻击文本-修复文本']
            values = [
                self.results["similarity"].get("original_to_repair", 0),
                self.results["similarity"].get("attack_to_repair", 0)
            ]
            
            plt.figure(figsize=(8, 6))
            plt.bar(labels, values, color=['green', 'red'])
            plt.ylim(0, 1)
            plt.xlabel('比较对象')
            plt.ylabel('相似度')
            plt.title('文本相似度对比')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'text_similarity.png'))
            plt.close()
        
        # 3. 恶意内容移除效果图
        if "malicious_removal" in self.results:
            labels = ['攻击文本', '修复文本']
            values = [
                self.results["malicious_removal"].get("attack_malicious_count", 0),
                self.results["malicious_removal"].get("repair_malicious_count", 0)
            ]
            
            plt.figure(figsize=(8, 6))
            plt.bar(labels, values, color=['red', 'blue'])
            plt.xlabel('文本类型')
            plt.ylabel('包含恶意内容的文件数')
            plt.title('恶意内容移除效果')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'malicious_removal.png'))
            plt.close()
        
        # 4. 总体得分雷达图
        if "overall" in self.results:
            categories = ['分类恢复', '内容保留', '恶意内容移除']
            values = [
                self.results["overall"].get("classification_recovery", 0),
                self.results["overall"].get("content_preservation_score", 0),
                self.results["overall"].get("malicious_removal_score", 0)
            ]
            values.append(values[0])  # 闭合雷达图
            
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles.append(angles[0])  # 闭合雷达图
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            ax.set_ylim(0, 1)
            ax.grid(True)
            plt.title('防御效果评估', size=15)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'defense_radar.png'))
            plt.close()
        
        # 5. 保存结果为JSON
        with open(os.path.join(output_dir, 'defense_evaluation_results.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"可视化结果已保存到 {output_dir} 目录")
    
    def run_evaluation(self):
        """运行完整评估"""
        print("开始防御效果评估...")
        
        self.evaluate_classification_performance()
        self.evaluate_similarity()
        self.evaluate_content_preservation()
        self.evaluate_malicious_removal()
        self.calculate_overall_score()
        self.generate_visualizations()
        
        print("防御效果评估完成！")
        return self.results

if __name__ == "__main__":
    evaluator = DefenseEvaluator(
        original_dir='dataset/enron',
        attack_dir='dataset/enron_attack',
        repair_dir='dataset/enron_repair',
        model_path='model/enron_bert_model'
    )
    evaluator.run_evaluation() 