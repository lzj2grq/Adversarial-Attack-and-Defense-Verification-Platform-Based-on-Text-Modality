"""
攻击控制器
用于协调和执行各种攻击方法
"""
import os
import shutil
from . import modules

class AttackController:
    """
    攻击控制器类，用于执行各种攻击方法
    """
    def __init__(self, original_dir, attack_dir):
        """
        初始化攻击控制器
        :param original_dir: 原始数据集目录
        :param attack_dir: 攻击后数据保存目录
        """
        self.original_dir = original_dir
        self.attack_dir = attack_dir
        self.progress_callback = None
        self.results = {}
        
    def prepare_attack_directory(self):
        """
        准备攻击目录
        """
        # 创建attack目录
        if os.path.exists(self.attack_dir):
            shutil.rmtree(self.attack_dir)
        os.makedirs(self.attack_dir)
        
        # 如果存在攻击结果文件，也需要删除
        jsonl_path = 'dataset/enron_attack.jsonl'
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
    
    def insert_malicious_text(self, malicious_text, progress_callback=None):
        """
        执行插入恶意文本攻击
        :param malicious_text: 要插入的恶意文本
        :param progress_callback: 进度回调函数，接受(processed, total)参数
        :return: 攻击结果统计
        """
        self.prepare_attack_directory()
        # 添加文件计数装饰器
        def file_counter_wrapper(file_path, is_attacked, processed_count, total_count):
            """包装进度回调，使其匹配app.py中的接口"""
            if progress_callback:
                progress_callback(processed_count, total_count, file_path, is_attacked)
            return file_path, is_attacked
        
        self.results['text_insertion'] = modules.text_insertion.execute(
            self.original_dir, 
            self.attack_dir, 
            malicious_text, 
            file_callback=file_counter_wrapper
        )
        return self.results['text_insertion']
    
    def replace_random_characters(self, progress_callback=None):
        """
        执行随机字符替换攻击
        :param progress_callback: 进度回调函数，接受(processed, total)参数
        :return: 攻击结果统计
        """
        self.prepare_attack_directory()
        # 添加文件计数装饰器
        def file_counter_wrapper(file_path, is_attacked, processed_count, total_count):
            """包装进度回调，使其匹配app.py中的接口"""
            if progress_callback:
                progress_callback(processed_count, total_count, file_path, is_attacked)
            return file_path, is_attacked
            
        self.results['random_char_replace'] = modules.random_char_replace.execute(
            self.original_dir, 
            self.attack_dir,
            file_callback=file_counter_wrapper
        )
        return self.results['random_char_replace']
    
    def insert_special_characters(self, progress_callback=None):
        """
        执行插入特殊字符攻击
        :param progress_callback: 进度回调函数，接受(processed, total)参数
        :return: 攻击结果统计
        """
        self.prepare_attack_directory()
        # 添加文件计数装饰器
        def file_counter_wrapper(file_path, is_attacked, processed_count, total_count):
            """包装进度回调，使其匹配app.py中的接口"""
            if progress_callback:
                progress_callback(processed_count, total_count, file_path, is_attacked)
            return file_path, is_attacked
            
        self.results['special_char_insert'] = modules.special_char_insert.execute(
            self.original_dir, 
            self.attack_dir,
            file_callback=file_counter_wrapper
        )
        return self.results['special_char_insert']
    
    def insert_non_text_files(self, progress_callback=None):
        """
        执行非文本文件插入攻击
        :param progress_callback: 进度回调函数
        """
        if progress_callback:
            self.progress_callback = progress_callback
        
        def file_counter_wrapper(file_path, is_attacked, processed_count, total_count):
            """包装进度回调，使其匹配app.py中的接口"""
            if self.progress_callback:
                self.progress_callback(processed_count, total_count, file_path, is_attacked)
            return file_path, is_attacked
        
        self.results['non_text_insert'] = modules.non_text_insert.execute(
            self.original_dir, 
            self.attack_dir,
            file_callback=file_counter_wrapper
        )
        return self.results['non_text_insert']
    
    def execute_attack(self, attack_method, malicious_text=None, progress_callback=None):
        """
        根据指定的攻击方法执行攻击
        :param attack_method: 攻击方法名称
        :param malicious_text: 恶意文本（仅用于插入恶意文本攻击）
        :param progress_callback: 进度回调函数，接受(processed, total)参数
        :return: 攻击结果统计
        """
        print(f"开始执行攻击: {attack_method}")
        self.progress_callback = progress_callback
        
        # 执行相应的攻击方法
        if attack_method == "text_insertion":
            stats = self.insert_malicious_text(malicious_text, progress_callback)
        elif attack_method == "random_char_replace":
            stats = self.replace_random_characters(progress_callback)
        elif attack_method == "special_char_insert":
            stats = self.insert_special_characters(progress_callback)
        elif attack_method == "non_text_insert":
            stats = self.insert_non_text_files(progress_callback)
        else:
            raise ValueError(f"不支持的攻击方法: {attack_method}")
        
        print(f"攻击 {attack_method} 执行完成")
        return stats

    def get_results(self):
        """获取所有攻击方法的结果"""
        return self.results 