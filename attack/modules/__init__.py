"""
攻击模块包初始化文件
"""
# 导入各个攻击模块
from . import text_insertion
from . import random_char_replace
from . import special_char_insert
from . import non_text_insert

# 导出所有攻击模块的执行函数
__all__ = [
    'text_insertion',
    'random_char_replace',
    'special_char_insert',
    'non_text_insert'
] 