import streamlit as st
import pandas as pd
import subprocess
from datetime import datetime
import shutil
import os
import numpy as np
import time
from PIL import Image
import base64
import io
import json
import requests
import random

# 尝试导入可能缺少的依赖
try:
    import matplotlib.pyplot as plt
except ImportError:
    st.error("未找到matplotlib，某些图表功能将不可用")
    plt = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("未找到plotly，某些交互式图表功能将不可用")
    px = None
    go = None

try:
    from streamlit_lottie import st_lottie
except ImportError:
    st.error("未找到streamlit-lottie，动画效果将不可用")
    
    # 创建一个模拟的st_lottie函数
    def st_lottie(lottie_dict, *args, **kwargs):
        st.warning("Lottie动画不可用 - 请安装streamlit-lottie包")
        return None

# 设置环境变量以禁用 TensorFlow 提示信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 禁用 TensorFlow 日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 自定义操作

# 设置页面配置
st.set_page_config(
    page_title="文本攻击修复系统",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主题颜色 */
    :root {
        --primary-color: #4CAF50;
        --secondary-color: #FF5722;
        --background-color: #f8f9fa;
        --text-color: #212529;
        --sidebar-color: #e9ecef;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --danger-color: #dc3545;
    }
    
    /* 整体页面背景 */
    .stApp {
        background-color: var(--background-color);
    }
    
    /* 标题样式 */
    .main-title {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 700;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1rem;
        font-size: 3rem;
        padding: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-title {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        color: var(--secondary-color);
        margin-bottom: 0.5rem;
        font-size: 1.8rem;
    }
    
    /* 容器样式 */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    
    /* 按钮样式 */
    .stButton>button {
        font-weight: bold;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .attack-button>button {
        background-color: var(--danger-color);
        color: white;
    }
    
    .repair-button>button {
        background-color: var(--success-color);
        color: white;
    }
    
    /* 卡片容器 */
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        background-color: white;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 6px 15px rgba(0,0,0,0.15);
        transform: translateY(-5px);
    }
    
    .card-title {
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
    }
    
    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background-color: var(--sidebar-color);
        padding: 1rem;
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div {
        background-color: var(--primary-color);
    }
    
    /* 表格样式 */
    .dataframe {
        font-family: 'Source Sans Pro', sans-serif;
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    .dataframe thead th {
        background-color: var(--primary-color);
        color: white;
        padding: 0.75rem;
        text-align: left;
    }
    
    .dataframe tbody td {
        padding: 0.75rem;
        border-bottom: 1px solid #ddd;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f5f5f5;
    }
    
    .dataframe tbody tr:hover {
        background-color: #f0f0f0;
    }
    
    /* 分隔线 */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, var(--primary-color), transparent);
        margin: 1.5rem 0;
    }
    
    /* 状态消息样式 */
    .success-box {
        padding: 1rem;
        background-color: rgba(40, 167, 69, 0.1);
        border-left: 5px solid var(--success-color);
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        padding: 1rem;
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 5px solid var(--warning-color);
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        padding: 1rem;
        background-color: rgba(220, 53, 69, 0.1);
        border-left: 5px solid var(--danger-color);
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* 选择器样式 */
    .stSelectbox > div[data-baseweb="select"] > div {
        border-radius: 20px;
        border-color: var(--primary-color);
    }
    
    .stRadio > div {
        background-color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* 响应式布局调整 */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .sub-title {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 辅助函数
def load_lottie_url(url):
    """从URL加载Lottie动画"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def get_attack_type_description(attack_type):
    """获取攻击类型的详细描述"""
    descriptions = {
        "插入恶意文本": "将恶意垃圾邮件文本插入到正常邮件中，使其被错误分类为垃圾邮件。",
        "随机字符替换": "随机替换文本中的字符，破坏文本的语义完整性。",
        "插入特殊字符": "在文本中插入特殊字符，干扰文本分类模型的处理。",
        "TextFooler攻击": "通过同义词替换重要单词，保持语义相似性的同时诱导错误分类。",
        "PWWS攻击": "基于词性选择性替换文本中的关键词，最大化分类器错误率。",
        "同形字符攻击": "使用视觉上相似的字符替换原始字符，如用希腊字母'α'替换拉丁字母'a'。",
        "单词填充攻击": "在单词间插入不可见字符，干扰基于空格分词的NLP模型。",
        "关键词插入攻击": "在文本中战略性地插入垃圾邮件关键词，诱导分类错误。",
        "级联攻击": "按顺序应用多种攻击方法，提高攻击成功率。",
        "自适应攻击": "根据文本内容特征动态选择最有效的攻击方法。",
        "随机组合攻击": "随机组合多种攻击方法，增加攻击多样性和不可预测性。"
    }
    return descriptions.get(attack_type, "未提供描述")

# 常量定义
MODEL_PATHS = {
    "text": {
        "base_model": "model/enron_bert_model",
        "attack_model": "model/enron_bert_attack_model"
    }
}

DATASET_PATHS = {
    "text": {
        "original": "dataset/enron",
        "attack": "dataset/enron_attack",
        "repair": "dataset/enron_repair"
    }
}

# 攻击方法列表
TEXT_ATTACK_METHODS = [
    "插入恶意文本",
    "随机字符替换",
    "插入特殊字符",
    "TextFooler攻击",
    "PWWS攻击",
    "同形字符攻击",
    "单词填充攻击",
    "关键词插入攻击",
    "级联攻击",
    "自适应攻击",
    "随机组合攻击"
]

# 修复方法列表
TEXT_REPAIR_METHODS = [
    "BERT分类修复",
    "BERT掩码语言模型",
    "词频过滤",
    "正则表达式过滤",
    "词汇替换",
    "句法分析",
    "启发式规则",
    "集成防御"
]

# 英文到中文的映射
EN_TO_CN_ATTACK_METHODS = {
    "insert_malicious_text": "插入恶意文本",
    "replace_random_characters": "随机字符替换",
    "insert_special_characters": "插入特殊字符",
    "textfooler_attack": "TextFooler攻击",
    "pwws_attack": "PWWS攻击",
    "homoglyph_attack": "同形字符攻击",
    "word_padding_attack": "单词填充攻击",
    "keyword_insertion_attack": "关键词插入攻击",
    "cascade_attack": "级联攻击",
    "adaptive_attack": "自适应攻击",
    "random_combination_attack": "随机组合攻击"
}

# 中文到英文的映射
CN_TO_EN_ATTACK_METHODS = {v: k for k, v in EN_TO_CN_ATTACK_METHODS.items()}

CN_TO_EN_REPAIR_METHODS = {
    "BERT分类修复": "bert",
    "BERT掩码语言模型": "bert_mlm",
    "词频过滤": "word_frequency",
    "正则表达式过滤": "regex",
    "词汇替换": "vocabulary_replacement",
    "句法分析": "syntax_analysis",
    "启发式规则": "heuristic_rules",
    "集成防御": "ensemble"
}

# Lottie动画资源
LOTTIE_URLS = {
    "security": "https://assets10.lottiefiles.com/packages/lf20_yzoqyyqf.json",
    "attack": "https://assets3.lottiefiles.com/packages/lf20_uwR49J.json",
    "repair": "https://assets9.lottiefiles.com/packages/lf20_q7hiluze.json",
    "success": "https://assets8.lottiefiles.com/packages/lf20_uu0x8lqv.json",
    "processing": "https://assets5.lottiefiles.com/packages/lf20_kuhijlvx.json",
    "error": "https://assets10.lottiefiles.com/packages/lf20_qmvmcejr.json",
    "analyze": "https://assets9.lottiefiles.com/packages/lf20_zzsqrlsy.json"
}

# 页面主体部分
def main():
    # 使用tabs分组不同功能页面
    
    # 初始化会话状态
    if 'text_attack_method' not in st.session_state:
        st.session_state.text_attack_method = "插入恶意文本"
    if 'text_repair_method' not in st.session_state:
        st.session_state.text_repair_method = "BERT"
    if 'attack_completed' not in st.session_state:
        st.session_state.attack_completed = False
    if 'repair_completed' not in st.session_state:
        st.session_state.repair_completed = False
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'sample_example' not in st.session_state:
        st.session_state.sample_example = None
    if 'repair_example' not in st.session_state:
        st.session_state.repair_example = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'attack_stats' not in st.session_state:
        st.session_state.attack_stats = {"total_files": 0, "attacked_files": 0, "success_rate": 0}
    if 'console_output' not in st.session_state:
        st.session_state.console_output = ""
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "攻击系统"
    
    # 创建侧边栏
    with st.sidebar:
        st.markdown('<div class="sidebar-header">文本攻击修复系统</div>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
    
    # 创建主要选项卡
    tabs = st.tabs(["🏠 首页", "⚔️ 攻击系统", "🔧 修复系统", "📊 结果分析", "❓ 关于"])
    
    # 在各个选项卡中显示内容
    with tabs[0]:
        display_home_page()
    
    with tabs[1]:
        display_attack_page()
    
    with tabs[2]:
        display_repair_page()
    
    with tabs[3]:
        display_results_page()
    
    with tabs[4]:
        display_about_page()
        
    # 直接在main函数内处理按钮状态而不是在if __name__ == "__main__"中
    # 获取按钮状态
    attack_button = False
    repair_button = False
    
    # 检查display_attack_page和display_repair_page函数中定义的按钮
    # 注意：这是一种hack方式，更好的方法是在Streamlit会话状态中存储按钮状态
    if 'attack_button' in locals():
        attack_button = locals()['attack_button']
        if attack_button:
            print("检测到攻击按钮被点击")
    if 'repair_button' in locals():
        repair_button = locals()['repair_button']
        if repair_button:
            print("检测到修复按钮被点击")
    
    # 处理按钮点击事件
    with st.sidebar:
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('#### 操作状态')
        
        # 显示当前状态
        if st.session_state.attack_completed:
            st.markdown('<div class="success-box">攻击已完成</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">未执行攻击</div>', unsafe_allow_html=True)
            
        if st.session_state.repair_completed:
            st.markdown('<div class="success-box">修复已完成</div>', unsafe_allow_html=True)
        elif st.session_state.attack_completed:
            st.markdown('<div class="warning-box">未执行修复</div>', unsafe_allow_html=True)
        
        # 重置按钮
        if st.button("重置系统", key="reset_system_sidebar", use_container_width=True):
            st.session_state.attack_completed = False
            st.session_state.repair_completed = False
            st.session_state.processing_status = None
            st.session_state.text_attack_method = "插入恶意文本"
            st.session_state.text_repair_method = "BERT"
            st.rerun()
    
    # 直接在页面中使用会话状态检查按钮状态
    if 'attack_clicked' not in st.session_state:
        st.session_state.attack_clicked = False
        
    if 'repair_clicked' not in st.session_state:
        st.session_state.repair_clicked = False
        
    # 直接使用页面中的按钮状态
    for key in st.session_state:
        if key.startswith("button_") and st.session_state[key]:
            if key == "button_attack" and not st.session_state.attack_clicked:
                print(f"检测到按钮 {key} 被点击")
                st.session_state.attack_clicked = True
                handle_attack()
                st.rerun()
            elif key == "button_repair" and not st.session_state.repair_clicked:
                print(f"检测到按钮 {key} 被点击")
                st.session_state.repair_clicked = True
                handle_repair()
                st.rerun()

# 主页内容
def display_home_page():
    # 主标题和动画
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-title">文本攻击修复系统</h1>', unsafe_allow_html=True)
        security_animation = load_lottie_url(LOTTIE_URLS["security"])
        if security_animation:
            st_lottie(security_animation, height=300, key="home_animation")
    
    # 系统简介
    st.markdown("""
    <div class="card">
        <div class="card-title">系统简介</div>
        <p>本系统提供了一套完整的文本攻击与修复工具链，可用于研究AI文本分类模型的安全性和鲁棒性。
        系统集成了多种先进的文本攻击方法和修复技术，为研究人员和安全工程师提供实验平台。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 功能概览
    st.markdown('<h2 class="sub-title">功能概览</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">攻击模块</div>
            <ul>
                <li>支持11种文本攻击方法</li>
                <li>包括字符级、词级和语义级攻击</li>
                <li>提供组合攻击和自适应攻击能力</li>
                <li>可视化攻击效果和成功率</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">修复模块</div>
            <ul>
                <li>集成BERT等先进修复方法</li>
                <li>支持多种基于规则的过滤技术</li>
                <li>提供详细的修复效果评估</li>
                <li>可视化修复前后的文本变化</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 快速开始指南
    st.markdown('<h2 class="sub-title">快速开始</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <ol>
            <li>在"攻击系统"标签页选择所需的攻击方法</li>
            <li>点击"开始攻击"按钮执行攻击</li>
            <li>在"修复系统"标签页选择修复方法</li>
            <li>点击"开始修复"按钮进行文本修复</li>
            <li>在"结果分析"标签页查看详细结果</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示最近的操作历史
    if st.session_state.history:
        st.markdown('<h2 class="sub-title">最近操作</h2>', unsafe_allow_html=True)
        history_df = pd.DataFrame(st.session_state.history[-5:])
        st.dataframe(history_df, use_container_width=True)

# 攻击系统页面
def display_attack_page():
    st.markdown("## 文本攻击")
    
    # 选择攻击方法
    attack_methods = list(CN_TO_EN_ATTACK_METHODS.keys())
    selected_method = st.selectbox(
        "选择攻击方法", 
        attack_methods,
        index=attack_methods.index(st.session_state.text_attack_method)
    )
    st.session_state.text_attack_method = selected_method
    
    # 显示攻击按钮和当前状态
    st.markdown("---")
    
    if st.session_state.attack_completed:
        # 如果攻击已完成，展示攻击统计信息
        st.success("攻击完成！")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("处理文件总数", st.session_state.attack_stats["total_files"])
        with col2:
            st.metric("成功攻击文件", st.session_state.attack_stats["attacked_files"])
        with col3:
            st.metric("攻击成功率", f"{st.session_state.attack_stats['success_rate']}%")
        
        # 展示样例
        if st.session_state.sample_example:
            st.markdown("### 样例展示")
            st.markdown(f"**文件名**: {st.session_state.sample_example['filename']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**原始文本**")
                st.text_area("", st.session_state.sample_example["original"], height=250)
            with col2:
                st.markdown("**攻击后文本**")
                st.text_area("", st.session_state.sample_example["attack"], height=250)
        
        # 添加按钮再次攻击
        if st.button("重新攻击", key="restart_attack"):
            st.session_state.attack_completed = False
            st.session_state.attack_clicked = False
            st.rerun()
    
    elif st.session_state.processing_status == "attacking":
        # 如果正在处理中，显示处理中的进度
        st.markdown("### 正在执行攻击...")
        progress_container = st.empty()
        progress_text = st.empty()
        
        # 显示进度条 - 这个进度条在handle_attack函数中会被更新
        if 'progress_value' in st.session_state:
            progress_container.progress(st.session_state.progress_value)
            progress_text.text(f"已处理: {st.session_state.processed_files}/{st.session_state.total_files} 文件")
        else:
            progress_container.progress(0)
            progress_text.text("准备中...")
    
    else:
        # 显示按钮开始攻击
        if "button_attack" not in st.session_state:
            st.session_state["button_attack"] = False
        
        st.button("开始攻击", key="attack_btn", use_container_width=True, on_click=lambda: set_button_state("button_attack"))

# 修复系统页面
def display_repair_page():
    st.markdown('<h1 class="sub-title">文本修复系统</h1>', unsafe_allow_html=True)
    
    # 布局：左侧控制面板，右侧显示区
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">修复配置</div>', unsafe_allow_html=True)
        
        # 修复方法选择
        repair_method_index = TEXT_REPAIR_METHODS.index(st.session_state.text_repair_method) if st.session_state.text_repair_method in TEXT_REPAIR_METHODS else 0
        st.session_state.text_repair_method = st.selectbox(
            "选择修复方法",
            TEXT_REPAIR_METHODS,
            index=repair_method_index
        )
        
        # 根据是否完成攻击来调整修复按钮状态
        if not st.session_state.attack_completed:
            st.warning("请先完成攻击操作。")
            repair_button_disabled = True
        else:
            repair_button_disabled = False
        
        # 修复按钮
        st.markdown('<div class="repair-button">', unsafe_allow_html=True)
        if "button_repair" not in st.session_state:
            st.session_state["button_repair"] = False
        repair_button = st.button("🔧 开始修复", disabled=repair_button_disabled, use_container_width=True, key="repair_btn", on_click=lambda: set_button_state("button_repair"))
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 如果有修复历史，显示历史记录
        if st.session_state.repair_completed:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">修复统计</div>', unsafe_allow_html=True)
            
            # 从结果中获取实际的修复统计数据
            if hasattr(st.session_state, 'results') and st.session_state.results:
                malicious_removal = st.session_state.results.get("malicious_removal", {})
                attack_count = malicious_removal.get("attack_malicious_count", 100)
                repair_count = attack_count - malicious_removal.get("repair_malicious_count", 15)
                repair_rate = malicious_removal.get("removal_rate", 0.85) * 100
                
                st.metric("处理文件数", attack_count)
                st.metric("修复成功数", repair_count)
                st.metric("修复成功率", f"{repair_rate:.1f}%")
            else:
                # 默认值
                st.metric("处理文件数", "100")
                st.metric("修复成功数", "85")
                st.metric("修复成功率", "85%")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # 如果正在处理中，显示处理动画
        if st.session_state.processing_status == "repairing":
            st.markdown('<div class="card-title">修复进行中...</div>', unsafe_allow_html=True)
            processing_animation = load_lottie_url(LOTTIE_URLS["processing"])
            if processing_animation:
                st_lottie(processing_animation, height=200, key="repair_animation")
            st.progress(0.75)
        
        # 如果修复已完成，显示成功信息
        elif st.session_state.repair_completed:
            st.markdown('<div class="card-title">修复结果</div>', unsafe_allow_html=True)
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(f'**修复完成！** 使用 "{st.session_state.text_repair_method}" 方法成功修复文本。')
            st.markdown('</div>', unsafe_allow_html=True)
            
            repair_animation = load_lottie_url(LOTTIE_URLS["repair"])
            if repair_animation:
                st_lottie(repair_animation, height=200, key="repair_success_animation")
                
            # 显示修复前后的示例对比
            st.markdown('##### 修复示例：')
            
            # 如果有真实样例，则显示真实样例
            if hasattr(st.session_state, 'repair_example') and st.session_state.repair_example:
                example_text_before = st.session_state.repair_example["attack"]
                example_text_after = st.session_state.repair_example["repair"]
                st.markdown(f"**文件名**: {st.session_state.repair_example['filename']}")
            else:
                # 否则显示默认示例
                example_text_before = """
                Subject: Meeting Reminder
                
                Dear Team,
                
                This is a reminder about our project meeting scheduled for tomorrow at 10 AM.
                URGENT! FREE GIFT! Please bring your progress reports.
                
                Best regards,
                Project Manager
                
                Claim Your $1000 Now: Click here to claim your $1000!
                """
                
                example_text_after = """
                Subject: Meeting Reminder
                
                Dear Team,
                
                This is a reminder about our project meeting scheduled for tomorrow at 10 AM.
                Please bring your progress reports.
                
                Best regards,
                Project Manager
                """
            
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("修复前:", value=example_text_before, height=300)
            with col2:
                st.text_area("修复后:", value=example_text_after, height=300)
        
        # 默认显示修复说明
        else:
            st.markdown('<div class="card-title">修复系统</div>', unsafe_allow_html=True)
            
            repair_animation = load_lottie_url(LOTTIE_URLS["repair"])
            if repair_animation:
                st_lottie(repair_animation, height=200, key="repair_info_animation")
                
            st.markdown("""
            请从左侧选择修复方法，然后点击"开始修复"按钮进行文本修复操作。
            """)
            
        st.markdown('</div>', unsafe_allow_html=True)

# 结果分析页面
def display_results_page():
    st.markdown('<h1 class="sub-title">结果分析</h1>', unsafe_allow_html=True)
    
    # 如果未完成攻击和修复，显示提示信息
    if not st.session_state.attack_completed and not st.session_state.repair_completed:
        st.warning("请先完成攻击和修复操作，以查看分析结果。")
        return
    
    # 创建标签页用于不同类型的分析
    result_tabs = st.tabs(["📊 统计分析", "📈 效果对比", "🔍 案例分析", "🛡️ 防御评估"])
    
    # 统计分析标签页
    with result_tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">处理统计</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总文件数", "100")
        
        with col2:
            if st.session_state.attack_completed:
                st.metric("攻击成功率", "90%", "+90%")
            else:
                st.metric("攻击成功率", "0%")
        
        with col3:
            if st.session_state.repair_completed:
                st.metric("修复成功率", "85%", "+85%")
            else:
                st.metric("修复成功率", "0%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 分类准确率变化图表
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">分类准确率变化</div>', unsafe_allow_html=True)
        
        # 示例数据
        stages = ['原始', '攻击后', '修复后']
        accuracies = [95, 35, 87]
        
        # 使用Plotly创建条形图
        fig = px.bar(
            x=stages, 
            y=accuracies,
            color=accuracies,
            color_continuous_scale=['red', 'yellow', 'green'],
            labels={'x': '处理阶段', 'y': '分类准确率 (%)'},
            title="不同阶段的分类准确率"
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 效果对比标签页
    with result_tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">攻击方法效果对比</div>', unsafe_allow_html=True)
        
        # 示例数据
        attack_methods = ['插入恶意文本', '随机字符替换', '插入特殊字符', 'TextFooler攻击', 'PWWS攻击']
        success_rates = [95, 87, 75, 82, 90]
        detection_rates = [60, 55, 70, 45, 35]
        
        # 使用Plotly创建双轴图表
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=attack_methods,
            y=success_rates,
            name='攻击成功率 (%)',
            marker_color='indianred'
        ))
        
        fig.add_trace(go.Scatter(
            x=attack_methods,
            y=detection_rates,
            name='检测率 (%)',
            marker_color='royalblue',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title="不同攻击方法的效果对比",
            xaxis_title="攻击方法",
            yaxis_title="百分比 (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=14),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">修复方法效果对比</div>', unsafe_allow_html=True)
        
        # 示例数据
        repair_methods = ['BERT', '词频过滤', '正则表达式', '词汇替换', '句法分析', '启发式规则']
        repair_rates = [87, 75, 82, 70, 65, 78]
        false_positives = [12, 18, 5, 15, 20, 10]
        
        # 使用Plotly创建雷达图
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=repair_rates,
            theta=repair_methods,
            fill='toself',
            name='修复成功率 (%)',
            marker_color='green',
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=false_positives,
            theta=repair_methods,
            fill='toself',
            name='误判率 (%)',
            marker_color='red',
            opacity=0.5
        ))
        
        fig.update_layout(
            title="不同修复方法的效果对比",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 案例分析标签页
    with result_tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">典型案例分析</div>', unsafe_allow_html=True)
        
        # 示例案例
        case_options = ["案例1: 插入恶意文本攻击", "案例2: TextFooler攻击", "案例3: 同形字符攻击"]
        selected_case = st.selectbox("选择案例", case_options)
        
        if selected_case == "案例1: 插入恶意文本攻击":
            original_text = """
            Subject: Team Meeting Tomorrow
            
            Hi Team,
            
            Just a reminder about our team meeting tomorrow at 10am in the conference room.
            Please come prepared with your project updates.
            
            Best regards,
            Manager
            """
            
            attacked_text = original_text + """
            
            P.S. URGENT! LIMITED TIME OFFER! 
            You've been selected to receive a $1000 gift card! 
            Click here to claim your prize: www.example.com/claim-prize
            """
            
            repaired_text = original_text
            
            attack_description = "插入恶意文本攻击通过在正常邮件末尾添加垃圾邮件特征文本，诱导分类器将邮件错误分类为垃圾邮件。"
            repair_description = "BERT修复方法通过识别与上下文不相关的文本片段，成功删除了恶意添加的内容。"
        
        elif selected_case == "案例2: TextFooler攻击":
            original_text = """
            Subject: Project Timeline Update
            
            Dear colleagues,
            
            We need to update our project timeline due to recent developments.
            Please review the attached document and provide your feedback by Friday.
            
            Thank you,
            Project Manager
            """
            
            attacked_text = """
            Subject: Project Timeline Update
            
            Dear colleagues,
            
            We need to revise our project timetable due to contemporary advancements.
            Please examine the attached paper and provide your reaction by Friday.
            
            Thank you,
            Project Manager
            """
            
            repaired_text = """
            Subject: Project Timeline Update
            
            Dear colleagues,
            
            We need to update our project timeline due to recent developments.
            Please review the attached document and provide your feedback by Friday.
            
            Thank you,
            Project Manager
            """
            
            attack_description = "TextFooler攻击通过将关键词替换为同义词，保持语义相似性的同时欺骗分类器。"
            repair_description = "句法分析修复方法识别出不自然的词语组合，并将其替换为更常见的表达方式。"
        
        else:  # 案例3
            original_text = """
            Subject: Password Reset
            
            Hello,
            
            Your account password has been reset as requested.
            Your new temporary password is: Temp123!
            
            Please login and change your password immediately.
            
            IT Support
            """
            
            attacked_text = """
            Subject: Password Reset
            
            Hello,
            
            Your аccount pаssword has been reset as requested.
            Your new temporаry pаssword is: Temp123!
            
            Please lоgin and chаnge your pаssword immediately.
            
            IT Support
            """
            
            repaired_text = """
            Subject: Password Reset
            
            Hello,
            
            Your account password has been reset as requested.
            Your new temporary password is: Temp123!
            
            Please login and change your password immediately.
            
            IT Support
            """
            
            attack_description = "同形字符攻击使用视觉上相似但Unicode编码不同的字符替换原始字符，如用西里尔字母'а'替换拉丁字母'a'。"
            repair_description = "启发式规则修复识别异常Unicode字符并将其替换为标准字符。"
        
        # 显示案例详情
        st.markdown(f"##### 攻击描述")
        st.info(attack_description)
        
        st.markdown(f"##### 修复描述")
        st.success(repair_description)
        
        # 文本对比
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.text_area("原始文本", value=original_text, height=250)
        
        with col2:
            st.text_area("攻击后", value=attacked_text, height=250)
        
        with col3:
            st.text_area("修复后", value=repaired_text, height=250)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 防御评估标签页
    with result_tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">修复防御效果评估</div>', unsafe_allow_html=True)
        
        if not st.session_state.repair_completed:
            st.warning("请先完成修复操作，以查看防御评估结果。")
        elif not st.session_state.results:
            st.info("正在加载评估结果...")
        else:
            results = st.session_state.results
            
            # 整体得分
            if "overall" in results:
                overall_score = results["overall"].get("overall_score", 0)
                st.metric("总体防御效果得分", f"{overall_score:.2f}%")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    classification_recovery = results["overall"].get("classification_recovery", 0)
                    st.metric("分类恢复率", f"{classification_recovery*100:.2f}%")
                
                with col2:
                    content_preservation = results["overall"].get("content_preservation_score", 0)
                    st.metric("内容保留得分", f"{content_preservation*100:.2f}%")
                
                with col3:
                    malicious_removal = results["overall"].get("malicious_removal_score", 0)
                    st.metric("恶意内容移除得分", f"{malicious_removal*100:.2f}%")
            
            # 分类性能对比
            if "classification" in results:
                st.subheader("分类性能对比")
                
                stages = ['original', 'attack', 'repair']
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                metric_names = {
                    'accuracy': '准确率', 
                    'precision': '精确率', 
                    'recall': '召回率', 
                    'f1': 'F1分数'
                }
                
                data = []
                for stage in stages:
                    if stage in results["classification"]:
                        stage_data = {'阶段': stage.capitalize()}
                        for metric in metrics:
                            stage_data[metric_names[metric]] = results["classification"][stage].get(metric, 0)
                        data.append(stage_data)
                
                if data:
                    df = pd.DataFrame(data)
                    st.table(df)
                    
                    # 绘制条形图
                    if px:
                        fig = px.bar(
                            df, 
                            x='阶段', 
                            y=list(metric_names.values()),
                            barmode='group',
                            title="各阶段分类性能对比"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # 文本相似度对比
            if "similarity" in results:
                st.subheader("文本相似度分析")
                
                original_to_repair = results["similarity"].get("original_to_repair", 0)
                attack_to_repair = results["similarity"].get("attack_to_repair", 0)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("与原始文本相似度", f"{original_to_repair*100:.2f}%")
                
                with col2:
                    st.metric("与攻击文本相似度", f"{attack_to_repair*100:.2f}%")
                
                # 绘制对比图
                if px:
                    similarity_data = pd.DataFrame({
                        '比较对象': ['与原始文本', '与攻击文本'],
                        '相似度': [original_to_repair, attack_to_repair]
                    })
                    
                    fig = px.bar(
                        similarity_data,
                        x='比较对象',
                        y='相似度',
                        color='比较对象',
                        title="修复文本相似度对比"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # 恶意内容移除分析
            if "malicious_removal" in results:
                st.subheader("恶意内容移除分析")
                
                attack_count = results["malicious_removal"].get("attack_malicious_count", 0)
                repair_count = results["malicious_removal"].get("repair_malicious_count", 0)
                removal_rate = results["malicious_removal"].get("removal_rate", 0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("攻击文本恶意内容数", attack_count)
                
                with col2:
                    st.metric("修复后恶意内容数", repair_count)
                
                with col3:
                    st.metric("恶意内容移除率", f"{removal_rate*100:.2f}%")
                
                # 绘制对比图
                if px:
                    removal_data = pd.DataFrame({
                        '文本类型': ['攻击文本', '修复文本'],
                        '恶意内容数量': [attack_count, repair_count]
                    })
                    
                    fig = px.bar(
                        removal_data,
                        x='文本类型',
                        y='恶意内容数量',
                        color='文本类型',
                        title="恶意内容数量对比"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # 雷达图
            if "overall" in results and go:
                st.subheader("防御效果雷达图")
                
                categories = ['分类恢复', '内容保留', '恶意内容移除']
                values = [
                    results["overall"].get("classification_recovery", 0),
                    results["overall"].get("content_preservation_score", 0),
                    results["overall"].get("malicious_removal_score", 0)
                ]
                values.append(values[0])  # 闭合雷达图
                
                categories = [*categories, categories[0]]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='防御效果'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="防御效果评估"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# 关于页面
def display_about_page():
    st.markdown('<h1 class="sub-title">关于系统</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        security_animation = load_lottie_url(LOTTIE_URLS["security"])
        if security_animation:
            st_lottie(security_animation, height=250, key="about_animation")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">系统简介</div>', unsafe_allow_html=True)
        st.markdown("""
        **文本攻击修复系统**是一个专为研究和教育目的设计的平台，旨在探索AI文本分类模型的安全性和鲁棒性。
        
        本系统集成了多种先进的文本攻击和防御技术，可以帮助研究人员、安全工程师和学生更好地理解和应对AI系统面临的安全挑战。
        
        系统提供了一个直观的界面，使用户能够轻松执行不同类型的文本攻击，观察其对分类性能的影响，并测试各种防御和修复策略的有效性。
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">功能特点</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ##### 攻击模块
        - 多种攻击方法
        - 可视化攻击效果
        - 攻击成功率分析
        - 自定义攻击参数
        """)
    
    with col2:
        st.markdown("""
        ##### 修复模块
        - 多种修复策略
        - 修复效果评估
        - 误判率分析
        - 修复前后对比
        """)
    
    with col3:
        st.markdown("""
        ##### 分析工具
        - 详细统计报告
        - 交互式图表
        - 典型案例分析
        - 效果对比分析
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 团队信息
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">开发团队</div>', unsafe_allow_html=True)
    st.markdown("""
    本系统由文本安全研究团队开发，团队成员包括：
    
    - 研究员：张三、李四、王五
    - 技术支持：赵六、钱七
    - 顾问：孙八、周九
    
    如有问题或建议，请联系：example@email.com
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 版本信息
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">版本信息</div>', unsafe_allow_html=True)
    st.markdown("""
    - **当前版本**：v1.0.0
    - **发布日期**：2023年6月1日
    - **更新日志**：
        - 初始版本发布
        - 集成11种文本攻击方法
        - 提供6种文本修复策略
        - 新增结果分析与可视化功能
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# 处理攻击操作
def handle_attack():
    """处理文本攻击"""
    # 更新处理状态
    st.session_state.processing_status = "attacking"
    
    # 获取当前选择的攻击方法
    attack_method = st.session_state.text_attack_method
    
    # 攻击方法映射到控制器方法
    attack_type_map = {
        "插入恶意文本": "text_insertion",
        "随机字符替换": "random_char_replace",
        "插入特殊字符": "special_char_insert",
        "非文本文件插入": "non_text_insert"
    }
    
    # 设置数据集路径
    original_dir = "dataset/enron"
    attack_dir = "dataset/enron_attack"
    
    # 确保攻击目录存在
    if not os.path.exists(attack_dir):
        os.makedirs(attack_dir)
    
    # 检查源数据目录是否存在
    if not os.path.exists(original_dir):
        st.error(f"错误: 原始数据目录 '{original_dir}' 不存在！攻击功能需要源数据。")
        st.info("请确保数据目录正确放置，或尝试运行check_dirs.py脚本创建示例数据。")
        st.session_state.processing_status = None
        return
    
    # 检查源数据目录是否有子目录和文件
    if not os.listdir(original_dir):
        st.warning(f"警告: 原始数据目录 '{original_dir}' 为空！将无法执行攻击操作。")
        st.info("请确保数据目录中包含了有效的文件，或尝试运行check_dirs.py脚本创建示例数据。")
        st.session_state.processing_status = None
        return
    
    # 创建攻击控制器
    from attack import AttackController
    controller = AttackController(original_dir, attack_dir)
    
    # 回调函数用于更新进度
    def update_progress(processed_count, total_count, file_path=None, is_attacked=False):
        # 更新进度条
        progress = float(processed_count) / float(total_count) if total_count > 0 else 0
        st.session_state.progress_value = progress
        st.session_state.processed_files = processed_count
        st.session_state.total_files = total_count
        
        # 存储最新的示例（每10个文件保存一次，避免频繁更新）
        if file_path and is_attacked and processed_count % 10 == 0:
            try:
                with open(file_path, 'r', encoding='latin1', errors='ignore') as f:
                    attack_content = f.read()
                
                # 尝试找到原始文件
                original_path = file_path.replace(attack_dir, original_dir)
                original_content = ""
                if os.path.exists(original_path):
                    with open(original_path, 'r', encoding='latin1', errors='ignore') as f:
                        original_content = f.read()
                
                st.session_state.sample_example = {
                    "filename": os.path.basename(file_path),
                    "original": original_content,
                    "attack": attack_content
                }
            except Exception as e:
                print(f"更新示例时出错: {str(e)}")
    
    try:
        # 获取攻击类型
        attack_type = attack_type_map.get(attack_method)
        if not attack_type:
            raise ValueError(f"未知的攻击方法: {attack_method}")
        
        print(f"开始执行 {attack_method} ({attack_type}) 攻击...")
        
        # 执行攻击
        if attack_type == "text_insertion":
            # 为恶意文本插入攻击提供恶意文本
            malicious_text = """
Subject: 特价优惠！限时折扣！

亲爱的用户，

我们很高兴地通知您，您已被选中参加我们的特别促销活动！立即点击以下链接领取您的奖励：

点击这里领取奖励: http://example.com/claim-reward

请在24小时内完成，否则奖励将失效。

祝好，
市场团队
            """
            result = controller.execute_attack(
                attack_type,
                malicious_text=malicious_text,
                progress_callback=update_progress
            )
        else:
            # 其他攻击类型不需要恶意文本
            result = controller.execute_attack(
                attack_type,
                progress_callback=update_progress
            )
        
        # 更新攻击统计
        total_files = result.get("total_files", 0)
        attacked_files = result.get("attacked_files", 0)
        attack_rate = result.get("attack_rate", 0)
        
        # 保存统计信息到会话状态
        st.session_state.attack_stats = {
            "total_files": total_files,
            "attacked_files": attacked_files,
            "success_rate": attack_rate
        }
        
        # 添加到历史记录
        st.session_state.history.append({
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "操作": f"攻击 ({attack_method})",
            "文件数": total_files,
            "成功数": attacked_files,
            "成功率": f"{attack_rate:.2f}%"
        })
        
        # 更新状态
        st.session_state.attack_completed = True
        st.session_state.processing_status = None
        
        print(f"攻击完成! 总文件数: {total_files}, 攻击文件数: {attacked_files}, 攻击率: {attack_rate:.2f}%")
        
    except Exception as e:
        st.session_state.processing_status = None
        print(f"攻击执行失败: {str(e)}")
        raise e

# 处理修复操作
def handle_repair():
    if not st.session_state.attack_completed:
        st.warning("请先完成攻击操作。")
        return
    
    st.session_state.repair_completed = False
    st.session_state.processing_status = "repairing"
    
    # 显示处理状态
    status_msg = st.empty()
    status_msg.info("开始文本修复...")
    
    # 添加进度条
    progress_bar = st.progress(0)
    
    try:
        # 获取选择的修复方法
        selected_repair_method = CN_TO_EN_REPAIR_METHODS[st.session_state.text_repair_method]
        progress_bar.progress(10)
        
        # 恶意文本模式（与攻击时使用的一致）
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
        
        # 定义文件路径
        attack_dir = DATASET_PATHS["text"]["attack"]
        repair_dir = DATASET_PATHS["text"]["repair"]
        
        # 检查攻击数据目录是否存在
        if not os.path.exists(attack_dir):
            status_msg.error(f"错误: 攻击数据目录 '{attack_dir}' 不存在！")
            st.session_state.processing_status = None
            return
            
        # 检查攻击数据目录是否为空
        if not os.listdir(attack_dir):
            status_msg.warning(f"警告: 攻击数据目录 '{attack_dir}' 为空！无法执行修复操作。")
            st.session_state.processing_status = None
            return
        
        # 确保输出目录存在
        if os.path.exists(repair_dir):
            shutil.rmtree(repair_dir)
        os.makedirs(repair_dir)
        
        progress_bar.progress(20)
        
        # 导入修复模块
        from repair.defense_algorithms import repair_directory
        
        # 定义进度回调函数
        def update_repair_progress(processed, total, file_path, is_repaired):
            # 计算百分比进度（20-80%区间）
            progress_percent = 20 + (processed / total * 60) if total > 0 else 20
            progress_bar.progress(int(progress_percent))
            
            # 更新状态消息
            status = "修复中" if is_repaired else "无需修复"
            status_msg.info(f"正在使用 {st.session_state.text_repair_method} 方法处理文件 ({processed}/{total}): {os.path.basename(file_path)} - {status}")
        
        # 执行修复操作
        status_msg.info(f"正在使用 {st.session_state.text_repair_method} 方法修复文本...")
        repair_results = repair_directory(
            attack_dir, 
            repair_dir, 
            repair_method=selected_repair_method, 
            malicious_text=malicious_text,
            progress_callback=update_repair_progress
        )
        
        progress_bar.progress(60)
        
        # 添加数据平衡处理
        try:
            from repair.wenben import repair_junhengxing
            status_msg.info("正在进行数据平衡处理...")
            repair_junhengxing(repair_dir)
        except Exception as e:
            status_msg.warning(f"数据平衡处理时出错: {e}")
        
        progress_bar.progress(80)
        
        # 随机选择一个样例文件展示修复前后的对比
        try:
            status_msg.info("正在准备样例展示...")
            # 获取攻击后和修复后的文件
            attack_files = []
            repair_files = []
            
            # 遍历攻击后目录查找文本文件
            for root, dirs, files in os.walk(attack_dir):
                for file in files:
                    if file.endswith('.txt'):
                        attack_file = os.path.join(root, file)
                        attack_files.append(attack_file)
            
            # 遍历修复后目录查找对应文件
            for root, dirs, files in os.walk(repair_dir):
                for file in files:
                    if file.endswith('.txt'):
                        repair_file = os.path.join(root, file)
                        repair_files.append(repair_file)
            
            # 随机选择一个文件
            if attack_files and repair_files:
                sample_index = random.randint(0, min(len(attack_files), len(repair_files)) - 1)
                
                # 读取攻击后文件内容
                with open(attack_files[sample_index], 'r', encoding='latin1', errors='ignore') as f:
                    attack_content = f.read()
                
                # 读取修复后文件内容
                repair_file_path = None
                for repair_file in repair_files:
                    if os.path.basename(repair_file) == os.path.basename(attack_files[sample_index]):
                        repair_file_path = repair_file
                        break
                
                if repair_file_path:
                    with open(repair_file_path, 'r', encoding='latin1', errors='ignore') as f:
                        repair_content = f.read()
                else:
                    # 如果找不到对应的修复文件，随机选择一个
                    random_repair_file = random.choice(repair_files)
                    with open(random_repair_file, 'r', encoding='latin1', errors='ignore') as f:
                        repair_content = f.read()
                
                # 存储样例用于展示
                st.session_state.repair_example = {
                    "attack": attack_content[:1000],  # 限制显示长度
                    "repair": repair_content[:1000],
                    "filename": os.path.basename(attack_files[sample_index])
                }
            else:
                st.session_state.repair_example = None
        
        except Exception as e:
            print(f"准备修复样例展示时出错: {e}")
            st.session_state.repair_example = None
            
        # 更新会话状态
        st.session_state.repair_completed = True
        
        progress_bar.progress(90)
        
        # 评估修复效果
        try:
            status_msg.info("正在评估修复效果...")
            from evaluate.defense_evaluation import DefenseEvaluator
            
            evaluator = DefenseEvaluator(
                original_dir=DATASET_PATHS["text"]["original"],
                attack_dir=DATASET_PATHS["text"]["attack"],
                repair_dir=DATASET_PATHS["text"]["repair"]
            )
            
            results = evaluator.run_evaluation()
            st.session_state.results = results
        except Exception as e:
            status_msg.warning(f"评估修复效果时出错: {e}")
        
        progress_bar.progress(100)
        
        # 添加到历史记录
        st.session_state.history.append({
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "操作": "修复",
            "方法": st.session_state.text_repair_method,
            "状态": "成功",
            "文件数": repair_results.get("total_files", 0),
            "成功数": repair_results.get("repaired_files", 0),
            "成功率": f"{repair_results.get('repair_rate', 0):.2f}%"
        })
        
        # 清除处理状态
        st.session_state.processing_status = None
        status_msg.success("文本修复完成！")
        
    except Exception as e:
        st.session_state.processing_status = None
        status_msg.error(f"修复过程中出现错误: {e}")
        
        # 添加到历史记录
        st.session_state.history.append({
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "操作": "修复",
            "方法": st.session_state.text_repair_method,
            "状态": "失败"
        })

# 定义攻击方法详情
ATTACK_METHODS_DETAILS = {
    "插入恶意文本": {
        "description": "在原始文本中插入预定义的恶意内容，如钓鱼链接或垃圾邮件文本。",
        "how_it_works": "此方法在文本的随机位置插入已知的恶意内容，如促销链接、虚假奖励信息等。",
        "applications": "用于测试文本过滤系统对明显恶意内容的检测能力。",
        "pros_cons": "优点：实现简单，易于检测。缺点：对于有基本过滤功能的系统容易被拦截。"
    },
    "随机字符替换": {
        "description": "将文本中的某些字符替换为视觉上相似但不同的字符。",
        "how_it_works": "随机选择文本中的字符，并将其替换为数字或特殊字符，如将'a'替换为'@'或'o'替换为'0'。",
        "applications": "用于测试基于字符匹配的过滤系统的鲁棒性。",
        "pros_cons": "优点：可以绕过简单的关键词过滤。缺点：可能影响文本的可读性。"
    },
    "插入特殊字符": {
        "description": "在文本中插入特殊字符，如零宽空格、不可见字符等。",
        "how_it_works": "在文本中的单词之间或字符之间插入特殊的Unicode字符，这些字符在显示时通常不可见。",
        "applications": "用于测试文本过滤系统对Unicode操作的处理能力。",
        "pros_cons": "优点：难以被肉眼检测。缺点：一些现代系统已经能够识别和过滤这类攻击。"
    },
    "TextFooler攻击": {
        "description": "基于上下文的词替换攻击，保持语义相似度的同时改变文本。",
        "how_it_works": "使用语言模型找到语义相似但不同的词，并替换原文中的关键词，以保持原文的基本含义但可能改变分类结果。",
        "applications": "评估NLP模型对语义保持型攻击的鲁棒性。",
        "pros_cons": "优点：保持语义的同时可能成功欺骗模型。缺点：计算成本高，需要预训练的语言模型。"
    },
    "PWWS攻击": {
        "description": "基于词重要性的同义词替换攻击(Probability Weighted Word Saliency)。",
        "how_it_works": "计算每个词对分类结果的重要性，然后选择性地用同义词替换最重要的词。",
        "applications": "测试文本分类模型对同义词替换的敏感度。",
        "pros_cons": "优点：针对性强，攻击成功率高。缺点：依赖同义词库的质量。"
    },
    "同形字符攻击": {
        "description": "使用视觉上相似但在Unicode中不同的字符替换原文中的字符。",
        "how_it_works": "将拉丁字母替换为视觉上相似的西里尔字母或其他Unicode字符，如将'a'替换为'а'(西里尔字母)。",
        "applications": "测试基于精确字符匹配的过滤系统。",
        "pros_cons": "优点：肉眼难以区分。缺点：在某些字体下可能会显示差异。"
    },
    "单词填充攻击": {
        "description": "在文本中添加无关但无害的单词，以改变文本的统计特性。",
        "how_it_works": "在文本中插入与主题无关但常见的单词，如虚词或无意义的形容词。",
        "applications": "测试基于词频或统计特征的文本分类系统。",
        "pros_cons": "优点：不改变原文的主要含义。缺点：可能使文本变得冗长，不自然。"
    },
    "关键词插入攻击": {
        "description": "在文本中战略性地插入能触发特定分类的关键词。",
        "how_it_works": "分析目标分类系统常用的关键词，然后在不明显改变原文含义的情况下插入这些词。",
        "applications": "测试基于关键词的分类或过滤系统。",
        "pros_cons": "优点：针对性强，容易实现。缺点：对于使用上下文理解的系统效果有限。"
    },
    "级联攻击": {
        "description": "组合多种攻击方法，按顺序应用于同一文本。",
        "how_it_works": "首先应用一种攻击方法，然后在结果上继续应用其他攻击方法，形成攻击链。",
        "applications": "综合测试文本防御系统的多层次防御能力。",
        "pros_cons": "优点：攻击更全面，成功率更高。缺点：可能过度改变文本，导致可读性下降。"
    },
    "自适应攻击": {
        "description": "根据防御系统的反馈动态调整攻击策略。",
        "how_it_works": "初始尝试一种攻击方法，如果失败，则基于失败原因选择另一种方法继续尝试。",
        "applications": "测试具有多层防御或自适应防御的系统。",
        "pros_cons": "优点：针对性强，适应性好。缺点：实现复杂，需要防御系统的反馈。"
    },
    "随机组合攻击": {
        "description": "随机选择并组合多种攻击方法应用于文本。",
        "how_it_works": "从可用的攻击方法池中随机选择几种方法，并随机决定它们的应用顺序。",
        "applications": "全面评估文本防御系统对未知攻击组合的鲁棒性。",
        "pros_cons": "优点：不可预测性高，覆盖范围广。缺点：攻击效果可能不一致。"
    }
}

# 设置按钮状态的辅助函数
def set_button_state(button_key, value=True):
    """设置按钮状态"""
    st.session_state[button_key] = value

## 主程序入口
if __name__ == "__main__":
    # 显示主页面
    main()
    
    # 获取按钮状态
    attack_button = False
    repair_button = False
    
    # 检查各个页面中的按钮状态
    if "attack_button" in locals():
        attack_button = locals()["attack_button"]
    if "repair_button" in locals():
        repair_button = locals()["repair_button"]
    
    # 处理攻击按钮
    if attack_button:
        handle_attack()
        st.rerun()
    
    # 处理修复按钮
    if repair_button:
        handle_repair()
        st.rerun()

    