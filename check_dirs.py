"""
检查并创建必要的目录结构
"""
import os
import sys

def check_and_create_dirs():
    """检查并创建必要的目录结构"""
    
    # 必要的目录列表
    required_dirs = [
        "dataset",
        "dataset/enron",
        "dataset/enron_attack",
        "dataset/repair_results",
        "dataset/attack_results",
        "results"
    ]
    
    print("检查必要的目录结构...")
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"创建目录: {directory}")
            except Exception as e:
                print(f"创建目录 {directory} 失败: {str(e)}")
                return False
        else:
            print(f"目录已存在: {directory}")
    
    # 检查enron数据集是否为空
    if not os.listdir("dataset/enron"):
        print("警告: enron数据集目录为空，攻击功能可能无法正常工作")
        
        # 创建示例子目录和文件
        try:
            sample_dirs = [
                "dataset/enron/enron1/ham",
                "dataset/enron/enron1/spam"
            ]
            
            for d in sample_dirs:
                if not os.path.exists(d):
                    os.makedirs(d)
                    print(f"创建示例目录: {d}")
            
            # 创建示例文件
            with open("dataset/enron/enron1/ham/sample.txt", "w", encoding="utf-8") as f:
                f.write("这是一个示例的正常邮件内容。\n" * 5)
            
            with open("dataset/enron/enron1/spam/sample.txt", "w", encoding="utf-8") as f:
                f.write("这是一个示例的垃圾邮件内容。\n特价促销！立即点击！\n" * 5)
                
            print("已创建示例文件")
        except Exception as e:
            print(f"创建示例文件失败: {str(e)}")
    
    return True

if __name__ == "__main__":
    if check_and_create_dirs():
        print("目录结构检查完成，系统准备就绪")
        sys.exit(0)  # 成功退出
    else:
        print("目录结构检查失败，请手动修复")
        sys.exit(1)  # 失败退出 