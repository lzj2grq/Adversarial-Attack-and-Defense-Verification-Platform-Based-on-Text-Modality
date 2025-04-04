import os
import random
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from sklearn.metrics import accuracy_score
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
                        if os.path.isfile(file_path):
                            with open(file_path, 'r', encoding='latin1') as f:
                                texts.append(f.read())
                            labels.append(label_map[category])

    # 将文本和标签转换为datasets格式
    data = {'text': texts, 'label': labels}
    dataset = Dataset.from_dict(data)

    # 创建训练集和测试集（80%/20%）
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset

# 载入数据
enron_attack_dir = 'dataset/enron'
dataset = load_data_from_enron(enron_attack_dir)

print(f"训练集大小: {len(dataset['train'])}, 测试集大小: {len(dataset['test'])}")


# 加载BERT的tokenizer和模型
model_name = "model/bert/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

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

# 训练模型
trainer.train()

# 保存模型
model.save_pretrained('model/enron_bert_model')
tokenizer.save_pretrained('model/enron_bert_model')


enron_attack_dir = 'dataset/enron_attack'
# from transformers import BertForSequenceClassification, BertTokenizer
# from datasets import load_metric

# # 加载保存的模型和tokenizer
# model = BertForSequenceClassification.from_pretrained('model/enron_bert_model')
# tokenizer = BertTokenizer.from_pretrained('model/enron_bert_model')

# # 加载数据集
# dataset = load_data_from_enron(enron_attack_dir)

# # 对测试集进行tokenization
# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# # 加载metric用于计算准确率
# metric = load_metric("accuracy")

# # 计算准确率
# def compute_metrics(p):
#     return metric.compute(predictions=p.predictions.argmax(axis=-1), references=p.label_ids)

# # 使用Trainer评估模型
# trainer = Trainer(
#     model=model,                         # 训练的模型
#     eval_dataset=tokenized_datasets['test'],    # 测试数据集
#     compute_metrics=compute_metrics,     # 计算准确率的函数
# )

# # 评估模型
# results = trainer.evaluate()
# print(f"准确率: {results['eval_accuracy']}")

