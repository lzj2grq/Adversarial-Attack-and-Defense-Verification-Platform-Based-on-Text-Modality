import torch
from sklearn.metrics import accuracy_score, recall_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# 加载数据集（假设数据已保存为 jsonl 格式）
def load_data():
    dataset = load_dataset('json', data_files={'train': 'dataset/enron_attack.jsonl', 'test': 'dataset/enron_attack.jsonl'})
    return dataset

# 数据预处理：分词
def preprocess_function(examples):
    tokenizer = BertTokenizer.from_pretrained('model/bert/bert-base-uncased')
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

# 创建模型
def create_model():
    model = BertForSequenceClassification.from_pretrained('model/bert/bert-base-uncased', num_labels=2)
    return model

# 计算评价指标
def compute_metrics(p):
    preds, labels = p
    preds = torch.argmax(torch.tensor(preds), axis=1)
    accuracy = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    return {'accuracy': accuracy, 'recall': recall}

# 训练模型
def train():
    # 加载数据
    dataset = load_data()

    # 数据预处理
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # 创建模型
    model = create_model()
    print(model)
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir='model/bert/results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    model.save_pretrained('model/enron_bert_attack_model')

# 运行训练
if __name__ == "__main__":
    train()
