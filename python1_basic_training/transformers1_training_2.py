import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 1️⃣ 检查 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2️⃣ 加载 GPT-2 模型和分词器
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 解决 pad_token 问题
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# 3️⃣ 加载数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 4️⃣ 数据预处理（确保有 `labels`）
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()  # 关键点：labels = input_ids
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 5️⃣ 训练/评估数据集拆分
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]  # 提供 eval 数据集

# 6️⃣ 训练配置
training_args = TrainingArguments(
    output_dir="../gpt2_finetune",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,  # 适配小显存
    per_device_eval_batch_size=2,
    fp16=True,  # 启用混合精度
    logging_steps=10,
    save_total_limit=2,
    num_train_epochs=1
)

# 7️⃣ 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset  # 解决 eval_dataset 缺失问题
)

trainer.train()
