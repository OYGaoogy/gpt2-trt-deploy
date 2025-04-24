import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# 启用 gradient_checkpointing 降低显存占用
# 节省显存，适合大模型/小显存机器！
# 正常训练时，PyTorch 会保存每一层的中间结果（forward 的输出），用于之后反向传播（backward）。
# 这个策略会使，不保存中间结果，反向传播时 重新再算一遍 forward。
# 用一点计算时间，换取很多显存。
model.gradient_checkpointing_enable()

# 使用 torch.compile() 加速，XLA 编译，可以减少计算冗余
# PyTorch 原本是动态图（eager mode），执行速度没那么快。
# torch.compile() 会把你的模型“转成优化后的静态图”，执行更快！
# PyTorch 的 torch.compile() 依赖 Triton 进行编译优化，但 Triton 不支持 Windows。
# model = torch.compile(model) # ❌ 这行代码会导致错误

# 加载数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 降低 max_length 会减少显存消耗，但未必会变慢，反而通常会更快！
# 含义是：每条文本最多只保留 64 个 token，超过的截断，不足的 padding。
# 此时 batch_size 可调大，存省了，可以一次送更多句子（batch_size↑）
# 缺点：语义被截断，如果原文本太长，64 token 不够用，可能损失信息
# 缺点：对长文本训练效果稍弱，特别是语言模型会“忘记”后面内容
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)  # max_length 128 → 64
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# 训练配置
# 优化一些配置（比如开启 torch.compile 或 gradient checkpointing）
# 可能触发了对输入的更严格检查。
# 传给 Trainer 的数据集包含了这些字段：
# ['attention_mask', 'input_ids', 'labels', 'text']
# Trainer 默认只保留 model.forward() 里明确写明的参数
# GPT-2 的 forward() 方法接受 input_ids, attention_mask, labels，被错误忽略
# remove_unused_columns=True Trainer 默认有个参数，
# 如果数据集中有字段不在 model.forward() 参数里，就自动移除。
# 从而导致模型根本收不到输入 → 报错
# DeepSpeed 依赖 aio.lib 和 cufile.lib，但 Windows 不支持这些库。
# 彻底禁用 DeepSpeed   deepspeed=None
training_args = TrainingArguments(
    output_dir="../gpt2_finetune",
    evaluation_strategy="epoch",    # 每个 epoch 做一次验证
    save_strategy="epoch",          # 每个 epoch 存一次模型
    per_device_train_batch_size=4,  # 提高 batch size（如果 OOM 降回 2）
    per_device_eval_batch_size=2,   # 验证时的 batch_size
    fp16=True,                      # 开启半精度训练（更快 + 节省显存）
    logging_steps=10,               # 每 10 步输出一次日志
    save_total_limit=2,             # 最多保留 2 个 checkpoint
    num_train_epochs=1,             # 只训练 1 轮
    remove_unused_columns=False,    # 不移除数据中未用的字段（解决兼容性问题）
    deepspeed=None                  # 明确禁用 DeepSpeed（因为 Windows 不支持）
)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

# 把训练/微调好的 GPT-2 模型参数（包括权重、配置）保存在本地
# pytorch_model.bin → 模型的权重，Hugging Face 新版推荐使用的一种更安全的模型权重格式 model.safetensors
# config.json → 模型结构的配置信息（比如层数、hidden size 等）
# 存档保存游戏进度
model.save_pretrained("../gpt2_finetune")
# 用的 tokenizer（分词器） 也保存下来，保证之后推理用的分词方式和训练时完全一样
# 存下游戏设置、按键布局
tokenizer.save_pretrained("../gpt2_finetune")
