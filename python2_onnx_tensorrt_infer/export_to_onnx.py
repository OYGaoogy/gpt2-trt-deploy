import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

model_path = "../gpt2_finetune"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# 构造包装类，避免 trace 时访问 past_key_values 报错
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.config.use_cache = False  # 显式关闭缓存

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return outputs.logits

wrapped_model = Wrapper(model)

# 构造 dummy 输入
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

onnx_path = "../model/gpt2.onnx"
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

# 导出 ONNX 模型
torch.onnx.export(
    wrapped_model,
    (input_ids, attention_mask),
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"}
    },
    opset_version=14  # ✅ 提升为 14，支持 Flash Attention
)

print(f"✅ 模型成功导出为 ONNX：{onnx_path}")
