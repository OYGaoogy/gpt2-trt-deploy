from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 tokenizer 和 model
# model.train() 用于训练阶段，会更新权重，模型默认值
# model.eval() 用于验证/推理，不会更新权重
model_path = "../gpt2_finetune"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # ✅ 提前设置 pad_token，避免 warning
model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device)
model.eval()

# 推理函数
def generate_response1(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    inputs['attention_mask'] = inputs['attention_mask']  # ✅ 显式传入 attention_mask
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id  # 用 pad_token 防止 generate 出错
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_response2(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs['attention_mask'] = inputs['attention_mask']  # ✅ 显式传入 attention_mask
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=80,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 示例调用
#print(generate_response1("Hi, you're the first AI I've ever trained"))
#print(generate_response1("今天天气很好，适合"))
print(generate_response1("who are you?"))
