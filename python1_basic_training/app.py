# copied from: python1_basic_training/transformers3_DeploymentFlask.py
# modified for: docker deployment
# date: 2025-04-27

from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only ❌")

# 初始化模型
# model.train() 用于训练阶段，会更新权重，梯度计算后更新参数，模型默认值
# model.eval() 用于验证/推理，不会更新权重
model_path = "../gpt2_finetune"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token   # ✅ 提前设置 pad_token，避免 warning
model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()
# model = GPT2LMHeadModel.from_pretrained(model_path)
# model = model.to(device)
# model.eval()

# 推理函数
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    inputs['attention_mask'] = inputs['attention_mask']  # ✅ 显式传入 attention_mask
    with torch.no_grad():   # 临时作用域，用于临时关闭梯度计算（提高速度，节省内存）
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

# 构造 Flask 应用，初始化 Flask Web 服务器
app = Flask(__name__)

# 加载 templates/index.html
@app.route("/")
def index():
    return render_template("index.html")

# API 路由
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    output = generate_response(prompt)
    return jsonify({"response": output})

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "gpu": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    })

# 启动 Flask 服务（监听全部 IP）
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
