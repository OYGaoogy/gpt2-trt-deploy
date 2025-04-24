FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 先只复制 requirements.txt 并安装依赖（缓存依赖层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制代码和模型（不影响缓存依赖）
COPY app.py .
COPY gpt2_finetune ./gpt2_finetune

# ✅ 加上模板目录（HTML 界面）
COPY templates ./templates
COPY static ./static

# 暴露端口
EXPOSE 5000

# 启动 Flask 服务
CMD ["python3", "app.py"]
