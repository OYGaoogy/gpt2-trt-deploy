# copied from: python2_onnx_tensorrt_infer/infer_api_server.py
# modified for: docker deployment
# date: 2025-04-27

from flask import Flask, request, jsonify
from trt_infer import infer_tensorrt, TOKENIZER
import logging, os

# === 屏蔽警告日志输出 ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel(logging.ERROR)

app = Flask(__name__)

@app.route("/infer", methods=["POST"])
def infer():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        print(f"📥 收到 prompt: {prompt}")

        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        '''
        logits[0, -1]：取出最后一个时间步（sequence最后一个 token）的预测分数。
        .argmax()：找到最高分的 token id（也就是最有可能生成的词）。
        TOKENIZER.decode()：把 token id 还原成真实的文字。
        '''
        logits, _ = infer_tensorrt(prompt)
        top1_token_id = int(logits[0, -1].argmax())
        top1_token = TOKENIZER.decode([top1_token_id], clean_up_tokenization_spaces=True).strip()

        if not top1_token:
            top1_token = f"[token_id={top1_token_id}]"

        return jsonify({"response": f"模型生成词: {top1_token}"})

    except Exception as e:
        print("❌ 推理失败:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6006)
