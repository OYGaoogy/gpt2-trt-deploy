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
