import time
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# ====== 通用配置 ======
model_path = "../gpt2_finetune"
onnx_path = "../model/gpt2.onnx"
prompt = "Hello world"


# ====== PyTorch 推理 ======
def infer_pytorch(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).eval()
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        end = time.time()

    logits = outputs.logits.detach().numpy()
    elapsed_ms = (end - start) * 1000
    return logits, elapsed_ms


# ====== TensorRT 推理 ======
def build_trt_engine(onnx_path):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        profile = builder.create_optimization_profile()
        profile.set_shape("input_ids", (1, 8), (1, 10), (1, 16))
        profile.set_shape("attention_mask", (1, 8), (1, 10), (1, 16))
        config.add_optimization_profile(profile)

        with open(onnx_path, "rb") as model_file:
            parser.parse(model_file.read())

        return builder.build_engine(network, config)


def infer_tensorrt(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=10)
    input_ids = inputs["input_ids"].astype(np.int32)
    attention_mask = inputs["attention_mask"].astype(np.int32)

    engine = build_trt_engine(onnx_path)
    context = engine.create_execution_context()

    input_ids_bind = engine.get_binding_index("input_ids")
    attention_bind = engine.get_binding_index("attention_mask")
    output_bind = engine.get_binding_index("logits")

    context.set_binding_shape(input_ids_bind, input_ids.shape)
    context.set_binding_shape(attention_bind, attention_mask.shape)

    d_input_ids = cuda.mem_alloc(input_ids.nbytes)
    d_attention = cuda.mem_alloc(attention_mask.nbytes)
    output_shape = (1, 10, engine.get_binding_shape(output_bind)[-1])
    output = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(d_input_ids, input_ids)
    cuda.memcpy_htod(d_attention, attention_mask)

    start = time.time()
    context.execute_v2([int(d_input_ids), int(d_attention), int(d_output)])
    end = time.time()

    cuda.memcpy_dtoh(output, d_output)

    elapsed_ms = (end - start) * 1000
    return output, elapsed_ms


# ====== 对比函数 ======
def compare_infer(prompt):
    print(f"\n🚀 对比推理结果（prompt: '{prompt}'）")

    logits_pt, time_pt = infer_pytorch(prompt)
    logits_trt, time_trt = infer_tensorrt(prompt)

    print(f"🔹 PyTorch 耗时：{time_pt:.2f} ms")
    print(f"🔹 TensorRT 耗时：{time_trt:.2f} ms")

    top1_pt = logits_pt[0, -1].argmax()
    top1_trt = logits_trt[0, -1].argmax()
    print(f"🔹 Top-1 Token (PyTorch)   : {top1_pt}")
    print(f"🔹 Top-1 Token (TensorRT) : {top1_trt}")
    print(f"✅ 输出是否一致：{'✅ 是' if top1_pt == top1_trt else '❌ 否'}")


# ====== 运行测试 ======
if __name__ == "__main__":
    compare_infer(prompt)
