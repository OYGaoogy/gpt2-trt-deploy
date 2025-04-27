import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
from transformers import GPT2Tokenizer

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
onnx_path = "../model/gpt2.onnx"
model_path = "../gpt2_finetune"

# === 初始化显式 CUDA context ===
cuda.init()
DEVICE = cuda.Device(0)
CTX = DEVICE.make_context()

# === 构建 TensorRT 引擎 ===
def build_engine(onnx_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        profile = builder.create_optimization_profile()
        profile.set_shape("input_ids", (1, 8), (1, 16), (1, 32))
        profile.set_shape("attention_mask", (1, 8), (1, 16), (1, 32))
        config.add_optimization_profile(profile)

        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                print("❌ ONNX 解析失败")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        return builder.build_engine(network, config)

print("🚀 初始化 TensorRT 引擎和分词器...")
ENGINE = build_engine(onnx_path)
CONTEXT = ENGINE.create_execution_context()
TOKENIZER = GPT2Tokenizer.from_pretrained(model_path)

# === 推理函数 ===
def infer_tensorrt(prompt):
    CTX.push()  # ✅ 显式进入上下文

    try:
        inputs = TOKENIZER(prompt, return_tensors="np", padding="max_length", truncation=True, max_length=16)
        input_ids = inputs["input_ids"].astype(np.int32)
        attention_mask = inputs["attention_mask"].astype(np.int32)

        input_ids_bind = ENGINE.get_binding_index("input_ids")
        attention_bind = ENGINE.get_binding_index("attention_mask")
        output_bind = ENGINE.get_binding_index("logits")

        CONTEXT.set_binding_shape(input_ids_bind, input_ids.shape)
        CONTEXT.set_binding_shape(attention_bind, attention_mask.shape)

        d_input_ids = cuda.mem_alloc(input_ids.nbytes)
        d_attention = cuda.mem_alloc(attention_mask.nbytes)
        output_shape = (1, 16, ENGINE.get_binding_shape(output_bind)[-1])
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        cuda.memcpy_htod(d_input_ids, input_ids)
        cuda.memcpy_htod(d_attention, attention_mask)

        start = time.time()
        CONTEXT.execute_v2([int(d_input_ids), int(d_attention), int(d_output)])
        end = time.time()

        cuda.memcpy_dtoh(output, d_output)
        elapsed = (end - start) * 1000

        return output, elapsed

    finally:
        CTX.pop()  # ✅ 推理完成后释放上下文
