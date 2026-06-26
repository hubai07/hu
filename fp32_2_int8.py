from onnxruntime.quantization import QuantType, quantize_dynamic

# 换成你绝对路径，避免相对路径找不到文件
model_fp32 = "./best.onnx"
model_quant = "./model_int8.onnx"

# 动态量化，只量化卷积、全连接层，保护YOLO检测输出精度
quantize_dynamic(
    model_input=model_fp32,
    model_output=model_quant,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["Conv", "MatMul"],
)
print(f"量化完成，量化模型保存路径：{model_quant}")
