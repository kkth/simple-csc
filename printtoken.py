from transformers import AutoTokenizer

# 选择模型，比如 Qwen2.5-1.5B
model_name = "Qwen/Qwen2.5-1.5B"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 打印 eos_token 及其 ID
print("模型名称:", model_name)
print("eos_token:", repr(tokenizer.eos_token))
print("eos_token_id:", tokenizer.eos_token_id)

# 也可以打印其他特殊 token 信息
print("bos_token:", repr(tokenizer.bos_token), "id:", tokenizer.bos_token_id)
print("pad_token:", repr(tokenizer.pad_token), "id:", tokenizer.pad_token_id)

