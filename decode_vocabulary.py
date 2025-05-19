from transformers import AutoTokenizer

# 加载 Qwen 的 tokenizer（以 qwen-7b 为例）
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

# 要查询的 ID
#token_ids = [56940,102065,9370,121705,120612,103437,103541,100739]
token_ids = [100677,  18493,  46306,  60548,  21894, 104448,  22382,  17714,  18600,
          44063,  99590,  35946,  31207, 107887,  32664,  14880,  99535,  30534,
          42140,  49238,  45181, 100873,  39165,  50285,  75882,  14777,  36667,
          99438,  56568,  39973,  30709,  31991,  32648,  99652, 102349,  99991,
          28726,  99360,  19793,  89012,  73038,  99622,  16530,  99841, 100372,
          29437,  18830,  23031,  30440,  15946,  67071,  16872,  17447,  50511,
          63431,  99611,  17177,  33872,  99415,  35987,  57218,  40820,  75223,
          16038,  22243,  42192,  63367,  99200,  33126,  43288,  87026,  42411,
          68756,  11622,  12857,  99704,  29524,  29635,  64754,  73157]

# 打印对应的 token
for token_id in token_ids:
	token = tokenizer.convert_ids_to_tokens(token_id)
	print(f"Token ID {token_id} -> Token: {token}")
	# 先看看原始 token
	token = tokenizer.convert_ids_to_tokens(token_id) 
	# 将 token id 解码为字符串（可以多个一起）
	decoded = tokenizer.decode([token_id])
	print(f"Decoded string: {decoded}")


