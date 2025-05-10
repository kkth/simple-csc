from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

token_id = [109733, 100431, 109949, 101047,  28726,  62922,  18600, 101034,  42140,
          18600,  82647,  18600,  32100,  90395, 107485,  25177, 104813, 109949,
           8997,  31196,   5122, 100677, 104615,  17447,  22243,  99185,  32648,
          99473,   8997,  66017,   5122]
text = tokenizer.decode(token_id, skip_special_tokens=False)
print(f"Token ID {token_id} 对应字符是: {repr(text)}")

