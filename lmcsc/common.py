from string import punctuation
import re

RADICAL_INDEX = 0
COMPONENT_INDEX = 1
OOV_CHAR = "□"

MIN = -1e32
HALF_MIN = -1e4
MAX = 1e32
EPS = 1e-7

chinese_punct = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟–—‘'‛“”„‟…‧."
english_punct = punctuation
PUNCT = set(chinese_punct + english_punct)

consonant_inits = {"q", "w", "r", "t", "y", "o", "p", "s", "d", "f", "g", "h", "j", "k", "l", "z", "x", "c", "b", "n", "m"}
reAlNUM = re.compile(r"^[a-zA-Z0-9]+$")

import gc
import torch

def print_tensor_info():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                device = obj.device if hasattr(obj, 'device') else obj.data.device
                print(f"Tensor shape: {obj.shape}, dtype: {obj.dtype}, device: {device}")
        except Exception as e:
            pass  # 有些对象可能访问失败