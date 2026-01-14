import math
from typing import List

# 假设您的类保存在 encoder_module.py 中
# 如果在同一个文件，可以直接忽略这行 import
from src.memory.encoder import OpenAIEmbeddingEncoder 
import os
import openai

os.environ.setdefault("OPENAI_API_KEY", "sk-fcngxSP4xHGTcaWTGHfnK1BHcQsGMuK6qyAmtnEDGtzJOU2m")
os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    计算两个向量的余弦相似度。
    公式: (A . B) / (||A|| * ||B||)
    """
    # 确保向量长度一致 (OpenAIEmbeddingEncoder 保证了这一点)
    if len(vec_a) != len(vec_b):
        raise ValueError(f"向量维度不匹配: {len(vec_a)} vs {len(vec_b)}")

    # 计算点积
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    
    # 计算模长 (Magnitude)
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    # 防止除以零 (例如 API 报错返回全 0 向量时)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)

def main():
    # 1. 初始化编码器 (确保环境变量 OPENAI_API_KEY 已设置)
    try:
        encoder = OpenAIEmbeddingEncoder()
    except RuntimeError as e:
        print(f"初始化失败: {e}")
        return

    # 2. 定义两组输入
    text1 = "left box"
    text2 = "middle box"

    # 3. 获取 Embedding
    print("正在调用 OpenAI API 生成 Embedding...")
    vector1 = encoder.encode_text(text1)
    vector2 = encoder.encode_text(text2)

    # 4. 计算相似度
    similarity_1_2 = cosine_similarity(vector1, vector2)

    # 5. 输出结果
    print(f"\n文本 1: {text1}")
    print(f"文本 2: {text2}")
    print(f"相似度: {similarity_1_2:.4f}")


if __name__ == "__main__":
    main()