import os
from openai import OpenAI

# ========= 1. 环境变量配置（示例） =========
os.environ.setdefault("OPENAI_API_KEY", "sk-fcngxSP4xHGTcaWTGHfnK1BHcQsGMuK6qyAmtnEDGtzJOU2m")
os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")

# ========= 2. 初始化 OpenAI client =========
client = OpenAI()  # 自动从环境变量读取 KEY / BASE_URL

# ========= 3. 调用 text-embedding-3-large =========
def demo_embedding():
    text = "Hello, this is a test for text-embedding-3-large."

    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )

    emb = resp.data[0].embedding
    print("=== embedding info ===")
    print("dim:", len(emb))
    print("first 8 values:", emb[:8])

    # 如果你想看完整向量，可以打印：
    # print(emb)

if __name__ == "__main__":
    demo_embedding()