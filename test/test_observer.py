# src/test.py
import os
import openai

from src.client.base_vlm_client import BaseVLMClient
from src.client.observer_vlm import ObserverVLM
from src.agent.observer import ObserverAgent


def main():
    # ========= 1. 配置 API =========
    # （如果你已经在环境变量里配好了，可以省略以下两行）
    os.environ.setdefault("OPENAI_API_KEY", "sk-fcngxSP4xHGTcaWTGHfnK1BHcQsGMuK6qyAmtnEDGtzJOU2m")
    os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")

    client = openai.OpenAI()

    # ========= 2. 初始化 BaseVLMClient & ObserverVLM =========
    base_client = BaseVLMClient(
        model="gpt-4o",   # 或你的模型名
        client=client,
    )
    observer_vlm = ObserverVLM(base_client=base_client)

    # ========= 3. 初始化 ObserverAgent =========
    observer = ObserverAgent(vlm=observer_vlm)

    # ========= 4. 准备图片路径（示例，按你自己的改） =========
    image_paths = [
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000000.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000001.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000002.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000003.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000004.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000005.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000006.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000007.png",
    ]

    # ========= 5. 准备 plan_list =========
    plan_list = """
        [current] Inspect the left box.
        [pending] Inspect the right box.
        [pending] Pick up the toy duck on the table and place it in the box that contains the toy duck.
        [pending] Pick up the snacks on the table and place it in the box that contains the snacks.
        """.strip()

    # ========= 6. 调用 ObserverAgent =========
    print("=== 调用 ObserverAgent ===")
    result = observer.run(
        image_paths=image_paths,
        plan_list=plan_list,
        max_tokens=256,
    )

    # ========= 7. 打印结果 =========
    print("\n=== Observer 结果 ===")
    print("status:", result.status)
    print("\n--- 原始模型输出 (raw_xml) ---")
    print(result.raw_xml)


if __name__ == "__main__":
    main()