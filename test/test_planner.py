# src/test_planner.py
import os
import openai

from src.memory.encoder import ZeroEncoder, OpenAIEmbeddingEncoder
from src.memory.recorder import Memory

from src.client.base_vlm_client import BaseVLMClient
from src.client.planner_vlm import PlannerVLM
from src.agent.planner import PlannerAgent


def build_memory() -> Memory:
    """
    可选：初始化一个 Memory 实例。这里给一个最简单的空 Memory。
    你也可以在这里预置一些常识性记忆。
    """
    encoder = OpenAIEmbeddingEncoder(
        embedding_dim=3072,
        model="text-embedding-3-large",
        api_key_env="OPENAI_API_KEY",  # 与之前相同的 key 读取方式
    )
    mem = Memory(encoder=encoder)

    # # 示例：预置一点关于左/右箱子的经验
    # mem.create(
    #     obj_name="left_box",
    #     data_type="text",
    #     data_value="The left box is usually used to store toy ducks.",
    #     text="The left box is usually used to store toy ducks.",
    # )
    # mem.create(
    #     obj_name="right_box",
    #     data_type="text",
    #     data_value="The right box is usually used to store snacks.",
    #     text="The right box is usually used to store snacks.",
    # )

    return mem


def main():
    # ========= 1. 配置 API =========
    os.environ.setdefault("OPENAI_API_KEY", "sk-fcngxSP4xHGTcaWTGHfnK1BHcQsGMuK6qyAmtnEDGtzJOU2m")
    os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")

    client = openai.OpenAI()

    # ========= 2. 初始化 BaseVLMClient & PlannerVLM =========
    base_client = BaseVLMClient(
        model="claude-sonnet-4-5-20250929",   # 换成你的真实模型名
        # model="gpt-4o",
        client=client,
    )
    planner_vlm = PlannerVLM(base_client=base_client)

    # ========= 3. 初始化 Memory & PlannerAgent =========
    memory = build_memory()
    planner = PlannerAgent(
        vlm=planner_vlm,
        memory=memory,
    )

    # ========= 4. 准备多轮 frames（与你的伪代码一致） =========
    image_paths = [
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000030.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000038.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000042.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000050.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000058.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000066.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000084.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000092.png",
    ]

    # image_paths = [
    #     "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000000.png",
    #     "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000001.png",
    #     "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000002.png",
    #     "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000003.png",
    #     "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000004.png",
    #     "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000005.png",
    #     "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000006.png",
    #     "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000007.png",
    # ]
    
    # user_instruction=""
    user_instruction = """
    Use the robot arm to move the toy duck and the snack from the table into the box containing toy ducks and into the box containing snacks.
    """.strip()

    # 初始 plan_list 为空，让 planner 在多轮推理后给出最终规划
    initial_plan_list = """
        [current] inspect the left box (determine if it is for toy ducks)
        [pending] inspect the right box (determine if it is for snacks)
        [pending] pick up the toy duck and place it in the box for toy ducks (once identified)
        [pending] pick up the snack and place it in the box for snacks (once identified)
        """
    # initial_plan_list = ""

    # ========= 5. 多轮调用 PlannerAgent =========
    print("=== 调用 PlannerAgent.run_multi_rounds ===")
    result = planner.run_refine(
        image_paths=image_paths,
        initial_plan_list=initial_plan_list,
        user_instruction=user_instruction,
        max_tokens=4096,
    )

    # # ========= 6. 打印结果 =========
    # print("\n=== Planner 最终结果 ===")
    # print("---- FINAL PLAN_LIST ----")
    # print(result.plan_text)

    # print("\n---- SUMMARY ----")
    # print(result.summary)

    # print("\n---- MEMORY OPERATIONS (last round) ----")
    # for op in result.memory_operations:
    #     print(op)

    # print("\n---- 原始模型输出 (raw_xml, last round) ----")
    # print(result.raw_xml)


if __name__ == "__main__":
    main()