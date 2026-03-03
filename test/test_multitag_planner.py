# src/test_multitag_planner.py
import os
import openai

from src.memory.encoder import OpenAIEmbeddingEncoder
from src.memory.multitag_recorder import MultiTagMemory

from src.client.base_vlm_client import BaseVLMClient
from src.client.planner_vlm import PlannerVLM
from src.agent.multitag_planner import PlannerAgent


def build_multitag_memory() -> MultiTagMemory:
    """
    初始化一个 MultiTagMemory 实例
    
    ✅ 特点：
    - 支持多标签（tags）
    - 支持图片记忆
    - 语义搜索
    """
    encoder = OpenAIEmbeddingEncoder(
        embedding_dim=3072,
        model="text-embedding-3-large",
        api_key_env="OPENAI_API_KEY",
    )
    mem = MultiTagMemory(encoder=encoder)

    # ========= 预置一些多标签记忆（可选） =========
    
    # 示例 1: 关于箱子的经验（纯文本）
    mem.create(
        tags=["left_box", "storage", "rule"],
        data_type="text",
        data_value="The left box is designated for toy ducks. Always verify by checking the label.",
        text="The left box is designated for toy ducks. Always verify by checking the label.",
    )
    
    mem.create(
        tags=["right_box", "storage", "rule"],
        data_type="text",
        data_value="The right box is designated for snacks. Check the sticker on the side.",
        text="The right box is designated for snacks. Check the sticker on the side.",
    )
    

    print(f"✅ Initialized MultiTagMemory with {len(mem.all())} pre-loaded records\n")
    return mem


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_memory_state(memory: MultiTagMemory):
    """打印当前 Memory 的状态"""
    print_separator("CURRENT MEMORY STATE")
    
    if not memory.all():
        print("(Memory is empty)")
        return
    
    for rec in memory.all():
        tags_str = ", ".join(rec.tags)
        data_type = rec.data_type
        
        print(f"ID: {rec.id}")
        print(f"  Tags: [{tags_str}]")
        print(f"  Type: {data_type}")
        
        if data_type == "text":
            text = rec.text
            print(f"  Text: {text[:100]}..." if len(text) > 100 else f"  Text: {text}")
        elif data_type == "image":
            desc = rec.text
            img_path = rec.image_path
            print(f"  Description: {desc}")
            print(f"  Image: {img_path}")
        
        print()


def main():
    # ========= 1. 配置 API =========
    os.environ.setdefault("OPENAI_API_KEY", "sk-fcngxSP4xHGTcaWTGHfnK1BHcQsGMuK6qyAmtnEDGtzJOU2m")
    os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")

    client = openai.OpenAI()

    # ========= 2. 初始化 BaseVLMClient & PlannerVLM =========
    print_separator("INITIALIZING PLANNER SYSTEM")
    
    base_client = BaseVLMClient(
        model="claude-sonnet-4-5-20250929",
        client=client,
    )
    planner_vlm = PlannerVLM(base_client=base_client)
    print("✅ VLM Client initialized")

    # ========= 3. 初始化 MultiTagMemory & PlannerAgent =========
    memory = build_multitag_memory()
    
    planner = PlannerAgent(
        vlm=planner_vlm,
        memory=memory,
    )
    print("✅ PlannerAgent initialized with MultiTagMemory\n")

    # ========= 4. 准备测试图片 =========
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
    
    user_instruction = """
    Use the robot arm to move the toy duck and the snack from the table into the correct boxes.
    The left box should contain toy ducks, and the right box should contain snacks.
    Please verify the boxes before placing items.
    """.strip()

    initial_plan_list = ""

    # ========= 5. 执行多轮规划 =========
    print_separator("STARTING MULTI-ROUND PLANNING")
    
    print(f"📸 Input: {len(image_paths)} images")
    print(f"📝 Instruction: {user_instruction}\n")
    
    result = planner.run_refine(
        image_paths=image_paths,
        initial_plan_list=initial_plan_list,
        user_instruction=user_instruction,
        max_tokens=4096,
        max_inner_rounds=10,
    )

    # ========= 6. 打印完整对话历史 =========
    planner.print_conversation()

    # ========= 7. 打印最终结果 =========
    print_separator("FINAL PLANNING RESULT")
    
    print("📋 FINAL PLAN:")
    print("-" * 80)
    print(result.plan_text if result.plan_text else "(No plan generated)")
    print()
    
    print("📝 SUMMARY:")
    print("-" * 80)
    print(result.summary if result.summary else "(No summary)")
    print()
    
    print("🔧 MEMORY OPERATIONS (Last Round):")
    print("-" * 80)
    if result.memory_operations:
        for i, op in enumerate(result.memory_operations, 1):
            print(f"{i}. Type: {op.type}")
            if op.tags:
                print(f"   Tags: {op.tags}")
            if op.obj_name:
                print(f"   Object: {op.obj_name}")
            if op.text:
                text_preview = op.text[:60] + "..." if len(op.text) > 60 else op.text
                print(f"   Text: {text_preview}")
            if op.image_path:
                print(f"   Image: {op.image_path}")
            if op.query:
                print(f"   Query: {op.query}")
            if op.reason:
                print(f"   Reason: {op.reason}")
            print()
    else:
        print("(No memory operations in last round)")
    
    # ========= 8. 打印更新后的 Memory 状态 =========
    print_memory_state(memory)
    
    print_separator("TEST COMPLETED")
    print(f"all tags: {memory.get_all_tags()}")
    print(f"✅ Total memory records: {len(memory.all())}")
    print(f"✅ Planning completed successfully")


if __name__ == "__main__":
    main()
