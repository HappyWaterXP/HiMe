# test_openai_multi_images.py
import os
import openai
from typing import List
import base64
from pathlib import Path


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_multi_images_distinction(
    image_paths: List[str],
    model: str = "gpt-4o",
    prompt: str = "Please describe each image separately. Number them as Image 1, Image 2, etc., and tell me what you see in each one."
):
    """
    测试模型能否区分多张图片
    
    Args:
        image_paths: 图片路径列表
        model: 模型名称
        prompt: 提示词
    """
    
    # 构建消息内容
    content = [
        {"type": "text", "text": prompt}
    ]
    
    # 添加所有图片
    for i, img_path in enumerate(image_paths, 1):
        if not Path(img_path).exists():
            print(f"Warning: 图片不存在 - {img_path}")
            continue
            
        img_base64 = encode_image_to_base64(img_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            }
        })
        print(f"已添加图片 {i}: {Path(img_path).name}")
    
    # 调用 API
    print(f"\n正在调用模型: {model}")
    print(f"图片数量: {len(image_paths)}")
    print("-" * 80)
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=2000,
    )
    
    # 打印结果
    print("\n=== 模型回复 ===\n")
    print(response.choices[0].message.content)
    print("\n" + "=" * 80)
    
    return response.choices[0].message.content


def main():
    # ========= 1. 配置 API =========
    os.environ.setdefault("OPENAI_API_KEY", "sk-fcngxSP4xHGTcaWTGHfnK1BHcQsGMuK6qyAmtnEDGtzJOU2m")
    os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")
    
    # ========= 2. 准备测试图片 =========
    image_paths = [
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000000.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000001/frame_000000.png",
        "/inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000002/frame_000000.png",
    ]
    
    # ========= 3. 测试不同的 prompts =========
    
    # 测试 1: 基础区分测试
    print("\n" + "=" * 80)
    print("测试 1: 基础图片区分")
    print("=" * 80)
    test_multi_images_distinction(
        image_paths=image_paths,
        model="claude-sonnet-4-20250514",  # 或 "gpt-4o"
        prompt="Please describe each image separately. Number them as Image 1, Image 2, etc.,tell me the position of robot arm, which box is the arm being inspected?"
    )
    


if __name__ == "__main__":
    main()