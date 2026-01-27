import time
import base64
from openai import OpenAI
import os
import openai
client = OpenAI(
    api_key="EMPTY",  # vllm 未开启 api-key 校验的话随便写
    base_url="http://10.11.18.134:8000/v1",
    timeout=3600,
)

# os.environ.setdefault("OPENAI_API_KEY", "xx")
# os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")
# client = openai.OpenAI()
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 8 张图片路径
# image_paths = [
#     "/Users/makabaka/code/mem_vla/_server_data/task_20260120_005357_2fc609ab/images/step_1768841692643.png",
#     "/Users/makabaka/code/mem_vla/_server_data/task_20260120_005357_2fc609ab/images/step_1768841692643.png",
#     "/Users/makabaka/code/mem_vla/_server_data/task_20260120_005357_2fc609ab/images/step_1768841692643.png",
#     "/Users/makabaka/code/mem_vla/_server_data/task_20260120_005357_2fc609ab/images/step_1768841692643.png",
#     "/Users/makabaka/code/mem_vla/_server_data/task_20260120_005357_2fc609ab/images/step_1768841692643.png",
#     "/Users/makabaka/code/mem_vla/_server_data/task_20260120_005357_2fc609ab/images/step_1768841692643.png",
#     "/Users/makabaka/code/mem_vla/_server_data/task_20260120_005357_2fc609ab/images/step_1768841692643.png",
#     "/Users/makabaka/code/mem_vla/_server_data/task_20260120_005357_2fc609ab/images/step_1768841692643.png",
#     "/Users/makabaka/code/mem_vla/_server_data/task_20260120_005357_2fc609ab/images/step_1768841692643.png",
# ]

image_paths = ['/Users/makabaka/code/mem_vla/_server_data/task1_baseline_2/images/init_1769196326755.png']

image_b64_list = [encode_image(p) for p in image_paths]

# 先构造 8 个 image_url 块
image_contents = [
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{b64}"
        },
    }
    for b64 in image_b64_list
]

# 再加上文本块
messages = [
    {
        "role": "user",
        "content": image_contents + [
            {
                "type": "text",
                "text": (
                    # "The current task is pick up the toy croissant on the plate "
                    # "and place it in the box. "
                    # "Tell me if the task is done. "
                    # "Output yes or no without explaination"
                    "the robot arm home pose is: in both images, the arm is aligned parallel to what appears to be a rail or track. The end effector (or gripper) is open and facing downwards towards the table."
                    "Does the robot arm in home pose?"
                ),
            },
        ],
    }
]

start = time.time()
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    # model="gpt-4o-2024-08-06", 
    messages=messages,
    max_tokens=512,
)
print(f"Response costs: {time.time() - start:.2f}s")
print("Generated text:", response.choices[0].message.content)