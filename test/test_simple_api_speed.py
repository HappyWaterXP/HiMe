#!/usr/bin/env python3
"""
Simple VLM API Performance Test

Tests API call latency with multiple images in a single OpenAI-format request.
Direct OpenAI client usage with no wrapper classes.

IMPORTANT: The server (https://aigc.x-see.cn) does NOT support the 'detail' field
in image_url, so we use standard format WITHOUT 'detail'.

Usage:
    uv run python src/test/test_simple_api_speed.py --model qwen3-vl-235b-a22b-instruct
    uv run python src/test/test_simple_api_speed.py --model claude-sonnet-4-5-20250929
    uv run python src/test/test_simple_api_speed.py --model gpt-4o
"""

import os
import time
import argparse
import openai
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image


def create_test_images(count: int) -> list[Image.Image]:
    """Create test images with varying colors."""
    images = []
    for i in range(count):
        color = (100 + i * 10, 150 + i * 5, 200 - i * 8)
        img = Image.new('RGB', (640, 480), color=color)
        images.append(img)
    return images


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_data}"


def test_api_speed():
    """Test API call speed with 8 images."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-235b-a22b-instruct",
        help="Model name to test"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=8,
        help="Number of images to send"
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"Simple VLM API Speed Test")
    print(f"Model: {args.model}")
    print(f"Images: {args.num_images}")
    print("=" * 70)

    # Setup OpenAI client
    os.environ.setdefault("OPENAI_API_KEY", "xx")
    os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")
    client = openai.OpenAI()

    # Create test images
    images = create_test_images(args.num_images)
    image_blocks = [
        {
            "type": "text",
            "text": "You are watching a robot task. Return only 'yes' or 'no' to indicate if task is done."
        }
    ]

    # Add all images to message
    for i, img in enumerate(images):
        print(f"Preparing image {i+1}/{len(images)}...", end="")
        image_url = image_to_base64(img)
        image_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": image_url
                # NOTE: 'detail' field removed - server doesn't support it
            }
        })
        print(" ✅")

    # Message structure
    messages = [
        {
            "role": "user",
            "content": image_blocks
        }
    ]

    print("\n" + "-"*70)
    print("Sending API request...")
    print(f"{len(images)} images in one request")

    # Time the API call
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=50
        )
        elapsed = time.perf_counter() - start

        print("="*70)
        print("RESULTS:")
        print("="*70)
        print(f"API call succeeded: {response.choices[0].message.content}")
        print(f"Total time: {elapsed:.3f} seconds")
        print(f"Average per image: {elapsed/len(images):.3f} seconds")
        print(f"No blocks: API call completed immediately")
        return 0

    except openai.APIError as e:
        elapsed = time.perf_counter() - start
        print("="*70)
        print(f"API error after {elapsed:.3f}s")
        print(f"Error: {e}")
        return 1

    except Exception as e:
        elapsed = time.perf_counter() - start
        print("="*70)
        print(f"Request failed after {elapsed:.3f}s")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(test_api_speed())