# -*- coding: utf-8 -*-
"""
vlm_selector.py

这个脚本通过一个兼容OpenAI的API来分析带标签的图片，并根据文本描述找出最匹配的图片区域。
这允许用户连接到自托管的视觉语言模型（VLM）服务。

依赖:
- openai: 通过 `pip install openai` 安装。
- Pillow: 通过 `pip install Pillow` 安装。
- python-dotenv: 通过 `pip install python-dotenv` 安装。

环境配置:
1. 将 `.env.example` 文件复制为 `.env`。
2. 在 `.env` 文件中填入你的配置信息：
   - OPENAI_API_KEY: 你的API密钥。
   - OPENAI_API_BASE: 你的自托管API端点URL。
   - OPENAI_MODEL_NAME: 你要使用的模型名称。

用法:
python vlm_selector.py <图片路径> "<文本描述>"

示例:
python vlm_selector.py labeled_screenshot.png "一只正在打哈欠的猫"
"""

import os
import sys
import base64
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

def encode_image_to_base64(image_path):
    """将图片文件编码为Base64字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"错误：图片文件未找到于 '{image_path}'。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误：编码图片时出错: {e}", file=sys.stderr)
        sys.exit(1)


def call_vlm_api(image_path, text_prompt):
    """
    调用兼容OpenAI的VLM API来分析图片。

    Args:
        image_path (str): 要分析的图片文件路径。
        text_prompt (str): 用于描述要寻找的图片的文本。

    Returns:
        str: 模型返回的最匹配图片的数字标签。如果出错则返回错误信息。
    """
    # 从环境变量获取配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("OPENAI_MODEL_NAME")

    if not all([api_key, base_url, model_name]):
        return "错误：请确保 OPENAI_API_KEY, OPENAI_API_BASE, 和 OPENAI_MODEL_NAME 环境变量都已设置。"

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 将图片编码为Base64
        base64_image = encode_image_to_base64(image_path)
        image_url = f"data:image/png;base64,{base64_image}"

        # 构造精确的提示
        prompt_template = f"""
        这是一张带有数字标签的图片搜索结果截图。请仔细观察图片中的每个带标签的区域。
        根据以下描述：'{text_prompt}'，找出最匹配的图片。
        你的回答必须**仅仅**是那张最匹配图片的数字标签，不要包含任何其他文字、解释或标点符号。
        """

        # 调用API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_template},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=10, # 限制token数量，因为我们只需要一个数字
        )

        # 解析响应并提取数字
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"调用API时发生未知错误: {e}"

def main():
    """
    主函数，处理命令行参数并驱动脚本流程。
    """
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("用法: python vlm_selector.py <图片路径> \"<文本描述>\"", file=sys.stderr)
        sys.exit(1)

    image_file_path = sys.argv[1]
    user_text_prompt = sys.argv[2]

    # 调用API并获取结果
    selected_label = call_vlm_api(image_file_path, user_text_prompt)

    # 打印结果到标准输出
    print(selected_label)

if __name__ == "__main__":
    main()