"""Hugging Face Spaces 入口：SDK=Gradio 时自动识别 demo。"""

from agent_eval.demo import create_demo

demo = create_demo()

if __name__ == "__main__":
    demo.launch()
