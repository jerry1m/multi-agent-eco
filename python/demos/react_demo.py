"""ReAct 范式示例（离线模拟）

展示：
- LLM 在 `Reason` 步骤生成思考（文本）
- LLM 在 `Action` 步骤选择工具或环境动作
- 环境返回观察（Observation），LLM 根据观察继续

此示例使用简单的 loop 演示推理与动作。
"""

import json
import random


def mock_llm_react(history):
    # 根据简单规则返回 Reason 和 Action
    if "need_sum" in history:
        reason = "需要计算两个数之和以回答用户的问题。"
        action = {"tool": "calculator", "input": "sum: 7,5"}
    else:
        reason = "先询问目标是否是求和或查询事实。"
        action = {"tool": "ask", "input": "请确认是求和吗？"}
    return {"reason": reason, "action": action}


def run_demo():
    history = []
    print("=== ReAct Demo ===")
    # 第一步 LLM 思考并行动
    out = mock_llm_react(history)
    print("LLM Reason:", out["reason"])
    print("LLM Action:", json.dumps(out["action"], ensure_ascii=False))

    # 环境根据 action 返回 observation
    if out["action"]["tool"] == "ask":
        obs = "用户：是的，我要计算两个数的和。"
        print("Observation:", obs)
        history.append("need_sum")
    elif out["action"]["tool"] == "calculator":
        obs = "观察：计算结果为12"
        print("Observation:", obs)
    else:
        obs = "Observation: 未知"

    # 第二轮
    out2 = mock_llm_react(history)
    print("LLM Reason:", out2["reason"])
    print("LLM Action:", json.dumps(out2["action"], ensure_ascii=False))

    # 模拟执行计算器
    if out2["action"]["tool"] == "calculator":
        parts = out2["action"]["input"].split(":", 1)[1].strip()
        a, b = [int(x) for x in parts.split(",")]
        print("工具(计算器) 输出:", a + b)


if __name__ == "__main__":
    run_demo()
