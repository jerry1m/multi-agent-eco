"""Plan-and-Solve 范式示例（离线模拟）

展示：
- LLM 先生成一个明确的分步计划（JSON list）
- 逐步执行计划，每步可调用工具或内部函数
"""

import json


def mock_llm_plan(problem_text):
    # 简单策略：若问题是求数列之和，计划为解析、计算、汇报
    plan = [
        {"step": 1, "action": "parse", "detail": "解析输入，提取数字"},
        {"step": 2, "action": "compute", "detail": "计算数字之和"},
        {"step": 3, "action": "format", "detail": "生成最终回答"},
    ]
    return plan


def execute_plan(plan, context):
    state = {}
    for p in plan:
        print(f"执行步骤 {p['step']}: {p['action']} — {p['detail']}")
        if p["action"] == "parse":
            nums = [int(x) for x in context.split() if x.isdigit()]
            state["nums"] = nums
            print("解析结果:", nums)
        elif p["action"] == "compute":
            state["sum"] = sum(state.get("nums", []))
            print("计算结果:", state["sum"]) 
        elif p["action"] == "format":
            print("最终回答: 输入数字之和为", state.get("sum"))
    return state


def run_demo():
    print("=== Plan-and-Solve Demo ===")
    problem = "请计算以下数字的和： 3 4 8"
    plan = mock_llm_plan(problem)
    print("LLM 生成计划(严格 JSON 格式):")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    execute_plan(plan, "3 4 8")


if __name__ == "__main__":
    run_demo()
