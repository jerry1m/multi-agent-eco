"""Function Calling / Tool Use 示例（离线模拟）

展示：
- 通过 System Prompt 风格约束，要求模型仅输出严格的 JSON
- 解析 JSON 并调用相应本地工具函数（搜索/计算）
"""

import json


def search_tool(query):
    # 模拟搜索结果
    return {"top_result": f"Search result for '{query}'", "score": 0.9}


def calculator_tool(expr):
    # 极简计算器，仅支持加法
    parts = [int(x.strip()) for x in expr.split("+")]
    return sum(parts)


def mock_llm_function_call(prompt):
    # 模拟 LLM 在 system prompt 约束下输出 JSON
    # 输出两种形式之一：调用 search 或 calculator
    if "查找" in prompt:
        return json.dumps({"tool": "search", "args": {"q": "anomaly detection"}}, ensure_ascii=False)
    else:
        return json.dumps({"tool": "calculator", "args": {"expr": "10+20+3"}}, ensure_ascii=False)


def run_demo():
    print("=== Function Calling Demo ===")
    prompts = ["请帮我查找有关异常检测的资源。", "请计算 10+20+3 的结果，严格输出 JSON 调用格式。"]
    for p in prompts:
        print("\nPrompt:", p)
        llm_out = mock_llm_function_call(p)
        print("LLM 输出(应为纯 JSON):", llm_out)
        try:
            payload = json.loads(llm_out)
        except Exception as e:
            print("解析 JSON 失败:", e)
            continue

        tool = payload.get("tool")
        args = payload.get("args", {})
        if tool == "search":
            res = search_tool(args.get("q", ""))
            print("调用 search_tool，返回:", res)
        elif tool == "calculator":
            res = calculator_tool(args.get("expr", "0"))
            print("调用 calculator_tool，返回:", res)
        else:
            print("未知工具:", tool)


if __name__ == "__main__":
    run_demo()
