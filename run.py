"""Run the autonomous QA agent against the sample app.

Usage:
    python run.py             # offline demo brain (no API key needed)
    python run.py deepseek    # real model via DEEPSEEK_API_KEY
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.qa_agent.agent import build_agent, build_mock_llm


def main():
    use_real = len(sys.argv) > 1 and sys.argv[1] == "deepseek"
    if use_real:
        from core.llm import DeepSeekClient
        llm = DeepSeekClient()
    else:
        llm = build_mock_llm()
    agent = build_agent(llm)
    trace = agent.run({
        "scenarios": "agents/qa_agent/scenarios.json",
        "app": "agents/qa_agent/sample_app",
    })
    print(f"outcome: {trace.outcome}")
    print(f"summary: {trace.summary}")
    print(f"steps:   {trace.iteration_count}")
    trace.save("trace.json")
    print("trace saved to trace.json")


if __name__ == "__main__":
    main()
