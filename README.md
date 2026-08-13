# Autonomous QA Agent

An agent that starts a live web app, runs a test scenario suite against it,
detects failing behavior, and files bug reports until every failure is
covered:

1. **Plan** - run the full scenario suite.
2. **Act** - probe the app, detect failures (also does its own extra
   probing).
3. **Observe** - read actual vs expected results.
4. **Verify** - file a bug report with reproduction steps for every failure
   and confirm nothing is missing before finishing.

Zero third-party dependencies - pure Python standard library.

## Run it

```cmd
python run.py
```

Offline demo brain by default (no API key). The sample app contains three
planted bugs: equal-operand sums are off by one, and search is
case-sensitive.

For a real model, create a `.env` file (git-ignored) and run:

```text
DEEPSEEK_API_KEY=sk-your-key-here
```

```cmd
python run.py deepseek
```

## Outputs

- `trace.json` - the full step-by-step trace.
- The interactive step-through of a real DeepSeek run:
  [visuals/qa-agent.html](visuals/qa-agent.html)

Part of the [Agentic Workflow Lab](https://github.com/ryanreo/agentic-workflow-lab).
