"""LLM backends.

Two implementations share one contract: `decide(...)` returns a JSON-ish dict
with {"thought", "action", "args"}.

* OpenAIClient - talks to any OpenAI-compatible /chat/completions endpoint.
* DeepSeekClient - same contract, pointed at DeepSeek's API.
* MockLLM - deterministic, offline "demo brain". Lets the whole project run
  without an API key, and makes every run reproducible for interviews.
"""

import json
import os
import re
import urllib.error
import urllib.request


def load_env_file():
    """Load KEY=VALUE pairs from a project-local .env file (if present).
    Never overrides variables that are already set in the environment."""
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def build_system_prompt(agent_prompt, tools):
    lines = [agent_prompt, "", "AVAILABLE TOOLS:", ""]
    for tool in tools.list():
        lines.append(f"- {tool.name}: {tool.description}")
        lines.append(f"  args schema: {json.dumps(tool.args)}")
    lines.extend([
        "",
        "You work in a loop. Each turn pick ONE tool and respond with JSON only:",
        '{"thought": "...", "action": "<tool name>", "args": {...}}',
        'When the task is fully complete respond with "finish":',
        '{"thought": "...", "action": "finish", "args": {"summary": "..."}}',
        "Your work is checked after every step. If you finish prematurely you",
        "will receive feedback describing what remains and must continue.",
        "Never call finish until your own verification passes.",
    ])
    return "\n".join(lines)


class LLM:
    name = "base"

    def decide(self, task, tools, state, history, verifier_feedback):
        raise NotImplementedError


class OpenAIClient(LLM):
    name = "openai"

    def __init__(self, model=None, api_key=None, endpoint=None, timeout=120):
        load_env_file()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Use the mock LLM for an offline "
                "demo, or export OPENAI_API_KEY to use a real model."
            )
        self.endpoint = endpoint or os.environ.get(
            "OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions"
        )
        self.timeout = timeout

    def decide(self, task, tools, state, history, verifier_feedback):
        messages = [
            {"role": "system",
             "content": build_system_prompt(
                 task.get("system_prompt", ""), tools)},
            {"role": "user",
             "content": f"TASK: {json.dumps(task, default=str)}"},
        ]
        for step in history:
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "thought": step["thought"],
                    "action": step["action"],
                    "args": step["args"],
                }),
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {step['observation']}",
            })
        if verifier_feedback:
            messages.append({
                "role": "user",
                "content": f"VERIFIER: {verifier_feedback} - do not call "
                           "finish yet; keep working.",
            })
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
            content = data["choices"][0]["message"]["content"]
            return self._parse(content)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"LLM API error {exc.code}: {body[:500]}") from exc

    @staticmethod
    def _parse(content):
        text = content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Could not parse LLM output: {text[:300]}")


class DeepSeekClient(OpenAIClient):
    """OpenAI-compatible client pointed at DeepSeek's API.

    Uses DEEPSEEK_API_KEY by default and deepseek-chat unless DEEPSEEK_MODEL
    is set. Point DEEPSEEK_ENDPOINT elsewhere for a DeepSeek-compatible proxy.
    """

    name = "deepseek"

    def __init__(self, model=None, api_key=None, endpoint=None, timeout=180):
        load_env_file()
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Export DEEPSEEK_API_KEY to use "
                "DeepSeek, or use the mock LLM for an offline demo."
            )
        super().__init__(
            model=model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=api_key,
            endpoint=endpoint or os.environ.get(
                "DEEPSEEK_ENDPOINT",
                "https://api.deepseek.com/chat/completions"),
            timeout=timeout,
        )


class MockLLM(LLM):
    """Deterministic stand-in for a real model.

    The policy is a callable with the same contract as decide(). It plays the
    role of the reasoning model so the loop, tools, tracing and eval harness
    can be demonstrated end-to-end without an API key.
    """

    name = "mock"

    def __init__(self, policy):
        self.policy = policy

    def decide(self, task, tools, state, history, verifier_feedback):
        return self.policy(task, state, history, verifier_feedback)
