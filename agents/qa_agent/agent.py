"""Agent #4: autonomous QA agent.

Starts a local web app, runs a suite of test scenarios against it, detects
failing behavior, and files bug reports with reproduction steps, expected and
actual results - verifying that every failure is covered before finishing.
"""

import json
import os
import re
import urllib.parse
import urllib.request

from core.agent import Agent
from core.llm import MockLLM
from core.tools import Tool, ToolRegistry

from .sample_app.app import start_server

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))
SYSTEM_PROMPT = (
    "You are an autonomous QA engineer. Run the test scenarios against the "
    "application, and for every failing scenario file a bug report containing "
    "reproduction steps, expected result and actual result. Do not finish "
    "until every failing scenario has a complete bug report."
)


def _resolve(task, key, default):
    value = task.get(key) or default
    if not os.path.isabs(value):
        value = os.path.join(ROOT, value)
    return os.path.abspath(value)


def load_scenarios(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def make_tools():
    def http_get(state, args):
        query = urllib.parse.urlencode(args.get("params", {}))
        url = state["base_url"] + args["path"]
        if query:
            url += "?" + query
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = resp.read().decode()
                return f"{resp.status} {body}"
        except urllib.error.HTTPError as exc:
            return f"{exc.code} {exc.read().decode(errors='replace')}"

    def _run_one(state, scenario):
        path = scenario["path"]
        params = scenario.get("params", {})
        query = urllib.parse.urlencode(params)
        url = state["base_url"] + path + (("?" + query) if query else "")
        actual_status = None
        actual_body = ""
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                actual_status = resp.status
                actual_body = resp.read().decode()
        except urllib.error.HTTPError as exc:
            actual_status = exc.code
            actual_body = exc.read().decode(errors="replace")
        expect = scenario["expect"]
        ok = actual_status == expect["status"]
        if ok and "body_is" in expect:
            ok = actual_body == expect["body_is"]
        if ok and "contains" in expect:
            ok = expect["contains"] in actual_body
        repro = f"GET {path}"
        if query:
            repro += "?" + query
        result = {
            "ok": ok, "repro": repro,
            "expected": expect.get("body_is") or expect.get("contains") or
                        str(expect["status"]),
            "actual": f"{actual_status} {actual_body}",
        }
        state["results"][scenario["id"]] = result
        if ok:
            return f"PASS: {scenario['id']} — {repro} -> {actual_status} {actual_body}"
        return (f"FAIL: {scenario['id']} — {repro} -> "
                f"got '{actual_body}', expected "
                f"'{result['expected']}'")

    def run_scenario(state, args):
        scenarios = load_scenarios(state["scenarios_path"])
        scenario = next((s for s in scenarios if s["id"] == args["id"]), None)
        if scenario is None:
            return f"ERROR: unknown scenario '{args['id']}'"
        return _run_one(state, scenario)

    def run_all_scenarios(state, args):
        scenarios = load_scenarios(state["scenarios_path"])
        lines = [_run_one(state, s) for s in scenarios]
        passed = sum(1 for line in lines if line.startswith("PASS"))
        return (f"{passed}/{len(lines)} scenarios passed\n" +
                "\n".join(lines))

    def report_bug(state, args):
        state["bugs"].append({
            "id": args["id"],
            "repro": args["repro"],
            "expected": args["expected"],
            "actual": args["actual"],
        })
        return f"bug reported: {args['id']} ({args['repro']})"

    def show_bugs(state, args):
        return json.dumps(state["bugs"], indent=2)

    return ToolRegistry([
        Tool("http_get", "Issue a GET request against the application.",
             http_get, {"path": "URL path", "params": "query params"}),
        Tool("run_scenario", "Run a single test scenario by id.",
             run_scenario, {"id": "scenario id"}),
        Tool("run_all_scenarios",
             "Run every test scenario and report pass/fail for each.",
             run_all_scenarios),
        Tool("report_bug",
             "File a bug report for a failing scenario.",
             report_bug, {"id": "scenario id", "repro": "reproduction steps",
                          "expected": "expected result",
                          "actual": "actual result"}),
        Tool("show_bugs", "Show all filed bug reports.", show_bugs),
    ])


def verifier(task, state, history):
    scenarios = load_scenarios(state["scenarios_path"])
    results = state.get("results", {})
    if len(results) < len(scenarios):
        return False, (f"self-check: only {len(results)}/{len(scenarios)} "
                       "scenarios have been run")
    failing = [s["id"] for s in scenarios if not results[s["id"]]["ok"]]
    reported = {b["id"] for b in state.get("bugs", [])}
    missing = [f for f in failing if f not in reported]
    if missing:
        return False, (f"self-check: failing scenarios without bug reports: "
                       f"{missing}")
    return True, (f"self-check: all {len(failing)} failing scenarios have "
                  "complete bug reports (repro, expected, actual)")


class QAPolicy:
    """Demo brain: run the whole suite, then document every failure."""

    def __init__(self):
        self.pending = []

    def __call__(self, task, state, history, feedback):
        last = history[-1] if history else None
        if not history:
            return {"thought": "Run the full scenario suite against the app.",
                    "action": "run_all_scenarios", "args": {}}
        if last["action"] == "run_all_scenarios":
            self.pending = []
            for line in last["observation"].splitlines():
                match = re.match(
                    r"^FAIL: (\w+) — (GET [^ ]+) -> got '(.+?)', "
                    r"expected '(.+?)'$",
                    line.strip())
                if match:
                    self.pending.append({
                        "id": match.group(1),
                        "repro": match.group(2),
                        "actual": match.group(3),
                        "expected": match.group(4),
                    })
            if not self.pending:
                return {"thought": "All scenarios pass; nothing to report.",
                        "action": "finish",
                        "args": {"summary": "All scenarios passed."}}
            bug = self.pending[0]
            return {"thought": f"Scenario '{bug['id']}' fails; filing a bug "
                               "report with reproduction steps.",
                    "action": "report_bug", "args": bug}
        if last["action"] == "report_bug":
            self.pending = self.pending[1:]
            if self.pending:
                bug = self.pending[0]
                return {"thought": f"Scenario '{bug['id']}' also fails; "
                                   "documenting it too.",
                        "action": "report_bug", "args": bug}
            return {"thought": "Every failure is documented with a repro.",
                    "action": "finish",
                    "args": {"summary": "Filed bug reports for all failing "
                                        "scenarios."}}
        return {"thought": "Re-running the suite to confirm state.",
                "action": "run_all_scenarios", "args": {}}


def build_mock_llm():
    return MockLLM(QAPolicy())


def build_agent(llm, max_iterations=15):
    def state_factory(task):
        server = start_server()
        return {
            "server": server,
            "base_url": f"http://127.0.0.1:{server.server_port}",
            "scenarios_path": _resolve(
                task, "scenarios",
                os.path.join("agents", "qa_agent", "scenarios.json")),
            "results": {},
            "bugs": [],
        }

    def teardown(state):
        server = state.get("server")
        if server:
            server.shutdown()
            server.server_close()

    return Agent("qa_agent", make_tools(), verifier, llm,
                 max_iterations=max_iterations,
                 system_prompt=SYSTEM_PROMPT,
                 state_factory=state_factory,
                 teardown=teardown)
