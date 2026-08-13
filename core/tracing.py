"""Per-run step tracing: the observable record of what the agent did."""

import json
import os
import time
import uuid


class Step:
    def __init__(self, iteration, thought, action, args,
                 observation=None, verifier=None, done=None):
        self.iteration = iteration
        self.thought = thought
        self.action = action
        self.args = args
        self.observation = observation
        self.verifier = verifier
        self.done = done

    def to_dict(self):
        return {
            "iteration": self.iteration,
            "thought": self.thought,
            "action": self.action,
            "args": self.args,
            "observation": self.observation,
            "verifier": self.verifier,
            "done": self.done,
        }


class Trace:
    def __init__(self, agent, task, run_id=None):
        self.agent = agent
        self.task = task
        self.run_id = run_id or uuid.uuid4().hex[:10]
        self.started = time.time()
        self.finished = None
        self.steps = []
        self.outcome = "running"
        self.summary = ""

    def add(self, step):
        self.steps.append(step)

    def finish(self, outcome, summary):
        self.outcome = outcome
        self.summary = summary
        self.finished = time.time()

    @property
    def duration(self):
        return round((self.finished or time.time()) - self.started, 2)

    @property
    def iteration_count(self):
        return len(self.steps)

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "agent": self.agent,
            "task": self.task,
            "outcome": self.outcome,
            "summary": self.summary,
            "duration_s": self.duration,
            "iterations": self.iteration_count,
            "steps": [s.to_dict() for s in self.steps],
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
