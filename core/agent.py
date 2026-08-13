"""The agent loop: plan -> act -> observe -> verify -> retry."""

from .tracing import Step, Trace


class Agent:
    def __init__(self, name, tools, verifier, llm, max_iterations=15,
                 system_prompt="", state_factory=None, teardown=None):
        self.name = name
        self.registry = tools
        self.verifier = verifier
        self.llm = llm
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.state_factory = state_factory or (lambda task: {})
        self.teardown = teardown

    def run(self, task, run_id=None):
        trace = Trace(self.name, task, run_id)
        state = self.state_factory(task)
        history = []
        try:
            for iteration in range(1, self.max_iterations + 1):
                llm_task = dict(task)
                llm_task.setdefault("system_prompt", self.system_prompt)
                decision = self.llm.decide(
                    llm_task, self.registry, state, history,
                    verifier_feedback="")
                thought = decision.get("thought", "")
                action = decision.get("action", "")
                args = decision.get("args", {}) or {}

                if action == "finish":
                    done, feedback = self.verifier(task, state, history)
                    step = Step(iteration, thought, action, args,
                                None, feedback, done)
                    trace.add(step)
                    if done:
                        trace.finish(
                            "success",
                            args.get("summary") or feedback or "task complete")
                        return trace
                    # Premature finish: feed the verifier's critique back in.
                    history.append({
                        "thought": thought,
                        "action": action,
                        "args": args,
                        "observation": f"VERIFIER: {feedback}",
                    })
                    continue

                observation = self.registry.run(action, state, args)
                step = Step(iteration, thought, action, args, observation)
                done, check_note = self.verifier(
                    task, state,
                    history + [{
                        "thought": thought,
                        "action": action,
                        "args": args,
                        "observation": observation,
                    }])
                step.verifier = check_note
                step.done = done
                trace.add(step)
                history.append({
                    "thought": thought,
                    "action": action,
                    "args": args,
                    "observation": observation,
                })
                if done:
                    trace.finish("success", check_note or "task complete")
                    return trace

            done, feedback = self.verifier(task, state, history)
            if done:
                trace.finish("success", feedback or "task complete")
            else:
                trace.finish(
                    "failure",
                    f"max iterations ({self.max_iterations}) reached. {feedback}")
            return trace
        finally:
            if self.teardown:
                self.teardown(state)
