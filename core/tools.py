"""Tool definitions and the registry the agent can call."""


class Tool:
    def __init__(self, name, description, fn, args=None):
        self.name = name
        self.description = description
        self.fn = fn  # fn(state, args) -> str observation
        self.args = args or {}

    def run(self, state, args):
        return str(self.fn(state, args or {}))


class ToolRegistry:
    def __init__(self, tools):
        self.tools = {t.name: t for t in tools}

    def list(self):
        return list(self.tools.values())

    def names(self):
        return list(self.tools)

    def has(self, name):
        return name in self.tools

    def run(self, name, state, args):
        if not self.has(name):
            return (
                f"ERROR: unknown tool '{name}'. "
                f"Available tools: {', '.join(self.names())}"
            )
        try:
            return self.tools[name].run(state, args)
        except Exception as exc:  # tool failures are observations, not crashes
            return f"ERROR: tool '{name}' failed: {exc}"
