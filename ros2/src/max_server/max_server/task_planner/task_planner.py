"""TaskPlanner: decompose high-level command into primitive skills (Phase 2 placeholder)."""


class TaskPlanner:

    def __init__(self, node):
        self._node = node

    def plan(self, command: str) -> list[str]:
        # TODO: real decomposition. For now, identity.
        return [command]
