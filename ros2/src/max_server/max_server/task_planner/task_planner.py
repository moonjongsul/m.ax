"""TaskPlanner: decompose high-level command into primitive skills.

Phase 2 placeholder for real decomposition. Already wired to dispatch the
picking cell: planned steps can trigger picking-cell services (registered as
generic service clients on the communicator) — both from the inference loop
(automatic) and from manual web/server triggers.
"""

from std_srvs.srv import Trigger


class TaskPlanner:

    # Service client name registered in the communicator (see config
    # picking_cell.service_client_list). Kept here so the trigger path has a
    # single source of truth.
    PICKING_PICK_CLIENT = "picking_pick"

    def __init__(self, node):
        self._node = node
        self._logger = node.get_logger()

    @property
    def _communicator(self):
        return self._node.communicator

    def plan(self, command: str) -> list[str]:
        # TODO: real decomposition. For now, identity.
        return [command]

    # ─── Picking cell dispatch ────────────────────────────────────────────

    def trigger_picking(self, timeout_sec: float = 60.0) -> tuple[bool, str]:
        """Trigger one picking-cell pick cycle (std_srvs/Trigger).

        Returns (success, message). Used by both the automatic (inference loop)
        and the manual (server-side) trigger paths.
        """
        comm = self._communicator
        if not comm.has_service_client(self.PICKING_PICK_CLIENT):
            msg = f"picking client '{self.PICKING_PICK_CLIENT}' not configured"
            self._logger.warn(f"[task_planner] {msg}")
            return False, msg

        self._logger.info("[task_planner] triggering picking cell")
        response = comm.call_service(
            self.PICKING_PICK_CLIENT, Trigger.Request(), timeout_sec=timeout_sec
        )
        if response is None:
            return False, "picking service call failed (unavailable/timeout)"
        return bool(response.success), str(response.message)
