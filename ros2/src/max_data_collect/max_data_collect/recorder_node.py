"""max_data_collect -- demonstration and rollout recorder.

Samples every configured camera, the arm, the gripper and the action
stream at a fixed rate and writes one episode per START/SAVE cycle as
HDF5 plus one mp4 per camera.

The recorder owns the episode state machine and is the single source of
truth for it:

    IDLE -> RECORDING -> PENDING -> IDLE

A command that does not apply to the current state is rejected with
success=false rather than silently changing anything -- the teleop node
does not track state, it reacts to RecorderStatus. STOP is not a save:
whether a demonstration was any good is only known once it is over, so
the episode stays pending until SAVE or DELETE.
"""

import shutil
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
)

from max_interfaces.msg import RecorderStatus
from max_interfaces.srv import RecordCommand

from max_data_collect import config_loader
from max_data_collect.collector.communicator import Communicator
from max_data_collect.collector.frame_builder import FrameBuilder
from max_data_collect.writer.dataset import Dataset
from max_data_collect.writer.episode_writer import EpisodeWriter, QueueOverflow
from max_data_collect.writer.video_writer import probe_codec


# Latched so a teleop node that starts late learns the state at once
# instead of waiting for the next heartbeat.
STATUS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)


class RecorderNode(Node):

    def __init__(self):
        super().__init__(
            "max_data_collect",
            automatically_declare_parameters_from_overrides=True,
        )
        self._cfg = config_loader.load(self)
        recording = self._cfg["recording"]
        interface = self._cfg["record_interface"]

        self._hz = recording["collect_hz"]
        self._warmup = float(recording["warmup_sec"])
        self._staleness = float(recording["staleness_timeout"])
        self._require_all = bool(recording["require_all_sources"])
        self._async_save = bool(self._cfg["storage"]["async_save"])
        self._undo_depth = int(self._cfg["storage"]["undo_depth"])

        codec = self._cfg["storage"]["video_codec"]
        if not probe_codec(codec):
            raise RuntimeError(
                f"ffmpeg cannot encode with '{codec}' on this machine; "
                "set storage.video_codec to one it has"
            )

        self._dataset = Dataset(self.get_logger(), self._cfg)
        self._dataset.open()

        # Guarded by _lock: every service call, the sampling timer and the
        # save thread take it. The timer's work is short -- build a row,
        # enqueue it -- and the writing happens on the writer thread.
        self._lock = threading.RLock()
        self._state = RecorderStatus.IDLE
        self._episode_id = ""
        self._episode_start = 0.0
        self._episode_duration = 0.0
        self._message = ""
        self._writer: EpisodeWriter | None = None
        self._saved: list = []
        self._warmup_until = 0.0
        self._warmed = False

        self._comm = Communicator(self, self.get_logger(), self._cfg)
        self._frames = FrameBuilder(self.get_logger(), self._cfg)

        self._status_pub = self.create_publisher(
            RecorderStatus, interface["status_topic"], STATUS_QOS
        )
        self._command_srv = self.create_service(
            RecordCommand, interface["command_service"], self._on_command
        )
        self.create_timer(1.0 / self._hz, self._on_tick)
        # The teleop node treats silence longer than 3 s as a dead
        # recorder, so this beats even while idle.
        self.create_timer(float(interface["status_period"]), self._publish_status)

        self.get_logger().info(
            f"[collect] ready at {self._hz:g} Hz -> {self._dataset.root} "
            f"(source={self._cfg['source']}, codec={codec})"
        )
        self._publish_status()

    # ── sampling ───────────────────────────────────────────────────────

    def _on_tick(self) -> None:
        with self._lock:
            if self._state != RecorderStatus.RECORDING:
                return
            now = time.monotonic()
            if now < self._warmup_until:
                return
            if not self._warmed:
                self._warmed = True
                self.get_logger().info(
                    f"[collect] RECORDING {self._episode_id} -- warmup over, "
                    f"keeping frames at {self._hz:g} Hz"
                )

            self._episode_duration = now - self._episode_start

            # Every stream matters, so a single one going quiet ends the
            # episode straight away. Recording through it would pair a
            # frozen camera (or a held state value) with a moving action
            # and quietly poison the dataset -- the operator needs to
            # know now, while the demonstration can still be redone.
            stale = self._comm.stale(self._staleness)
            if stale:
                episode_id = self._episode_id
                reason = f"source(s) stopped publishing: {stale}"
                self.get_logger().error(
                    f"[collect] {episode_id}: {reason}; episode aborted and "
                    "discarded"
                )
                self._drop_pending()
                self._message = f"aborted: {reason}"
                return

            row = self._frames.build(
                self._comm.snapshot(), now, self._episode_start
            )
            try:
                self._writer.put(row)
            except (QueueOverflow, RuntimeError) as exc:
                self.get_logger().error(f"[collect] {exc}; aborting episode")
                self._drop_pending()
                self._message = f"aborted: {exc}"

    # ── command service ────────────────────────────────────────────────

    def _on_command(self, request, response):
        handlers = {
            RecordCommand.Request.START: self._cmd_start,
            RecordCommand.Request.STOP: self._cmd_stop,
            RecordCommand.Request.SAVE: self._cmd_save,
            RecordCommand.Request.DELETE: self._cmd_delete,
        }
        handler = handlers.get(request.command)
        with self._lock:
            if handler is None:
                response.success = False
                response.message = f"unknown command {request.command}"
                response.episode_id = ""
            else:
                success, message, episode_id = handler()
                response.success = success
                response.message = message
                response.episode_id = episode_id
                self._message = message
            response.state = self._state
        self._publish_status()
        return response

    def _cmd_start(self):
        if self._state != RecorderStatus.IDLE:
            return (False, f"cannot START from state {self._state}", "")

        missing = self._comm.never_received()
        if missing:
            if self._require_all:
                return (False, f"sources have not published yet: {missing}", "")
            self.get_logger().warning(
                f"[collect] starting with silent sources: {missing}"
            )

        index, episode_id, path = self._dataset.allocate()
        try:
            writer = EpisodeWriter(
                self.get_logger(), self._cfg, path, episode_id, index,
                self._frames.stale_bits,
            )
            writer.start()
        except Exception as exc:                    # noqa: BLE001
            self.get_logger().error(f"[collect] cannot start writer: {exc}")
            shutil.rmtree(path, ignore_errors=True)
            return (False, f"cannot start writer: {exc}", "")

        self._writer = writer
        self._episode_id = episode_id
        self._episode_start = time.monotonic()
        self._warmup_until = self._episode_start + self._warmup
        self._warmed = False
        self._episode_duration = 0.0
        self._frames.reset()
        self._state = RecorderStatus.RECORDING
        self.get_logger().info(
            f"[collect] START {episode_id} (warmup {self._warmup:g}s)"
        )
        return (True, f"recording {episode_id}", episode_id)

    def _cmd_stop(self):
        if self._state != RecorderStatus.RECORDING:
            return (False, f"cannot STOP from state {self._state}", "")
        episode_id = self._episode_id
        self._stop_recording("stopped by operator")
        return (True, f"{episode_id} buffered, awaiting SAVE or DELETE",
                episode_id)

    def _cmd_save(self):
        if self._state == RecorderStatus.RECORDING:
            # A good demo ends with a single press: stop, then commit.
            self._stop_recording("stopped by SAVE")
        if self._state != RecorderStatus.PENDING:
            return (False, f"nothing to SAVE (state {self._state})", "")

        if self._writer.written == 0:
            episode_id = self._episode_id
            self._drop_pending()
            return (False, f"{episode_id} captured no frames; discarded",
                    episode_id)

        episode_id = self._episode_id
        writer = self._writer
        duration = self._episode_duration

        if self._async_save:
            self._state = RecorderStatus.SAVING
            threading.Thread(
                target=self._finish_save, args=(writer, duration, episode_id),
                daemon=True, name=f"max-collect-save-{episode_id}",
            ).start()
            return (True, f"saving {episode_id}", episode_id)

        self._finish_save(writer, duration, episode_id)
        return (True, f"saved {episode_id}", episode_id)

    def _cmd_delete(self):
        if self._state in (RecorderStatus.RECORDING, RecorderStatus.PENDING):
            episode_id = self._episode_id
            if self._state == RecorderStatus.RECORDING:
                self._stop_recording("stopped by DELETE")
            self._drop_pending()
            self.get_logger().info(f"[collect] DELETE (buffered) {episode_id}")
            return (True, f"discarded {episode_id}", episode_id)

        if self._state != RecorderStatus.IDLE:
            return (False, f"cannot DELETE from state {self._state}", "")
        if not self._saved:
            return (False, "nothing left to undo", "")

        path = self._saved.pop()
        episode_id = path.name
        shutil.rmtree(path, ignore_errors=True)
        self._dataset.unregister(episode_id)
        self.get_logger().info(f"[collect] DELETE (saved) {episode_id}")
        return (True, f"deleted {episode_id}", episode_id)

    # ── transitions ────────────────────────────────────────────────────

    def _stop_recording(self, reason: str) -> None:
        self._episode_duration = time.monotonic() - self._episode_start
        self._state = RecorderStatus.PENDING
        self._message = reason
        frames = self._writer.written if self._writer else 0
        self.get_logger().info(
            f"[collect] STOP {self._episode_id} "
            f"({self._episode_duration:.2f}s, {frames} frames): {reason}"
        )

    def _drop_pending(self) -> None:
        if self._writer is not None:
            try:
                self._writer.discard()
            except Exception as exc:                # noqa: BLE001
                self.get_logger().warning(f"[collect] discard failed: {exc}")
        self._writer = None
        self._episode_id = ""
        self._episode_duration = 0.0
        self._state = RecorderStatus.IDLE

    def _finish_save(self, writer, duration: float, episode_id: str) -> None:
        """Close the encoders and write the files, then return to IDLE."""
        try:
            entry = writer.finish(duration)
            self._dataset.register(entry)
            with self._lock:
                if self._undo_depth > 0:
                    self._saved.append(self._dataset.episodes_dir / episode_id)
                    del self._saved[:-self._undo_depth]
                else:
                    self._saved.clear()
                self._episode_id = episode_id
                self._message = (
                    f"saved {episode_id} ({entry['num_frames']} frames)"
                )
                if entry["dropped"]:
                    self._message += f", {entry['dropped']} dropped"
            self.get_logger().info(f"[collect] SAVE {self._message}")
        except Exception as exc:                    # noqa: BLE001
            self.get_logger().error(f"[collect] save failed: {exc}")
            shutil.rmtree(
                self._dataset.episodes_dir / episode_id, ignore_errors=True
            )
            with self._lock:
                self._message = f"save failed: {exc}"
        finally:
            with self._lock:
                self._writer = None
                self._state = RecorderStatus.IDLE
            self._publish_status()

    # ── status ─────────────────────────────────────────────────────────

    def _publish_status(self) -> None:
        msg = RecorderStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        with self._lock:
            msg.state = self._state
            msg.episode_id = self._episode_id
            msg.duration = float(self._episode_duration)
            msg.frame_count = self._writer.written if self._writer else 0
            message = self._message
            # While RECORDING this can no longer fire -- the sampling
            # tick aborts on the first stale source. Reported while IDLE
            # instead, so the operator sees a dead camera before starting
            # a demonstration rather than losing one to it.
            if self._state == RecorderStatus.IDLE:
                stale = self._comm.stale(self._staleness)
                if stale:
                    message = f"not ready, no data from: {stale}"
        msg.message = message
        self._status_pub.publish(msg)

    # ── shutdown ───────────────────────────────────────────────────────

    def destroy_node(self):
        with self._lock:
            if self._writer is not None:
                self.get_logger().warning(
                    "[collect] shutting down with an episode in flight; "
                    "discarding it"
                )
                self._drop_pending()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
