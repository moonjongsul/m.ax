"""환경 카메라에서 흰색 트레이만 잘라 위에서 본 영상으로 편다.

    ros2 run max_ai max_ai_env_tray_span --ros-args \\
        --params-file $(ros2 pkg prefix max_ai)/share/max_ai/config/env_tray_span_config.yaml

    /observation/env/color/image_raw/compressed        CompressedImage (구독)
    /observation/env_tray/color/image_raw/compressed   CompressedImage (퍼블리시)

이미지 콜백 하나로 돈다. 프레임마다 트레이 윗면 모서리 4점을 찾고, 원근
변환으로 펴서 90° 반시계로 돌린 640x480을 내보낸다. `scripts/tray_span`을
노드로 옮긴 것이다.

**모서리 추적.** 카메라가 조금씩 움직여도 모서리를 따라가도록 기준 이미지와
ORB 특징점을 매칭해 호모그래피를 구한다. 특징점은 트레이 테두리 둘레에서만
뽑는다 -- 포켓 안은 부품이 들어가면서 바뀌고, 로봇은 매 프레임 다르다.
매칭이 부족하면 직전 모서리를 그대로 쓴다.

원본 스크립트는 `max_jump`보다 크게 튄 추정을 항상 버려서, 카메라가 한 번에
크게 움직이면 옛 모서리에 영영 갇힌다. 여기서는 튄 추정이 `relock_frames`
프레임 연속으로 서로 일치하면 새 위치로 다시 고정한다.

출력 stamp는 입력 프레임의 것을 복사한다. 원본 env 영상과 같은 시각으로
맞춰 쓸 수 있어야 한다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from .ros_util import sensor_qos

_ROTATE = {
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
    "none": None,
}


class TrayTracker:
    """기준 프레임의 모서리를 현재 프레임으로 옮긴다. 좌표는 기준 이미지 픽셀."""

    def __init__(self, ref_img, ref_corners, n_features=1500, margin=60.0, inner_shrink=0.12,
                 min_inliers=25, max_jump=40.0, smooth=0.5, relock_frames=10):
        self.ref_corners = np.float32(ref_corners).reshape(4, 2)
        self.min_inliers = min_inliers
        self.max_jump = max_jump            # px. 직전보다 이만큼 넘게 튀면 바로 받지 않는다
        self.smooth = smooth                # 0 = 평활 없음, 1에 가까울수록 느리게 따라감
        self.relock_frames = relock_frames  # 튄 추정이 이만큼 연속 일치하면 재고정. 0 = 안 함
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        mask = self._feature_mask(ref_img.shape[:2], self.ref_corners, margin, inner_shrink)
        gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray, mask)
        if self.ref_des is None or len(self.ref_kp) < min_inliers:
            raise ValueError(f"only {len(self.ref_kp)} features around the tray in the reference image")
        self.corners = self.ref_corners.copy()  # 마지막으로 받은 추정
        self._cand = None                       # 재고정 후보
        self._cand_n = 0

    @staticmethod
    def _feature_mask(shape, corners, margin, inner_shrink):
        h, w = shape
        c = corners.mean(0)
        outer = c + (corners - c) * (1 + margin / np.linalg.norm(corners - c, axis=1, keepdims=True))
        inner = c + (corners - c) * (1 - inner_shrink)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, outer.astype(np.int32), 255)
        cv2.fillConvexPoly(mask, inner.astype(np.int32), 0)  # 포켓: 내용물이 바뀐다
        return mask

    def update(self, img):
        """(corners, ok). 실패하면 직전 모서리와 ok=False."""
        new = self._estimate(img)
        if new is None:
            return self.corners, False

        if np.abs(new - self.corners).max() > self.max_jump:
            if self._cand is not None and np.abs(new - self._cand).max() <= self.max_jump:
                self._cand_n += 1
            else:
                self._cand_n = 1
            self._cand = new
            if self.relock_frames <= 0 or self._cand_n < self.relock_frames:
                return self.corners, False
            # 카메라가 실제로 옮겨졌다. 평활 없이 새 위치로 바로 간다.
            self.corners = new
            self._cand, self._cand_n = None, 0
            return self.corners, True

        self._cand, self._cand_n = None, 0
        self.corners = self.smooth * self.corners + (1 - self.smooth) * new
        return self.corners, True

    def _estimate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < self.min_inliers:
            return None

        pairs = self.matcher.knnMatch(self.ref_des, des, k=2)
        good = [m for m, *n in pairs if n and m.distance < 0.75 * n[0].distance]
        if len(good) < self.min_inliers:
            return None

        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 2.0)
        if H is None or inl.sum() < self.min_inliers:
            return None
        return cv2.perspectiveTransform(self.ref_corners[None], H)[0]


def warp_tray(img, corners, out_w=640, out_h=480, rotate="ccw", interpolation=cv2.INTER_LINEAR):
    """모서리(TL, TR, BR, BL)를 펴서 회전한 뒤 out_w x out_h로 만든다."""
    code = _ROTATE[rotate]
    quarter = code in (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE)
    # 90° 회전이면 가로세로를 바꿔 편 뒤 돌려야 최종 크기가 out_w x out_h가 된다.
    w, h = (out_h, out_w) if quarter else (out_w, out_h)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(np.float32(corners), dst)
    top = cv2.warpPerspective(img, H, (w, h), flags=interpolation)
    return top if code is None else cv2.rotate(top, code)


def resolve_path(path: str) -> str:
    """상대 경로는 설치된 max_ai/config 기준."""
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.join(get_package_share_directory("max_ai"), "config", path)


class EnvTraySpanNode(Node):
    def __init__(self):
        super().__init__("max_ai_env_tray_span")
        p = self._declare()

        self.proc_size = (p["proc_width"], p["proc_height"])
        self.out_size = (p["out_width"], p["out_height"])
        self.rotate = p["out_rotate"]
        if self.rotate not in _ROTATE:
            raise ValueError(f"out_rotate must be one of {list(_ROTATE)}, got {self.rotate!r}")
        self.jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_quality"]]

        corners = np.float32(p["ref_corners"])
        if corners.size != 8:
            raise ValueError(f"ref_corners needs 8 numbers (TL TR BR BL as x, y), got {corners.size}")
        corners = corners.reshape(4, 2)

        self.tracker = None
        self.fixed_corners = corners
        if p["tracking"]:
            ref_path = resolve_path(p["ref_image"])
            ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
            if ref is None:
                raise FileNotFoundError(f"cannot read reference image {ref_path}")
            if (ref.shape[1], ref.shape[0]) != self.proc_size:
                raise ValueError(f"reference image is {ref.shape[1]}x{ref.shape[0]}, "
                                 f"but proc size is {self.proc_size[0]}x{self.proc_size[1]}")
            self.tracker = TrayTracker(
                ref, corners, n_features=p["n_features"], margin=p["margin"],
                inner_shrink=p["inner_shrink"], min_inliers=p["min_inliers"],
                max_jump=p["max_jump"], smooth=p["smooth"], relock_frames=p["relock_frames"],
            )
            self.get_logger().info(f"tracking against {ref_path} ({len(self.tracker.ref_kp)} features)")
        else:
            self.get_logger().info("tracking off: fixed corners")

        qos = sensor_qos(p["qos_depth"])
        self.pub = self.create_publisher(CompressedImage, p["output_topic"], qos)
        self.debug_pub = None
        if p["publish_debug"]:
            self.debug_pub = self.create_publisher(CompressedImage, p["debug_topic"], qos)
        self.create_subscription(CompressedImage, p["input_topic"], self._on_image, qos)

        self._lost = False
        self.get_logger().info(f"{p['input_topic']} -> {p['output_topic']}")

    def _declare(self) -> dict:
        defaults = {
            "input_topic": "/observation/env/color/image_raw/compressed",
            "output_topic": "/observation/env_tray/color/image_raw/compressed",
            "qos_depth": 1,
            "publish_debug": False,
            "debug_topic": "/observation/env_tray/debug/compressed",
            "ref_image": "env_img.png",
            "ref_corners": [392.0, 119.0, 562.0, 133.0, 603.0, 307.0, 382.0, 321.0],
            "proc_width": 640,
            "proc_height": 480,
            "tracking": True,
            "n_features": 1500,
            "margin": 60.0,
            "inner_shrink": 0.12,
            "min_inliers": 25,
            "max_jump": 40.0,
            "smooth": 0.5,
            "relock_frames": 10,
            "out_width": 640,
            "out_height": 480,
            "out_rotate": "ccw",
            "jpeg_quality": 90,
        }
        return {k: self.declare_parameter(k, v).value for k, v in defaults.items()}

    def _on_image(self, msg: CompressedImage) -> None:
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn("failed to decode input image", throttle_duration_sec=5.0)
            return

        # 모서리는 proc 크기 좌표다. 추적은 줄인 영상으로 하고, 펴기는 원본
        # 해상도에서 해서 입력이 더 크면 그 해상도를 살린다.
        h, w = frame.shape[:2]
        small = frame if (w, h) == self.proc_size else cv2.resize(frame, self.proc_size,
                                                                     interpolation=cv2.INTER_AREA)
        if self.tracker is None:
            corners, ok = self.fixed_corners, True
        else:
            corners, ok = self.tracker.update(small)
            if ok == self._lost:
                self._lost = not ok
                if self._lost:
                    self.get_logger().warn("tray tracking lost: holding last corners")
                else:
                    self.get_logger().info("tray tracking recovered")

        scale = np.float32([w / self.proc_size[0], h / self.proc_size[1]])
        out = warp_tray(frame, corners * scale, *self.out_size, rotate=self.rotate)
        self._publish(self.pub, msg, out)

        if self.debug_pub is not None:
            dbg = small.copy()
            cv2.polylines(dbg, [corners.astype(np.int32)], True, (0, 255, 0) if ok else (0, 0, 255), 2)
            self._publish(self.debug_pub, msg, dbg)

    def _publish(self, pub, src: CompressedImage, img) -> None:
        ok, buf = cv2.imencode(".jpg", img, self.jpeg_params)
        if not ok:
            self.get_logger().warn("failed to encode output image", throttle_duration_sec=5.0)
            return
        out = CompressedImage()
        out.header = src.header
        out.format = "jpeg"
        out.data = buf.tobytes()
        pub.publish(out)


def main(argv=None):
    rclpy.init(args=argv)
    node = EnvTraySpanNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
