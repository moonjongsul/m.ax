import cv2
import numpy as np


class TrayTracker:
    """Track tray corners across camera pose changes.

    Matches ORB features between a reference frame (whose tray corners are known)
    and the current frame, estimates a homography with RANSAC, and maps the
    reference corners into the current frame. Only the table/tray-rim area around
    the tray is used for features, since the tray contents and the robot change.
    """

    def __init__(self, ref_img, ref_corners, n_features=1500, margin=60, inner_shrink=0.12,
                 min_inliers=25, max_jump=40.0, smooth=0.5):
        self.ref_corners = np.float32(ref_corners).reshape(4, 2)
        self.min_inliers = min_inliers
        self.max_jump = max_jump      # px; reject a frame whose corners jump more than this
        self.smooth = smooth          # 0 = no smoothing, closer to 1 = smoother/slower
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        mask = self._feature_mask(ref_img.shape[:2], self.ref_corners, margin, inner_shrink)
        gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray, mask)
        self.corners = self.ref_corners.copy()  # last good estimate

    @staticmethod
    def _feature_mask(shape, corners, margin, inner_shrink):
        h, w = shape
        c = corners.mean(0)
        outer = c + (corners - c) * (1 + margin / np.linalg.norm(corners - c, axis=1, keepdims=True))
        inner = c + (corners - c) * (1 - inner_shrink)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, outer.astype(np.int32), 255)
        cv2.fillConvexPoly(mask, inner.astype(np.int32), 0)  # tray pockets: contents change
        return mask

    def update(self, img):
        """Return (corners, ok). On failure the last good corners are returned with ok=False."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < self.min_inliers:
            return self.corners, False

        pairs = self.matcher.knnMatch(self.ref_des, des, k=2)
        good = [m for m, *n in pairs if n and m.distance < 0.75 * n[0].distance]
        if len(good) < self.min_inliers:
            return self.corners, False

        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 2.0)
        if H is None or inl.sum() < self.min_inliers:
            return self.corners, False

        new = cv2.perspectiveTransform(self.ref_corners[None], H)[0]
        if np.abs(new - self.corners).max() > self.max_jump:
            return self.corners, False

        self.corners = self.smooth * self.corners + (1 - self.smooth) * new
        return self.corners, True


def warp_tray(img, corners, out_w=640, out_h=480):
    """Top-down view of the tray, rotated 90° CCW, sized out_w x out_h."""
    w, h = out_h, out_w
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(np.float32(corners), dst)
    top = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
    return cv2.rotate(top, cv2.ROTATE_90_COUNTERCLOCKWISE)
