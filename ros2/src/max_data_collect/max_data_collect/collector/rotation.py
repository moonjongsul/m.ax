"""Quaternion -> 6D rotation representation.

rot6d is the first two columns of the rotation matrix. It is stored
alongside the raw quaternion rather than instead of it: quaternions are
the lossless record of what the robot reported, but they are a poor
learning target because q and -q are the same rotation, so a sign flip
between consecutive frames looks like a discontinuity that isn't there.
rot6d is continuous everywhere, which is why policies train on it.

Recovering the full matrix is Gram-Schmidt on the two columns; the third
column is their cross product. See Zhou et al., "On the Continuity of
Rotation Representations in Neural Networks" (CVPR 2019).
"""

import numpy as np


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit-normalised quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def quat_to_rot6d(qx: float, qy: float, qz: float, qw: float) -> list:
    """Quaternion -> [r11, r21, r31, r12, r22, r32].

    Column-major order: the first three values are the matrix's first
    column, the next three its second.
    """
    R = quat_to_matrix(qx, qy, qz, qw)
    return [float(R[0, 0]), float(R[1, 0]), float(R[2, 0]),
            float(R[0, 1]), float(R[1, 1]), float(R[2, 1])]


def rot6d_to_matrix(rot6d) -> np.ndarray:
    """Inverse of `quat_to_rot6d`, by Gram-Schmidt. For consumers/tests."""
    a1 = np.asarray(rot6d[:3], dtype=float)
    a2 = np.asarray(rot6d[3:6], dtype=float)
    b1 = a1 / max(np.linalg.norm(a1), 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-12)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)
