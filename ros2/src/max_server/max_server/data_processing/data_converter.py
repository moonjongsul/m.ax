"""Rotation format conversions.

Quaternion convention: [qx, qy, qz, qw] (scalar-last, ROS/geometry_msgs order).
RPY convention: [roll, pitch, yaw] intrinsic ZYX (yaw about Z, pitch about Y',
roll about X'') — matches tf2's getRPY.
Rot6D convention: first two columns of the rotation matrix flattened as
[r00, r10, r20, r01, r11, r21] — Zhou et al. "On the Continuity of Rotation
Representations in Neural Networks" (CVPR 2019).
"""

import numpy as np

# quaternion order: qx, qy, qz, qw


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """[qx, qy, qz, qw] -> 3x3 rotation matrix. Assumes the input is unit-norm."""
    qx, qy, qz, qw = quat
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float64)


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> [qx, qy, qz, qw]. Uses Shepperd's method for stability."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def convert_quat_to_rot6d(quat) -> np.ndarray:
    """[qx, qy, qz, qw] -> 6D rotation (first two columns of R, column-major)."""
    q = np.asarray(quat, dtype=np.float64)
    R = _quat_to_rotmat(q)
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float64)


def convert_rot6d_to_quat(rot6d) -> np.ndarray:
    """6D rotation -> [qx, qy, qz, qw]. Reconstructs R via Gram-Schmidt."""
    r = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1, a2 = r[:3], r[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 /= np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)
    return _rotmat_to_quat(R)


def convert_quat_to_rpy(quat) -> np.ndarray:
    """[qx, qy, qz, qw] -> [roll, pitch, yaw] (intrinsic ZYX, radians)."""
    qx, qy, qz, qw = np.asarray(quat, dtype=np.float64)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Clamp sinp to avoid NaN near gimbal lock.
    sinp = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def convert_rpy_to_quat(rpy) -> np.ndarray:
    """[roll, pitch, yaw] (intrinsic ZYX, radians) -> [qx, qy, qz, qw]."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qx, qy, qz, qw], dtype=np.float64)
