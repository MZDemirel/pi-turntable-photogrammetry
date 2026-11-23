"""Temel geometri: kuaterniyon, doğrusal üçgenleme ve benzerlik dönüşümü kestirimi."""
from __future__ import annotations

import numpy as np


def kuaterniyon_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def ucgenle(normlar: np.ndarray, pozlar: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Doğrusal (DLT) üçgenleme.

    normlar: (k, 2) bozulması giderilmiş normalize görüntü koordinatları
    pozlar:  k adet (R, t); dünya -> kamera dönüşümü (X_kam = R X + t)
    """
    A = []
    for (u, v), (R, t) in zip(normlar, pozlar):
        P = np.hstack([R, np.reshape(t, (3, 1))])
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3]


def umeyama(kaynak: np.ndarray, hedef: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """hedef ≈ s · R · kaynak + t olacak şekilde (s, R, t) kestirir (Umeyama, 1991).
    Düzlemsel nokta kümeleriyle de çalışır."""
    kaynak = np.asarray(kaynak, dtype=np.float64)
    hedef = np.asarray(hedef, dtype=np.float64)
    n = len(kaynak)
    mk, mh = kaynak.mean(0), hedef.mean(0)
    xk, xh = kaynak - mk, hedef - mh
    U, D, Vt = np.linalg.svd(xh.T @ xk / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / ((xk ** 2).sum() / n))
    t = mh - s * R @ mk
    return s, R, t
