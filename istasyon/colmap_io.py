"""COLMAP metin (TXT) modelinin okunması ve kamera modelleri.

COLMAP piksel kuralı: sol üst pikselin merkezi (0.5, 0.5).
OpenCV piksel kuralı:  sol üst pikselin merkezi (0, 0).
OpenCV'den gelen koordinatlar COLMAP kamerasıyla kullanılmadan önce 0.5 eklenmelidir.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometri import kuaterniyon_R

DESTEKLENEN = ("SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV")


def _bozulma(u, v, k1, k2, p1, p2):
    r2 = u * u + v * v
    rad = k1 * r2 + k2 * r2 * r2
    du = u * rad + 2 * p1 * u * v + p2 * (r2 + 2 * u * u)
    dv = v * rad + 2 * p2 * u * v + p1 * (r2 + 2 * v * v)
    return du, dv


@dataclass
class Kamera:
    id: int
    model: str
    genislik: int
    yukseklik: int
    params: np.ndarray

    def ic(self):
        """(fx, fy, cx, cy, (k1, k2, p1, p2))"""
        p = self.params
        if self.model == "SIMPLE_PINHOLE":
            return p[0], p[0], p[1], p[2], (0.0, 0.0, 0.0, 0.0)
        if self.model == "PINHOLE":
            return p[0], p[1], p[2], p[3], (0.0, 0.0, 0.0, 0.0)
        if self.model == "SIMPLE_RADIAL":
            return p[0], p[0], p[1], p[2], (p[3], 0.0, 0.0, 0.0)
        if self.model == "RADIAL":
            return p[0], p[0], p[1], p[2], (p[3], p[4], 0.0, 0.0)
        if self.model == "OPENCV":
            return p[0], p[1], p[2], p[3], (p[4], p[5], p[6], p[7])
        raise ValueError(f"Desteklenmeyen kamera modeli: {self.model} (desteklenenler: {DESTEKLENEN})")

    def piksel_norm(self, xy: np.ndarray) -> np.ndarray:
        """COLMAP kuralındaki piksel(ler)i bozulması giderilmiş normalize koordinata çevirir."""
        fx, fy, cx, cy, d = self.ic()
        xy = np.asarray(xy, dtype=np.float64)
        u = (xy[..., 0] - cx) / fx
        v = (xy[..., 1] - cy) / fy
        uu, vv = u.copy(), v.copy()
        for _ in range(100):
            du, dv = _bozulma(uu, vv, *d)
            uu, vv = u - du, v - dv
        return np.stack([uu, vv], axis=-1)

    def norm_piksel(self, uv: np.ndarray) -> np.ndarray:
        """Normalize koordinat(lar)ı bozulma uygulayarak COLMAP kuralındaki piksele çevirir."""
        fx, fy, cx, cy, d = self.ic()
        uv = np.asarray(uv, dtype=np.float64)
        du, dv = _bozulma(uv[..., 0], uv[..., 1], *d)
        return np.stack([fx * (uv[..., 0] + du) + cx, fy * (uv[..., 1] + dv) + cy], axis=-1)


@dataclass
class Goruntu:
    id: int
    R: np.ndarray       # dünya -> kamera
    t: np.ndarray
    kamera_id: int
    ad: str

    @property
    def merkez(self) -> np.ndarray:
        return -self.R.T @ self.t


def _satirlar(yol: Path) -> list[str]:
    return [s for s in Path(yol).read_text(encoding="utf-8").split("\n") if not s.startswith("#")]


def kameralari_oku(yol: str | Path) -> dict[int, Kamera]:
    kameralar = {}
    for s in _satirlar(yol):
        p = s.split()
        if not p:
            continue
        kameralar[int(p[0])] = Kamera(int(p[0]), p[1], int(p[2]), int(p[3]), np.array(p[4:], dtype=np.float64))
    return kameralar


def goruntuleri_oku(yol: str | Path) -> dict[int, Goruntu]:
    """images.txt: her görüntü iki satırdır; ikinci satır (2B noktalar) boş olabilir."""
    satirlar = _satirlar(yol)
    goruntuler = {}
    i = 0
    while i < len(satirlar):
        p = satirlar[i].split()
        if not p:
            i += 1
            continue
        gid = int(p[0])
        R = kuaterniyon_R(*map(float, p[1:5]))
        t = np.array(p[5:8], dtype=np.float64)
        goruntuler[gid] = Goruntu(gid, R, t, int(p[8]), " ".join(p[9:]))
        i += 2
    return goruntuler


def noktalari_oku(yol: str | Path):
    """points3D.txt -> (xyz (N,3), rgb (N,3) uint8, hata (N,), iz_uzunlugu (N,))"""
    xyz, rgb, hata, iz = [], [], [], []
    for s in _satirlar(yol):
        p = s.split()
        if not p:
            continue
        xyz.append([float(v) for v in p[1:4]])
        rgb.append([int(v) for v in p[4:7]])
        hata.append(float(p[7]))
        iz.append((len(p) - 8) // 2)
    return (np.array(xyz).reshape(-1, 3), np.array(rgb, dtype=np.uint8).reshape(-1, 3),
            np.array(hata), np.array(iz, dtype=int))
