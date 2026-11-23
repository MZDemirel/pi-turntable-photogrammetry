"""Basit PLY okuma/yazma (yalnızca köşe/vertex elemanı, liste özelliği yok).
COLMAP'in points3D ve fused.ply çıktıları için yeterlidir."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_TIP = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def oku(yol: str | Path) -> dict[str, np.ndarray]:
    with open(yol, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"PLY dosyası değil: {yol}")
        bicim, n, ozellikler, eleman = None, 0, [], None
        while True:
            s = f.readline().decode("ascii").strip()
            if s == "end_header":
                break
            p = s.split()
            if not p:
                continue
            if p[0] == "format":
                bicim = p[1]
            elif p[0] == "element":
                eleman = p[1]
                if eleman == "vertex":
                    n = int(p[2])
            elif p[0] == "property" and eleman == "vertex":
                if p[1] == "list":
                    raise ValueError("Liste özelliği desteklenmiyor")
                ozellikler.append((p[2], _TIP[p[1]]))
        dt = np.dtype(ozellikler)
        if bicim == "binary_little_endian":
            veri = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
        elif bicim == "ascii":
            tablo = np.loadtxt(f, max_rows=n, ndmin=2)
            veri = np.zeros(n, dtype=dt)
            for j, (ad, _) in enumerate(ozellikler):
                veri[ad] = tablo[:, j]
        else:
            raise ValueError(f"Desteklenmeyen PLY biçimi: {bicim}")
    return {ad: np.asarray(veri[ad]) for ad, _ in ozellikler}


def yaz(yol: str | Path, xyz: np.ndarray, rgb: np.ndarray | None = None, normal: np.ndarray | None = None) -> None:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    alanlar = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normal is not None:
        alanlar += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if rgb is not None:
        alanlar += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    v = np.zeros(len(xyz), dtype=np.dtype(alanlar))
    v["x"], v["y"], v["z"] = xyz.T
    if normal is not None:
        v["nx"], v["ny"], v["nz"] = np.asarray(normal, dtype=np.float32).reshape(-1, 3).T
    if rgb is not None:
        v["red"], v["green"], v["blue"] = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3).T
    tip_adi = {"<f4": "float", "u1": "uchar"}
    baslik = ["ply", "format binary_little_endian 1.0", f"element vertex {len(v)}"]
    baslik += [f"property {tip_adi[t]} {ad}" for ad, t in alanlar]
    baslik.append("end_header")
    with open(yol, "wb") as f:
        f.write(("\n".join(baslik) + "\n").encode("ascii"))
        f.write(v.tobytes())
