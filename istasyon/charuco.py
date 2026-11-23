"""ChArUco deseninin tanımı, kaydı ve görüntüde köşe tespiti.

Desen ayarları bir JSON dosyasında tutulur; deseni üreten betik ile köşeleri bulan betik
aynı dosyayı okuyarak aynı deseni kullanır.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DesenAyar:
    kare_x: int = 7            # yatay kare sayısı
    kare_y: int = 5            # dikey kare sayısı
    kare_mm: float = 20.0      # kare kenar uzunluğu (mm)
    isaret_mm: float = 15.0    # ArUco işaretçi kenar uzunluğu (mm)
    sozluk: str = "DICT_5X5_100"

    def kaydet(self, yol: str | Path) -> None:
        Path(yol).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def yukle(cls, yol: str | Path) -> "DesenAyar":
        return cls(**json.loads(Path(yol).read_text(encoding="utf-8")))

    def board(self) -> "cv2.aruco.CharucoBoard":
        sozluk = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.sozluk))
        return cv2.aruco.CharucoBoard((self.kare_x, self.kare_y), self.kare_mm, self.isaret_mm, sozluk)

    def kose_konumlari(self) -> np.ndarray:
        """Desen köşelerinin desen düzlemindeki konumları (N, 3), mm; z = 0."""
        return np.asarray(self.board().getChessboardCorners(), dtype=np.float64).reshape(-1, 3)

    @property
    def kose_sayisi(self) -> int:
        return (self.kare_x - 1) * (self.kare_y - 1)


class KoseBulucu:
    """Görüntüde ChArUco köşelerini kimlikleriyle birlikte bulur."""

    def __init__(self, ayar: DesenAyar):
        self.ayar = ayar
        self._detector = cv2.aruco.CharucoDetector(ayar.board())

    def bul(self, gri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Dönüş: (kimlikler (N,), pikseller (N, 2)). Piksel koordinatları OpenCV kuralındadır:
        sol üst pikselin merkezi (0, 0)."""
        koseler, kimlikler, _, _ = self._detector.detectBoard(gri)
        if kimlikler is None or len(kimlikler) == 0:
            return np.empty(0, dtype=int), np.empty((0, 2))
        return kimlikler.ravel().astype(int), koseler.reshape(-1, 2).astype(np.float64)
