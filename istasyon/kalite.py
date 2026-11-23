"""Kare kalitesi ölçütleri: netlik (Laplace varyansı) ve pozlama (kırpılma oranı)."""
from __future__ import annotations

import cv2
import numpy as np


def olcekle(gri: np.ndarray, genislik: int | None = 1600) -> np.ndarray:
    """Görüntüyü verilen genişliğe küçültür. Laplace varyansı çözünürlüğe bağlı olduğundan,
    farklı kameralardan gelen skorların karşılaştırılabilmesi için analiz sabit genişlikte yapılır."""
    if genislik is None or gri.shape[1] <= genislik:
        return gri
    oran = genislik / gri.shape[1]
    return cv2.resize(gri, (genislik, round(gri.shape[0] * oran)), interpolation=cv2.INTER_AREA)


def netlik_skoru(gri: np.ndarray) -> float:
    """Laplace operatörü yanıtının varyansı. Büyük değer = net görüntü (Pech-Pacheco vd., 2000)."""
    return float(cv2.Laplacian(gri, cv2.CV_64F).var())


def kirpilma_orani(gri: np.ndarray, alt: int = 5, ust: int = 250) -> tuple[float, float]:
    """(karanlık oran, parlak oran): gri seviyesi <= alt ve >= ust olan piksellerin oranı."""
    n = gri.size
    return float(np.count_nonzero(gri <= alt) / n), float(np.count_nonzero(gri >= ust) / n)
