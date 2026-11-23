"""ChArUco köşeleri kullanılarak SfM modelinin metrik ölçeğe getirilmesi.

1. Her köşe, göründüğü görüntülerdeki konumları ve COLMAP kamera pozları ile üçgenlenir.
2. Yeniden izdüşüm hatası yüksek köşeler atılır.
3. Modeldeki köşeler ile desendeki gerçek konumları (mm) arasında benzerlik dönüşümü
   (ölçek s, dönme R, öteleme t) kestirilir. Dönüşüm modeli mm birimine ve desen
   koordinat sistemine (tabla düzlemi z = 0) taşır.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .colmap_io import Goruntu, Kamera
from .geometri import ucgenle, umeyama


@dataclass
class OlcekSonucu:
    s: float                      # model birimi -> mm
    R: np.ndarray
    t: np.ndarray
    kose_sayisi: int
    rms_mm: float                 # hizalama sonrası köşe artıklarının RMS'i
    max_mm: float
    ikili_olcek_medyan: float     # köşe çiftlerinden ayrı ayrı hesaplanan ölçeklerin medyanı
    ikili_olcek_std: float
    ort_izdusum_hatasi_px: float
    kimlikler: list[int] = field(default_factory=list)

    def uygula(self, X: np.ndarray) -> np.ndarray:
        return self.s * (np.asarray(X) @ self.R.T) + self.t

    def ozet(self) -> dict:
        return {
            "olcek_mm_birim": self.s,
            "kose_sayisi": self.kose_sayisi,
            "hizalama_rms_mm": self.rms_mm,
            "hizalama_max_mm": self.max_mm,
            "ikili_olcek_medyan": self.ikili_olcek_medyan,
            "ikili_olcek_std": self.ikili_olcek_std,
            "ikili_olcek_degisim_katsayisi_yuzde": 100 * self.ikili_olcek_std / self.ikili_olcek_medyan,
            "ort_izdusum_hatasi_px": self.ort_izdusum_hatasi_px,
            "R": self.R.tolist(),
            "t": self.t.tolist(),
        }


def koseleri_ucgenle(gozlemler: dict[int, list[tuple[int, np.ndarray]]],
                     kameralar: dict[int, Kamera], goruntuler: dict[int, Goruntu],
                     min_gorus: int = 3, max_hata_px: float = 2.0):
    """gozlemler: kose_id -> [(goruntu_id, (x, y) OpenCV pikseli), ...]
    Dönüş: {kose_id: (X (3,), RMS yeniden izdüşüm hatası px)}"""
    sonuc = {}
    for kid, obs in gozlemler.items():
        obs = [(g, xy) for g, xy in obs if g in goruntuler]
        if len(obs) < min_gorus:
            continue
        normlar, pozlar, kams = [], [], []
        for gid, xy in obs:
            g = goruntuler[gid]
            k = kameralar[g.kamera_id]
            normlar.append(k.piksel_norm(np.asarray(xy) + 0.5))
            pozlar.append((g.R, g.t))
            kams.append(k)
        X = ucgenle(np.asarray(normlar), pozlar)
        hatalar, onde = [], True
        for (R, t), k, (_, xy) in zip(pozlar, kams, obs):
            Xc = R @ X + t
            if Xc[2] <= 0:
                onde = False
                break
            p = k.norm_piksel(Xc[:2] / Xc[2]) - 0.5
            hatalar.append(np.linalg.norm(p - np.asarray(xy)))
        if not onde:
            continue
        e = float(np.sqrt(np.mean(np.square(hatalar))))
        if e <= max_hata_px:
            sonuc[kid] = (X, e)
    return sonuc


def olcek_hesapla(gozlemler, kameralar, goruntuler, desen_noktalari: np.ndarray,
                  min_gorus: int = 3, max_hata_px: float = 2.0, min_cift_mm: float | None = None) -> OlcekSonucu:
    ucg = koseleri_ucgenle(gozlemler, kameralar, goruntuler, min_gorus, max_hata_px)
    if len(ucg) < 4:
        raise RuntimeError(f"Ölçekleme için yeterli köşe üçgenlenemedi ({len(ucg)} köşe; en az 4 gerekli).")
    ids = sorted(ucg)
    model = np.array([ucg[i][0] for i in ids])
    desen = np.asarray(desen_noktalari)[ids]

    s, R, t = umeyama(model, desen)
    artik = np.linalg.norm(desen - (s * model @ R.T + t), axis=1)
    # bir kez aykırı değer ayıklama: medyanın 3 katından büyük artıklar
    iyi = artik <= max(3 * np.median(artik), 1e-9)
    if iyi.sum() >= 4 and not iyi.all():
        ids = [i for i, ok in zip(ids, iyi) if ok]
        model, desen = model[iyi], desen[iyi]
        s, R, t = umeyama(model, desen)
        artik = np.linalg.norm(desen - (s * model @ R.T + t), axis=1)

    # OpenCV desen koordinatlarında z ekseni desenin arkasına bakar. Kameralar (ve nesne) +z tarafında
    # kalsın diye gerekirse x ekseni etrafında 180° çevrilir: tabla düzlemi z = 0, nesne z > 0.
    merkezler = np.array([g.merkez for g in goruntuler.values()])
    if len(merkezler) and np.median((s * merkezler @ R.T + t)[:, 2]) < 0:
        F = np.diag([1.0, -1.0, -1.0])
        R, t = F @ R, F @ t

    if min_cift_mm is None:
        min_cift_mm = 0.25 * float(np.ptp(desen[:, :2], axis=0).max())
    olcekler = []
    for a, b in combinations(range(len(ids)), 2):
        gercek = np.linalg.norm(desen[a] - desen[b])
        mod = np.linalg.norm(model[a] - model[b])
        if gercek >= min_cift_mm and mod > 0:
            olcekler.append(gercek / mod)
    olcekler = np.asarray(olcekler) if olcekler else np.array([s])

    return OlcekSonucu(
        s=s, R=R, t=t, kose_sayisi=len(ids),
        rms_mm=float(np.sqrt(np.mean(artik ** 2))), max_mm=float(artik.max()),
        ikili_olcek_medyan=float(np.median(olcekler)), ikili_olcek_std=float(np.std(olcekler)),
        ort_izdusum_hatasi_px=float(np.mean([ucg[i][1] for i in ids])), kimlikler=ids,
    )
