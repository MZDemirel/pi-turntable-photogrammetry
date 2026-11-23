"""Ölçekleme hattının sentetik veriyle uçtan uca sınanması.

Bilinen ölçek ve konumda bir desen, bilinen kameralarla (bozulmalı) izdüşürülür; hattın
gerçek ölçeği geri bulması beklenir."""
import numpy as np
import pytest

from istasyon.charuco import DesenAyar
from istasyon.colmap_io import Goruntu, Kamera
from istasyon.geometri import kuaterniyon_R, umeyama
from istasyon.olcek import olcek_hesapla


def rastgele_R(rng):
    q = rng.normal(size=4)
    return kuaterniyon_R(*q)


def bakis(C, hedef, yukari):
    z = hedef - C
    z /= np.linalg.norm(z)
    x = np.cross(z, yukari)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ C


def sentetik(rng, s_gercek=37.5, gurultu_px=0.3, k=-0.05):
    ayar = DesenAyar()
    P = ayar.kose_konumlari()                     # mm, z = 0
    R0, t0 = rastgele_R(rng), rng.normal(size=3)
    X = (P @ R0.T) / s_gercek + t0                # model koordinatları
    merkez = X.mean(0)
    a, b, n = R0[:, 0], R0[:, 1], R0[:, 2]
    yaricap = 300 / s_gercek

    kam = Kamera(1, "SIMPLE_RADIAL", 4000, 3000, np.array([3000.0, 2000.0, 1500.0, k]))
    goruntuler, gozlemler = {}, {}
    for i, phi in enumerate(np.deg2rad(np.arange(0, 360, 10))):
        e = np.deg2rad(40)
        C = merkez + yaricap * (np.cos(e) * (np.cos(phi) * a + np.sin(phi) * b) + np.sin(e) * n)
        R, t = bakis(C, merkez, n)
        goruntuler[i + 1] = Goruntu(i + 1, R, t, 1, f"{i:03d}.jpg")
        Xc = X @ R.T + t
        pix = kam.norm_piksel(Xc[:, :2] / Xc[:, 2:3]) - 0.5 + rng.normal(scale=gurultu_px, size=(len(X), 2))
        for kid, p in enumerate(pix):
            if 0 <= p[0] < 4000 and 0 <= p[1] < 3000:
                gozlemler.setdefault(kid, []).append((i + 1, p))
    return ayar, P, {1: kam}, goruntuler, gozlemler, s_gercek


def test_olcek_geri_bulunur():
    rng = np.random.default_rng(0)
    ayar, P, kameralar, goruntuler, gozlemler, s_gercek = sentetik(rng)
    sonuc = olcek_hesapla(gozlemler, kameralar, goruntuler, P)
    assert sonuc.kose_sayisi == ayar.kose_sayisi
    assert sonuc.s == pytest.approx(s_gercek, rel=1e-3)
    assert sonuc.rms_mm < 0.1
    assert sonuc.ikili_olcek_medyan == pytest.approx(s_gercek, rel=2e-3)
    # kameralar tabla düzleminin üstünde (z > 0)
    merkezler = sonuc.uygula(np.array([g.merkez for g in goruntuler.values()]))
    assert (merkezler[:, 2] > 0).all()


def test_yetersiz_kose_hata_verir():
    rng = np.random.default_rng(1)
    _, P, kameralar, goruntuler, gozlemler, _ = sentetik(rng)
    az = dict(list(gozlemler.items())[:3])
    with pytest.raises(RuntimeError):
        olcek_hesapla(az, kameralar, goruntuler, P)


def test_bozulma_gidip_gelir():
    kam = Kamera(1, "OPENCV", 4000, 3000, np.array([3000, 3050, 2000, 1500, -0.1, 0.02, 0.001, -0.0005]))
    uv = np.random.default_rng(2).uniform(-0.6, 0.6, size=(50, 2))
    assert np.allclose(kam.piksel_norm(kam.norm_piksel(uv)), uv, atol=1e-9)


def test_umeyama_tam_cozum():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(20, 3))
    R = rastgele_R(rng)
    B = 2.5 * A @ R.T + np.array([1.0, -2.0, 3.0])
    s, R2, t = umeyama(A, B)
    assert s == pytest.approx(2.5)
    assert np.allclose(R2, R)
    assert np.allclose(t, [1.0, -2.0, 3.0])
