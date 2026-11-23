import cv2
import numpy as np

from istasyon import ply
from istasyon.charuco import DesenAyar, KoseBulucu
from istasyon.kalite import kirpilma_orani, netlik_skoru


def test_bulaniklik_skoru_dusurur():
    desen = (np.indices((400, 600)).sum(0) // 20 % 2 * 255).astype(np.uint8)
    skorlar = [netlik_skoru(desen if s == 0 else cv2.GaussianBlur(desen, (0, 0), s)) for s in (0, 1, 2, 4)]
    assert all(a > b for a, b in zip(skorlar, skorlar[1:]))


def test_kirpilma_orani():
    g = np.full((10, 10), 128, np.uint8)
    g[0, :] = 0
    g[1:3, :] = 255
    assert kirpilma_orani(g) == (0.1, 0.2)


def test_uretilen_desende_tum_koseler_bulunur():
    ayar = DesenAyar()
    img = ayar.board().generateImage((1400, 1000), marginSize=60, borderBits=1)
    kimlikler, pikseller = KoseBulucu(ayar).bul(img)
    assert sorted(kimlikler.tolist()) == list(range(ayar.kose_sayisi))
    assert pikseller.shape == (ayar.kose_sayisi, 2)


def test_desen_json_gidip_gelir(tmp_path):
    ayar = DesenAyar(kare_x=9, kare_y=6, kare_mm=25.0, isaret_mm=18.0)
    ayar.kaydet(tmp_path / "d.json")
    assert DesenAyar.yukle(tmp_path / "d.json") == ayar


def test_ply_gidip_gelir(tmp_path):
    xyz = np.random.default_rng(0).normal(size=(100, 3)).astype(np.float32)
    rgb = np.random.default_rng(1).integers(0, 255, size=(100, 3), dtype=np.uint8)
    ply.yaz(tmp_path / "a.ply", xyz, rgb)
    v = ply.oku(tmp_path / "a.ply")
    assert np.allclose(np.stack([v["x"], v["y"], v["z"]], 1), xyz)
    assert (np.stack([v["red"], v["green"], v["blue"]], 1) == rgb).all()
