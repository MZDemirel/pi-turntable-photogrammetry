"""Yazdırılmaya hazır ChArUco deseni üretir.

Çıktılar:
  desen.pdf   - gerçek boyutta (%100 ölçekle) yazdırılacak sayfa: desen, 10°'lik açı işaretleri, 100 mm kontrol çizgisi
  desen.png   - desenin yüksek çözünürlüklü görüntüsü
  desen.json  - desen ayarları; 02_colmap_olcekle.py bu dosyayı okur

Örnek:
  python on_calisma/01_charuco_desen.py --cikti veri/desen
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from istasyon.charuco import DesenAyar

KAGIT_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0)}
DPI = 600


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kare-x", type=int, default=7, help="yatay kare sayısı (varsayılan 7)")
    ap.add_argument("--kare-y", type=int, default=5, help="dikey kare sayısı (varsayılan 5)")
    ap.add_argument("--kare-mm", type=float, default=20.0, help="kare kenarı, mm (varsayılan 20)")
    ap.add_argument("--isaret-mm", type=float, default=15.0, help="ArUco işaretçi kenarı, mm (varsayılan 15)")
    ap.add_argument("--sozluk", default="DICT_5X5_100", help="ArUco sözlüğü (varsayılan DICT_5X5_100)")
    ap.add_argument("--kagit", choices=KAGIT_MM, default="A4")
    ap.add_argument("--aci-adimi", type=float, default=10.0, help="açı işaretleri aralığı, derece (varsayılan 10)")
    ap.add_argument("--cikti", default="veri/desen", help="çıktı klasörü")
    a = ap.parse_args()

    ayar = DesenAyar(a.kare_x, a.kare_y, a.kare_mm, a.isaret_mm, a.sozluk)
    if ayar.isaret_mm >= ayar.kare_mm:
        sys.exit("Hata: işaretçi kenarı kare kenarından küçük olmalı.")
    cikti = Path(a.cikti)
    cikti.mkdir(parents=True, exist_ok=True)

    bw, bh = ayar.kare_x * ayar.kare_mm, ayar.kare_y * ayar.kare_mm      # desen boyutu, mm
    px_mm = DPI / 25.4
    img = ayar.board().generateImage((round(bw * px_mm), round(bh * px_mm)), marginSize=0, borderBits=1)
    cv2.imwrite(str(cikti / "desen.png"), img)

    pw, ph = KAGIT_MM[a.kagit]
    if bw > pw - 20 or bh > ph - 60:
        sys.exit(f"Hata: {bw:.0f}x{bh:.0f} mm desen {a.kagit} kâğıda sığmıyor.")
    cx, cy = pw / 2, ph / 2

    fig = plt.figure(figsize=(pw / 25.4, ph / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, pw)
    ax.set_ylim(0, ph)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=[cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2])

    # Açı işaretleri: desenin çevresinde bir çember; kâğıt döndürülürken sabit bir işarete hizalanır.
    r = np.hypot(bw, bh) / 2 + 4
    if r + 14 <= min(pw, ph) / 2:
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, lw=0.4, color="0.4"))
        for aci in np.arange(0, 360, a.aci_adimi):
            buyuk = abs(aci % 30) < 1e-9
            uz = 6 if buyuk else 3.5
            th = np.deg2rad(90 - aci)          # 0° üstte, saat yönünde artar
            ax.plot([cx + r * np.cos(th), cx + (r + uz) * np.cos(th)],
                    [cy + r * np.sin(th), cy + (r + uz) * np.sin(th)], color="k", lw=0.6 if buyuk else 0.4)
            if buyuk:
                ax.text(cx + (r + 9.5) * np.cos(th), cy + (r + 9.5) * np.sin(th), f"{aci:.0f}°",
                        ha="center", va="center", fontsize=6)
    else:
        print("Uyarı: açı işaretleri kâğıda sığmadığı için çizilmedi; daha büyük kâğıt veya daha küçük desen seçin.")

    # 100 mm kontrol çizgisi
    y0 = 14
    ax.plot([cx - 50, cx + 50], [y0, y0], color="k", lw=1.0)
    for x in (cx - 50, cx + 50):
        ax.plot([x, x], [y0 - 2, y0 + 2], color="k", lw=1.0)
    ax.text(cx, y0 + 3.5, "100 mm — yazdırdıktan sonra cetvel/kumpasla kontrol edin", ha="center", fontsize=6.5)
    ax.text(cx, ph - 10,
            f"ChArUco {ayar.kare_x}x{ayar.kare_y} · kare {ayar.kare_mm:g} mm · işaretçi {ayar.isaret_mm:g} mm · "
            f"{ayar.sozluk} · %100 ölçekle (gerçek boyut) yazdırın",
            ha="center", fontsize=6.5)

    fig.savefig(cikti / "desen.pdf")
    plt.close(fig)
    ayar.kaydet(cikti / "desen.json")
    print(f"Desen kaydedildi: {cikti}/desen.pdf, desen.png, desen.json")
    print(f"Desen boyutu: {bw:g} x {bh:g} mm, {ayar.kose_sayisi} köşe.")
    print("Yazdırdıktan sonra bir karenin kenarını kumpasla ölçün; 02 betiğine --kare-mm ile verin.")


if __name__ == "__main__":
    main()
