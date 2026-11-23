"""Laplace varyansına dayalı netlik skorunun bulanık kareleri ayırt edip edemediğini sınar.

Her fotoğrafa artan şiddette Gauss bulanıklığı uygulanır ve netlik skoru hesaplanır.
Ayrıca her fotoğrafın pozlama kırpılma oranları raporlanır.

Çıktılar:
  netlik.csv            fotoğraf x bulanıklık şiddeti skor tablosu ve kırpılma oranları
  netlik_grafik.png     bulanıklık şiddetine göre netlik skoru (Şekil 4.3)
  ozet.txt              eşik değeri ve tespit oranları

Örnek:
  python on_calisma/03_netlik_testi.py --goruntuler veri/on_calisma_1/goruntuler --cikti veri/on_calisma_1/netlik
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from istasyon.kalite import kirpilma_orani, netlik_skoru, olcekle

GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goruntuler", required=True)
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--sigmalar", default="0,0.5,1,1.5,2,3,4",
                    help="Gauss bulanıklığı standart sapmaları, analiz çözünürlüğünde piksel (0 = orijinal)")
    ap.add_argument("--genislik", type=int, default=1600, help="analiz genişliği, piksel (varsayılan 1600)")
    ap.add_argument("--esik", type=float, help="netlik eşiği; verilmezse orijinal skorların medyanının yarısı önerilir")
    a = ap.parse_args()

    sigmalar = [float(s) for s in a.sigmalar.split(",")]
    if sigmalar[0] != 0:
        sigmalar = [0.0] + sigmalar
    dosyalar = sorted(p for p in Path(a.goruntuler).iterdir() if p.suffix.lower() in GORUNTU_UZANTILARI)
    if not dosyalar:
        sys.exit("Klasörde fotoğraf bulunamadı.")
    cikti = Path(a.cikti)
    cikti.mkdir(parents=True, exist_ok=True)

    skorlar = np.zeros((len(dosyalar), len(sigmalar)))
    kirpilma = []
    for i, d in enumerate(dosyalar):
        gri = cv2.imread(str(d), cv2.IMREAD_GRAYSCALE)
        if gri is None:
            sys.exit(f"Okunamadı: {d}")
        gri = olcekle(gri, a.genislik)
        kirpilma.append(kirpilma_orani(gri))
        for j, s in enumerate(sigmalar):
            g = gri if s == 0 else cv2.GaussianBlur(gri, (0, 0), s)
            skorlar[i, j] = netlik_skoru(g)
        print(f"{d.name}: netlik {skorlar[i, 0]:.1f}")

    esik = a.esik if a.esik is not None else 0.5 * float(np.median(skorlar[:, 0]))
    tespit = (skorlar < esik).mean(axis=0) * 100

    with open(cikti / "netlik.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["goruntu"] + [f"sigma_{s:g}" for s in sigmalar] + ["karanlik_oran", "parlak_oran"])
        for d, sk, (k, p) in zip(dosyalar, skorlar, kirpilma):
            w.writerow([d.name] + [f"{v:.3f}" for v in sk] + [f"{k:.5f}", f"{p:.5f}"])

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ort = skorlar.mean(axis=0)
    ax.fill_between(sigmalar, skorlar.min(axis=0), skorlar.max(axis=0), alpha=0.2, color="tab:blue",
                    label="en küçük–en büyük")
    ax.plot(sigmalar, ort, "o-", color="tab:blue", label="ortalama")
    ax.axhline(esik, color="tab:red", ls="--", lw=1, label=f"eşik = {esik:.1f}")
    ax.set_yscale("log")
    ax.set_xlabel(f"Gauss bulanıklığı σ (piksel, {a.genislik} px genişlikte)")
    ax.set_ylabel("Netlik skoru (Laplace varyansı)")
    ax.set_title(f"Bulanıklık şiddetine göre netlik skoru (n = {len(dosyalar)})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cikti / "netlik_grafik.png")
    plt.close(fig)

    k = np.array(kirpilma)
    satirlar = [
        f"Fotoğraf sayısı: {len(dosyalar)}",
        f"Analiz genişliği: {a.genislik} px",
        f"Orijinal netlik skoru: medyan {np.median(skorlar[:, 0]):.1f}, en küçük {skorlar[:, 0].min():.1f}, "
        f"en büyük {skorlar[:, 0].max():.1f}",
        f"Eşik: {esik:.1f}" + ("" if a.esik is not None else " (öneri: orijinal skorların medyanının yarısı)"),
        "",
        "Bulanık kabul edilen fotoğraf oranı:",
    ]
    satirlar += [f"  σ = {s:<4g}: %{t:5.1f}" for s, t in zip(sigmalar, tespit)]
    satirlar += [
        "",
        "σ = 0 satırı yanlış alarm oranıdır (net fotoğrafların bulanık sayılması).",
        f"Pozlama kırpılması: karanlık en fazla %{100 * k[:, 0].max():.2f}, parlak en fazla %{100 * k[:, 1].max():.2f}",
    ]
    (cikti / "ozet.txt").write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print("\n" + "\n".join(satirlar))
    print(f"\nÇıktılar: {cikti}/netlik.csv, netlik_grafik.png, ozet.txt")


if __name__ == "__main__":
    main()
