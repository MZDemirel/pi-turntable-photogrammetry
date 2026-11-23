"""Fotoğraflardan COLMAP ile 3B model üretir ve ChArUco deseniyle metrik ölçeğe (mm) getirir.

Adımlar:
  1. COLMAP: öznitelik çıkarımı -> eşleştirme -> seyrek yeniden yapılandırma (SfM) -> TXT dönüşümü
  2. (--yogun) COLMAP: görüntü düzeltme -> PatchMatch stereo -> füzyon (CUDA gerektirir)
  3. Her fotoğrafta ChArUco köşelerinin bulunması ve üçgenlenmesi
  4. Model ile desen arasında benzerlik dönüşümü: model mm birimine ve tabla koordinatlarına taşınır

Çıktılar (çalışma klasöründe):
  istatistik.json         Tablo 4.1 için COLMAP istatistikleri
  olcek.json              ölçek, dönüşüm ve ölçekleme hatası
  seyrek_mm.ply           ölçeklenmiş seyrek nokta bulutu
  yogun_mm.ply            ölçeklenmiş yoğun nokta bulutu (--yogun ile)
  kose_tespiti.csv        fotoğraf başına bulunan köşe sayısı

Örnekler:
  python on_calisma/02_colmap_olcekle.py --goruntuler veri/on_calisma_1/goruntuler \\
      --calisma veri/on_calisma_1/colmap --desen veri/desen/desen.json --kare-mm 19.95 --yogun
  # COLMAP'i yeniden çalıştırmadan sadece ölçekleme:
  python on_calisma/02_colmap_olcekle.py --goruntuler ... --calisma ... --desen ... --sadece-olcek
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from istasyon import colmap_io, ply
from istasyon.charuco import DesenAyar, KoseBulucu
from istasyon.olcek import olcek_hesapla

GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def calistir(komut: list[str]) -> float:
    print("\n$ " + " ".join(komut), flush=True)
    t0 = time.time()
    r = subprocess.run(komut)
    if r.returncode != 0:
        sys.exit(f"COLMAP adımı başarısız oldu (çıkış kodu {r.returncode}).")
    return time.time() - t0


def en_buyuk_model(seyrek: Path) -> Path:
    modeller = sorted(p for p in seyrek.iterdir() if p.is_dir())
    if not modeller:
        sys.exit("COLMAP hiçbir model oluşturamadı; fotoğraflar arasında yeterli eşleşme yok.")
    boyut = {m: (m / "images.bin").stat().st_size if (m / "images.bin").exists() else 0 for m in modeller}
    secilen = max(modeller, key=boyut.get)
    if len(modeller) > 1:
        print(f"Uyarı: COLMAP {len(modeller)} ayrı model üretti; en büyüğü ({secilen.name}) kullanılıyor. "
              "Bu, bazı fotoğrafların diğerlerine bağlanamadığını gösterir.")
    return secilen


def colmap_calistir(a, goruntuler: Path, calisma: Path) -> dict:
    sureler = {}
    vt = calisma / "database.db"
    seyrek = calisma / "sparse"
    if vt.exists():
        vt.unlink()
    if seyrek.exists():
        shutil.rmtree(seyrek)
    seyrek.mkdir(parents=True)

    c = a.colmap
    sureler["oznitelik_s"] = calistir([c, "feature_extractor", "--database_path", str(vt), "--image_path", str(goruntuler),
                                       "--ImageReader.single_camera", "1", "--ImageReader.camera_model", a.kamera_modeli])
    sureler["eslestirme_s"] = calistir([c, f"{a.eslestirme}_matcher", "--database_path", str(vt)])
    sureler["sfm_s"] = calistir([c, "mapper", "--database_path", str(vt), "--image_path", str(goruntuler),
                                 "--output_path", str(seyrek)])
    model = en_buyuk_model(seyrek)
    txt = calisma / "sparse_txt"
    txt.mkdir(exist_ok=True)
    calistir([c, "model_converter", "--input_path", str(model), "--output_path", str(txt), "--output_type", "TXT"])

    if a.yogun:
        yogun = calisma / "dense"
        if yogun.exists():
            shutil.rmtree(yogun)
        sureler["duzeltme_s"] = calistir([c, "image_undistorter", "--image_path", str(goruntuler), "--input_path", str(model),
                                          "--output_path", str(yogun), "--output_type", "COLMAP"])
        sureler["patchmatch_s"] = calistir([c, "patch_match_stereo", "--workspace_path", str(yogun),
                                            "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true"])
        sureler["fuzyon_s"] = calistir([c, "stereo_fusion", "--workspace_path", str(yogun), "--workspace_format", "COLMAP",
                                        "--input_type", "geometric", "--output_path", str(yogun / "fused.ply")])
    return sureler


def goruntu_oku(yol: Path, genislik: int, yukseklik: int) -> np.ndarray | None:
    """Fotoğrafı COLMAP'in gördüğü piksel düzeninde okur (EXIF yönlendirmesi farkını yakalar)."""
    for bayrak in (cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION, cv2.IMREAD_GRAYSCALE):
        g = cv2.imread(str(yol), bayrak)
        if g is not None and g.shape[1] == genislik and g.shape[0] == yukseklik:
            return g
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goruntuler", required=True, help="fotoğrafların bulunduğu klasör")
    ap.add_argument("--calisma", required=True, help="COLMAP çalışma ve çıktı klasörü")
    ap.add_argument("--desen", required=True, help="01_charuco_desen.py'nin ürettiği desen.json")
    ap.add_argument("--kare-mm", type=float, help="yazdırılan desende kumpasla ölçülen kare kenarı (mm)")
    ap.add_argument("--yogun", action="store_true", help="yoğun nokta bulutu da üret (CUDA'lı COLMAP gerekir)")
    ap.add_argument("--sadece-olcek", action="store_true", help="COLMAP'i çalıştırma, mevcut modeli ölçekle")
    ap.add_argument("--colmap", default="colmap", help="colmap programının yolu")
    ap.add_argument("--kamera-modeli", default="SIMPLE_RADIAL", choices=colmap_io.DESTEKLENEN)
    ap.add_argument("--eslestirme", default="exhaustive", choices=["exhaustive", "sequential"])
    ap.add_argument("--max-hata-px", type=float, default=2.0, help="köşe kabulü için en büyük yeniden izdüşüm hatası")
    a = ap.parse_args()

    goruntuler, calisma = Path(a.goruntuler), Path(a.calisma)
    calisma.mkdir(parents=True, exist_ok=True)
    ayar = DesenAyar.yukle(a.desen)
    desen_noktalari = ayar.kose_konumlari()
    if a.kare_mm:
        desen_noktalari *= a.kare_mm / ayar.kare_mm
        print(f"Kare kenarı ölçülen değere göre alındı: {a.kare_mm} mm (nominal {ayar.kare_mm} mm)")

    istat_yol = calisma / "istatistik.json"
    if a.sadece_olcek:
        istat = json.loads(istat_yol.read_text()) if istat_yol.exists() else {}
    else:
        if shutil.which(a.colmap) is None:
            sys.exit(f"'{a.colmap}' bulunamadı. COLMAP'i kurun veya --colmap ile yolunu verin.")
        t0 = time.time()
        istat = {"sureler": colmap_calistir(a, goruntuler, calisma)}
        istat["toplam_sure_dk"] = (time.time() - t0) / 60

    txt = calisma / "sparse_txt"
    kameralar = colmap_io.kameralari_oku(txt / "cameras.txt")
    gor = colmap_io.goruntuleri_oku(txt / "images.txt")
    xyz, rgb, hata, iz = colmap_io.noktalari_oku(txt / "points3D.txt")
    toplam = sum(1 for p in goruntuler.iterdir() if p.suffix.lower() in GORUNTU_UZANTILARI)
    istat.update({
        "toplam_goruntu": toplam,
        "kaydedilen_goruntu": len(gor),
        "seyrek_nokta": int(len(xyz)),
        "ort_yeniden_izdusum_hatasi_px": float(hata.mean()) if len(hata) else None,
        "ort_iz_uzunlugu": float(iz.mean()) if len(iz) else None,
        "kamera": {k.model: k.params.tolist() for k in kameralar.values()},
    })

    # ChArUco köşeleri
    bulucu = KoseBulucu(ayar)
    gozlemler: dict[int, list] = defaultdict(list)
    with open(calisma / "kose_tespiti.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["goruntu", "kose_sayisi"])
        for g in gor.values():
            k = kameralar[g.kamera_id]
            img = goruntu_oku(goruntuler / g.ad, k.genislik, k.yukseklik)
            if img is None:
                print(f"Uyarı: {g.ad} okunamadı veya boyutu COLMAP kamerasıyla uyuşmuyor; atlandı.")
                continue
            kimlikler, pikseller = bulucu.bul(img)
            w.writerow([g.ad, len(kimlikler)])
            for kid, xy in zip(kimlikler, pikseller):
                gozlemler[int(kid)].append((g.id, xy))
    print(f"\n{len(gozlemler)} farklı ChArUco köşesi en az bir fotoğrafta bulundu (desende {ayar.kose_sayisi} köşe var).")

    sonuc = olcek_hesapla(gozlemler, kameralar, gor, desen_noktalari, max_hata_px=a.max_hata_px)
    (calisma / "olcek.json").write_text(json.dumps(sonuc.ozet(), indent=2, ensure_ascii=False), encoding="utf-8")

    ply.yaz(calisma / "seyrek_mm.ply", sonuc.uygula(xyz), rgb)
    yogun = calisma / "dense" / "fused.ply"
    if yogun.exists():
        v = ply.oku(yogun)
        X = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        renk = np.stack([v["red"], v["green"], v["blue"]], axis=1) if "red" in v else None
        normal = np.stack([v["nx"], v["ny"], v["nz"]], axis=1) @ sonuc.R.T if "nx" in v else None
        ply.yaz(calisma / "yogun_mm.ply", sonuc.uygula(X), renk, normal)
        istat["yogun_nokta"] = int(len(X))
    istat_yol.write_text(json.dumps(istat, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== COLMAP istatistikleri (Tablo 4.1) ===")
    print(f"Kaydedilen görüntü        : {istat['kaydedilen_goruntu']} / {istat['toplam_goruntu']}")
    print(f"Seyrek nokta sayısı       : {istat['seyrek_nokta']}")
    print(f"Ort. yeniden izdüşüm hatası: {istat['ort_yeniden_izdusum_hatasi_px']:.3f} px")
    if "yogun_nokta" in istat:
        print(f"Yoğun nokta sayısı        : {istat['yogun_nokta']}")
    if "toplam_sure_dk" in istat:
        print(f"Toplam COLMAP süresi      : {istat['toplam_sure_dk']:.1f} dk")
    print("\n=== Ölçekleme ===")
    print(f"Kullanılan köşe           : {sonuc.kose_sayisi}")
    print(f"Ölçek                     : {sonuc.s:.6f} mm / model birimi")
    print(f"Hizalama hatası (RMS/max) : {sonuc.rms_mm:.3f} / {sonuc.max_mm:.3f} mm")
    print(f"Köşe çiftlerinden ölçek   : {sonuc.ikili_olcek_medyan:.6f} ± {sonuc.ikili_olcek_std:.6f} "
          f"(%{100 * sonuc.ikili_olcek_std / sonuc.ikili_olcek_medyan:.2f})")
    print(f"\nÖlçeklenmiş bulut(lar): {calisma}/seyrek_mm.ply" + (", yogun_mm.ply" if yogun.exists() else ""))
    print("Bulutlar mm biriminde; tabla düzlemi z = 0. Nesne ölçüleri CloudCompare'de alınabilir.")


if __name__ == "__main__":
    main()
