# Akıllı Döner Tablalı Fotogrametri İstasyonu

Döner tabla üzerine konan 5–25 cm boyutlarındaki nesnelerin fotoğraflarını insan müdahalesi olmadan çeken
ve bu görüntülerden metrik ölçeği doğru, dokulu 3B modeller üreten düşük maliyetli (< 200 USD), açık kaynaklı
bir fotogrametri istasyonu.

Marmara Üniversitesi Teknoloji Fakültesi Bilgisayar Mühendisliği Bölümü bitirme projesi.

- **Öğrenciler:** Muhammed Zahid Demirel, Yusuf Tunahan Ergin
- **Danışman:** Dr. Öğr. Üyesi Anıl Baş

## Sistem

| Katman | Donanım / Yazılım | Görev |
|---|---|---|
| Hareket | Arduino, TMC2209, NEMA 17 | Döner tabla ve kamera eğim kolunun sürülmesi, LED kontrolü |
| Çekim | Raspberry Pi, Python, OpenCV | Çekim planı, kamera kontrolü, kare başına netlik/pozlama denetimi |
| Yeniden yapılandırma | Bilgisayar, COLMAP, Open3D | SfM/MVS, ChArUco ile metrik ölçekleme, ICP ile birleştirme |
| Doğrulama | CloudCompare | Referans nesnelerde C2M / RMS hata analizi |

## Klasör yapısı

```
istasyon/        Ortak Python paketi (ChArUco, kalite denetimi, COLMAP okuma, ölçekleme)
on_calisma/      Ön çalışma betikleri (elle döndürülen tabla ile pilot deney)
tests/           Birim testleri (pytest)
veri/            Fotoğraflar ve çıktılar — git'e gönderilmez
```

İlerleyen dönemlerde eklenecek klasörler:

```
arduino/         Motor ve LED kontrol yazılımı (İP-1)
pi/              Raspberry Pi çekim yazılımı (İP-2)
boru_hatti/      Otomatik yeniden yapılandırma ve ICP birleştirme (İP-3, İP-4)
mekanik/         CAD ve STL dosyaları (İP-1)
dogrulama/       Test nesneleri ve analiz betikleri (İP-5)
```

## Kurulum

Python 3.10 veya üstü gerekir.

```bash
python -m venv .venv
source .venv/bin/activate          # fish kabuğunda: source .venv/bin/activate.fish
pip install -r requirements.txt
```

COLMAP ayrıca kurulmalıdır: <https://colmap.github.io/install.html>. Yoğun nokta bulutu (MVS) adımı
CUDA destekli bir COLMAP sürümü ve NVIDIA ekran kartı gerektirir; seyrek yeniden yapılandırma ve ölçekleme
CUDA olmadan da çalışır.

Testler:

```bash
python -m pytest
```

## Ön çalışma

Ayrıntılı adımlar için [on_calisma/README.md](on_calisma/README.md).

## Lisans

MIT — ayrıntılar için [LICENSE](LICENSE).
