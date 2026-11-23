# Ön çalışma: elle döndürülen tabla ile pilot deney

Amaç: donanım hazır olmadan önce yazılım hattının (COLMAP), ChArUco ile ölçeklemenin ve
Laplace varyansına dayalı netlik denetiminin çalıştığını göstermek. Sonuçlar 1. ara raporun
4. bölümünü doldurur.

Komutlar depo kök klasöründe, sanal ortam etkinken çalıştırılır.

## 1. Deseni üret ve yazdır

```bash
python on_calisma/01_charuco_desen.py --cikti veri/desen
```

- `veri/desen/desen.pdf` dosyasını **%100 ölçekle / gerçek boyutta** yazdırın ("sayfaya sığdır" kapalı).
- Alttaki 100 mm çizgisini cetvelle kontrol edin.
- Bir karenin kenarını kumpasla ölçün ve not edin (ör. 19,95 mm). Bu değer 2. adımda `--kare-mm` ile verilir.
- Kâğıdı düz tutmak için sert bir kartona yapıştırın; kâğıdın kıvrılması ölçek hatasına yol açar.

## 2. Fotoğrafları çek

| Konu | Yapılacak |
|---|---|
| Fon | Düz, desensiz bir fon (tek renk karton/kumaş). Arka plandaki desenler SfM'i yanıltır, çünkü nesne döndüğü hâlde arka plan sabit kalır. |
| Kamera | Telefonu bir sehpaya/desteğe sabitleyin. Yakınlaştırma (zoom) kullanmayın, hep aynı lensle çekin. |
| Odak ve pozlama | Nesneye dokunup basılı tutarak odak ve pozlamayı kilitleyin (AE/AF kilidi). Çekim boyunca değişmemeli. |
| Işık | Yumuşak, homojen ışık; doğrudan güneş ve sert gölge yok. Flaş kapalı. |
| Açı | Kâğıdı her seferinde 10° döndürün; çemberdeki çentikleri masadaki sabit bir işarete hizalayın. Tam tur = 36 fotoğraf. |
| Yükseklik | Mümkünse ikinci bir turu farklı bir kamera yüksekliğinde (ör. daha yukarıdan) çekin; 72 fotoğraf modeli belirgin biçimde iyileştirir. |
| Desen | Desenin büyük kısmı her fotoğrafta görünmeli; nesne desenin ortasını kapatabilir, kenarları açık kalsın. |

Fotoğrafları `veri/on_calisma_1/goruntuler/` klasörüne kopyalayın. Not alın: telefon modeli,
çözünürlük, nesne ve gerçek ölçüleri, çekim mesafesi.

## 3. 3B model ve ölçekleme

```bash
python on_calisma/02_colmap_olcekle.py \
    --goruntuler veri/on_calisma_1/goruntuler \
    --calisma veri/on_calisma_1/colmap \
    --desen veri/desen/desen.json \
    --kare-mm 19.95 \
    --yogun
```

- `--yogun` CUDA'lı COLMAP ister. CUDA'sız COLMAP'te bu bayrağı kaldırın; seyrek model ve ölçekleme yine çalışır.
- Ekrandaki "COLMAP istatistikleri" **Tablo 4.1**'i doldurur.
- `seyrek_mm.ply` / `yogun_mm.ply` dosyalarını CloudCompare'de açın. Birim mm'dir, tabla düzlemi z = 0'dır.
  Nesnenin kenarlarını CloudCompare'in nokta seçme aracıyla ölçüp kumpas değerleriyle karşılaştırın (**Tablo 4.2**).
- "Hizalama hatası" desen köşelerinin ölçekleme sonrası ne kadar doğru yere oturduğunu gösterir; ölçeklemenin
  doğruluğu için doğrudan bir göstergedir.

## 4. Netlik testi

```bash
python on_calisma/03_netlik_testi.py \
    --goruntuler veri/on_calisma_1/goruntuler \
    --cikti veri/on_calisma_1/netlik
```

- `netlik_grafik.png` → **Şekil 4.3**.
- `ozet.txt` → her bulanıklık düzeyinde kaç fotoğrafın bulanık olarak yakalandığı; σ = 0 satırı yanlış alarm oranıdır.
- Önerilen eşik ilk tahmindir; sonraki dönemde gerçek bulanık çekimlerle ayarlanacaktır.

## Sorun giderme

| Belirti | Olası neden / çözüm |
|---|---|
| Kaydedilen görüntü sayısı düşük, birden fazla model | Fon desenli, ışık değişken ya da adım çok büyük. Fonu sadeleştirin, ikinci yükseklikten tur ekleyin. |
| "Yeterli köşe üçgenlenemedi" | Desen fotoğraflarda çok küçük veya bulanık. Kamerayı yaklaştırın, odağı desene ayarlayın. |
| Hizalama hatası > 0,5 mm | Kâğıt düz değil, `--kare-mm` ölçülmemiş ya da yazıcı ölçeği bozuk. |
