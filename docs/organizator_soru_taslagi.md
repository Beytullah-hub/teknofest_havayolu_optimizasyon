# Organizatör Q&A — Cevaplanan ve Açık Sorular

## A. Organizatörce CEVAPLANAN sorular (modele işlendi)

- **Rakip tanımı → TAŞIYICI BAZLI (onaylandı).** Organizatör: *"Rekabet listesinde gün bazında aynı havayoluna ait yalnızca bir uçuş (en kısası) olmalı; THY'nin de en kısa uçuşu rekabet sırasını belirlemeli."* → Modelimizdeki `--competitor_mode carrier` (her O-D/gün/taşıyıcı için tek kayıt = min journey time) birebir uyumludur. `N_od` = o O-D/gündeki farklı taşıyıcı sayısı.
- **Sıralamada beraberlik → aynı sırayı paylaşır (onaylandı).** Aynı journey time'a sahip rakipler aynı rank'ta sayılır; `Jπ ≤ Tcomp` yenme kuralıyla THY beraberlikte de yener. Model formülü `r = N − yenilen + 1` bu davranışı korur.
- **Hub kova sınırları → `00–09, 10–19, …` (onaylandı).** Kova = `dakika // 10`. Modelimiz zaten bu ayrımı kullanır (tam 10/20/30 dakikalar bir sonraki kovanın başı değil, ilgili kovanın parçasıdır: 12:10 → ikinci kova).
- **Çözücü → Gurobi (onaylandı).** Değerlendirme ortamında Gurobi lisansı vardır; jüri, Pyomo ile yazılmış kodu **Gurobi çözücüsüyle** çözerek değerlendirir (Pyomo zorunlu). Kodumuz solver-agnostiktir (`--solver gurobi`); Gurobi option'ları (TimeLimit/MIPGap/Seed/Threads) desteklenir. HiGHS açık kaynak yedektir.

## B. Hâlâ AÇIK sorular (iletilmesi önerilir)

Son tarih 16 Temmuz. Cevaplara göre kod tek bayrakla uyarlanabilir.

1. **Aday bağlantı evreni:** Brief 2.1'de π = (r1, r2) herhangi bir inbound×outbound çifti olarak tanımlanıyor. Saat kaydırmalarıyla bağlantı penceresine giren ancak tabloda satırı bulunmayan TK inbound×outbound kombinasyonları geçerli bağlantı sayılır mı? (Tablo mevcut tarifeye göre üretilmiş görünüyor.)

2. **H kısıtı:** Kick-off sunumunda 8. kısıt grubu olarak "Slot Pencereleri — düzenlenmiş havalimanlarında varış pencereleri" geçiyor, ancak Brief'te ve paylaşılan veride karşılığı yok. Gizli test setinde bu kısıt/veri olacak mı; olacaksa format nedir?

3. **Çıktı dosyası formatı:** Teslim maddesi "çıktı dosyası ... (belirtilen formatta)" diyor ancak paylaşılan dokümanlarda açık bir çıktı şablonu/şeması yok. Beklenen belirli bir kolon şeması, dosya adı veya şablon var mı? Varsa paylaşılabilir mi? (Şu an seçilen bağlantılar, ayarlanan IST saatleri, O-D bazında yenilen rakipler ve toplam amaç değeri ayrı Excel dosyaları olarak üretiliyor.)

4. **Hub kapasitesi rezidüeli:** Mevcut tarifede bazı 10 dakikalık kovalarda TK etkinlik sayısı nominal kapasitenin (varış 15 / kalkış 10) üzerinde. Bu kovalar için rezidüel kapasite 0 mı kabul edilmelidir, yoksa kapasite yalnızca ayarlanabilir uçuş alt kümesine mi uygulanır?

5. **Hafta sarması:** Pazar gecesi IST'e varıp Pazartesi sabahı devam eden bağlantılar geçerli aktarma sayılır mı? Sayılıyorsa bu bağlantının "günü" hangi gündür (inbound günü mü, outbound günü mü)?

6. **E1 tanımı (sayı mı skor mu?):** Brief 4.5'te E1 başlığı "Skor/bağlantı-sayısı dengesi". Bağıl dengesizlik ham bağlantı **sayısı** üzerinden mi, yoksa azalan getirili bağlantı **skoru** (1 + 0.5 + 0.25 + ...) üzerinden mi hesaplanmalıdır? (Skor bazlı yorum, azalan getiri nedeniyle küçük pazarlarda çok daha az kısıtlayıcıdır.)

7. **Teslim kanalı ve formatı:** Teslim paketi (kod, rapor PDF'leri, çıktı dosyaları) hangi kanala ve hangi formatta (ZIP, repo linki vb.) yüklenecektir? Q&A soruları için resmi iletişim adresi nedir? Paylaşılan brief/sunum/veri dokümanlarında bu bilgi yer almıyor.

8. **Zorunlu seçim ile E1/E2 çelişkisi (tam kapsam):** Brief 2.3 "aktarma süresi uygun olan tüm uçuşlar seçilmek zorundadır" der; 4.5 E1/E2 denge kısıtları koyar. Tam veri setinde, penceresi kaydırma sınırları içinde her koşulda pencere içinde kalan (kilitli) bağlantılar nedeniyle bazı O-D çiftlerinde iki kural aynı anda sağlanamıyor (model kanıtlanabilir infeasible). Bu durumda beklenen davranış nedir: (a) denge kısıtı bu çiftlerde gevşetilir/raporlanır, (b) zorunlu seçim gevşetilir, (c) tek yönlü pazarlar E1/E2 kapsamı dışıdır? Bizim uygulamamız (a)'dır: kaçınılabilir hiçbir ihlale izin verilmez, kaçınılmaz minimum set raporlanır.
