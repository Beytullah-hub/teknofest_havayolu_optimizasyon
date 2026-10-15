# Teknik Rapor

**TEKNOFEST — Yapay Zeka Destekli Havayolu Optimizasyonu**
Tarife Planlamalarında Rakip Odaklı Ağ Optimizasyonu

## 1. Yaklaşım Özeti

Problem, IST hub'ı üzerinden aktarmalı O-D pazarlarında, ayarlanabilir uçuşların IST kalkış/varış saatlerini belirleyerek gelir ağırlıklı rekabetçilik faydasını (azalan getirili bağlantı ödülü + lookup tabanlı sıralama ödülü) enbüyükleyen bir karışık tamsayılı programlama (MIP) modeli olarak kurulmuştur. Model Pyomo ile ifade edilir, HiGHS ile deterministik ayarlarla (sabit seed, tek thread) çözülür.

**Çalışma noktası (1 saat bütçesi).** Şartnamede çözüm süresi bir başarı kriteridir (Kriter 3: Hesaplama Performansı ve Ölçeklenebilirlik); kısa sürede iyi sonuç üretmek esastır. Model, gelir-ağırlıklı amaç fonksiyonunun domine edildiği **en yüksek gelirli 130 O-D pazarını** (`ρ` sıralamasına göre) **tam 7 gün** kapsamında çözer. **A (rotasyon), B (bağlantı penceresi), F (hub kapasitesi) ile E1 ve E2 (yönsel denge) hard kısıt olarak, hiçbir ihlal olmadan** sağlanır; tarife düzenliliği (G) uçuş-bazlı haftalık kaydırmayla içselleşir ve yalnızca **veri kaynaklı** istisnalar (orijinal tarifesi zaten X_dev'i aşan uçuşlar) raporlanır — optimizasyon **sıfır yeni regularity ihlali** üretir. Çözüm **ispatlı optimaldir** ve dakikalar içinde tamamlanır. Kapsam seçiminin gerekçesi — zorunlu seçim (2.3) ile yönsel denge (E1) arasındaki, veri kaynaklı ve kanıtlanmış çelişki — §4'tedir.

İki temel tasarım kararı vardır. Birincisi, saatleri mutlak değişkenler yerine **kaydırma (shift) değişkenleri** ile modellemektir: bağlantı boşluğu `gap = gap⁰ + shift_out − shift_in` ile, journey time `J = J⁰ + (gap − gap⁰)` ile doğrusal güncellenir; tüm big-M değerleri veriden sıkı türetilir. İkincisi, kaydırma granülaritesinin **uçuş-bazlı haftalık** seçilmesidir: aynı uçuş numarası/yön/istasyonun haftadaki tüm örnekleri tek shift değişkenini paylaşır. Bu, tarife düzenliliği (G) kısıtını değişken uzayında içselleştirir — haftalık paterni düzenli her uçuş için G kurulum gereği sağlanır — ve model boyutunu küçültürken kısmi çözümlerde görülen günler-arası uygunsuzlukları yapısal olarak imkânsız kılar.

## 2. Varsayımlar

1. **Rakip tanımı (taşıyıcı bazlı).** Her (O-D, gün, taşıyıcı) tek rakiptir ve `Tcomp` o taşıyıcının en iyi (min) journey time'ıdır. Bu yorumu veri zorunlu kılar: satır bazlı sayımda rakip sayısı bazı O-D/günlerde **90'a** çıkar, oysa `change_ranking_input.xlsx` sıralama ödülü tablosu **Rakip Sayısı ≤ 40** ile sınırlıdır — satır bazlı sayım tabloda karşılığı olmayan anahtarlar üretir. Taşıyıcı bazlı sayım (ort. 3.45, maks. 17) tablo aralığıyla tutarlıdır. Kod `--competitor_mode row` ile alternatif yoruma dönebilir.
2. **Sıralama formülü ve beraberlik.** Güncellenen sıralama `Son Rank = clip(N − yenilen + 1, 1, N)` ile lookup tablosuna eşlenir (tabloda ranklar 1..N aralığındadır). Aynı journey time'a sahip rakipler sıralamada aynı konumu paylaşır; `Jπ ≤ Tcomp` yenme kuralı gereği THY beraberlikte de yener ve formül bu davranışı korur.
3. **Aday bağlantı evreni.** Brief 2.1 π = (r1, r2) tanımını herhangi bir inbound×outbound çifti olarak verdiğinden, tablo satırlarına ek olarak aynı O-D/gün içindeki tüm TK kombinasyonları aday üretilir (cross-product); yolculuk süresi bacak sürelerinden (ElapsedTime1/2) kurulur. `--candidate_mode table` ile yalnızca tablo evrenine dönülebilir.
4. **Hafta sarması.** Pazar gecesi varıp Pazartesi devam eden bağlantılarda dakika-of-week farkı ~−10080 çıkar; yarım haftadan derin negatif farklara +10080 dk eklenir. Veri setinde bu düzeltme 845 satırı (384'ü geçerli 60-300 penceresinde TK bağlantısı) kurtarmıştır.
5. **Kaydırma sınırı.** Brief "Standard: sınır yok, gerekirse tanımlanabilir" der; teslim koşusunda operasyonel makullük ve hesaplama sıkılığı için `max_shift = 60 dk` tanımlanmıştır (CLI ile değiştirilebilir).
6. **Rezidüel hub kapasitesi.** Kapsam içindeki ayarlanabilir etkinlikler kova kapasitesini, tam ağdaki ayarlanamayan TK etkinliklerinin işgal ettiği slotlar düşüldükten sonra kullanır. Bazı kovalarda mevcut tarifenin kendisi nominal kapasiteyi aşmaktadır; bu kovalarda rezidüel 0'a kırpılır (ayarlanabilir etkinlik giremez) ve durum raporlanır.
7. **Veri kaynaklı regularity ihlalleri.** Bazı uçuşların mevcut tarifedeki günler arası yayılımı zaten X_dev=15'i aşar (ör. TK1304 AGP-IST, 85 dk). Bu uçuşların saatleri dondurulur; ihlal "veriden gelen" olarak ayrı raporlanır ve optimizasyonca kötüleştirilmez.
8. **Yönsel denge (E1/E2) — sıfır ihlalli, hard.** Primary çözümde E1 (α=0.20, ek tolerans yok) ve E2 (Γ=30) **hard kısıt** olarak modellenir ve tam sağlanır (checker: 0 E1, 0 E2). Tek yönlü pazarlar (ters yönde hiç TK adayı olmayanlar) E1/E2 kapsamı dışıdır ve sebep koduyla raporlanır. **Kapsam seçiminin özü budur:** zorunlu seçim (Brief 2.3, "uygun tüm uçuşlar seçilmek zorundadır") ile E1 hard, bu veri üzerinde kapsam büyüdükçe çelişir — HiGHS presolve, top-200 ve üzeri kapsamda modelin **infeasible** olduğunu <1 saniyede kanıtlar (pencere içinde kilitli bağlantılar `(n_f,n_r)=(2,1)` yaparken `1 > α·3`). Bu çelişki **veriden gelir, formülasyondan değil** (bkz. §4). Bu nedenle primary kapsam, her iki kuralın da **birlikte ve hard** sağlandığı ve çözümün hızlı-optimal kaldığı top-130 seçilmiştir (feasibility sınırı top-135; ispatlı optimal, E1/E2'de sıfır ihlal).

## 3. Doğrusallaştırma Seçimleri

- **Zorunlu seçim (x ⇔ pencere içi):** Üç-durum yapısı `x + low + high = 1` ve iki taraflı sıkı big-M pencere kısıtları. Böylece pencereye giren bağlantı pasif bırakılamaz (Brief 2.3), pencere dışı bağlantı sunulamaz (B kısıtı).
- **Azalan getiri:** Monoton slot göstergeleri `z_{k,s} ≤ z_{k,s−1}` ve `Σx = Σz`; slot sayısı aday sayısına eşittir (yapay tavan yok).
- **En iyi süre / rakip yenme:** `best_time` üç kısıt ailesiyle aktif bağlantıların minimumuna bağlanır; rakip yenme göstergeleri rakip bazlı sıkı big-M ile (`Jmax_k − Tcomp`, `Tcomp + ε`). Yenilen sayısı SOS1-benzeri seviye göstergeleriyle lookup ödülüne bağlanır; kötüleşen rank'lerin negatif ağırlıkları da modele yansır.
- **Koşullu E2:** `(2 − key_active_f − key_active_r)` aktivasyon terimi ile yalnızca iki yön de aktifken bağlayıcı.
- **Kova ataması:** Event başına ±max_shift içindeki kovalarla sınırlı atama değişkenleri (`~13 kova/event`), `M_b = 2·max_shift + 11`.
- **Big-M politikası:** Global sabit yok; gap big-M'leri aday bazında, journey big-M'leri pazar bazında (`Jmax_k`), rakip big-M'leri rakip bazında türetilir.

## 4. Kapsam Seçimi ve Ölçeklenebilirlik

**Neden top-130 O-D × 7 gün?** İki kısıtın kesişimi kapsamı belirler:
1. **Gelir ağırlığı:** Amaç `ρ` ile ölçeklidir (örnek veride 1000'e kadar); toplam faydanın büyük kısmı en yüksek gelirli pazarlarca üretilir, dolayısıyla top-N kapsamı bu payı yakalar.
2. **Zorunlu seçim ⇔ E1 çelişkisi (veri kaynaklı, kanıtlı):** Brief 2.3 pencere içindeki tüm uçuşların seçilmesini zorunlu kılar; Brief 4.5-E ise yönsel bağlantı-sayısı dengesini α=0.20 ile sınırlar. Bu veride, kapsam büyüdükçe pencereye kilitli bağlantılar bu iki kuralı **aynı anda sağlanamaz** hale getirir. HiGHS presolve, **top-200 ve üzerinde modelin infeasible olduğunu <1 saniyede kanıtlar**; feasibility sınırı **top-135**'tir (top-140 için 1 saatte feasible çözüm bulunamaz).

En geniş feasible kapsam top-135'tir; ancak çözüm süresi top-130'a (~16 sn) göre ~27× artar (~430 sn). Çözüm süresi bir başarı kriteri (Kriter 3) ve 1 saat sınırı jüri makinesinde çalışacağından, %1 objective farkı için bu süre riski alınmamış ve **her iki kuralı da hard sağlayan, hızlı-optimal top-130 primary seçilmiştir** — sonuç ispatlı optimal olup A/B/F ve E1/E2 kısıtlarında sıfır ihlallidir (G: yalnızca veri kaynaklı istisnalar, optimizasyon yeni ihlal üretmez). Şartname alt küme yaklaşımına açıkça izin verir: *"Çözümün elde edilememesi durumunda (computational complexity, memory limit vs.) tüm uçuşların bir alt kümesi ayarlanabilir olarak tanımlanabilir."*

**Çözücü bağımsızlığı.** Model Pyomo ile kurulduğundan çözücüden bağımsızdır: `--solver` ile HiGHS (varsayılan, açık kaynak), Gurobi, CPLEX veya CBC kullanılabilir; ticari çözücüler için option adları (TimeLimit/MIPGap/Seed/Threads) kodda ayrıca ele alınır. Primary top-130 çözümü açık kaynak HiGHS ile **ispatlı optimale** ulaşır; daha güçlü bir çözücü, çelişkiyi çözmeden (kapsam sabitken) aynı optimal değeri daha hızlı verir, çelişki-dışı daha büyük kapsamlarda ise daha geniş çözümlere imkân tanır.

**Ölçek analizi (deneysel).** Zorunlu seçim + E1 hard birlikte, kapsam büyüdükçe infeasible olur:

| Kapsam | Zorunlu seçim + E1/E2 hard | Objective |
|---|---|---|
| top-100 × 7 gün | ispatlı optimal, E1/E2 sıfır ihlal, ~sn | 630.299 |
| **top-130 × 7 gün (primary)** | **ispatlı optimal, E1/E2 sıfır ihlal, ~16 sn** | **810.340** |
| top-135 × 7 gün | ispatlı optimal, E1/E2 sıfır ihlal, ~430 sn | 818.118 |
| top-140 × 7 gün | 1 saatte feasible çözüm bulunamıyor | — |
| top-200 / top-300 | presolve **infeasible** (<1 sn kanıt) | — |

Feasibility sınırı top-135'tir; top-130 ise hem her iki kuralı hard sağlar hem de ~16 sn'de optimale ulaşır (top-135'in ~430 sn'sine karşı), bu yüzden süre güvenliği için primary seçilmiştir. (Not: E1'i soft'a alıp ihlale izin verilseydi top-300'e kadar çıkıp ~1.68M objective elde edilebilirdi; ancak bu, listelenmiş bir kısıtın — 4.5-E — ihlali anlamına geleceğinden ve diskalifiye riski taşıdığından tercih edilmemiştir.)


**Uçuş-bazlı haftalık kaydırma** tarife düzenliliğini (G) değişken uzayında içselleştirir: haftalık paterni düzenli (yayılım ≤ X_dev) her uçuşa haftada tek shift değişkeni verilir, G kurulum gereği sağlanır; açık günler-arası kısıt yalnızca az sayıdaki düzeltilebilir düzensiz uçuş için gerekir. Onarılamaz düzensiz gruplar dondurulur ve veri-verili ihlal olarak raporlanır. Bilinçli ödünleşim: aynı uçuşun günler arası ±X_dev bağımsız ince ayarı feda edilir; karşılığında model küçülür ve determinist tek-komut çalıştırma basitleşir (`--shift_granularity event` bu serbestliği açar).

## 5. Sonuçlar

Primary teslim çözümü (`outputs/final_delivery/`, top-130 O-D × 7 gün):

| Metrik | Değer |
|---|---|
| Amaç değeri (ödül) | **810.340** (bağlantı 556.288 + sıralama 254.052) |
| Aktif bağlantı / yenilen rakip | 742 / 650 |
| Çözüm kalitesi | **OPTIMAL** (ispatlı; dakikalar) |
| Sert (hard) ihlal | **0** (A rotasyon, B pencere, F kapasite, zorunlu seçim, E1, E2 — hepsi hard) |
| Kısıt ihlali E1 / E2 (hard) | **0 / 0** |
| Regularity (G) | Optimizasyon kaynaklı ihlal **0**. G, uçuş-bazlı haftalık kaydırmayla içselleştirilir; onarılamaz/veri kaynaklı istisnalar soft-pre-existing olarak raporlanır (13 grup, saatleri dondurulmuş — ör. TK2014 IST-ASR orijinalde 175 dk yayılım). |

**A, B, F ve E1/E2 hard olarak sıfır ihlalle sağlanır;** G regularity uçuş-bazlı haftalık kaydırmayla içselleştirilir, veri kaynaklı/onarılamaz istisnalar soft/pre-existing olarak raporlanır ve **optimizasyon yeni regularity ihlali üretmez** (`--regularity_mode soft`; checker: introduced/hard regularity ihlali 0). Çözüm **ispatlı optimaldir** — diskalifiye/ağır-puan-kaybı eşiklerinin dışındadır.

- **Amaç değeri ayrışımı:** `objective_value.txt` içinde `reward_before_violation_penalty` (jürinin kıyasladığı saf şartname amacı), `violation_penalty_total` (yalnızca çözücü yönlendirmesi için ceza) ve `net_solver_objective` ayrı raporlanır.
- **Denge kısıtlarının fiyatı:** Mevcut tarife (baseline) E1/E2'yi yaygın ihlal eder (~93.000 ihlal birimi); çözümümüz bu kısıtları karşılanabilir her çiftte sağlar. Denge kısıtları kapatıldığında model baseline'ı aşar (gün 1 deneyi: 620.171 vs 597.366, +%3.8) — fark, denge kısıtlarının gerçek fiyatıdır.
- **Uygunluk:** Bağımsız checker pencere, zorunlu seçim, rotasyon, rezidüel kapasite, regularity (dairesel yayılım, introduced/pre-existing ayrımı), E1/E2 ve amaç yeniden hesabını doğrular.
- **Determinizm:** Sabit HiGHS seed (0), tek thread, pinlenmiş bağımlılık sürümleri; aynı girdiyle aynı çıktı.

## 6. Çıktılar ve Yeniden Üretim

Tek komut (primary teslim, ≤1 saat sınırına uygun):

```bash
python main.py --data_dir ./data --output_dir ./outputs --model_level final \
  --solver highs --time_limit 3600 --max_od 130 --top_importance_only --max_shift 60 \
  --regularity_mode soft --regularity_penalty_weight 100000 \
  --directional_e1_mode hard --directional_e1_formula ratio_total \
  --directional_e2_mode hard \
  --run_id final_delivery
```

`time_limit 3600`, çözüm süresi bir başarı kriteri olduğundan benimsenen 1 saatlik üst bütçedir. Çıktılar `outputs/final_delivery/` altında (optimized_schedule, selected_connections, beaten_competitors, od_metrics, objective_value, feasibility_report, run_summary ve denetim raporları). Bağımsız denetim: `python check_solution.py --output_dir ./outputs --run_id final_delivery`.

## 7. Sınırlamalar ve Gelecek İş

- Primary kapsam top-130 O-D'dir; zorunlu seçim ve E1/E2'nin **birlikte hard sağlandığı** feasibility sınırı top-135, primary ise süre güvenliği için top-130'dur (top-140+ 1 saatte çözülemez/infeasible). Daha yüksek objective (~1.68M, top-300) yalnızca E1'i soft'a alıp 4.5-E kısıtını ihlal ederek mümkündür; diskalifiye riski nedeniyle tercih edilmemiştir. Q&A ile bu çelişkiye açıklık gelirse kapsam tek bayrakla genişletilebilir.
- Haftalık paylaşılan shift, aynı uçuşun günler arası bağımsız ince ayarını feda eder (`--shift_granularity event` açar).
- Üretilmiş adaylarda detour benzeri ticari filtreler (veri sağlanmadığından) uygulanmamıştır; `--candidate_mode table` muhafazakâr alternatiftir.
- Sunumda geçen H kısıtı (düzenlenmiş havalimanlarında varış pencereleri) için veri sağlanmamıştır; event bazlı saat bandı mekanizması bu kısıtı veri geldiğinde doğrudan destekler.
