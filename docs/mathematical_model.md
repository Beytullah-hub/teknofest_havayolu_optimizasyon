# Matematiksel Model Dokümanı

**TEKNOFEST — Yapay Zeka Destekli Havayolu Optimizasyonu**
Tarife Planlamalarında Rakip Odaklı Ağ Optimizasyonu

Bu doküman teslim edilen nihai modelin kümelerini, parametrelerini, karar değişkenlerini, amaç fonksiyonunu ve tüm kısıtlarını tanımlar. Model Python + Pyomo ile kurulur ve HiGHS ile çözülür. Primary teslim, gelir-ağırlıklı amaç fonksiyonunca domine edilen **en yüksek gelirli 130 O-D pazarını × tam 7 gün** tek MIP olarak çözer; **A, B, F kısıtları ile E1/E2 (D/E) hard olarak sıfır ihlalle** sağlanır, tarife düzenliliği (G) uçuş-bazlı haftalık kaydırmayla içselleşir (yalnızca veri kaynaklı istisnalar raporlanır, optimizasyon sıfır yeni ihlal üretir) ve çözüm ispatlı optimaldir. Kapsam gerekçesi ve ölçek analizi Bölüm 10'dadır.

## 1. Kümeler

- `S`: Dış istasyonlar (outstation).
- `K`: O-D/gün pazar anahtarları `k = (o, d, h)`, `o ≠ d`, `h ∈ {1..7}`.
- `R`: Ayarlanabilir TK IST uçuş etkinlikleri (event). Her event `r`, bir uçuş numarasının belirli bir *fiziksel* gündeki IST varışı (inbound) veya IST kalkışı (outbound)'dır. Hafta sarmalı bağlantılarda event günü, pazarlama günü (Gün kolonu) değil gerçek dakika-of-week'ten türetilir; böylece aynı fiziksel uçuş tek bir event/karar değişkeni alır.
- `C`: Aday TK aktarma bağlantıları `π = (r_in, r_out)`. Aday kümesi iki kaynaktan oluşur:
  - **Tablo adayları:** Girdi tablosundaki mevcut TK bağlantı satırları.
  - **Üretilmiş adaylar (cross-product):** Aynı O-D/gün için tüm TK inbound × outbound kombinasyonları; gap erişilebilirlik ön filtresi `L − 2·max_shift ≤ gap_eski ≤ U + 2·max_shift` uygulanır. (Brief 2.1'deki π tanımı herhangi bir (r1, r2) çiftini kapsar.)
- `C_k ⊆ C`: `k` pazarına ait adaylar.
- `S_k`: `k` için bağlantı slotları, `|S_k| = |C_k|` (yapay tavan yok).
- `J_k`: `k` için rakip kayıtları. **Rakip = taşıyıcı**: her (O-D, gün, taşıyıcı) için tek kayıt; `Tcomp` o taşıyıcının en iyi (min) journey time değeridir. `N_k = |J_k|`.
- `Q`: Uçak rotasyonu kısıt çiftleri (Flight Pairs eşleşmelerinden).
- `B`: 10 dakikalık hub zaman kovaları, `b ∈ {0..143}`, gün ve yön (varış/kalkış) bazında.
- `P`: Yönsel denge çiftleri `p = (k, rev(k))`.

## 2. Parametreler

- `ρ_k`: O-D gelir/önem katsayısı (Yolcu Verisi; O-D bazında toplanır).
- `gap⁰_c`: Adayın mevcut tarifedeki bağlantı boşluğu (hafta sarması düzeltilmiş: ham fark yarım haftadan derin negatifse +10080 dk eklenir).
- `J⁰_c`: Adayın mevcut journey time'ı. Tablo adaylarında ElapsedTime1 + gap + ElapsedTime2 (yoksa gate-to-gate); üretilmiş adaylarda TIB + gap⁰ + TOB (bacak süreleri ElapsedTime1/2 medyanlarından).
- `L = 60`, `U = 300`: Bağlantı penceresi (dk).
- `max_shift`: Event başına izin verilen maksimum kaydırma (teslim koşusunda 60 dk; Brief "gerekirse tanımlanabilir" der).
- `τ = 45`: Minimum yer süresi (dk).
- `X_dev = 15`: Gün-içi tutarlılık (regularity) limiti (dk).
- `cap_arr = 15`, `cap_dep = 10`: Kova başına nominal varış/kalkış kapasitesi. **Rezidüel kapasite** kullanılır: modelde ayarlanamayan (kapsam dışı/sabit) TK etkinliklerinin işgal ettiği slotlar nominal kapasiteden düşülür; negatif rezidüel 0'a kırpılır.
- `α = 0.20`: E1 bağıl dengesizlik eşiği (ek tolerans YOK; şartname birebir).
- `Γ = 30`: E2 en iyi journey time fark eşiği (dk).
- `W^c_s = 0.5^(s−1)`: Azalan getirili slot ödülü (ilk bağlantı 1, sonrakiler yarısı).
- `W^r_{k,b}`: `b` rakip yenildiğinde ranking ödülü; `change_ranking_input.xlsx` lookup tablosundan (Rakip Sayısı, İlk Rank, Son Rank) → Weight. `İlk Rank` mevcut tarifeden hesaplanır, `Son Rank = clip(N_k − b + 1, 1, N_k)`.
- `Tcomp_{k,j}`: Rakip `j`'nin journey time'ı (taşıyıcının en iyisi).
- `Jmax_k = max_{c∈C_k} (J⁰_c + 2·max_shift)`: Pazar bazında sıkı big-M değeri.

## 3. Karar Değişkenleri

- `shift_g ∈ Z, −max_shift ≤ shift_g ≤ max_shift`: Kaydırma değişkeni (dk). **Granülarite uçuş-bazlı haftalıktır**: `g`, bir (yön, uçuş numarası, istasyon) grubudur ve grubun haftadaki TÜM event'leri aynı `shift_g` değerini paylaşır (`t_r = t⁰_r + shift_{g(r)}`). Haftalık saat paterni `X_dev`'den fazla dağılan gruplar bu paylaşımdan çıkarılır: `2·max_shift` ile banda çekilebiliyorsa event bazında ayrı değişken alır, çekilemiyorsa donduruluyor (`shift = 0`) ve veri-verili ihlal olarak raporlanır. Fiziksel karar olan "IST kalkış/varış saatleri" bu yapıdan türetilir.
- `gap_c ∈ R`: Adayın yeni bağlantı boşluğu.
- `x_c ∈ {0,1}`: Aday bağlantı pazara sunuluyor mu.
- `low_c, high_c ∈ {0,1}`: Gap pencerenin altında/üstünde göstergeleri (zorunlu seçim doğrusallaştırması).
- `z_{k,s} ∈ {0,1}`: `k` pazarında `s`. slot dolu mu (monoton).
- `key_active_k ∈ {0,1}`: `k`'da en az bir aktif bağlantı var mı.
- `best_conn_c ∈ {0,1}`, `best_time_k ∈ [0, Jmax_k]`: En iyi journey time seçimi ve değeri.
- `beat^L_{k,j} ∈ {0,1}`: Rakip `j` yenildi mi (`best_time_k ≤ Tcomp_{k,j}`).
- `level_{k,b} ∈ {0,1}`: Yenilen rakip sayısı `b` seviyesinin göstergesi.
- `bucket_{r,b} ∈ {0,1}`: Event `r`, kova `b`'ye atandı mı.
- `dd_p ≥ 0`: Yönsel bağlantı sayısı farkı.

## 4. Amaç Fonksiyonu

```text
maximize   Σ_k ρ_k · [ Σ_{s∈S_k} W^c_s · z_{k,s}  +  Σ_b W^r_{k,b} · level_{k,b} ]
```

Amaç, şartnamedeki `max Σ ρ_od (bağlantı-sayısı ödülü + sıralama ödülü)` ifadesinin birebir doğrusallaştırmasıdır. **Ek ceza terimi yoktur** (shift penalty teslim modelinde kapalıdır, `w_shift = 0`).

## 5. Bağlantı Uygunluğu ve Zorunlu Seçim (B, C kısıtları)

Gap tanımı (tutarlı bağlama):

```text
gap_c = gap⁰_c + shift_out(c) − shift_in(c)                          ∀c
```

Pencere bağlama (aday bazında sıkı big-M'lerle; `gapmin_c = gap⁰_c − 2·max_shift`, `gapmax_c = gap⁰_c + 2·max_shift`):

```text
gap_c ≥ L − (L − gapmin_c)⁺ · (1 − x_c)                              ∀c
gap_c ≤ U + (gapmax_c − U)⁺ · (1 − x_c)                              ∀c
```

Zorunlu seçim — "aktarma süresi uygun olan tüm uçuşlar seçilmek zorundadır" (Brief 2.3):

```text
x_c + low_c + high_c = 1                                             ∀c
gap_c ≤ (L − 1) + (gapmax_c − L + 1)⁺ · (1 − low_c)                  ∀c
gap_c ≥ (U + 1) − (U + 1 − gapmin_c)⁺ · (1 − high_c)                 ∀c
```

Böylece `x_c = 1 ⇔ L ≤ gap_c ≤ U`: pencere içindeki hiçbir bağlantı pasif bırakılamaz, pencere dışındaki hiçbir bağlantı sunulamaz.

Journey time (Brief 2.2): `J_c = J⁰_c + (gap_c − gap⁰_c)`; blok süreleri karara bağlı olmadığından journey time yalnızca gap üzerinden güncellenir (sabit-sabit bağlantılarda sabittir, veri olarak ele alınır).

## 6. Azalan Getirili Slot Yapısı (C kısıtı)

```text
Σ_{c∈C_k} x_c = Σ_{s∈S_k} z_{k,s}                                    ∀k
z_{k,s} ≤ z_{k,s−1}                                                  ∀k, s ≥ 2
```

Monoton slotlar hem doğruluğu hem LP gevşetme kalitesini iyileştirir (Brief 7 ipucu). `|S_k| = |C_k|` olduğundan zorunlu seçimle çelişen yapay tavan yoktur.

## 7. Rakip Yenme ve Sıralama (D kısıtı)

En iyi journey time seçimi:

```text
Σ_{c∈C_k} x_c ≤ |C_k| · key_active_k ;   Σ_{c∈C_k} x_c ≥ key_active_k        ∀k
Σ_{c∈C_k} best_conn_c = key_active_k ;   best_conn_c ≤ x_c                   ∀k, c
best_time_k ≤ J_c + Jmax_k · (1 − x_c)                                       ∀c   (best ≤ tüm aktifler)
best_time_k ≥ J_c − Jmax_k · (1 − best_conn_c)                               ∀c   (best = seçilen)
best_time_k ≤ J_c + Jmax_k · (1 − best_conn_c)                               ∀c
```

Rakip yenme (`J ≤ Tcomp` ⇒ yenilmiş; Brief 4.4):

```text
beat^L_{k,j} ≤ key_active_k                                                  ∀k, j
best_time_k ≤ Tcomp_{k,j} + (Jmax_k − Tcomp_{k,j})⁺ · (1 − beat^L_{k,j})     ∀k, j
best_time_k ≥ Tcomp_{k,j} + ε − (Tcomp_{k,j} + ε) · (beat^L_{k,j} + 1 − key_active_k)   ∀k, j
```

(ε = 0.001; ikinci kısıt yenilmemiş rakibin gösterge tutarlılığını sağlar.) Bir rakibi yenmek, onu yenen en az bir sunulan bağlantı gerektirir; `best_time` aktif bağlantıların minimumu olduğundan bu birebir sağlanır.

Sıralama seviyesi ve ödül bağlama:

```text
Σ_b level_{k,b} = 1 ;    Σ_j beat^L_{k,j} = Σ_b b · level_{k,b}              ∀k
```

Güncellenen sıralama `r_k = clip(N_k − b + 1, 1, N_k)` lookup tablosunun rank alanıyla eşlenir (tabloda ranklar 1..N aralığındadır; uç durum yorumu teknik raporda belgelenmiştir). Kötüleşen rank'ler için tablodaki negatif ağırlıklar da modele aynen yansır.

## 8. Zaman ve Operasyon Kısıtları (A, F, G)

**(A) Rotasyon** — Flight Pairs eşleşmesi bulunan her çift `q ∈ Q` için:

```text
t⁰_in(q) + shift_in(q) ≥ t⁰_out(q) + shift_out(q) + TOB_q + τ + TIB_q
```

Blok süreleri O&D tablosundaki ElapsedTime1/2 ve rota/uçuş numarası eşleşmelerinden türetilir; eşleşmeyenler raporlanır.

**(F) Hub kapasitesi (rezidüel)** — Her event tam bir kovaya atanır ve atandığı kovayla saat tutarlıdır:

```text
Σ_b bucket_{r,b} = 1                                                          ∀r
dakika_r ≥ 10·b − M_b · (1 − bucket_{r,b})                                    ∀(r,b)
dakika_r ≤ 10·(b+1) − 1 + M_b · (1 − bucket_{r,b})                            ∀(r,b)
Σ_{r: varış} bucket_{r,b} ≤ (cap_arr − sabit_kullanım_{h,b,arr})⁺              ∀(h,b)
Σ_{r: kalkış} bucket_{r,b} ≤ (cap_dep − sabit_kullanım_{h,b,dep})⁺             ∀(h,b)
```

`M_b = 2·max_shift + 10 + 1`. Sabit kullanım, tam ağdaki (kapsam dışı dahil) ayarlanamayan TK etkinliklerinden hesaplanır.

**(G) Tarife düzenliliği** — Gereksinim: aynı uçuş numarasının farklı günlerdeki IST saatleri birbirinden en fazla `X_dev` sapabilir:

```text
|(t⁰_a + shift_a) − (t⁰_b + shift_b)| ≤ X_dev        (gün-içi dakika bazında)
```

Uygulama üç katmanlıdır:

1. **Yapısal garanti (esas mekanizma):** Uçuş-bazlı haftalık kaydırma (Bölüm 3) sayesinde bir grubun tüm günleri aynı `shift_g` ile kayar; günler arası sapma orijinal değerinde sabit kalır. Orijinal tarifede `≤ X_dev` olan her grup için G kısıtı **kurulum gereği** sağlanır — açık kısıta gerek yoktur. Bu, hem modeli küçültür hem de kısmi çözümlerden kaynaklanabilecek günler-arası uygunsuzlukları imkânsız kılar.
2. **Düzeltilebilir düzensiz gruplar:** Orijinal sapması `X_dev`'i aşan ancak `2·max_shift` ile banda çekilebilen gruplar event bazında değişken alır ve yukarıdaki pairwise kısıt açıkça eklenir; model bu uçuşları haftalık banda **çeker** (veri ihlalini onarır).
3. **Onarılamaz gruplar:** `2·max_shift` ile bile banda sığmayan gruplar orijinal saatlerinde dondurulur ve `pre_existing_data_violation_frozen` statüsüyle raporlanır; optimizasyon hiçbir ihlali kötüleştirmez ve yeni ihlal üretmez.

Kısmi gün koşularında (yalnızca alt küme çözülürken) kapsam dışı günlerin sabit saatleri ek **çapa (anchor)** sınırı olarak uygulanır: yeni saat, kapsam dışı referansların `[max_ref − X_dev, min_ref + X_dev]` bandında tutulur.

## 9. Yönsel Denge (E1, E2)

Primary teslimde E1 ve E2 **hard kısıt** olarak uygulanır (slack yok). `n_f = Σ_{c∈C_k} x_c`, `n_r = Σ_{c∈C_rev(k)} x_c`:

```text
dd_p ≥ n_f − n_r ;  dd_p ≥ n_r − n_f
dd_p ≤ α · (n_f + n_r)                                    (E1; α = 0.20, ek tolerans yok)
```

İki yön de boşsa E1 kendiliğinden sağlanır (0 ≤ 0). E2 yalnızca iki yön de aktifken bağlayıcıdır (koşullu aktivasyon):

```text
best_time_k − best_time_rev(k) ≤ Γ + Mp · (2 − key_active_k − key_active_rev(k))
best_time_rev(k) − best_time_k ≤ Γ + Mp · (2 − key_active_k − key_active_rev(k))
Mp = max(Jmax_k, Jmax_rev(k))
```

**Kapsam ve uygulanabilirlik.** E1/E2 yalnızca her iki yönün de en az bir aday bağlantısı olan çiftlere uygulanır; ters yönü yapısal olarak hiç aday içeremeyen (tek yönlü) pazarlar kısıt kapsamı dışında tutulur ve sebep koduyla raporlanır (`one_direction_without_tk_candidates`).

**Zorunlu seçim ⇔ E1 çelişkisi ve kapsam.** Zorunlu seçim (Bölüm 5, `x_c = 1 ⇔ pencere içi`) ile E1 hard, bu veri üzerinde kapsam büyüdükçe çelişir: pencereye kilitli bağlantılar `(n_f, n_r) = (2, 1)` yaparken `1 > α·3 = 0.6` olur ve model **infeasible** hale gelir (HiGHS presolve top-200+ için <1 sn'de kanıtlar). Bu nedenle primary kapsam, her iki kuralın da hard sağlandığı en geniş-hızlı nokta olan **top-130** seçilir (Bölüm 10). Kodda E1/E2 soft moda alınabilir (`--directional_e1_mode soft`), ancak bu listelenmiş bir kısıtın ihlali anlamına geldiğinden primary teslimde kullanılmaz.

## 10. Çözüm Stratejisi ve Kapsam (1 Saat Çalışma Noktası)

Şartnamede çözüm süresi bir başarı kriteridir (Kriter 3); koşu 1 saatlik üst bütçeyle sınırlanmıştır. Primary çözüm, gelir-ağırlıklı amaç fonksiyonunun domine edildiği **en yüksek gelirli 130 O-D pazarını × tam 7 gün** tek MIP olarak çözer; **A, B, F kısıtları ile E1, E2 (D/E) hard olarak, hiçbir ihlal olmadan** sağlanır ve çözüm **ispatlı optimaldir** (~16 sn). Tarife düzenliliği (G) uçuş-bazlı haftalık kaydırmayla (Bölüm 3/8) değişken uzayında içselleşir; kapsam-içi her düzenli uçuş için G kurulum gereği sağlanır. G'de raporlanan tek istisna **veri kaynaklı** ihlallerdir: orijinal tarifesi zaten X_dev=15'i aşan uçuşlar (ör. TK2014 IST-ASR, 175 dk) dondurulur; optimizasyon bunları **kötüleştirmez ve yeni ihlal üretmez** (introduced/hard regularity ihlali = 0).

Çözüm deterministik ayarlarla (sabit seed=0, tek thread) HiGHS ile alınır ve mip_gap toleransına dakikalar içinde ulaşıp süre sınırından önce durur. Model sıkılaştırma (rakip binary önişleme: TK-adaysız pazar + never-beatable eleme; veri-türevli sıkı big-M'ler) LP gevşemesini iyileştirir.

**Kapsam seçimi ve ölçek (dürüst not).** Zorunlu seçim (Bölüm 5) ile E1 hard bu veri üzerinde çelişir; kapsam büyüdükçe model infeasible olur (HiGHS presolve top-200+ için <1 sn'de kanıtlar). Feasibility sınırı top-135'tir; primary top-130, her iki kuralı hard sağlarken ~16 sn'de optimale ulaştığı için (top-135'in ~430 sn'sine karşı) süre güvenliğiyle seçilmiştir. E1'i soft'a alarak top-300'e kadar ~1.68M objective mümkündür ancak bu, listelenmiş 4.5-E kısıtının ihlali olduğundan tercih edilmez. Kod solver-agnostiktir (`--solver` ile HiGHS/Gurobi/CPLEX/CBC).

## 11. Doğrusallaştırma ve Sıkılık Notları

- Tüm big-M'ler örnek verisinden türetilir: gap big-M'leri aday bazında (`gapmin/gapmax`), journey big-M'leri pazar bazında (`Jmax_k`), rakip kısıtlarında rakip bazında. Global gevşek sabit yoktur.
- Ulaşılamaz adaylar (gap penceresi hiçbir shift kombinasyonuyla yakalanamayan) ve kapasitesi sıfır kovalar ön işlemede elenir.
- Azalan getiri monoton slot göstergeleriyle; zorunlu seçim üç-durum (x/low/high) yapısıyla kurulur.
- Sabit-sabit bağlantıların journey time ve yenme durumu veriye sabitlenebilir; mevcut kurulumda tüm TK etkinlikleri ayarlanabilir olduğundan bu yol kullanılmaz.

## 12. Uygunluk Denetimi (Checker)

Modelden bağımsız checker şunları yeniden hesaplar: pencere ihlalleri, zorunlu seçim tutarlılığı (feasible-but-inactive / infeasible-but-active), rotasyon, rezidüel hub kapasitesi, haftalık regularity, E1/E2 ve amaç değerinin bileşen bazında yeniden hesabı. Checker ile model arasında fark varsa checker esas alınır.
