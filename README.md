# TEKNOFEST — Yapay Zeka Destekli Havayolu Optimizasyonu

IST hub'ı üzerinden aktarmalı O-D pazarlarında, ayarlanabilir uçuşların IST kalkış/varış saatlerini optimize ederek gelir ağırlıklı rekabetçilik faydasını (azalan getirili bağlantı ödülü + lookup tabanlı sıralama ödülü) enbüyükleyen, uçtan uca çalışan Python + Pyomo + HiGHS kod tabanı.

## Proje Yapısı

```text
.
├── main.py                 # Komut satırı giriş noktası ve deney akışları
├── check_solution.py       # Üretilen çözümün bağımsız denetimi
├── src/                    # Veri, model, çözücü, metrik ve çıktı modülleri
├── data/                   # Yerel Excel girdileri (Git tarafından izlenmez)
├── outputs/final_delivery/ # Doğrulanmış örnek teslim çıktısı
└── docs/                   # Matematiksel model ve teknik rapor
```

Python 3.10 veya üzeri önerilir. Varsayılan açık kaynak çözücü HiGHS'tır; model ayrıca Gurobi, CPLEX ve CBC ile çalışabilecek şekilde tasarlanmıştır.

## Kurulum

```bash
pip install -r requirements.txt
```

Sürümler `requirements.txt` içinde pinlenmiştir (yeniden üretilebilirlik). Solver: HiGHS (`highspy`), sabit seed=0 ve tek thread ile deterministik koşar.

## Girdi Dosyaları

`--data_dir` altında aranır:

- `O&D Rakip Bağlantı Tablosu.xlsx`
- `Yolcu Verisi_masked.xlsx`
- `change_ranking_input.xlsx`
- `Flight Pairs.xlsx`

Dosya adları birebir bulunamazsa anahtar kelimeyle esnek eşleme yapılır (ör. `*Rakip*`, `*Yolcu*`, `*rank*`, `*Pair*` — gizli test setinde farklı adlar sorun çıkarmaz). Gerekirse CLI ile açıkça verilebilir: `--competitor_connections_file`, `--passenger_data_file`, `--ranking_input_file`, `--flight_pairs_file`.

## Tek Komutla Çalıştırma (teslim koşusu)

Primary teslim: en yüksek gelirli **130 O-D pazarı × tam 7 gün**. **A, B, F kısıtları ile E1/E2 hard olarak sıfır ihlalle** sağlanır; tarife düzenliliği (G) uçuş-bazlı haftalık kaydırmayla içselleşir ve yalnızca veri kaynaklı istisnalar raporlanır (optimizasyon sıfır yeni ihlal üretir). Çözüm ispatlı optimaldir. Şartnamede çözüm süresi bir başarı kriteri olduğundan (Kriter 3) koşu **1 saatlik** üst bütçeyle sınırlanmıştır. Kapsam gerekçesi için `docs/technical_report.md` §4.

```bash
python main.py --data_dir ./data --output_dir ./outputs --model_level final \
  --solver highs --time_limit 3600 --max_od 130 --top_importance_only --max_shift 60 \
  --regularity_mode soft --regularity_penalty_weight 100000 \
  --directional_e1_mode hard --directional_e1_formula ratio_total \
  --directional_e2_mode hard \
  --run_id final_delivery
```

`--time_limit 3600` üst sınırdır; top-130 çözümü E1/E2'yi hard sağlayarak (G veri-kaynaklı istisnalar hariç sıfır ihlal) **dakikalar içinde ispatlı optimale** ulaşır ve erken durur. Kapsam gerekçesi (zorunlu seçim ⇔ E1 çelişkisi) için `docs/technical_report.md` §4.

**Çözücü notu:** Model **solver-agnostiktir**; `--solver` ile HiGHS (varsayılan), Gurobi, CPLEX veya CBC kullanılabilir (Gurobi/CPLEX için option adları kodda ayrıca desteklenir). Teslim çıktısı açık kaynak HiGHS ile ispatlı optimale ulaşır.

**Kapsam notu:** Brief 2.3 (zorunlu seçim) ile Brief 4.5-E (E1 α=0.20 hard) bu veri üzerinde kapsam büyüdükçe çelişir — HiGHS presolve, top-200 ve üzerinde modelin **infeasible** olduğunu <1 sn'de kanıtlar. top-130, her iki kuralın da **hard ve sıfır ihlalle** sağlanabildiği en geniş kapsamdır; bu nedenle primary seçilmiştir (bkz. `docs/technical_report.md` §4). Bu çelişkiye Q&A ile açıklık gelirse kapsam tek bayrakla genişletilebilir.

Küçük/hızlı doğrulama örneği:

```bash
python main.py --data_dir ./data --output_dir ./outputs --model_level final \
  --solver highs --time_limit 300 --max_od 20 --days 1 --top_importance_only --max_shift 60
```

Bağımsız çözüm denetimi:

```bash
python check_solution.py --output_dir ./outputs --run_id final_delivery
```

## Önemli CLI Bayrakları

| Bayrak | Varsayılan | Açıklama |
|---|---|---|
| `--model_level` | baseline | `final`: tüm kısıt zinciri (rotasyon, rezidüel hub kapasitesi, regularity, E1/E2, zorunlu seçim, lookup ranking) |
| `--shift_granularity` | flight_week | `flight_week`: uçuş başına tek haftalık shift (G kısıtı yapısal); `event`: gün bazlı shift |
| `--sequential_days` | kapalı | Yedek: günleri sırayla çözer; birleşik çıktı `WEEKLY STATUS` ile etiketlenir |
| `--competitor_mode` | carrier | `carrier`: rakip = taşıyıcı, Tcomp = en iyi süresi; `row`: her tablo satırı ayrı rakip |
| `--candidate_mode` | cross_product | `cross_product`: tüm TK inbound×outbound kombinasyonları aday; `table`: yalnızca tablo satırları |
| `--regularity_anchor_mode` | auto | Kısmi gün koşularında kapsam dışı günlere göre X_dev çapası |
| `--max_shift` | 60 | Event başına maksimum kaydırma (dk) |
| `--solver_random_seed` / `--solver_threads` | 0 / 1 | Determinizm |

## Model Özeti

- **Karar:** uçuş-bazlı haftalık tamsayı `shift` (aynı uçuş no/yön/istasyonun tüm haftası tek değişken; haftalık paterni bozuk uçuşlar event bazında çözülür veya dondurulur); bağlantı boşluğu ve journey time doğrusal güncellenir.
- **Zorunlu seçim:** `x_c = 1 ⇔ 60 ≤ gap_c ≤ 300` (üç-durum doğrusallaştırması) — pencere içindeki bağlantı pasif bırakılamaz.
- **Azalan getiri:** monoton slot göstergeleri; slot sayısı = aday sayısı.
- **Sıralama ödülü:** taşıyıcı bazlı rakip yenme göstergeleri + SOS1 seviye seçimi + `change_ranking_input.xlsx` lookup.
- **Rotasyon:** Flight Pairs eşleşmeleri; TOB + 45 dk yer süresi + TIB.
- **Hub kapasitesi:** 10 dk kovalar, varış 15 / kalkış 10, **rezidüel** (sabit uçuşların slotları düşülür).
- **Regularity:** haftalık ±15 dk — uçuş-bazlı haftalık shift ile **yapısal olarak garanti**; düzeltilebilir düzensiz uçuşlar açık kısıtla banda çekilir, onarılamazlar dondurulup ayrı raporlanır.
- **Big-M politikası:** global sabit yok; tüm big-M'ler aday/pazar/rakip bazında veriden türetilir.

Ayrıntı: `docs/mathematical_model.md` (PDF: `docs/mathematical_model.pdf`), `docs/technical_report.md` (PDF: `docs/technical_report.pdf`).

## Çıktılar

Her koşu için `outputs/<run_id>/` altında `optimized_schedule.xlsx`, `selected_connections.xlsx`, `beaten_competitors.xlsx`, `od_metrics.xlsx`, `objective_value.txt` (amaç değeri ayrışımı: `reward_before_violation_penalty` / `violation_penalty_total` / `net_solver_objective` / `solution_quality`), `feasibility_report.txt`, `run_summary.txt`, `directional_balance_report.xlsx` ve denetim dosyaları üretilir.

Deney çıktıları varsayılan olarak Git dışında tutulur; depoda yalnızca doğrulanmış `outputs/final_delivery/` paketi saklanır. Girdi çalışma kitapları da veri gizliliği ve dosya boyutu nedeniyle commit edilmez.

## Doğrulama

Hızlı bir uçtan uca kontrol için küçük bir veri alt kümesi çözülebilir:

```bash
python main.py --data_dir ./data --output_dir ./outputs --model_level final \
  --solver highs --time_limit 60 --max_od 2 --days 1 \
  --top_importance_only --max_shift 60 --run_id smoke_test
python check_solution.py --output_dir ./outputs --run_id smoke_test
```

## Bilinen Veri Notları

- 845 satırda hafta sarması düzeltildi (Pazar→Pazartesi bağlantıları); 384'ü geçerli TK bağlantısıdır.
- Bazı uçuşların mevcut tarifedeki günler arası yayılımı X_dev=15'i zaten aşar (ör. TK1304 AGP-IST, 85 dk) — saatleri dondurulur, ayrı raporlanır.
- Bazı 10 dk kovalarında mevcut tarife nominal kapasitenin üzerindedir — rezidüel 0'a kırpılır ve raporlanır.
