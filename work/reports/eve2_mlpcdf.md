# eve2: mlpcdf — ординально-CDF голова (CREAD/RQ-Reg класс, идея №5 research-отчёта)

Новый член бленда: `work/scripts/train_mlpcdf.py` — скелет train_mlpziln.py
(та же загрузка якорей/фич, gap30-протокол, featprep clip99, es-metric raw/cal,
save_preds/log_score/save_meta/save_torch), заменена только голова.

## Конструкция головы
- z = log1p(GMV) дискретизируется в B=64 упорядоченных бакета по квантилям
  ОБУЧАЮЩЕГО распределения; нулевой бакет отдельный: e_0=0, e_1..e_63 —
  квантили j/63 положительных z трейна (тай-эджи дедупятся, K логируется).
- Голова: один Linear c K=63 выходами — логиты survival s_k = P(z > e_k);
  s_0 — это ровно zero-гейт P(y>0) (в трейне pos_rate ~0.54, у val-таргета ~46% нулей).
- Лосс: BCE по всем K сигмоидам, сглаживание меток eps=0.05
  (t -> t*0.95 + 0.025), среднее по строкам и порогам.
- Декод: E[z] = Σ_k (e_{k+1}-e_k) * s_k — сумма ширин бакетов, взвешенных
  кумулятивными вероятностями (тождество E[z] = ∫ P(z>t)dt). Сигмоиды
  независимы и монотонность не гарантируют, поэтому перед декодом s
  прогоняется через cummin (валидная survival-кривая). pred = expm1(E[z]).
- `--desmooth` (выкл. по умолчанию) обращает сглаживание на декоде
  (s-eps/2)/(1-eps): смещение уровня от сглаживания и так съедает binned
  log-shift калибровка, через которую проходит каждый член бленда.
- В stats npz дополнительно: cdf_edges + перцентили (s_0, E[z]) по сидам.

## Смоук (прогнан руками, лёгкий)
```
USE_V2=1 USE_V3=1 USE_V4=1 .venv/bin/python work/scripts/train_mlpcdf.py \
  --name cdf_smoke --smoke --n-anchors 1 --threads 2
```
[SMOKE] val_rmsle=1.719445, 7s (1 якорь, 200 шагов, batch 2048, mps);
63 порога, e1=1.740 e_med=4.362 e_max=10.067; s0_mean=0.564 при pos_rate 0.570.
mono_viol=1.0 на 200 шагах — ожидаемо для независимых сигмоид, декод
монотонен всегда (cummin).

## Полный прогон — очередь 991_mlpcdf
```
.venv/bin/python work/scripts/train_mlpcdf.py --name mlpcdf \
  --n-anchors 14 --gap-days 30 --seeds 42 --epochs 40 --batch 8192 --lr 1e-3
env: USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=4
```
Зеркало исторической команды mlpziln (260_mlpziln_full), но один сид 42.
Статус: очередь подхватила в 20:38:50 сразу после kostya46_s3.

РЕЗУЛЬТАТ 991: exit=0, 59s. Ранний стоп на ep1 (raw val ep1 1.70818, ep2-5 хуже,
patience 4), val_rmsle=1.708175, ep=[1]. BCE при этом ПАДАЛ все 5 эпох
(0.40892 -> 0.40185): raw-критерий уткнулся в дрейф УРОВНЯ от label smoothing
(при заострении сигмоид смещение декода растёт), а ранжирование ещё училось.

## Замер 991 (эталон: blend колонки work/preds_pack = 1.665647)
- margin.py mlpcdf: скор_cal 1.671912 (сырой был 1.708 — калибровка сняла
  уровень, диагноз подтверждён), корр 0.99710, ЗАПАС −0.00085 -> шум.
- joint_gain --each mlpcdf: выигрыш −0.000004, вес 0.000 -> шум.
Член на чекпойнте ep1 бленду не нужен.

## Вторая попытка — очередь 992_mlpcdf_esc (--es-metric cal)
Ровно под эту патологию в скелете есть --es-metric cal: ранняя остановка по
честному калиброванному val RMSLE (2-fold по юзерам, calibrate.py-сдвиги);
уровень, который перезапишет калибровка, перестаёт решать судьбу чекпойнта.
```
.venv/bin/python work/scripts/train_mlpcdf.py --name mlpcdf_esc \
  --n-anchors 14 --gap-days 30 --seeds 42 --epochs 40 --batch 8192 \
  --lr 1e-3 --es-metric cal
```
РЕЗУЛЬТАТ 992: (см. work/reports/job_mlpcdf_esc.log)

## Замер
```
.venv/bin/python work/scripts/margin.py mlpcdf_esc
.venv/bin/python work/scripts/joint_gain.py --each mlpcdf_esc
```
