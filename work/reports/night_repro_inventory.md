# Ночная ревизия: инвентарь воспроизводимости действующего бленда (21.08, ~04:00)

Действующий бленд = `work/reports/blend_reopt.json`, ключ `winner` (lib B_plus_cal,
nnls_free, val **1.665647**, OOF 1.665764). 14 членов, Σ|w| = **1.006117** (nnls без
ограничения суммы). Эталон подтверждён замером: колонка `blend` из
`work/preds_pack/val_preds.parquet` даёт RMSLE **1.665647** против target.

**Алгебра сходится побитово (в пределах float32):** Σ wᵢ·log1p(членᵢ) по колонкам пака
воспроизводит колонку `blend` с max|Δ| = 3.83e-07 (val) и 3.89e-07 (test), RMSLE
пересборки 1.665647. Пак пересобран этой ночью в 00:43 (после регенерации lagd28) и
**закоммичен** (pack-new, ветка sasha, дерево чистое) — то есть весь действующий бленд,
включая калиброванные версии всех 14 членов на val И test, лежит в git.

## Инвентарь по членам

Обозначения: (a) откуда val/test parquet; (b) веса/бустеры в `work/models` (сам каталог
в .gitignore — всё локально); (c) чем обучается; (d) вердикт.
«ночь» = закрывается ночной очередью 980–985 (сейчас в pending; 979 hazard_v1 работает с 01:08).

| # | член | вес | (a) preds | (b) веса в work/models | (c) обучение (команда) | (d) вердикт |
|---|------|-----|-----------|------------------------|------------------------|-------------|
| 2 | fusion_v3c_avg_cal | 0.228925 | базы fusion_v3c{42,555,7} только work/preds (локально); avg и _cal колонки в паке (git) | только *_stats.npz, .pt НЕТ; fusion_v3c_avg_cal.npz есть | done 833/832/835: `train_fusion3.py --name fusion_v3cXX --final --epochs 3 --batch 2048 --eval-batch 1024 --lr 1e-3 --seeds XX --threads 4 --eval-every 492 --n-ch 12 --es-metric cal` (1050/952/918s) → merge_seeds.py → calibrate.py | прогноз-артефакт (torch/MPS ретрейн не побитовый) — **закрывается этой ночью** (981–983 с save_torch p1/p2 + 985 merge+recal) |
| 3 | gseq_small_s42_cal | 0.108701 | **git**: work/colab/out/gseq_small_s42_{val,test}.parquet + json конфига | нет; gseq_small_s42_cal.npz есть | work/colab/gpu_seq.py (git) на Colab GPU; cfg L=112,d=96,2 слоя,4 головы,ff=192, lr 3e-4, 3 эпохи, 4.9 мин GPU; cal_rmsle 1.698129 | **только предикт-артефакт в git** (нужен внешний GPU; веса не сохранялись) |
| 4 | fusion_v3ctl_cal | 0.106124 | базы только work/preds; _cal в паке (git) | только fusion_v3ctl_stats.npz; fusion_v3ctl_cal.npz есть | done 851: `train_fusion3.py --name fusion_v3ctl --final --epochs 3 --batch 2048 --eval-batch 1024 --lr 1e-3 --seeds 42 --threads 4 --eval-every 984 --n-ch 8` (875s) | прогноз-артефакт — **закрывается этой ночью** (980 + 985) |
| 5 | wklin (сырой) | 0.070987 | work/preds локально; колонка в паке (git) | нет (train_wklin без model_io; ridge) | `train_wklin.py --name wklin --emit-tier` (91s, USE_V2/3/4; один запуск пишет wklin_base+wklin+wklin_wk) | **воспроизводим из чистого клона**: ridge детерминирован, сида нет; ночью 984 перегенерирует протокольно |
| 6 | weak_an_d_cal | 0.045493 | work/preds локально; _cal в паке (git) | **weak_an_d.txt (9.2MB) + meta + cal.npz — есть** | done 9035: `train_weak.py --name weak_an_d --mech anchors --k-anchors 4 --sel-seed 77 --anchor-pool 0 --model lgb --objective log_mse --params '{tweedie 1.45, n_est 6000}'` (123s) + calibrate | **воспроизводим из чистого клона** (плюс веса на диске) |
| 7 | weak_ft_recency_cal | 0.043429 | work/preds локально; _cal в паке (git) | **weak_ft_recency.txt + meta + cal.npz — есть** | done 902: `train_weak.py --mech ftype --ftype recency --n-anchors 14 ...` (~190s) + calibrate | **воспроизводим из чистого клона** |
| 8 | behavonly_avg_cal | 0.041373 | базы behavonly{,_s1337,_s7} в work/preds; avg_cal в паке (git) | s1337.txt и s7.txt есть; **бустер сида 42 отсутствует** (обучен до model_io); avg_cal.npz есть | done 260/776/777: `train_behavonly.py` (сид 42 с `--n-anchors 14`, 214s; s1337/s7 без флага = 27 якорей, 427/442s) → merge_seeds → calibrate | **воспроизводим из чистого клона** (LGB детерминирован; дешёвый ретрейн сида 42) |
| 9 | lagd28 (сырой) | 0.035049 | work/preds, перегенерирован этой ночью (00:28/00:38, job 978); обе колонки в паке (git) | весов нет ПО ПОСТРОЕНИЮ (lag-TTA: обучение при предикте) | done 978: `lag_tta.py --prefix lagd --lags 0,14,28,42,56,70 --test --seed 42` (787s); тест отдельно: 977 `--test-only 61` (171s) | **воспроизводим из чистого клона** (детерминированный LGB; ночная регенерация это подтвердила) |
| 10 | c_ts2_s42_cal | 0.033278 | базы в work/preds; _cal в паке (git) | бустеров нет (до model_io); **c_ts2_s42_cal.npz — единственная отсутствующая cal-таблица** (видно в --stage check) | done 130: `train_gbdt.py --name c_ts2_s42 --model lgb --objective two_stage --n-anchors 14 --seed 42 --gap-days 30 --params '{leaves 127/255,...}'` (303s) + `calibrate.py --pred c_ts2_s42` | **воспроизводим из чистого клона** (ретрейн теперь сохранит __stage1/__stage2 бустеры) |
| 11 | gseq_big_s42_cal | 0.024431 | **git**: work/colab/out/gseq_big_s42_{val,test}.parquet + json | нет; cal.npz есть | gpu_seq.py arm=big (L=364,d=256,6 слоёв); **обучение оборвано на step 4500/11736 (done:false), взят ckpt-avg 1.677205** | **только предикт-артефакт в git**: точный повтор невозможен даже на GPU (обрыв не воспроизводится) |
| 12 | fusion_f_cal | 0.010602 | базы в work/preds; _cal в паке (git) | только fusion_f_stats.npz; cal.npz есть | done 287: `train_fusion.py --name fusion_f --final --epochs 3 --batch 2048 --eval-batch 1024 --lr 1e-3 --seeds 42 --threads 5` (1844s, USE_V4) | прогноз-артефакт (torch) — **в ночной очереди НЕ стоит**, единственная torch-дыра, которая ночью не закроется |
| 13 | febspec2_cal | 0.008916 | базы в work/preds; _cal в паке (git) | бустер не сохраняется (без model_io); febspec2_cal.npz есть | done 623: `train_febspec2.py --name febspec2 --config auto --cohort 0.20 --threads 3` (429s; свой короткоисторический набор признаков собирает сам) | **воспроизводим из чистого клона** |
| 14 | wklin_wk_cal | 0.002788 | work/preds; _cal в паке (git) | нет; wklin_wk_cal.npz есть | тот же запуск train_wklin.py, что и wklin (отдельно не получается) | **воспроизводим из чистого клона**; ночью 984+985 |

## Суммарные веса по категориям (Σ|w| = 1.006117)

| категория | члены | Σ вес | доля |
|---|---|---|---|
| воспроизводим из чистого клона (детерминированный ретрейн, скрипт+команда в git) | wklin, weak_an_d_cal, weak_ft_recency_cal, behavonly_avg_cal, lagd28, c_ts2_s42_cal, febspec2_cal, wklin_wk_cal | **0.281313** | 28.0% |
| только предикт-артефакт в git — сегодня | kostya46_cal, gseq_small_s42_cal, gseq_big_s42_cal (внешние, 0.379153) + fusion_v3c_avg_cal, fusion_v3ctl_cal, fusion_f_cal (torch без весов, 0.345651) | **0.724804** | 72.0% |
| — из них закрывается этой ночью (980–983 save_torch + 985 recal) | fusion_v3c_avg_cal + fusion_v3ctl_cal | **0.335049** | 33.3% |
| — останется предикт-артефактом после ночи | kostya46_cal, gseq_small/big, fusion_f_cal | **0.389755** | 38.7% |
| невоспроизводим вовсе | — | **0.000000** | 0% |

Нижняя строка — главный итог: невоспроизводимых членов НЕТ. Все 14 (включая
калиброванные версии) лежат колонками в git-паке `work/preds_pack/{val,test}_preds.parquet`,
и бленд собирается из них с точностью 4e-07 без единого локального файла.

Разрез «есть ли веса модели на диске сейчас»: полные веса есть только у
weak_an_d + weak_ft_recency (Σ 0.0889) и 2/3 сидов behavonly; у 0.72 веса бленда
модельных весов нет нигде (у torch появятся утром). `work/models/` в .gitignore —
даже сохранённые веса живут только на этой машине.

## inference.py --stage check (запущен: лёгкий, exit=1 из-за нехваток; полный вывод в scratchpad)

- `--verify-blend` перечисляет расхождение с действующим отчётом: **12 членов только в
  пакете, 6 только в отчёте** (kostya46_cal 0.246, gseq_small 0.109, gseq_big 0.024,
  lagd28 0.035, fusion_f_cal 0.011, wklin_wk_cal 0.003), **8 весов разошлись**.
- По старому пакетному бленду: готово 19/25 базовых (вклад 0.752 из 1.005), не хватает
  11 артефактов, переобучение 5.8 ч на месте / 7.4 ч на чистой машине; seq2-тензоры
  (11 ГБ, для seq2tr_f) удалены; отсутствуют c_ts2_s42_cal.npz, seq2tr_f_cal.npz,
  hmmsim_cal.npz, twl_v7_cal.npz.

## Что произойдёт этой ночью (важно для эталона)

980–984 ПЕРЕЗАПИШУТ work/preds/fusion_v3ctl*, fusion_v3c{42,555,7}*, wklin* новой
реализацией (у torch — не побитовой), 985 пересоберёт avg и калибровки. Эталон
1.665647 после этого сместится: колонки пака и work/preds разойдутся до пересборки
пака. Старая реализация останется только в закоммиченном паке.

## Рекомендации (по убыванию срочности)

1. **Утром после 985**: пересобрать пак (`build_preds_pack.py`), перезапустить
   `blend_reopt.py --save`, зафиксировать новый эталон в scores.tsv; до этого никакие
   замеры margin/joint_gain со старым паком не смешивать с новыми preds.
2. **Синхронизировать inference.py с действующим блендом**: перенести 14-членный
   winner в BLEND_WEIGHTS/MEMBER_PARTS/BASES (появятся kostya46_cal, gseq_*, lagd28,
   fusion_f_cal; у kostya46/gseq вход — предикт-артефакты из git, их надо описать как
   persist="preds" с путями work_kostya/preds и work/colab/out). Пока это не сделано,
   пакет воспроизводит старый 20-членный бленд, а не действующий.
3. **Последняя torch-дыра**: поставить в очередь ретрейн fusion_f с save_torch
   (1844s, вес 0.0106) — после ночи это единственный член, чью точную реализацию
   нельзя восстановить локально из весов.
4. **Дешёвые закрытия** (в дневную очередь, суммарно ~10 мин): c_ts2_s42 ретрейн
   (303s, сохранит бустеры + восстановит отсутствующий c_ts2_s42_cal.npz),
   behavonly сид 42 (214s, сохранит бустер).
5. **Внешние члены (0.379 веса)**: попросить Костю сохранить бустеры kostya46 (или
   зафиксировать окружение), а у gseq принять как данность — gseq_big принципиально
   неповторим (оборванное обучение), защита — только parquet в git (уже есть).
6. **Упаковка**: work/models в .gitignore; перед отгрузкой веса копируются в
   final_submission/models (сейчас там только chain_test.npz) — после ночных
   ретрейнов включить копи-шаг в финализацию.

— Источники: blend_reopt.json (winner), work/queue/{979–985}.json + done/{130,260,287,
623,776,777,832,833,835,851,902,978,9035}.json, scores.tsv, model_io.py (контракт
имён), inference.py BASES/MEMBER_PARTS, reproduce_training.md, work_kostya/README.md,
work/colab/out/*.json. Замеры: .venv/bin/python, полный вывод check —
scratchpad/check_out.txt.
