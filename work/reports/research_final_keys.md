# Research: последние ключи к <1.645 (веб-скан, 2026-08-18)

Zero-CPU research по пяти вопросам. Учтено, что уже есть в стеке и в `external_tricks.md` (#1 multi-snapshot — сделано, 14 якорей; #2 DART; #3 whale — ИЗМЕРЕНО +0.0001, тупик; #4 бины — сделано; #10 дистилляция; #13 count×AOV; #14 сегментные модели), а также замеренные факты: Ф19 (оракул zero/nonzero стоит 0.448 RMSLE — главный резерв), Ф7 (recency заказа — сильнейший сепаратор), Ф20 (ошибки сильных моделей r=0.995 — нужны новые ИСТОЧНИКИ информации, не новые головы), Ф22 (потолок посегментной калибровки +0.0011), Ф23 (adversarial AUC=1.0, дрейф структурный).

---

## Q1. Intermittent demand (Croston / SBA / TSB / ADIDA / iMAPA)

**Что говорит литература.**
- Прямое применение классики как самостоятельных прогнозов — проигрывает GBDT: в M5 весь топ-50 — LightGBM, победитель — 220 LGBM c tweedie-NLL и усреднением recursive+direct по уровням пулинга ([M5 accuracy: results & findings](https://www.sciencedirect.com/science/article/pii/S0169207021001874), [разбор Artefact](https://www.artefact.com/blog/sales-forecasting-in-retail-what-we-learned-from-the-m5-competition-published-in-medium-tech-blog/)). Croston/SBA/TSB/ADIDA/iMAPA доступны в [Nixtla statsforecast](https://github.com/Nixtla/statsforecast) — считаются за минуты на 250k рядов.
- **Ключевой свежий результат — классика как ПРИЗНАКИ, а не как прогноз**: статья "Primacy of feature engineering over architectural complexity for intermittent demand" ([PMC12873174](https://pmc.ncbi.nlm.nih.gov/articles/PMC12873174/), Sci Rep 2026) — фреймворк SHOS: (a) экспоненциально сглаженная вероятность спроса p̂ (это ровно TSB-компонента), (b) сглаженный условный размер спроса ẑ (Croston-компонента), (c) их произведение как point-forecast — всё три ПОДАЮТСЯ КАК ФИЧИ в одностадийный LightGBM. Результат: MAE −50%, RMSE −28% vs тот же LGBM без этих фич (Wilcoxon p<0.001); одностадийный LGBM+SHOS-фичи ОБОШЁЛ двухстадийный hurdle. Масштаб эффекта на их данных не переносится 1:1, но направление и механика — прямо наш случай (46% нулей).
- TSB (Teunter–Syntetos–Babai) специально построен под «устаревание» покупателя: вероятность спроса экспоненциально затухает с каждым днём без покупки — это непрерывно-затухающая версия нашего recency, самого сильного сепаратора (Ф7).
- Двухстадийные ML-схемы occurrence×size для intermittent подтверждаются и в [двухэтапном подходе для fashion retail](https://doi.org/10.3390/forecast8040056) и [обзоре lumpy/intermittent](https://arxiv.org/pdf/2103.13812), но SHOS-вывод: сила — в признаках, не в стадийности.

**Применимость к нам.** У нас есть окна и `ord_gap_mean/std` (burstiness), но НЕТ: (1) экспоненциально сглаженных p̂_α и ẑ_α на дату якоря при нескольких α (0.02/0.05/0.1/0.3) — в отличие от окон, это бесконечная память с экспоненциальным ядром; (2) **hazard/renewal-фич**: `overdue_ratio = days_since_last_order / ADI_user`, и эмпирическая вероятность «покупка в ближайшие 30 дней при условии паузы d дней» из личного (и посегментного) распределения межпокупочных интервалов: `F(d+30)−F(d) / (1−F(d))`. Это перевод recency из «сырых дней» в «дни, нормированные на собственный ритм юзера» — целится ровно в Ф19/Ф7. (3) SB-классификация квадрантов (ADI×CV²: smooth/intermittent/erratic/lumpy) как категориальная фича и ось для калибровки.

**Вердикт: самый сильный кандидат.** Дёшево (один проход по дневным логам), новый источник информации для margin zero/nonzero.

## Q2. Декомпозиции таргета в победных решениях

**Свидетельства** (основной источник — [Bojer & Meldgaard, "Kaggle forecasting competitions: an overlooked learning opportunity"](https://arxiv.org/pdf/2009.07701), разбор 6 соревнований):
- **Favorita 1st place: одна модель на каждый под-горизонт** (16 LGBM + 16 FFNN по дням горизонта + всегоризонтный LGBM + CNN). Авторы обзора: «инновация решения — обучение одной модели на горизонт»; но в Recruit не все в топе использовали horizon-specific, «выигрыш может быть несущественным относительно роста числа моделей». Компромисс 11-го места Recruit: по модели на НЕДЕЛЮ (6 моделей).
- **Walmart Store Sales winner: SVD-денойзинг** матрицы рядов по департаментам → прогноз по очищенным рядам; «одной модели SVD+STL+ETS хватило бы для победы». Прямой прецедент коллаборативного сигнала (см. Q4).
- **Rossmann winner**: XGB + ridge-поправка тренда; ансамбль XGB на разных подвыборках/фичах давал ~5% к лучшей одиночной; фичи-агрегаты на нескольких уровнях иерархии.
- **Частота×размер vs прямой tweedie** (актуарная литература): [сравнение GLM freq/sev vs pure premium](https://resolve.cambridge.org/core/services/aop-cambridge-core/content/view/C79A0FAF521D324251AE2876D3B73492/9781139342681c2_p39-59_CBO.pdf/applying-generalized-linear-models-to-insurance-data-frequencyseverity-versus-pure-premium-modeling.pdf) — «явного победителя нет»; [ML-декомпозиция 2026](https://doi.org/10.3390/math14101640): XGB-частота × XGB-тяжесть дала наименьшую ошибку среди рассмотренных. Т.е. count×AOV (наш #13) — законный, но не гарантированный кандидат.
- **Elo 1st place** (зеркальная задача: таргет с ~1% выбросов −33): линейный стэкинг «классификатор выброса + регрессор без выбросов» дал **+0.015 CV** ([разборы](https://medium.com/codex/elo-merchant-category-recommendation-understand-customer-loyalty-f952438e6d17)). У нас whale-гейт уже измерен (+0.0001): наша версия была гейтом поверх base, у Elo — линейный стэкинг p как признака мета-модели; но при r=0.995 ошибок ждать большого выигрыша не стоит.

**Применимость.** Два варианта декомпозиции, которых НЕТ в нашем списке:
1. **По каналам: gmv = gmv_search + gmv_cat точно (Ф2)** — две отдельные модели (или две головы) на log1p(gmv_search_30d) и log1p(gmv_cat_30d), финал = log1p(expm1(ŷ_s)+expm1(ŷ_c)) с поправкой Йенсена, откалиброванной на валидации. Прецеденты: Enefit (раздельные consumption/production), актуарная декомпозиция. Search-канал доминирует в MSE (91%, Ф21) — специализированная search-модель может чуть выиграть, а cat-модель дёшева.
2. **Понедельная декомпозиция горизонта**: 4 головы (w1..w4+) с суммированием в линейном пространстве (Favorita-стиль, компромисс Recruit-11th). Даёт таргет-сайд разнообразие, которого в нашем переборе «непохожих моделей» не было (там была модельная, не таргетная новизна).

## Q3. Test-time adaptation / pseudo-labeling для таблиц

- **AmEx**: задокументировано — [Deotte, 15th: Transformer + knowledge distillation от LightGBM с pseudo-labels на тесте](https://realvincentyuan.github.io/Spacecraft/amex-default-prediction-kaggle-competition-summary/index.html); дистилляция закрывала разрыв NN↔GBDT и поднимала бленд. Выигрыши в топах — порядка тысячных долей метрики, не перевороты.
- [NVIDIA Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/): pseudo-labeling — стандартный приём (soft labels = регуляризация); там же: seed-ансамбли и дообучение на 100% данных давали +0.003 MAP@3 — т.е. типичный масштаб.
- **Self-training под covariate shift**: [Revisiting Self-Training with Regularized Pseudo-Labeling for Tabular Data](https://arxiv.org/pdf/2302.14013) — работает только с регуляризацией (curriculum/уверенность), иначе усиливает собственные ошибки (confirmation bias). Риск для нас: soft-таргеты наследуют наш замеренный сдвиг (тест недопредсказан всеми файлами, Ф17) — сдвиг надо чинить ДО дистилляции (LB-алгеброй), иначе студент его зацементирует.
- **Importance weighting под тестовое распределение — для нас мёртв**: adversarial AUC=1.0 структурный (Ф23) → веса вырождаются. Живой вариант ровно один: **дистилляция бленда на ФИЧИ теста** (студент видит тестовые x с soft-таргетами ансамбля) — это не reweighting и от AUC=1.0 не страдает.
- Jane Street RTMDF 2025: топы адаптировались онлайн (дообучение на раскрываемых лагах в ходе теста) — [walkthrough](https://www.youtube.com/watch?v=lfzzPZZyzjE); у нас тест одномоментный, онлайновая часть неприменима, переносится только идея «студент видит тестовые фичи».

**Вердикт:** ждать больше ~0.001 не стоит; делать на CUDA-машине в конце, после фиксации сдвига.

## Q4. Коллаборативные / непараметрические сигналы

- **SVD/матричные разложения — есть победный прецедент**: Walmart Store Sales winner использовал truncated SVD по матрице рядов как ДЕНОЙЗЕР перед прогнозом ([Bojer & Meldgaard](https://arxiv.org/pdf/2009.07701)). Для нас: SVD user×week (250k×58, log1p) → 8–16 латентных факторов юзера как фичи + статистики по реконструированной (очищенной) траектории (тренд/уровень последних недель без шума).
- **kNN как мета-фича — есть победный прецедент**: Otto 1st place (Titericz & Semenov) держал kNN-модели в 1-м уровне стэка; «среди лучших моделей — XGB и kNN», слабые пооодиночке kNN-мета-фичи давали вклад ([writeup](https://www.kaggle.com/competitions/otto-group-product-classification-challenge/writeups/gilberto-titericz-stanislav-semenov-1st-place-winn)). Перенос на нас — **«будущее похожих юзеров»**: для каждой строки (user, anchor) найти k соседей среди СТАРЫХ снапшотов (по 16–32 SVD-компонентам или топ-фичам, faiss), фича = mean/median реализованного forward-30d log-GMV соседей (+ доля нулей среди соседей). Темпорально чисто: будущее соседей уже в прошлом; для train-строк — соседи только из более ранних якорей (OOF по времени). Это непараметрическая оценка E[log1p|x] с иной indutive bias, чем деревья.
- Для LTV напрямую: embeddings полезны как фичи ([Shaped: user/item embeddings для churn/LTV](https://www.shaped.ai/blog/peering-inside-the-black-box-leveraging-user-item-embeddings)), но контролируемых сравнений «GBDT+RFM vs GBDT+RFM+MF» в соревновательных писаниях мало — уверенность средняя.
- **BTYD (BG/NBD, Pareto/NBD)**: P(alive) и E[транзакций за 30д] из R/F/T считаются за минуты ([Fader/Hardie BG/NBD](https://medium.com/geekculture/predicting-customer-life-time-value-cltv-via-beta-geometric-negative-binominal-distribution-59be07ac30bd), [CLVTools](https://www.clvtools.com/reference/pnbd.html)); ML в среднем сильнее BTYD ([ResearchGate: CLV modelling with GB](https://www.researchgate.net/publication/392531005_Customer_Lifetime_Value_Modelling_with_Gradient_Boosting)), но их выход — параметрическая аппроксимация того же hazard, что и Q1; если делать TSB/renewal-фичи, BG/NBD добавит мало. Опция «если останется время».

## Q5. Калибровка P(buy) при zero-inflation

- **Математика метрики**: RMSLE-оптимум = E[log1p(Y)|x] = p(x)·E[log1p(Y)|Y>0,x]. P(buy) входит ЛИНЕЙНО в лог-пространстве: относительная ошибка p умножает предикт 1:1, и качество p напрямую масштабирует RMSLE (у бинов/hurdle это уже заложено). Оракул zero/nonzero = 0.448 (Ф19) — резерв в дискриминации p огромен, но добывается ПРИЗНАКАМИ (Q1), а не калибровкой; потолок посегментной калибровки уже замерен: +0.0011 (Ф22).
- **Чем калибровать**, если калибровать: [beta calibration (Kull et al., AISTATS 2017)](https://proceedings.mlr.press/v54/kull17a.html) — 3 параметра, устойчива там, где isotonic переобучается на малых сегментах; [Venn-Abers](https://valeman.medium.com/how-to-calibrate-your-classifier-in-an-intelligent-way-a996a2faf718) — в эмпирических исследованиях ([RF-калибровка на 22 датасетах](https://link.springer.com/article/10.1007/s10994-018-5753-x), [SDM'19](https://epubs.siam.org/doi/abs/10.1137/1.9781611975673.4)) стабильно бьёт Platt и isotonic. Правильная цель — не ECE, а финальный RMSLE: 2-параметрическая поправка логита p (сдвиг+температура), подобранная по нашей LB-алгебре (Ф24) — это структурированная версия глобального сдвига, отдельно для p-головы.
- **Focal loss — не рекомендуется**: он classification-calibrated, но НЕ strictly proper — даёт и недо-, и переуверенность ([NeurIPS'20](https://papers.neurips.cc/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf), [теория](https://arxiv.org/pdf/2011.09172)); польза показана для сильного дисбаланса в DNN, наш 46/54 — не тот случай. Если пробовать — только с обязательной пост-калибровкой, ожидание ~0.
- Google ZILN ([arXiv:1912.07753](https://arxiv.org/abs/1912.07753)): для оценки p рекомендует decile-диагностику калибровки — дешёвый чек, есть ли вообще что чинить в p-голове.

## Бонус: нестандартные ходы 2023–2026 вне нашего списка

1. **SVD-денойзинг траекторий** (победный механизм Walmart-2014, забытый приём) — см. Q4.
2. **kNN-«будущее соседей» как фича** (механизм Otto-1st в темпоральной версии) — см. Q4.
3. **Renewal-hazard фичи + TSB/SHOS** (Sci Rep 2026) — см. Q1; в kaggle-практике почти не встречается, в статьях — большой эффект.
4. **Канальная декомпозиция gmv_search+gmv_cat** — использует ТОЧНОЕ тождество данных (Ф2), аналог Enefit consumption/production. См. Q2.
5. TabPFN-2.5 сабсэмпл-ансамблем как «чужая» модель ([обзор](https://arxiv.org/html/2502.17361v1)): при r=0.995 ошибок и вердикте «непохожие слабые не дают вклада» — низкий приоритет, только как OOF-фича, не как член бленда.
6. M5-uncertainty инсайт ([results & findings](https://www.sciencedirect.com/science/article/pii/S0169207021001722)): суммирование гранулярных квантилей лучше прямых квантилей уровня — намёк, что наши понедельные/канальные головы стоит суммировать в линейном пространстве, а не блендить в логе.

---

## Топ-7 действий (выигрыш × вероятность / часы)

| # | Действие | Ожидание ΔRMSLE | P(успех) | Часы | Где |
|---|---|---|---|---|---|
| 1 | TSB/SHOS-фичи: p̂_α, ẑ_α (α=0.02/0.05/0.1/0.3), p̂·ẑ, overdue_ratio=days_since/ADI, renewal-P(order≤30d|пауза d) из личного+сегментного распределения интервалов, SB-квадрант → ретрейн чемпиона LGB + hurdle-MLP | −0.001..−0.003 | 0.5 | 6 | M1 (фичи CPU-лёгкие, 1 тяжёлый ретрейн в очередь) |
| 5 | Понедельные головы w1..w4+ с суммой в линейном пространстве (Favorita-механика) | −0.0005..−0.0015 | 0.35 | 8 | CUDA |
| 6 | P(buy)-тюнинг: beta-calibration/Venn-Abers p-головы + 2-параметрическая логит-поправка (сдвиг+температура) под финальный RMSLE через LB-алгебру; сначала decile-чек ZILN — есть ли вообще смещение | −0.0003..−0.0012 | 0.5 | 2 | M1 (CPU) |
| 7 | Дистилляция бленда на тестовые фичи (наш #10, с поправкой: сначала снять замеренный глобальный сдвиг, потом soft-targets; студент LGB+MLP; регуляризация по 2302.14013) | −0.0003..−0.0015 | 0.3 | 6 | CUDA |

Суммарно реалистичный сценарий: #1+#2+#6 на M1 (одна очередь тяжёлых ретрейнов) + #3/#4 на CUDA — матожидание ≈ −0.002..−0.004, что закрывает разрыв 1.6492→<1.645 при удаче на двух пунктах.

**Анти-выводы (не тратить время):** importance weighting/adversarial-перевзвешивание (Ф23, AUC=1.0 структурный); focal loss для p (не proper, наш дисбаланс мал); ещё сегментные калибровки выходов (потолок +0.0замерен); whale-гейт повторно (замерен +0.0001); чистые Croston/SBA как самостоятельные прогнозы (M5: проигрывают LGBM-tweedie).

## JSON

```json
{"top7":[
{"action":"TSB/SHOS features: exp-smoothed P(demand) & demand size (alpha 0.02-0.3), their product, overdue_ratio=days_since_last/ADI, renewal P(order in 30d | pause d) from personal+segment interval distribution, SB-quadrant; retrain champion LGB + hurdle-MLP","expected":"-0.001..-0.003","hours":6.0,"owner_hint":"FILE.csv queue (features cheap, 1 heavy retrain)"},
{"action":"Truncated SVD on user x week log1p matrix (8-16 factors) + denoised-trajectory stats as features; piggyback on retrain #1","expected":"-0.0005..-0.002","hours":4.0,"owner_hint":"FILE.csv"},
{"action":"Channel decomposition: separate heads for gmv_search and gmv_cat (identity gmv=search+cat), sum in linear space with Jensen correction tuned on val + LB algebra","expected":"-0.0005..-0.002","hours":5.0,"owner_hint":"CUDA or FILE.csv queue"},
{"action":"kNN 'neighbors' future' feature: faiss over 16-32 SVD dims, k=64/256, mean forward-30d log-GMV and zero-share of neighbors from strictly earlier anchors (time-OOF)","expected":"-0.001..-0.0025","hours":8.0,"owner_hint":"CUDA (faiss)"},
{"action":"Weekly sub-horizon heads w1..w4+ summed in linear space (Favorita per-horizon mechanics)","expected":"-0.0005..-0.0015","hours":8.0,"owner_hint":"CUDA"},
{"action":"P(buy) tuning: ZILN decile check, then beta-calibration/Venn-Abers of p-head + 2-param logit shift+temperature optimized for final RMSLE via LB algebra","expected":"-0.0003..-0.0012","hours":2.0,"owner_hint":"FILE.csv CPU"},
{"action":"Distill blend onto test features (soft targets AFTER removing measured global shift; LGB+MLP students; regularized per arXiv:2302.14013)","expected":"-0.0003..-0.0015","hours":6.0,"owner_hint":"CUDA, do last"}
],
"eureka":"Ф19: весь резерв — в margin zero/nonzero, а сильнейший сепаратор — recency (Ф7). Единственный незадействованный чистый апгрейд этого сепаратора — renewal/hazard-признаки: перевести days_since_last в 'дни, нормированные на личный ритм' (overdue_ratio=days_since/ADI) и в вероятность P(заказ в 30д | пауза d) из эмпирического распределения межпокупочных интервалов юзера/сегмента, плюс TSB-затухающая p̂ и Croston-ẑ как фичи (SHOS, Sci Rep 2026: одностадийный LGBM с такими фичами бьёт two-stage hurdle). В лог-пространстве p входит в предикт линейно, так что выигрыш в p переносится в RMSLE 1:1."}
```
