# Kaggle-сессия: tfm3b (GPU0) + событийный kevf (GPU1) за один заход

Цель: снять два GPU-долга команды одной сессией T4×2 (~6 ч из 9 доступных):
- **tfm3b** — честный ретрейн трансформера Кости М. (обучение по 378, тест без
  застоялости; наша главная известная ось, ожидание −0.0001..−0.0002 на LB);
- **kevf** — событийный трансформер (наше неиспробованное представление, только
  смоук был; потолок неизвестен).

## Подготовка (один раз)

1. `bash work/kaggle/make_bundle.sh` → `/tmp/tfm3b_bundle.zip`; залить его как
   приватный Kaggle Dataset **ozon-code**.
2. Датасет с данными (как в work/kaggle/README.md): `train.parquet` +
   `sample_submit.csv` → приватный Dataset **ozon-ecup-t3** (если уже есть — ок).
3. Новый Notebook → Accelerator **GPU T4 x2** → Add Input: оба датасета.
   Internet: можно Off (pip не нужен, если polars есть; если нет — On для ячейки 1).

## Ячейка 1 — код и раскладка

```python
import shutil, subprocess, sys, glob, os
# распаковка бандла с сохранением путей work/scripts/seq/...
subprocess.run(["unzip", "-oq", glob.glob("/kaggle/input/*/tfm3b_bundle.zip")[0],
                "-d", "/kaggle/working"], check=True)
os.chdir("/kaggle/working")
os.makedirs("work/preds", exist_ok=True)
try:
    import polars  # build_tensor/avg_seeds
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "polars"], check=True)
data = glob.glob("/kaggle/input/*/train.parquet")[0]
print("данные:", data)
```

## Ячейка 2 — тензор (~5-10 мин, дождаться конца)

```python
import subprocess, sys
subprocess.run([sys.executable, "work/scripts/seq/build_tensor.py",
                "--src", data, "--out", "tensor"], check=True)
subprocess.run([sys.executable, "work/scripts/seq/make_valid3.py",
                "--data", "tensor"], check=True)
print("тензор готов")
```

## Ячейка 3 — оба прогона параллельно (Popen, `!cmd &` на Kaggle запрещён)

```python
import shlex, subprocess, os, sys

def launch(cmd, log, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
    return subprocess.Popen(shlex.split(cmd), stdout=open(log, "w"),
                            stderr=subprocess.STDOUT, env=env, start_new_session=True)

# GPU0: фазы A+B, три сида, все фиксы уже внутри (--minutes 45 => ~4.8 ч суммарно)
p_tfm = launch(f"{sys.executable} work/scripts/seq/run_all.py --data tensor "
               f"--minutes 45 --out to_sasha", "runall.log", "0")
# GPU1: событийный, один сид на сессию (~4-6 ч)
subprocess.run([sys.executable, "work/kaggle/kaggle_seq.py", "build"], check=True)
p_kevf = launch(f"{sys.executable} work/kaggle/kaggle_seq.py train "
                f"--name kevf_s42 --seed 42 --device cuda:0", "kevf.log", "1")
procs = {"tfm3b": p_tfm, "kevf": p_kevf}
print({k: p.pid for k, p in procs.items()})
```

(`--device cuda:0` у kevf верен: внутри процесса с `CUDA_VISIBLE_DEVICES=1`
единственная видимая карта — физическая №1.)

## Ячейка 4 — мониторинг (перезапускать по вкусу)

```python
!tail -n 4 runall.log kevf.log; ls -la work/preds to_sasha out 2>/dev/null | tail -20
```

## Ячейка 5 — сторож + упаковка (обязательна последней при Save & Run All)

```python
import time, subprocess
while any(p.poll() is None for p in procs.values()):
    time.sleep(600)
    subprocess.run(["tail", "-n", "2", "runall.log", "kevf.log"])
# tfm3b: усреднение трёх сидов в пару по контракту хэндоффа
import sys
subprocess.run([sys.executable, "work/scripts/seq/avg_seeds.py", "--out",
                "work/preds/tfm3b_val.parquet"] +
               [f"work/preds/tfm2_s{s}_val.parquet" for s in (1, 2, 3)], check=True)
subprocess.run([sys.executable, "work/scripts/seq/avg_seeds.py", "--out",
                "work/preds/tfm3b_test.parquet"] +
               [f"work/preds/tfm2_s{s}_rt_test.parquet" for s in (1, 2, 3)], check=True)
subprocess.run(["zip", "-r", "results_gpu.zip", "to_sasha", "out",
                "work/preds/tfm3b_val.parquet", "work/preds/tfm3b_test.parquet",
                "runall.log", "kevf.log"], check=True)
print("СКАЧАТЬ: /kaggle/working/results_gpu.zip")
```

## После сессии

Скачанный `results_gpu.zip` → распаковать `tfm3b_{val,test}.parquet` в
`work/handoff/`, kevf-парой (`out/kevf_s42_{val,test}.parquet`) — в `work/colab/out/`,
и сказать агенту: приёмка tfm3b — одна команда
(`.venv/bin/python work/scripts/ingest_tfm3b.py`, ~5-7 мин до готового кандидата),
kevf меряется margin/joint_gain как обычно.

Вторая сессия (при остатке квоты): kevf сиды 1337+7 на двух картах параллельно —
ячейки из work/kaggle/README.md.
