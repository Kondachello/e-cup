"""budget.py — считалка трат попыток. Подставляешь состояние, получаешь следующее действие.

Константы взяты из части A и из замеров команды:
  σ_κ  = sd(e)/sqrt(n·G_mse)  — ЗАКОН из части C: SE замера κ НЕ константа,
                 она определяется валидационным выигрышем самой оси
  доза  w = V/(V+σ_κ²), V=0.042; перелом при G=0.0004: крупнее — верить замеру
  noise = 0.000022 фиктивный выигрыш одного подобранного направления на паблике
  SE_pub / SE_priv = 0.0056 / 0.0028 ; перенос public->private 0.584

Экономика (вывод в zhenya_B_budget.md):
  ценность зонда для ОСИ, ИЗМЕРИМОЙ ЛОКАЛЬНО = w·τ²·G = 0.0116·G,
  а простое применение по приору даёт m²·G = 0.111·G — в десять раз больше.
  => зондировать имеет смысл ТОЛЬКО оси, которых на валидации не существует.

Запуск:
    python work/zhenya_eda/scripts/budget.py --state work/zhenya_eda/state.json
    python work/zhenya_eda/scripts/budget.py --demo
"""
from __future__ import annotations
import argparse, json, sys
from datetime import date, datetime
from pathlib import Path

SD_E = 1.6664
V_PRIOR = 0.0416         # τ² по REML на 15 осях реестра (часть C); τ=0.204
TAU = V_PRIOR ** 0.5


def sigma_kappa(q: float, n_pub: int = 50_000) -> float:
    """ЗАКОН части C, проверен прямым замером (n1_sigma_arbiter, отношение 1.03-1.23):
        σ_κ = F0 / sqrt(n_публики · q),   q = mean_P(h²) из параболы S²=F0²-2bc+b²q
    ВНИМАНИЕ: параболическая σ из kappa_registry (шум_LB·(F0+S)/(2q)) занижена
    в 1.1-11 раз — она масштабируется как 1/q, а истинная как 1/sqrt(q)."""
    import math
    return SD_E / math.sqrt(n_pub * max(q, 1e-12))


def dose(q: float) -> float:
    """сколько верить свежему замеру κ: w = τ²/(τ²+σ²). Перелом при q=0.00134"""
    s = sigma_kappa(q)
    return V_PRIOR / (V_PRIOR + s * s)


def q_from_gain(G_rmsle: float) -> float:
    """грубый перевод: ось с валидационным выигрышем G при κ=1 имеет q≈2·F0·G"""
    return 2 * SD_E * G_rmsle


SE_K = 0.332             # оставлено для совместимости: это σ мелкой пробы (G~0.00015)
W = V_PRIOR / (V_PRIOR + SE_K ** 2)
M_PRIOR = 0.333
NOISE = 0.000022
SE_PUB, SE_PRIV = 0.0056, 0.0028
FREEZE = date(2026, 8, 31)
LAST_SUB = date(2026, 8, 30)
PER_DAY = 5

CLASS_PRIOR = {          # приор κ по классу оси (часть A, §5)
    "history_model": 0.90,   # модель, обученная на истории; вала не видела
    "blend_delta": 0.56,     # пересборка весов на валидационном окне
    "probe_applied": 0.85,   # коэффициент, замеренный прямо на лидерборде
    "level": 0.15,           # свойство окна
    "feature_stack": 0.20,   # подгонка под остаток вала
    "segment": 0.05,         # окно + состав сегментов
    "unknown": M_PRIOR,
}


def attempts_left(today: date) -> int:
    return max(0, (LAST_SUB - today).days + 1) * PER_DAY


def shrink(k_hat: float, q: float, mu: float = M_PRIOR) -> float:
    """усадка замеренной κ; приор ПЛОСКИЙ N(0.333, 0.205²) — классовый проиграл LOO"""
    w = dose(q)
    return w * k_hat + (1 - w) * mu


def apply_gain(G: float, kappa: float) -> float:
    """ожидаемый тестовый выигрыш при применении оси с долей c=kappa"""
    return max(0.0, kappa ** 2 * G)


def probe_value(ax: dict) -> float:
    """ценность ЗОНДА (сверх простого применения)"""
    if ax["type"] == "S":                      # локально не измерима — зонд единственный путь
        k = ax.get("k_expect", 0.5)
        a = max(0.0, 1 - SE_K ** 2 / (k ** 2 + SE_K ** 2))
        return a * k * k * ax["q"]
    # тип L: зонд лишь уточняет κ. Доза теперь СВОЯ у каждой оси (часть C).
    q = ax.get("q") or q_from_gain(ax["mdl_corund"])
    return dose(q) * V_PRIOR * ax["mdl_corund"]


def bank_value(ax: dict) -> float:
    """ценность ПРИМЕНЕНИЯ без зонда, по приору класса"""
    if ax["type"] == "S":
        return 0.0                              # без замера коэффициент неизвестен
    m = CLASS_PRIOR.get(ax.get("cls", "unknown"), M_PRIOR)
    return apply_gain(ax["mdl_corund"], m)


def decide(state: dict) -> list[dict]:
    today = date.fromisoformat(state["today"])
    A = state.get("attempts_left") or attempts_left(today)
    acts = []
    for ax in state["axes"]:
        if ax.get("done"):
            continue
        pv, bv = probe_value(ax), bank_value(ax)
        if ax["type"] == "S":
            acts.append(dict(action="ЗОНД", axis=ax["name"], ev=pv, cost=1,
                             why="локально не измерима, замер — единственный путь"))
        else:
            acts.append(dict(action="БАНК (применить по приору, БЕЗ попытки)", axis=ax["name"],
                             ev=bv, cost=0,
                             why=f"зонд добавил бы лишь {pv:.6f}, применение даёт {bv:.6f}"))
            if pv > NOISE and A > 6:
                acts.append(dict(action="зонд (опционально)", axis=ax["name"], ev=pv, cost=1,
                                 why="окупается только при избытке попыток"))
    for a in acts:
        a["ev_per_attempt"] = a["ev"] / a["cost"] if a["cost"] else float("inf")
    acts.sort(key=lambda a: (-a["ev_per_attempt"], -a["ev"]))
    return acts


def report(state: dict) -> None:
    today = date.fromisoformat(state["today"])
    A = state.get("attempts_left") or attempts_left(today)
    days = max(0, (LAST_SUB - today).days + 1)
    print(f"дата {today}, до последней заливки {days} дн, попыток {A}")
    print(f"текущий лучший замеренный: {state.get('best_measured', '—')}")
    print(f"финалисты с замером: {state.get('finalists_measured', 0)} из 2\n")

    reserve = 2 + (1 if state.get("line_improving", True) else 0)
    print(f"РЕЗЕРВ (не тратить): {reserve} попыток — {'2 на подтверждение финалистов' if reserve==2 else '2 на финалистов + 1 на обновление консерватора'}")
    print(f"свободных попыток: {A - reserve}\n")

    acts = decide(state)
    print(f"{'#':>2} {'действие':46s} {'ось':22s} {'E[выигрыш]':>12} {'поп.':>5}")
    free = A - reserve
    for i, a in enumerate(acts, 1):
        take = "" if a["cost"] == 0 else ("  <- берём" if free >= a["cost"] and a["ev"] > NOISE else "  (нет попыток/ниже шума)")
        if a["cost"] and free >= a["cost"] and a["ev"] > NOISE:
            free -= a["cost"]
        print(f"{i:>2} {a['action']:46s} {a['axis']:22s} {a['ev']:>12.6f} {a['cost']:>5}{take}")
    print()
    tot = sum(a["ev"] for a in acts if a["cost"] == 0)
    print(f"суммарно из БАНКА (без единой попытки): {tot:.6f}")
    print(f"порог осмысленности одного шага: {NOISE:.6f} (фиктивный выигрыш направления)")

    print(f"\n=== ПРАВИЛО ОСТАНОВКИ ===")
    print(f"прекратить зондировать, когда лучшая E[выигрыш] зонда < {NOISE:.6f},")
    print(f"или когда свободных попыток осталось {reserve} — они резерв.")

    print(f"\n=== ВЫБОР ДВУХ ФИНАЛОВ ===")
    d = state.get("delta_between_candidates", 0.0)
    if d:
        z = d / (SE_PRIV * (1 - 0.584) ** 0.5 + 1e-12)
        print(f"разница кандидатов на паблике {d:.6f}; на привате это {z:.1f} сигмы")
    print("правило: финал 1 = лучший ЗАМЕРЕННЫЙ (гарантия);")
    print("         финал 2 = лучший РАСЧЁТНЫЙ, но только если он отличается от первого")
    print(f"         направлением с корреляцией < 0.999 — иначе второй слот потрачен зря.")
    print(f"         оба обязаны иметь замеренный публичный скор.")


DEMO = {
    "today": "2026-08-20",
    "best_measured": 1.648093,
    "finalists_measured": 1,
    "line_improving": True,
    "delta_between_candidates": 0.0003,
    "axes": [
        {"name": "tfm3b", "type": "L", "cls": "history_model", "mdl_corund": 0.00090, "done": False},
        {"name": "kevf", "type": "L", "cls": "feature_stack", "mdl_corund": 0.00060, "done": False},
        {"name": "joint-fusion", "type": "L", "cls": "blend_delta", "mdl_corund": 0.00070, "done": False},
        {"name": "сезонный хвост (структурный)", "type": "S", "q": 0.0002, "k_expect": 0.6, "done": False},
        {"name": "поправка на молчащих 2", "type": "S", "q": 0.0002, "k_expect": 0.4, "done": False},
    ],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=str, default="")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo or not a.state:
        report(DEMO)
    else:
        report(json.loads(Path(a.state).read_text(encoding="utf-8")))
