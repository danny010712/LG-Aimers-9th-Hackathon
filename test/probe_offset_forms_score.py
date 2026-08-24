"""09 세션 O2a 주장 독립 검증.

주장: mr 단독(c=0)이 현행 V0(mr+wayoff)보다 out-of-year에서 낫다.
근거로 제시된 것: 2021->24 (V0 -39.4 / O2a +45.5), 2022->24 (V0 +28.6 / O2a +29.8)

확인할 것:
 1. 보고에서 빠진 2023->24 (2023은 2024/2025와 같은 체제 = 가장 가까운 전이)
 2. 전이 매트릭스 전체 (S != T 12개)
 3. 배포 관련성: 실제 배포는 2024에서 적합한다. 2024 출처 전이의 부호
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

YEARS = [2021, 2022, 2023, 2024]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bss(p, y):
    r = y.mean()
    return 100000 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


d = pd.read_csv("probe_offset_forms_preds.csv.gz")
Y = {t: d[d.season == t].reset_index(drop=True) for t in YEARS}
R = {t: (Y[t].game_type == "R").values for t in YEARS}


def fit(S, cols):
    """출처 연도 S에서 계수와 mu 적합. cols = ['mr'] 또는 ['mr','wayoff']"""
    g = Y[S]
    z = logit(g.p_success.values)
    t = g.y.values
    U = [logit(g["p_" + c].values) for c in cols]
    mu = [float(u.mean()) for u in U]
    U = [u - m for u, m in zip(U, mu)]

    def nll(w):
        p = np.clip(1 / (1 + np.exp(-(z + sum(wi * u for wi, u in zip(w, U))))),
                    1e-9, 1 - 1e-9)
        return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

    w = minimize(nll, [0.0] * len(cols), method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-9)).x
    return list(w), mu


def apply_score(T, cols, w, mu, mask=None):
    g = Y[T]
    z = logit(g.p_success.values)
    for c, wi, m in zip(cols, w, mu):
        z = z + wi * (logit(g["p_" + c].values) - m)
    p1 = 1 / (1 + np.exp(-z))
    p0 = g.p_success.values
    y = g.y.values
    if mask is not None:
        p0, p1, y = p0[mask], p1[mask], y[mask]
    return bss(p1, y) - bss(p0, y)


FORMS = {"V0 (mr+wayoff)": ["mr", "wayoff"], "O2a (mr only)": ["mr"]}

print("=== 계수 (출처 연도별 적합) ===")
for S in YEARS:
    row = []
    for name, cols in FORMS.items():
        w, mu = fit(S, cols)
        row.append(f"{name}: [{', '.join(f'{x:+.4f}' for x in w)}]")
    print(f" {S}: " + "   ".join(row))

for scope, mk in [("전체 채점", None), ("R만 채점", R)]:
    print(f"\n=== 전이 매트릭스 · {scope} (델타 BSS) ===")
    print(f" {'S->T':<12} {'V0':>9} {'O2a':>9} {'차이':>9}")
    diffs = []
    for S in YEARS:
        for T in YEARS:
            if S == T:
                continue
            m = None if mk is None else mk[T]
            v = {}
            for name, cols in FORMS.items():
                w, mu = fit(S, cols)
                v[name] = apply_score(T, cols, w, mu, m)
            a, b = v["V0 (mr+wayoff)"], v["O2a (mr only)"]
            diffs.append(b - a)
            star = " *" if T == 2024 else ""
            print(f" {S}->{T:<8} {a:>9.1f} {b:>9.1f} {b-a:>+9.1f}{star}")
    print(f" O2a-V0: 중앙 {np.median(diffs):+.1f}  평균 {np.mean(diffs):+.1f}  "
          f"양수 {sum(x>0 for x in diffs)}/{len(diffs)}")

print("\n=== 목표 2023·2024만 (2025와 같은 체제) · 전체 채점 ===")
for T in [2023, 2024]:
    for S in YEARS:
        if S == T:
            continue
        v = {}
        for name, cols in FORMS.items():
            w, mu = fit(S, cols)
            v[name] = apply_score(T, cols, w, mu)
        print(f" {S}->{T}  V0 {v['V0 (mr+wayoff)']:+8.1f}   "
              f"O2a {v['O2a (mr only)']:+8.1f}   차이 {v['O2a (mr only)']-v['V0 (mr+wayoff)']:+7.1f}")
