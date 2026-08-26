"""offset은 드리프트 축인가 판별력 축인가 — 08 §5-13 정정 (2026-08-25).

§5-13은 전이 배율 8개를 "2025 미관측성을 건드리는 축(0.79~2.40)" vs
"관측 데이터의 판별력만 올리는 축(0.26 이하)"으로 갈랐고 offset을 전자(2.40)에 넣었다.
**틀렸다.** 직접 재면 offset은 레벨을 전혀 안 건드린다:

    offset 없음 + 최적이동   797.9
    offset     + 최적이동   820.0    Δ +22.1   ← 레벨을 다 뺐는데 100% 남는다
    예측 평균 0.49446 → 0.49447 (Δ +0.00001)
    예측 sd  0.04170 → 0.04620 (Δ +10.8%)     ← 퍼짐 = RES 증가 = 판별력

→ **판별력 축인데 배율이 2.40이었다.** 배율을 가르는 것은 축의 종류가 아니라
   **베이스 강도**다(2요인 모형, 09 §1-N).

계수 축도 함께 닫는다: b의 목적함수가 최적 근처에서 극도로 평평하다(±20%에 0.6점).
연도별 |b|가 커지는 추세(−.068→−.091→−.097)를 외삽해도 **−0.37**.

입력: probe_offset_forms_preds.csv.gz (4연도 × 3라벨 out-of-year 예측)
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

SRC = "probe_offset_forms_preds.csv.gz"


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    d = pd.read_csv(SRC)
    years = sorted(d.season.unique())

    print("=== 연도별 최적 계수 (그 연도에서 적합 = 오라클) ===")
    print(f"{'연도':>6}{'base rate':>11}{'b':>10}{'c':>10}{'base BSS':>11}{'오라클 Δ':>11}")
    coef = {}
    for yv in years:
        g = d[d.season == yv]
        z, m, w = logit(g.p_success.values), logit(g.p_mr.values), logit(g.p_wayoff.values)
        y = g.y.values.astype(float)
        mm, mw = m.mean(), w.mean()
        r = y.mean(); U = r * (1 - r)
        f = lambda v: np.mean((sigmoid(z + v[0] * (m - mm) + v[1] * (w - mw)) - y) ** 2)
        b, c = minimize(f, [0., 0.], method="Nelder-Mead",
                        options={"xatol": 1e-5, "fatol": 1e-12}).x
        coef[yv] = (b, c)
        bs = 1e5 * (1 - np.mean((sigmoid(z) - y) ** 2) / U)
        of = 1e5 * (1 - np.mean((sigmoid(z + b * (m - mm) + c * (w - mw)) - y) ** 2) / U)
        print(f"{yv:>6}{r:>11.4f}{b:>10.4f}{c:>10.4f}{bs:>11.1f}{of-bs:>11.1f}")
    print("  ⚠️ 2021은 F 라벨 아티팩트 시대라 b=−0.30으로 튄다. 제외하고 볼 것.")

    g = d[d.season == 2024]
    z, m, w = logit(g.p_success.values), logit(g.p_mr.values), logit(g.p_wayoff.values)
    y = g.y.values.astype(float); mm, mw = m.mean(), w.mean()
    r = y.mean(); U = r * (1 - r)
    bss = lambda q: 1e5 * (1 - np.mean((q - y) ** 2) / U)
    b, c = coef[2024]
    zo = z + b * (m - mm) + c * (w - mw)

    def best_shift(zz):
        s = minimize_scalar(lambda s: np.mean((sigmoid(zz - s) - y) ** 2),
                            bounds=(-.5, .5), method="bounded").x
        return s, sigmoid(zz - s)

    s0, q0 = best_shift(z); s1, q1 = best_shift(zo)
    print("\n=== 🔴 offset은 레벨을 고치는가 판별력을 올리는가 (2024) ===")
    print(f"  offset 없음                    BSS {bss(sigmoid(z)):7.1f}")
    print(f"  offset                        BSS {bss(sigmoid(zo)):7.1f}"
          f"   Δ{bss(sigmoid(zo))-bss(sigmoid(z)):+6.1f}")
    print(f"  offset 없음 + 최적이동(s={s0:+.4f})  BSS {bss(q0):7.1f}")
    print(f"  offset     + 최적이동(s={s1:+.4f})  BSS {bss(q1):7.1f}"
          f"   Δ{bss(q1)-bss(q0):+6.1f}  ← 레벨을 다 뺀 뒤")
    keep = (bss(q1) - bss(q0)) / (bss(sigmoid(zo)) - bss(sigmoid(z)))
    print(f"  → 레벨 보정 후 {keep*100:.0f}% 잔존 = **순수 판별력 축**")
    print(f"  예측평균 {sigmoid(z).mean():.5f} → {sigmoid(zo).mean():.5f}"
          f"  (Δ{sigmoid(zo).mean()-sigmoid(z).mean():+.5f})  ← 레벨 안 건드림")
    print(f"  예측 sd  {sigmoid(z).std():.5f} → {sigmoid(zo).std():.5f}"
          f"  (Δ{(sigmoid(zo).std()/sigmoid(z).std()-1)*100:+.1f}%)  ← 퍼짐 = RES 증가")

    print("\n=== b 민감도 — 목적함수가 평평하다 (계수 축 종료) ===")
    print(f"{'b':>10}{'BSS':>10}{'최적 대비':>11}")
    ref = bss(sigmoid(zo))
    for bb in [-0.060, -0.070, -0.080, -0.090, b, -0.105, -0.115, -0.130, -0.150, -0.200]:
        q = sigmoid(z + bb * (m - mm) + c * (w - mw))
        tag = " ← 2024 최적" if bb == b else ""
        print(f"{bb:>10.4f}{bss(q):>10.1f}{bss(q)-ref:>11.2f}{tag}")
    ys = np.array([coef[k][0] for k in (2022, 2023, 2024)])
    sl = np.polyfit([0, 1, 2], ys, 1)[0]
    q = sigmoid(z + (ys[-1] + sl) * (m - mm) + c * (w - mw))
    print(f"\n  |b| 추세 {sl:+.5f}/yr → 2025 선형외삽 {ys[-1]+sl:+.5f}")
    print(f"  2024에 그 값을 썼다면 {bss(q)-ref:+.2f} BSS → **외삽 무의미**")


if __name__ == "__main__":
    main()
