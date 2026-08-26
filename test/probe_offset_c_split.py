"""offset의 `c`(wayoff 계수)를 2스트라이크 여부로 분리 — 09 §3-J (2026-08-25).

발단: 잔차 지도에서 2스트라이크 카운트의 BSS가 유독 낮았다(0-2 = 246 vs 3-0 = 1516).
offset을 세그먼트별로 재보니 **우리 최대 판별력 축이 거기서 해를 끼치고 있었다**:

    013 베이스, 전역계수
      2스트라이크  n= 72,782   654.3 → 624.6   Δ **−29.7**
      그 외       n=180,334   941.0 → 980.8   Δ +39.8

기전: 2스트라이크에선 투수가 **의도적으로 존 밖으로** 던진다(유인구).
같은 `p_wayoff`가 다른 뜻이므로 전역 `c` 하나로는 틀린다.

🔴 `b`는 분리하면 안 된다 — 연도별 차이가 +0.032 / −0.035 / +0.159로 **부호 2/3, 크기 5배**.
✅ `c`만 분리한다 — 차이가 −0.0512 / −0.0431 / −0.0320으로 **부호 3/3, 크기 안정**(§0-5 지속성).

검증(전부 out-of-year, 채택기준 4·5·6):
    2023→2024  **+5.21**  ← 배포 최근접
    2022→2024  +8.31
    2021→2022  +3.57      (2021 출처 = F 아티팩트 시대, 참고만)
    2022→2023  −0.07      (목표 2023 = 죽은 fold, base 10.1)
    013 강한 베이스 자기적합 오라클 **+12.36** — 포화로 안 줄었다

배포 형태: 파라미터 **1개 추가**. 표는 학습 때 계산해 meta에 저장. 재학습 없음 =
용량 증가 0(021과 같은 구조). `strikes_before`는 test에도 있는 컬럼.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import glob

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CACHE = "probe_offset_forms_preds.csv.gz"


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_global(z, m, w, y, mm, mw):
    f = lambda v: np.mean((sigmoid(z + v[0] * (m - mm) + v[1] * (w - mw)) - y) ** 2)
    return minimize(f, [0., 0.], method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-14}).x


def fit_split(z, m, w, y, s2, mm, mw):
    f = lambda v: np.mean((sigmoid(z + v[0] * (m - mm)
                                   + np.where(s2 == 1, v[2], v[1]) * (w - mw)) - y) ** 2)
    return minimize(f, [0., 0., 0.], method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-14, "maxiter": 4000}).x


def main():
    d = pd.read_csv(CACHE)
    cx = pd.read_csv("data/train.csv", usecols=["row_id", "strikes_before"])
    d = d.merge(cx, on="row_id")
    d["s2"] = (d.strikes_before == 2).astype(int)

    print("=== ① 지속성: 연도별로 c 차이가 이어지는가 (§0-5) ===")
    print(f"{'연도':>6}{'b(그외)':>10}{'b(2S)':>10}{'b차이':>9}"
          f"{'c(그외)':>10}{'c(2S)':>10}{'c차이':>9}")
    diffs = {}
    for yv in sorted(d.season.unique()):
        g = d[d.season == yv]
        z, m, w = logit(g.p_success.values), logit(g.p_mr.values), logit(g.p_wayoff.values)
        y = g.y.values.astype(float); mm, mw = m.mean(), w.mean(); s = g.s2.values
        b0, c0 = fit_global(z[s == 0], m[s == 0], w[s == 0], y[s == 0], mm, mw)
        b1, c1 = fit_global(z[s == 1], m[s == 1], w[s == 1], y[s == 1], mm, mw)
        diffs[yv] = (b1 - b0, c1 - c0)
        print(f"{yv:>6}{b0:>10.4f}{b1:>10.4f}{b1-b0:>9.4f}{c0:>10.4f}{c1:>10.4f}{c1-c0:>9.4f}")
    db = [diffs[y][0] for y in (2022, 2023, 2024)]
    dc = [diffs[y][1] for y in (2022, 2023, 2024)]
    print(f"\n  2022~24  b차이 부호 {sum(np.sign(db) == np.sign(np.mean(db)))}/3  "
          f"→ 🔴 b는 분리 금지")
    print(f"           c차이 부호 {sum(np.sign(dc) == np.sign(np.mean(dc)))}/3  "
          f"평균 {np.mean(dc):+.4f}  → ✅ c만 분리")

    print("\n=== ② out-of-year 전이 (채택기준 4·5·6) ===")
    print(f"{'전이':>12}{'base':>9}{'A 전역':>10}{'B c분리':>10}{'B−A':>8}")
    for S, T in [(2023, 2024), (2022, 2024), (2021, 2022), (2022, 2023)]:
        sg_, tg = d[d.season == S], d[d.season == T]
        zS, mS, wS = logit(sg_.p_success.values), logit(sg_.p_mr.values), logit(sg_.p_wayoff.values)
        mmS, mwS = mS.mean(), wS.mean()
        A = fit_global(zS, mS, wS, sg_.y.values.astype(float), mmS, mwS)
        B = fit_split(zS, mS, wS, sg_.y.values.astype(float), sg_.s2.values, mmS, mwS)
        zT, mT, wT = logit(tg.p_success.values), logit(tg.p_mr.values), logit(tg.p_wayoff.values)
        yT = tg.y.values.astype(float); mmT, mwT = mT.mean(), wT.mean(); sT = tg.s2.values
        r = yT.mean(); U = r * (1 - r)
        bss = lambda q: 1e5 * (1 - np.mean((q - yT) ** 2) / U)
        a = bss(sigmoid(zT + A[0] * (mT - mmT) + A[1] * (wT - mwT)))
        b_ = bss(sigmoid(zT + B[0] * (mT - mmT) + np.where(sT == 1, B[2], B[1]) * (wT - mwT)))
        note = " ← 배포 최근접" if (S, T) == (2023, 2024) else (
            " (2021 출처 = F 아티팩트)" if S == 2021 else (
                " (목표 2023 = 죽은 fold)" if T == 2023 else ""))
        print(f"{S}→{T:>6}{bss(sigmoid(zT)):>9.1f}{a:>10.1f}{b_:>10.1f}{b_-a:>8.2f}{note}")

    print("\n=== ③ 배포 베이스(013) 확인 — 포화로 줄어드는가 ===")
    tr = pd.read_csv("data/train.csv",
                     usecols=["row_id", "season", "strikes_before", "control_success"])
    L = pd.read_csv("recovered_labels.csv.gz", usecols=["row_id", "middle"])
    tr = tr.merge(L, on="row_id", how="left")
    va = (tr.season == 2024).values; have = tr["middle"].notna().values[va]
    v = tr[va][have].reset_index(drop=True)
    avg = lambda pat: np.mean([np.load(x) for x in sorted(glob.glob(pat))], axis=0)
    z = logit(avg("artifacts/auxpred_ins_013_backup/success_*.npy"))
    m = logit(avg("artifacts/auxpred/mr_2024_*.npy"))
    w = logit(avg("artifacts/auxpred/wayoff_2024_*.npy"))
    y = v.control_success.values.astype(float); s2 = (v.strikes_before.values == 2).astype(int)
    mm, mw = m.mean(), w.mean(); r = y.mean(); U = r * (1 - r)
    bss = lambda q: 1e5 * (1 - np.mean((q - y) ** 2) / U)
    A = fit_global(z, m, w, y, mm, mw); B = fit_split(z, m, w, y, s2, mm, mw)
    qa = sigmoid(z + A[0] * (m - mm) + A[1] * (w - mw))
    qb = sigmoid(z + B[0] * (m - mm) + np.where(s2 == 1, B[2], B[1]) * (w - mw))
    print(f"  offset 없음                                        {bss(sigmoid(z)):7.1f}")
    print(f"  A 전역  b={A[0]:+.4f} c={A[1]:+.4f}                     {bss(qa):7.1f}"
          f"   ← run 015 배포값(−0.0990, +0.0074)과 일치")
    print(f"  B c분리 b={B[0]:+.4f} c그외={B[1]:+.4f} c2S={B[2]:+.4f}  {bss(qb):7.1f}"
          f"   B−A = {bss(qb)-bss(qa):+.2f}")
    zo = z + A[0] * (m - mm) + A[1] * (w - mw)
    print("\n  세그먼트별 offset 효과 (전역계수):")
    for tag, i in [("2스트라이크", s2 == 1), ("그 외", s2 == 0)]:
        u = y[i].mean() * (1 - y[i].mean())
        b0 = 1e5 * (1 - np.mean((sigmoid(z[i]) - y[i]) ** 2) / u)
        b1 = 1e5 * (1 - np.mean((sigmoid(zo[i]) - y[i]) ** 2) / u)
        print(f"    {tag:<10} n={i.sum():>7,}  {b0:7.1f} → {b1:7.1f}   Δ{b1-b0:+7.1f}")


if __name__ == "__main__":
    main()
