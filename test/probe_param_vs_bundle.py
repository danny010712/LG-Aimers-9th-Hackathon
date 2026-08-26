"""cond가 진 원인 — 파라미터화인가 번들링인가 (039, 2026-08-25).

004는 cond 4표 번들(pc·ph·bc·pi)을 **트리 피처**로 줘서 LB −5.4.
021은 **같은 정보(투수×타자손)를 offset 표**로 줘서 LB +5.27.
08 §5-12는 그 차이를 "죽은 표 3개가 살아있는 1개를 희석"(번들링)으로 설명했지만
**파라미터화 탓인지 번들링 탓인지 분리된 적이 없다.**

039 = 013 + `cond_ph` **하나만** 트리 피처. 이걸로 갈린다.

결정적 측정: platoon offset을 두 베이스 위에 각각 얹어 이득을 비교한다.
  A) 013 성공예측 위          이득 = ?   (문서값 +20.5)
  B) 039 성공예측 위          이득 = ?
  ph 피처가 platoon 정보를 흡수했다면 B의 이득이 줄어야 한다.

판정:
  039 주모델 ≈ 858.6  &  platoon 이득 유지   → 트리가 못 쓴다. **파라미터화가 원인**
  039 주모델 >  858.6  &  platoon 이득 감소   → **번들링이 원인.** 파라미터화 축 재개방
  039 주모델 <  858.6                         → ph 단독으로도 해롭다. 파라미터화 확정
"""
import io
import sys

if getattr(sys.stdout, "encoding", "") != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import glob

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import gate

BASES = [("013 (기준)", "artifacts/auxpred_ins_013_backup"),
         ("039 (+cond_ph)", "artifacts/auxpred_ph")]


def fit_b(z, spl, y):
    f = lambda b: np.mean((gate.sigmoid(z + b * spl) - y) ** 2)
    return float(minimize_scalar(f, bounds=(-10, 10), method="bounded",
                                 options={"xatol": 1e-7}).x)


def main():
    v, y, _ = gate.deploy_base()
    df = gate.hist()
    # platoon 표는 <=2023으로만 (2024 자기적합 방지)
    tab = gate._split_table(df[df.season <= 2023], "plat")
    spl = pd.DataFrame({"pitcher_id": v["pitcher_id"].values,
                        "plat": v["plat"].values}).merge(
        tab, on=["pitcher_id", "plat"], how="left")["split"].fillna(0.0).values

    print(f"{'베이스':<16}{'주모델':>9}{'platoon b':>11}{'얹은 뒤':>9}{'이득':>8}")
    print("-" * 55)
    out = {}
    for tag, d in BASES:
        f = sorted(glob.glob(f"{d}/success_2024_*.npy"))
        if not f:
            print(f"{tag:<16}  캐시 없음 ({d}) — 학습 먼저")
            continue
        z = gate.logit(np.mean([np.load(x) for x in f], axis=0))
        b = fit_b(z, spl, y)
        a0 = gate.bss(gate.sigmoid(z), y)
        a1 = gate.bss(gate.sigmoid(z + b * spl), y)
        out[tag] = (a0, b, a1, a1 - a0)
        print(f"{tag:<16}{a0:>9.1f}{b:>11.3f}{a1:>9.1f}{a1-a0:>+8.2f}", flush=True)

    if len(out) == 2:
        (m0, _, _, g0), (m1, _, _, g1) = out.values()
        print(f"\n주모델 Δ  {m1-m0:+.1f}      platoon 이득 Δ  {g1-g0:+.2f}")
        print("  이득이 유지되면 → 트리가 ph를 못 쓴다 = 파라미터화가 원인")
        print("  이득이 줄면    → ph를 흡수했다 = 004의 패배는 번들링 탓")


if __name__ == "__main__":
    main()
