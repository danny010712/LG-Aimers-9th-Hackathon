"""RMSE 이득 +4.5가 판별력(RES)인가 레벨(REL)인가.

09 §3-J가 남긴 규칙: in-year 이득은 배포 shift를 넣으면 사라질 수 있다
(c 2스트라이크 분리 +12.36 -> +0.37). 레벨 몫을 제거하고 다시 잰다.

방법: 두 팔을 각각 **2024 실제 평균에 맞춰 로짓 재중심화**(오라클 레벨)한 뒤 비교.
배포에서는 logit_shift가 이 역할을 한다. 레벨이 원인이면 Δ가 무너진다.
"""
import io, sys, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from scipy.optimize import brentq

SEEDS = [42, 7, 2024]
y = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                usecols=["season", "control_success"])
yv = y.loc[y.season == 2024, "control_success"].values.astype(float)
r = yv.mean()
print(f"2024 실제 base rate {r:.6f}  n={len(yv):,}")


def bss(p):
    return max(0.0, 100000 * (1 - np.mean((p - yv) ** 2) / (r * (1 - r))))


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def recenter(p):
    """로짓을 상수만큼 밀어 예측평균을 실제 base rate에 맞춘다 (오라클 shift)."""
    z = lg(p)
    d = brentq(lambda d: 1 / (1 + np.exp(-(z + d))).mean() - r, -2, 2)
    return 1 / (1 + np.exp(-(z + d))), d


out = {}
for loss in ("Logloss", "RMSE"):
    p = np.mean([np.load(f"artifacts/rmse_probe/{loss}_2024_{s}.npy")
                 for s in SEEDS], axis=0)
    pr, d = recenter(p)
    out[loss] = (bss(p), bss(pr), d, p.mean())
    print(f"{loss:<8} raw {out[loss][0]:7.1f}   레벨보정후 {out[loss][1]:7.1f}   "
          f"shift {d:+.6f}   예측평균 {p.mean():.6f}")

d_raw = out["RMSE"][0] - out["Logloss"][0]
d_lvl = out["RMSE"][1] - out["Logloss"][1]
print(f"\nΔ raw        {d_raw:+.2f}")
print(f"Δ 레벨보정후  {d_lvl:+.2f}    <- 배포에는 이 몫만 남는다")
print(f"레벨 몫       {100*(1 - d_lvl/d_raw) if d_raw else float('nan'):.0f}%")

# 시드쌍 부호 (§4-2)
print("\n[시드별 짝지은 Δ, 레벨보정 후]")
ds = []
for s in SEEDS:
    a = recenter(np.load(f"artifacts/rmse_probe/Logloss_2024_{s}.npy"))[0]
    b = recenter(np.load(f"artifacts/rmse_probe/RMSE_2024_{s}.npy"))[0]
    ds.append(bss(b) - bss(a))
    print(f"  seed {s:<5} {bss(a):7.1f} -> {bss(b):7.1f}   Δ {ds[-1]:+6.2f}")
ds = np.array(ds)
print(f"  평균 {ds.mean():+.2f}  sd {ds.std(ddof=1):.2f}  "
      f"표준오차 {ds.std(ddof=1)/np.sqrt(len(ds)):.2f}  "
      f"t={ds.mean()/(ds.std(ddof=1)/np.sqrt(len(ds))):.2f}")
