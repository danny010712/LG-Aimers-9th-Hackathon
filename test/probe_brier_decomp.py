"""Murphy 분해 — 우리 손실이 어디에 있는가 (REL vs RES). 09 §1-M / §0-6.

    Brier = REL − RES + UNC
      REL(신뢰도) "0.52라 말한 것 중 52%가 성공했나" — 거짓말한 몫
      RES(해상도) 예측을 흩뜨려 실제로 갈라낸 몫
      UNC        상수예측의 Brier = r(1−r). 데이터 성질, 모델 무관

🔴 **REL이 작은 것은 자랑이 아니다.** 상수예측도 REL=0이고 BSS는 0이다.
   부스팅+147만행+전역 이동이면 자동으로 그렇게 된다. 어려운 것은 RES다.

핵심 산출 (013 검증예측, 전역 이동 후):
    REL ≲ 6 BSS   /   RES 0.00223   → RES가 REL의 약 148배
    정직한 split-half isotonic = **같은 연도인데도 −19.0**
  → calibration 축은 "고칠 게 6점뿐이라 보정기 노이즈가 그보다 크다"로 닫힌다.
    기존 근거(팀 LB −23)는 `calibration and smoothing` 2변수 동시 변경이라 교란돼 있었다.

⚠️ REL은 bin 개수에 민감하다. 100 bin에서는 노이즈 바닥(0.000098)이 신호(0.000086)보다
   커서 음수가 나온다. **한 bin 수만 보고 "0"이라 하면 안 된다** — 09 §0-2 위반.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import glob

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

LB = {"주최측 baseline": 549.51, "우리 021": 1057.00, "100위": 1124.70,
      "10위": 1171.22, "3위(제출 4회)": 1218.66, "1위": 1421.99}
B_LB = 0.249445          # 2025 추정 r(1−r)


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def decomp(p, y, r, K):
    b = pd.qcut(p, K, labels=False, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": y, "b": b}).groupby("b").agg(
        n=("y", "size"), pm=("p", "mean"), ym=("y", "mean"))
    w = t.n / t.n.sum()
    rel = float((w * (t.pm - t.ym) ** 2).sum())
    res = float((w * (t.ym - r) ** 2).sum())
    nz = float((w * (t.ym * (1 - t.ym)) / t.n).sum())
    return rel, res, nz, int(t.n.mean())


def main():
    d = pd.read_csv("data/train.csv", usecols=["row_id", "season", "control_success"])
    L = pd.read_csv("recovered_labels.csv.gz", usecols=["row_id", "middle"])
    d = d.merge(L, on="row_id", how="left")
    va = (d.season == 2024).values
    have = d["middle"].notna().values[va]
    p0 = np.mean([np.load(x) for x in
                  sorted(glob.glob("artifacts/auxpred_ins_013_backup/*.npy"))], axis=0)
    y = d[va][have].control_success.values.astype(float)
    r = float(y.mean()); UNC = r * (1 - r); n = len(y)
    bss = lambda q: 1e5 * (1 - np.mean((q - y) ** 2) / UNC)

    s = minimize_scalar(lambda s: np.mean((sigmoid(logit(p0) - s) - y) ** 2),
                        bounds=(-.5, .5), method="bounded").x
    p = sigmoid(logit(p0) - s)
    print(f"013 raw BSS {bss(p0):.1f} → 최적 전역이동(s={s:+.5f}) 후 {bss(p):.1f}")
    print(f"예측 sd={p.std():.5f} 범위 {p.min():.4f}~{p.max():.4f}  base rate={r:.4f}\n")

    print("=== REL은 bin 개수에 민감하다 (노이즈 바닥과 함께 볼 것) ===")
    print(f"{'bins':>6}{'bin당행':>10}{'REL(raw)':>12}{'노이즈':>12}{'차':>12}{'BSS':>8}")
    for K in [5, 10, 20, 50, 100, 200]:
        rel, res, nz, m = decomp(p, y, r, K)
        print(f"{K:>6}{m:>10,}{rel:>12.7f}{nz:>12.7f}{rel-nz:>12.7f}"
              f"{1e5*(rel-nz)/UNC:>8.1f}")
    rel, res, nz, _ = decomp(p, y, r, 20)
    print(f"\n→ REL ≲ 6 BSS.  RES(20bin, 노이즈 제거) = {res-nz:.7f}"
          f"  = REL의 약 {(res-nz)/max(rel-nz,1e-9):.0f}배")

    print("\n=== 상수예측도 REL=0이다 (REL이 작은 건 자랑이 아니다) ===")
    for tag, q in [("상수예측(=base rate)", np.full(n, r)), ("우리 모델(이동 후)", p)]:
        jitter = q + np.random.default_rng(1).normal(0, 1e-9, n)
        b = pd.qcut(jitter, 20, labels=False, duplicates="drop")
        t = pd.DataFrame({"p": q, "y": y, "b": b}).groupby("b").agg(
            n=("y", "size"), pm=("p", "mean"), ym=("y", "mean"))
        w = t.n / t.n.sum()
        rl = float((w * (t.pm - t.ym) ** 2).sum())
        rs = float((w * (t.ym - r) ** 2).sum())
        nzz = float((w * (t.ym * (1 - t.ym)) / t.n).sum())
        print(f"  {tag:<22} REL={max(rl-nzz,0):.7f}  RES={max(rs-nzz,0):.7f}"
              f"  BSS={bss(q):8.1f}")

    print("\n=== isotonic: 부정 vs 정직 (둘 다 같은 연도 2024) ===")
    dirty = IsotonicRegression(out_of_bounds="clip").fit_transform(p, y)
    h = np.random.default_rng(0).random(n) < 0.5
    out = np.empty(n)
    for m in (h, ~h):
        out[~m] = IsotonicRegression(out_of_bounds="clip").fit(p[m], y[m]).predict(p[~m])
    print(f"  적합=채점 (부정)     BSS {bss(dirty):7.1f}  Δ{bss(dirty)-bss(p):+6.1f}"
          f"   ← 팀 노트북 '+75'의 정체")
    print(f"  정직한 split-half  BSS {bss(out):7.1f}  Δ{bss(out)-bss(p):+6.1f}"
          f"   ← 같은 연도인데도 손해")

    print("\n=== LB: 격차는 전부 RES다 (REL≈0이므로 RES−REL ≈ RES) ===")
    print(f"{'':<18}{'BSS':>9}{'Brier':>11}{'RES−REL':>11}{'상수대비':>10}")
    for tag, v in LB.items():
        br = B_LB * (1 - v / 1e5)
        print(f"  {tag:<16}{v:>9.2f}{br:>11.6f}{B_LB-br:>11.6f}{v/1e3:>9.2f}%")
    print(f"\n  1위 − 우리 = {B_LB*(1421.99-1057.00)/1e5:.6f}"
          f"  (우리 대비 +{(1421.99-1057)/1057*100:.0f}%)")
    print("  🔴 1위조차 상수예측보다 1.42% 나을 뿐이다. 후처리로 건드릴 여지는 REL뿐이고 그게 ≲6.")


if __name__ == "__main__":
    main()
