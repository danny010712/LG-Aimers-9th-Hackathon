"""ph 키(투수x타자손)를 유지하고 **타깃만 바꾼다** — 복원 라벨 8종.

044가 산 유일한 표가 ph다. 같은 키로 다른 라벨의 편차표를 만들면
주모델에 없는 정보가 되는가? (주모델은 asof_pitcher_*_rate = 통산 **주변값**만 가진다)

판정: 지속진폭(corr x 가중sd)을 cond_ph(success, 0.0326)와 비교.
     + cond_ph와의 중복도(같은 칸끼리 상관) — 높으면 새 정보가 아니다.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd

M = 50
df = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                 usecols=["row_id", "season", "pitcher_id", "batter_hand"])
L = pd.read_csv("recovered_labels.csv.gz")
df = df.merge(L, on="row_id", how="left")
print(f"rows={len(df):,}  라벨 복원 {df['middle'].notna().sum():,}", flush=True)
df = df[df["middle"].notna()].copy()
df["mr"] = ((df.middle > 0) | (df.reverse > 0)).astype(int)
df["wayoff"] = (1 - df.success) * (1 - df.mr)
lg = lambda q: np.log(np.clip(q, 1e-9, 1-1e-9) / (1 - np.clip(q, 1e-9, 1-1e-9)))
K, PRIOR = ["pitcher_id", "batter_hand"], "pitcher_id"


def dev(d, tgt):
    g = d.groupby(K)[tgt].agg(["sum", "size"])
    b = d.groupby(PRIOR)[tgt].agg(["sum", "size"])
    pb = (b["sum"] / b["size"]).reindex(g.index.get_level_values(PRIOR)).values
    p = (g["sum"] + M * pb) / (g["size"] + M)
    return pd.DataFrame({"dev": lg(p) - lg(pb), "n": g["size"].values},
                        index=g.index).reset_index()


wsd = lambda t: float(np.sqrt(np.average(t.dev**2, weights=t.n)
                              - np.average(t.dev, weights=t.n)**2))
hist, tgt24 = df[df.season <= 2023], df[df.season == 2024]
base = None
print(f"\n{'타깃':<12} {'평균':>7} {'corr':>9} {'가중sd':>8} {'지속진폭':>9} "
      f"{'ph(success)와 상관':>18}")
print("-"*70)
for t in ("success", "mr", "wayoff", "middle", "reverse", "ball", "strike",
          "fastball", "breaking", "offspeed"):
    a, b = dev(hist, t), dev(tgt24, t)
    m = a.merge(b, on=K, suffixes=("_a", "_b"))
    m = m[(m.n_a >= 30) & (m.n_b >= 30)]
    c = np.corrcoef(m.dev_a, m.dev_b)[0, 1]
    amp = c * wsd(a)
    if t == "success":
        base = a.rename(columns={"dev": "dev0"})[K + ["dev0"]]
        red = 1.0
    else:
        j = a.merge(base, on=K)
        red = np.corrcoef(j.dev, j.dev0)[0, 1]
    print(f"{t:<12} {df[t].mean():>7.3f} {c:>+9.4f} {wsd(a):>8.4f} "
          f"{amp:>9.4f} {red:>+18.3f}")
print("\n기준: cond_ph(success) 지속진폭 0.0326 -> 주모델 +14.1 -> LB +5.55")
print("      |ph와의 상관| 이 크면 새 정보가 아니다")
