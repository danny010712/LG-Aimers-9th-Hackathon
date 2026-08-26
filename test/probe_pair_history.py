"""투수x타자 실제 대전 이력 표 — p_matchup(log5 공식)이 아니라 진짜 표.

판별식:
 ① 정보가 느는가   asof_*는 전부 **주변(marginal)** 통계다. 쌍(pair) 이력은 없다 -> 통과 가능
 ② 행마다 변하는가  자명
 ③ 해마다 이어지나  <=2023 쌍표 vs 2024 쌍표 corr

먼저 희소성부터 본다. 쌍당 표본이 너무 적으면 ③ 이전에 끝난다.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd

df = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                 usecols=["season", "pitcher_id", "batter_id", "control_success"])
print(f"rows={len(df):,}", flush=True)

pair = df.groupby(["pitcher_id", "batter_id"])["control_success"].size()
print(f"\n[희소성] 관측된 쌍 {len(pair):,}개 "
      f"(가능 {df.pitcher_id.nunique()*df.batter_id.nunique():,})")
print(f"  쌍당 투구수  중앙 {pair.median():.0f}  평균 {pair.mean():.1f}  "
      f"75% {pair.quantile(.75):.0f}  95% {pair.quantile(.95):.0f}  최대 {pair.max()}")
for th in (10, 20, 30, 50, 100):
    cov = df.groupby(["pitcher_id", "batter_id"])["control_success"].transform("size")
    print(f"  쌍 n>={th:<4} 인 **행** 비율 {100*(cov >= th).mean():5.1f}%  "
          f"쌍 개수 {int((pair >= th).sum()):,}")

# 2025 배포 조건: <=2024 쌍표를 2025 행에 붙인다. 2024를 목표로 모사.
hist = df[df.season <= 2023]
tgt = df[df.season == 2024]
hp = hist.groupby(["pitcher_id", "batter_id"])["control_success"].size()
cov = tgt.set_index(["pitcher_id", "batter_id"]).index.map(hp).to_series().fillna(0).values
print(f"\n[배포 모사] 2024 행 중 <=2023 쌍이력이 있는 비율 {100*(cov>0).mean():.1f}%  "
      f"n>=20 {100*(cov>=20).mean():.1f}%  n>=50 {100*(cov>=50).mean():.1f}%")

# ③ 지속성 — 투수 기준선 대비 편차(platoon과 같은 형태)
M = 270.0
def tab(d, minn):
    g = d.groupby(["pitcher_id", "batter_id"])["control_success"].agg(["sum", "size"])
    b = d.groupby("pitcher_id")["control_success"].agg(["sum", "size"])
    pb = (b["sum"] / b["size"]).reindex(g.index.get_level_values(0)).values
    p = (g["sum"] + M * pb) / (g["size"] + M)
    o = pd.DataFrame({"split": np.log(p/(1-p)) - np.log(pb/(1-pb)),
                      "n": g["size"].values}, index=g.index).reset_index()
    return o[o.n >= minn]

print("\n[③ 지속성  (기준: 투수x좌우 +0.412 채택)]")
for minn in (10, 20, 30):
    a, b = tab(hist, minn), tab(df[df.season == 2024], minn)
    m = a.merge(b, on=["pitcher_id", "batter_id"], suffixes=("_a", "_b"))
    if len(m) < 30:
        print(f"  n>={minn:<3} 겹치는 쌍 {len(m)} 개 — 판정 불가")
        continue
    c = np.corrcoef(m.split_a, m.split_b)[0, 1]
    print(f"  n>={minn:<3} corr {c:+.4f}  겹치는 쌍 {len(m):,}  "
          f"진폭비 {m.split_b.std()/m.split_a.std():.3f}")
