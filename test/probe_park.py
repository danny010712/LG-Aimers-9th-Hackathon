"""구장 축 — 홈팀 = 구장. 문서가 '컬럼이 없다'고 적었으나 파생된다.

top_bottom=='T'(초) -> 홈팀이 수비 -> pitcher_team_id 가 홈 = 구장
top_bottom=='B'(말) -> 홈팀이 공격 -> batter_team_id  가 홈 = 구장

검정 순서 (§7-1 판별식):
 ① 정보가 느는가   구장 단독 성공률 편차 + 기존 열로 설명되나
 ② 행마다 변하는가  자명 (구장은 경기마다)
 ③ 해마다 이어지나  <=2023 표 vs 2024 표 corr  (gate.G1 기준: 좌우 0.412 채택)
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd

C = ["season", "game_type", "top_bottom", "pitcher_team_id", "batter_team_id",
     "pitcher_id", "batter_id", "pitcher_hand", "batter_hand", "control_success"]
df = pd.read_csv("data/train.csv", encoding="utf-8-sig", usecols=C)
df["park"] = np.where(df.top_bottom == "T", df.pitcher_team_id, df.batter_team_id)
print(f"rows={len(df):,}  구장 {df.park.nunique()}개  "
      f"투수 {df.pitcher_id.nunique():,}  타자 {df.batter_id.nunique():,}")

# 정합성: 홈팀은 항상 두 팀 중 하나여야 하고, 같은 경기에서 일관돼야 한다
bad = (df.park != df.pitcher_team_id) & (df.park != df.batter_team_id)
print(f"  정합성 위반 {bad.sum():,}")

# ---- ① 구장 단독 효과 ----
print("\n[① 구장 단독 성공률]")
t = df.groupby("park")["control_success"].agg(["mean", "size"]).sort_values("mean")
print(t.to_string())
print(f"  전체 {df.control_success.mean():.4f}  구장간 sd {t['mean'].std():.4f}  "
      f"폭 {t['mean'].max()-t['mean'].min():.4f}")

# 원정/홈 분리 — 파크 팩터인가 홈어드밴티지인가
df["is_home_p"] = (df.top_bottom == "T").astype(int)
print("\n[홈/원정 분리]")
print(df.groupby(["park", "is_home_p"])["control_success"].mean().unstack()
      .assign(diff=lambda x: x[1] - x[0]).to_string())

# ---- ③ 지속성: 투수x구장 편차표 ----
def split_table(d, key):
    g = d.groupby(["pitcher_id", key])["control_success"].agg(["sum", "size"])
    base = d.groupby("pitcher_id")["control_success"].agg(["sum", "size"])
    M = 270.0
    p = (g["sum"] + M * (base["sum"] / base["size"]).reindex(
        g.index.get_level_values(0)).values) / (g["size"] + M)
    pb = (base["sum"] / base["size"]).reindex(g.index.get_level_values(0)).values
    out = pd.DataFrame({"split": np.log(p / (1 - p)) - np.log(pb / (1 - pb)),
                        "n": g["size"].values}, index=g.index).reset_index()
    return out

print("\n[③ 지속성 corr  (기준: 투수x좌우 0.412 채택 / 나머지 <=0.106 기각)]")
df["plat"] = (df.pitcher_hand.astype(str) + "_" + df.batter_hand.astype(str))
for key, name in (("park", "투수x구장"), ("plat", "투수x좌우(대조)"),
                  ("is_home_p", "투수x홈원정(대조)")):
    a = split_table(df[df.season <= 2023], key)
    b = split_table(df[df.season == 2024], key)
    m = a.merge(b, on=["pitcher_id", key], suffixes=("_a", "_b"))
    m = m[(m.n_a >= 50) & (m.n_b >= 50)]
    c = np.corrcoef(m.split_a, m.split_b)[0, 1]
    # 진폭까지 (09 §2-O: corr만 보면 속는다)
    amp = m.split_b.std() / m.split_a.std()
    print(f"  {name:<18} corr {c:+.4f}   n={len(m):,}   "
          f"진폭비 sd(2024)/sd(<=2023) {amp:.3f}")
