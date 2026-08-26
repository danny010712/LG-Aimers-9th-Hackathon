"""WE가 순수 상태 함수인가 — base_state 포함한 정확한 검정.

v1은 base_state를 빼먹어 num_runners_on=1의 세 배치(1b/2b/3b)가 다값으로 보였다.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd

C = ["season", "inning", "top_bottom", "outs_before", "balls_before",
     "strikes_before", "base_state", "run_top_before", "run_bot_before",
     "home_win_expectancy", "li", "pitcher_team_id", "batter_team_id"]
df = pd.read_csv("data/train.csv", encoding="utf-8-sig", usecols=C)

FULL = ["inning", "top_bottom", "outs_before", "base_state",
        "run_top_before", "run_bot_before"]
for name, key in [("상태(카운트 제외)", FULL),
                  ("상태+카운트", FULL + ["balls_before", "strikes_before"])]:
    for col in ("home_win_expectancy", "li"):
        g = df.groupby(key, observed=True)[col].agg(["nunique", "std", "size"])
        mu = g[g.nunique_ if False else g["nunique"] > 1]
        print(f"[{name} / {col}] 조합 {len(g):,}  다값 {len(mu):,} "
              f"({100*len(mu)/len(g):.2f}%)  "
              f"다값행 {mu['size'].sum():,} ({100*mu['size'].sum()/len(df):.2f}%)")
        if len(mu):
            print(f"    sd 중앙 {mu['std'].median():.4f}  최대 {mu['std'].max():.4f}")
    print()

# 카운트가 WE를 바꾸는가 (상태 고정하고 카운트만 변화)
g = df.groupby(FULL, observed=True)["home_win_expectancy"].nunique()
print(f"카운트 포함 여부: 상태만으로 WE 고유값 1인 조합 비율 "
      f"{100*(g==1).mean():.2f}%")

# 시즌/팀이 남은 분산을 설명하나
sub = df.groupby(FULL, observed=True)["home_win_expectancy"].transform("nunique")
rest = df[sub > 1]
if len(rest):
    print(f"\n남은 다값 행 {len(rest):,}")
    r = rest.copy()
    r["_dev"] = r["home_win_expectancy"] - r.groupby(FULL, observed=True)[
        "home_win_expectancy"].transform("mean")
    for c in ("season", "pitcher_team_id", "batter_team_id",
              "balls_before", "strikes_before"):
        e = r.groupby(c)["_dev"].mean()
        print(f"  {c:<18} 잔차평균 범위 {e.min():+.4f}~{e.max():+.4f}  "
              f"sd {e.std():.4f}")
