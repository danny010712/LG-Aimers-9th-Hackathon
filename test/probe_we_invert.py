"""home_win_expectancy 역산 — 팀 전력이 들어 있나?

first_game.csv 첫 투구가 정확히 50.0/50.0 이었다. 홈 어드밴티지도 팀 전력도
안 들어간 중립 표라는 뜻이다. 그게 **모든 경기**에서 그런지 확인한다.

경기 시작 상태 = inning 1 · top_bottom T · outs 0 · 주자 0 · 점수 0-0.
이 상태에서 WE가 경기마다 다르면 -> 상태 밖 정보(팀 전력/순위)가 들어 있다.
같으면 -> 순수 상태 함수 = 기존 컬럼의 결정론적 재조합 = 정보 증가 0.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd

C = ["season", "game_month", "game_dayofweek", "game_type", "inning",
     "top_bottom", "outs_before", "balls_before", "strikes_before",
     "num_runners_on", "run_top_before", "run_bot_before",
     "score_diff_home", "home_win_expectancy", "away_win_expectancy", "li",
     "pitcher_team_id", "batter_team_id", "control_success"]
df = pd.read_csv("data/train.csv", encoding="utf-8-sig", usecols=C)
print(f"rows={len(df):,}", flush=True)

# 1) home + away = 100 인가 (중복열 확인)
s = df["home_win_expectancy"] + df["away_win_expectancy"]
print(f"[1] home+away  평균 {s.mean():.4f}  최소 {s.min():.2f}  최대 {s.max():.2f}  "
      f"!=100 인 행 {(np.abs(s-100)>1e-6).sum():,}")

# 2) 경기 시작 상태만 뽑기
m = ((df.inning == 1) & (df.top_bottom == "T") & (df.outs_before == 0)
     & (df.balls_before == 0) & (df.strikes_before == 0)
     & (df.num_runners_on == 0) & (df.run_top_before == 0)
     & (df.run_bot_before == 0))
st = df[m]
print(f"\n[2] 경기 시작 상태 행 {len(st):,}개")
v = st["home_win_expectancy"]
print(f"    WE 고유값 {v.nunique()}개  범위 {v.min():.2f}~{v.max():.2f}  "
      f"sd {v.std():.4f}")
print(v.value_counts().head(10).to_string())

# 3) 상태가 완전히 같은데 WE가 다른 사례가 있나 (= 상태 밖 정보)
KEY = ["inning", "top_bottom", "outs_before", "balls_before", "strikes_before",
       "num_runners_on", "run_top_before", "run_bot_before", "score_diff_home"]
g = df.groupby(KEY, observed=True)["home_win_expectancy"].agg(["nunique", "std", "size"])
multi = g[g["nunique"] > 1]
print(f"\n[3] 상태조합 {len(g):,}개 중 WE가 여러 값인 조합 {len(multi):,}개 "
      f"({100*len(multi)/len(g):.1f}%)")
if len(multi):
    print(f"    그 조합들의 WE sd  중앙 {multi['std'].median():.4f}  "
          f"90분위 {multi['std'].quantile(.9):.4f}  최대 {multi['std'].max():.4f}")
    print(multi.sort_values("size", ascending=False).head(8).to_string())

# 4) base_state까지 넣어도 남나 (주자 배치 차이가 원인일 수 있다)
KEY2 = KEY + ["num_runners_on"]
g2 = df.groupby(KEY2 + ["run_top_before", "run_bot_before"], observed=True
                )["home_win_expectancy"].nunique()
print(f"\n[4] 점수 원본까지 넣은 조합 {len(g2):,}개 중 다값 "
      f"{(g2>1).sum():,} ({100*(g2>1).mean():.1f}%)")
