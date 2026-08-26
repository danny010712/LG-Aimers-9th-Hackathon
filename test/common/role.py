"""투수 역할 표 — 등판당 투구수. 학습(train_local.py)과 추론(script.py)이 공유한다.

`asof_pitcher_n`은 **통산 누적**만 준다. 여기서는 **경기당 소화량**을 만든다 —
경기 경계를 알아야 나오는 값이고, 그건 train 행을 봐야 계산된다.
모델이 스스로 도달할 수 없는 값이라는 점에서 시즌내 분해(08 §5-10)와 같은 구조다.

⚠️ 시점 규칙 (cond.py와 동일, leak 방지):
  학습 행: 시즌 S의 행에는 **S 미만 시즌**으로 만든 표를 붙인다.
  추론 행: 2025 test에는 **train 전체(2019~2024)** 로 만든 표를 붙인다.

역할 판별력 확인 (2026-08-25): 등판당 투구수 중앙 20 / 25% 16 / 75% 48 / 최대 99.
표 자체의 지속성 ≤2023 vs 2024 corr **+0.741** (09 §2-P).
"""
import numpy as np
import pandas as pd

# 경기 식별 대용. 실제 game_id가 없어 (시즌·월·요일·양팀)으로 대신한다.
GKEY = ["season", "game_month", "game_dayofweek", "pitcher_team_id", "batter_team_id"]
MIN_GAMES = 5                    # 등판 5경기 미만은 표에 넣지 않는다 (중앙값이 못 믿음)
ROLE_COLS = ["role_ppg"]


def build_table(hist):
    """hist(과거 행)에서 투수별 등판당 투구수 중앙값."""
    h = hist.copy()
    h["_g"] = h[GKEY].astype(str).agg("|".join, axis=1)
    per = h.groupby(["pitcher_id", "_g"]).size().rename("n").reset_index()
    t = per.groupby("pitcher_id").agg(role_ppg=("n", "median"), _games=("n", "size"))
    return t[t["_games"] >= MIN_GAMES][["role_ppg"]].reset_index()


def apply_table(d, table):
    """표를 조회해 role_ppg를 붙인다. 없는 투수는 NaN(CatBoost가 결측으로 처리)."""
    return d[["pitcher_id"]].merge(table, on="pitcher_id", how="left")["role_ppg"].values


def build_training_columns(df):
    """학습용: 시즌마다 '그 이전 시즌들'로 만든 표를 적용한 role_* 열만 반환."""
    out = pd.DataFrame(index=df.index, columns=ROLE_COLS, dtype=float)
    for s in sorted(df["season"].unique()):
        hist = df[df["season"] < s]
        if len(hist) == 0:
            continue                     # 첫 시즌은 참조할 과거가 없다 -> NaN
        t = build_table(hist)
        m = (df["season"] == s).values
        out.loc[m, "role_ppg"] = apply_table(df.loc[m], t)
    return out
