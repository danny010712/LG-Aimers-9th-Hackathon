"""피처 엔지니어링 — 학습(train_local.py)과 추론(script.py)이 공유한다.

각 행은 그 행의 정보만으로 파생된다 (test 내부 행 간 통계 사용 금지 규정 준수).
채택 근거는 08 문서 §3-E. 중요도 하위권으로 확인된 주자/압박 플래그는 넣지 않는다.
"""
import numpy as np

# 저cardinality 범주형만. 선수 ID(792/830종)는 범주형으로 주면 과적합
# (LightGBM 612.9→199.5, CatBoost 780.0→647.3). 수치형으로 남긴다.
BASE_CAT = ["top_bottom", "game_type", "base_state", "pitcher_hand",
            "batter_hand", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT + ["count_state"]


def engineer(df, global_mean, m=30):
    """원본 열은 하나도 버리지 않고 파생 열만 추가한다."""
    d = df.copy()

    # 1) 베이지안 스무딩 — 표본 적은 투수/타자를 리그평균 쪽으로 당겨 cold-start 완화
    for who in ("pitcher", "batter"):
        n = d[f"asof_{who}_n"].fillna(0)
        rate = d[f"asof_{who}_success_rate"].fillna(global_mean)
        d[f"smoothed_{who}_success_rate"] = (n * rate + m * global_mean) / (n + m)

    # 2) 좌우 상성 (같은 손 = 투수 유리)
    d["platoon_advantage"] = (d["pitcher_hand"] == d["batter_hand"]).astype(int)

    # 3) 볼카운트
    d["count_advantage"] = d["strikes_before"] - d["balls_before"]
    d["count_state"] = (d["balls_before"].astype(str) + "-"
                        + d["strikes_before"].astype(str))

    # 4) 최근 폼 - 통산 폼
    d["recent_control_momentum"] = (d["asof_pitcher_prev1_game_success_rate"]
                                    - d["asof_pitcher_success_rate"])
    d["form_trend_5_1"] = (d["asof_pitcher_prev1_game_success_rate"]
                           - d["asof_pitcher_prev5_game_success_rate"])

    # 5) 홈 여부: 홈팀이 초(T)에 던진다.
    #    score_diff 비교로 구하면 동점(전체 25.6%)일 때 항상 홈이 되는 버그.
    d["is_home"] = (d["top_bottom"] == "T").astype(int)
    d["pitcher_win_expectancy"] = np.where(d["is_home"] == 1,
                                           d["home_win_expectancy"],
                                           d["away_win_expectancy"])

    # 6) cold-start 자체를 신호로
    d["is_coldstart_pitcher"] = d["asof_pitcher_n"].isna().astype(int)
    return d


def prepare(df, feature_cols, cat_cols):
    """모델 입력 행렬. 학습·추론이 반드시 같은 변환을 쓰도록 여기 한 곳에 둔다."""
    import pandas as pd
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)          # NaN → "nan" 문자열로 통일
    for c in feature_cols:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X
