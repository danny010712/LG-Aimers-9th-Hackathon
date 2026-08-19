"""피처 엔지니어링 — 학습(train_local.py)과 추론(script.py)이 공유한다.

각 행은 그 행의 정보만으로 파생된다 (test 내부 행 간 통계 사용 금지 규정 준수).
채택 근거는 08 문서 §3-E. 중요도 하위권으로 확인된 주자/압박 플래그는 넣지 않는다.

=== EXTRA_FE 토글 추가 (2026-08) ===
개인 노트북(Train812.ipynb) feature importance 탐색에서 나온 후보 5종을
`extra_fe=True`로 켜면 추가할 수 있다. ⚠️ 08문서 §6-I "FE 추가 8종"이
전부 실패(노이즈 또는 악화)했던 전례가 있으므로, 이번에도 반드시
out-of-year로 단일 변수 검증할 것 — 켜고 끄는 게 즉시 가능하도록
기존 2-column 스무딩과 분리해서 별도 분기로 구현했다.

추가되는 것:
  - 10-column 베이지안 스무딩 (기존 2개 → 10개, SHRINK_PAIRS)
  - p_matchup      : log5 공식 (세이버메트릭스)
  - is_abs         : season>=2024 이진 플래그 (ABS 도입 여부)
  - pitchmix_entropy / pitchmix_concentration : 구종비율 Shannon 엔트로피 / Herfindahl 집중도

⚠️ rate_means(10개 컬럼 각각의 평균)는 **학습 구간에서 미리 계산해 전달**해야
   한다(global_mean과 같은 성격 — test에서 재계산하면 규정 위반). None으로
   두면 global_mean으로 전부 대체하지만, 이건 임시 fallback일 뿐 — 실제
   실행에서는 항상 compute_rate_means()로 만든 dict를 넘길 것.
"""
import numpy as np

# 저cardinality 범주형만. 선수 ID는 범주형으로 주면 과적합
BASE_CAT = ["top_bottom", "game_type", "base_state", "pitcher_hand",
            "batter_hand", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT + ["count_state"]

# 원본 2개(success_rate만) -> 10개로 확장.
# offspeed/breaking/fastball은 asof_pitcher_pitchmix_n을 쓴다(09문서 §1-B에서
# pitchmix_n == n임이 확인됐지만, 의미상 맞는 분모를 그대로 쓴다).
SHRINK_PAIRS = [
    ("asof_pitcher_success_rate", "asof_pitcher_n"),
    # ("asof_pitcher_reverse_rate", "asof_pitcher_n"),
    # ("asof_pitcher_middle_rate", "asof_pitcher_n"),
    # ("asof_pitcher_ball_rate", "asof_pitcher_n"),
    # ("asof_pitcher_strike_rate", "asof_pitcher_n"),
    ("asof_batter_success_rate", "asof_batter_n"),
    # ("asof_batter_middle_rate", "asof_batter_n"),
    # ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
    # ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
    # ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"),
]


def compute_rate_means(df, rate_pairs=SHRINK_PAIRS):
    """SHRINK_PAIRS 각 rate 컬럼 자신의 평균을 prior로 계산.
    반드시 학습 구간(예: tr = season<=2023)에서만 호출할 것 — 검증 누수 방지."""
    return {rate_col: float(df[rate_col].mean(skipna=True))
            for rate_col, _ in rate_pairs}


def engineer(df, global_mean, m=30, extra_fe=False, rate_means=None):

    d = df.copy()

    if extra_fe:
        # --- 10-column 베이지안 스무딩 (2-column을 대체) ---
        rate_means = rate_means or {}
        for rate_col, n_col in SHRINK_PAIRS:
            mean_val = rate_means.get(rate_col, global_mean)
            n = d[n_col].fillna(0)
            rate = d[rate_col].fillna(mean_val)
            d[f"smoothed_{rate_col}"] = (n * rate + m * mean_val) / (n + m)
    else:
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
    d["is_home"] = (d["top_bottom"] == "T").astype(int)
    d["pitcher_win_expectancy"] = np.where(d["is_home"] == 1,
                                           d["home_win_expectancy"],
                                           d["away_win_expectancy"])

    # 6) cold-start 자체를 신호로
    d["is_coldstart_pitcher"] = d["asof_pitcher_n"].isna().astype(int)

    if extra_fe:
        # # 7) ABS(자동 볼 판정 시스템) 도입 여부 — 2024시즌부터 KBO 전면 도입.
        # #    season을 범주형으로 주는 것(08 §6-I, 604.8로 대붕괴 확인)과는
        # #    메커니즘이 다르다 — 이진값이라 2025도 학습에서 이미 본 값(1)이라
        # #    미관측 카테고리 문제가 안 생긴다. 다만 이 프로젝트 문서 안에서
        # #    별도로 out-of-year 검증된 적은 없다 — 반드시 검증할 것.
        # d["is_abs"] = (d["season"] >= 2024).astype(int)

        # 8) p_matchup — log5 공식(세이버메트릭스). 08문서 "매치업 심화"(766.0,
        #    악화)와 이름·주제는 겹치나 정확히 동일 구현인지는 불확실.
        p_p = d[f"smoothed_asof_pitcher_success_rate"]
        p_b = d[f"smoothed_asof_batter_success_rate"]
        d["p_matchup"] = (p_p * p_b) / (p_p * p_b + (1 - p_p) * (1 - p_b) + 1e-12)

        # # 9-10) 구종믹스 엔트로피 / 집중도. 08문서 "구종믹스 편중"(776.1, 악화)과
        # #       이름·주제는 겹치나 정확히 동일 구현인지는 불확실.
        # p_fast = d["asof_pitcher_fastball_rate"].fillna(0).clip(0, 1)
        # p_break = d["asof_pitcher_breaking_rate"].fillna(0).clip(0, 1)
        # p_off = d["asof_pitcher_offspeed_rate"].fillna(0).clip(0, 1)
        # total_rate = p_fast + p_break + p_off + 1e-9
        # p_fast, p_break, p_off = p_fast / total_rate, p_break / total_rate, p_off / total_rate

        # eps = 1e-12
        # h_fast = np.where(p_fast > 0, -p_fast * np.log2(p_fast + eps), 0)
        # h_break = np.where(p_break > 0, -p_break * np.log2(p_break + eps), 0)
        # h_off = np.where(p_off > 0, -p_off * np.log2(p_off + eps), 0)
        # d["pitchmix_entropy"] = h_fast + h_break + h_off
        # d["pitchmix_concentration"] = p_fast ** 2 + p_break ** 2 + p_off ** 2

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
