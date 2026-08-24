"""피처 엔지니어링 — 학습(train_local.py)과 추론(script.py)이 공유한다.

각 행은 그 행의 정보만으로 파생된다 (test 내부 행 간 통계 사용 금지 규정 준수).
채택 근거는 08 문서 §3-E. 중요도 하위권으로 확인된 주자/압박 플래그는 넣지 않는다.

=== 시즌내 성적 분해 (08 §5-10) ===
`asof_*`는 시즌 리셋이 없는 **통산** 누적이라 리그 하락 추세만큼 항상 위로 치우친다
(2024 실측: 통산 .5114 vs 실제 .4861, 편차 +.0253). 통산은 누적값이므로
**직전 시즌 말 기준점을 빼면 그 시즌 성적만 남는다**:

    시즌내 = (n₁·r₁ − s₀) / (n₁ − n₀)      n₁,r₁ = 그 행의 asof / n₀,s₀ = 기준점

편차가 +.0253 → +.0033으로 줄고, 1변수 예측력이 (평균보정 후) 238 → 361이 된다.
모델은 이걸 스스로 못 만든다 — n₀,s₀가 어느 컬럼에도 없고 선수 ID를 조회키로
쓰는 건 금지돼 있다(§3). 즉 **기존 피처가 담지 못한 새 정보**다.

⚠️ 규정: 쓰는 것은 (a) 그 행 자신의 `asof_*` (b) 학습 데이터로 만든 기준점 표뿐이다.
   test의 다른 행은 0개 쓴다 — 한 행만 줘도 계산된다. 기준점 표는 `global_mean`·`mu`와
   같은 성격으로 학습 때 만들어 zip에 싣는다.
"""
import numpy as np
import pandas as pd

# 저cardinality 범주형만. 선수 ID는 범주형으로 주면 과적합
BASE_CAT = ["top_bottom", "game_type", "base_state", "pitcher_hand",
            "batter_hand", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT + ["count_state"]


ANCHOR_WHO = ("pitcher", "batter")


def build_anchor(df, target="control_success"):
    """시즌내 분해용 기준점 표. 반환 열: [apply_season, who, id, n0, s0].

    `apply_season = S` 행에는 **S−1 시즌 말** 통산을 붙인다. 그 기준점은 정의상
    season ≤ S−1 데이터로만 만들어지므로 누수가 구조적으로 불가능하다.
    배포용으로 마지막 시즌+1(=2025) 행도 만든다 — 그때 기준점은 train 전체다.

    df에는 season / {who}_id / asof_{who}_n / asof_{who}_success_rate / target 필요.
    """
    seasons = sorted(df["season"].unique())
    out = []
    for who in ANCHOR_WHO:
        idc, nc = f"{who}_id", f"asof_{who}_n"
        rc = f"asof_{who}_success_rate"
        for S in seasons[1:] + [seasons[-1] + 1]:
            prev = df[df["season"] <= S - 1]
            # 통산 n이 가장 큰 행 = 그 선수의 마지막 투구. 그 행의 asof는 '직전'
            # 값이므로 그 투구 자체(+1구, +결과)를 더해야 시즌 말 통산이 된다.
            last = prev.sort_values(nc).groupby(idc).tail(1)
            out.append(pd.DataFrame({
                "apply_season": S, "who": who, "id": last[idc].values,
                "n0": last[nc].values + 1,
                "s0": last[nc].values * last[rc].values + last[target].values,
            }))
    return pd.concat(out, ignore_index=True)


def _add_inseason(d, anchor, global_mean, m):
    """d에 시즌내 분해 열 4개를 붙인다. anchor는 build_anchor()의 반환값."""
    for who in ANCHOR_WHO:
        a = anchor[anchor["who"] == who]
        key = pd.MultiIndex.from_arrays([d["season"].values, d[f"{who}_id"].values])
        src = pd.MultiIndex.from_arrays([a["apply_season"].values, a["id"].values])
        n0 = pd.Series(a["n0"].values, index=src).reindex(key).fillna(0).values
        s0 = pd.Series(a["s0"].values, index=src).reindex(key).fillna(0).values

        n1 = d[f"asof_{who}_n"].fillna(0).values
        career = d[f"asof_{who}_success_rate"].fillna(global_mean).values
        # 기준점이 없는 신인은 n0=s0=0 → dn=n1, ds=n1·r₁ → 시즌내 = 통산.
        # 실제로 그게 정답이다(데뷔 시즌이면 통산이 곧 그 시즌 성적).
        dn = np.maximum(n1 - n0, 0)
        ds = np.clip(n1 * career - s0, 0, dn)
        ins = np.where(dn > 0, ds / np.where(dn > 0, dn, 1), career)
        # 통산 쪽으로 스무딩 — 시즌 초 표본이 얇을 때 분산을 억제한다(§6-M의 교훈:
        # 이 데이터에서 개인기록은 신선도보다 표본 수가 지배한다).
        d[f"ins_{who}_success_rate"] = (dn * ins + m * career) / (dn + m)
        d[f"ins_{who}_n"] = dn
    return d


def engineer(df, global_mean, m=30, anchor=None):

    d = df.copy()

    # 0) 시즌내 성적 분해 (anchor가 있을 때만 — 없으면 이전과 완전히 동일)
    if anchor is not None:
        d = _add_inseason(d, anchor, global_mean, m)

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
