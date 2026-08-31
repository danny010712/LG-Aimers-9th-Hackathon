"""2단계 사전 점검 — "과거 시즌별 평균 → 추세 기울기 하나"가 미래 성적에
원시 상관을 갖는지, 그리고 그게 (a) 통산평균 위에, (b) 이미 강한 축인
'이번 시즌 전반' 위에 각각 추가 정보를 주는지 확인한다.

probe_2024_form_validation.py와 같은 골격 — 2024를 전반/후반으로 쪼개서
후반을 '아직 안 온 미래'로 취급한다(train.csv라 우리만 정답을 알고 있음).

사용: python probe_season_trend_signal.py <train.csv 경로>
"""
import sys
import numpy as np
import pandas as pd

MIN_2024_PITCHES = 60
MIN_N_PER_SEASON = 30
PRIOR_SEASONS = [2019, 2020, 2021, 2022, 2023]


def season_trend_slope(season_avgs):
    """(season, avg) 쌍 2개 이상 있으면 선형회귀 기울기, 아니면 NaN."""
    valid = season_avgs.dropna()
    if len(valid) < 2:
        return np.nan
    x = valid.index.values.astype(float)
    y = valid.values.astype(float)
    slope, _ = np.polyfit(x, y, 1)
    return slope


def r2(y, pred):
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def fit_r2(X_cols, y):
    X = np.column_stack([np.ones(len(y))] + X_cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    return r2(y, pred), coef


def run(train_path):
    df = pd.read_csv(train_path)
    print(f"train.csv 로드: {len(df):,}행")

    d24 = df[df["season"] == 2024].sort_values(["pitcher_id", "row_id"]).copy()
    if "asof_pitcher_n" in d24.columns:
        d24 = df[df["season"] == 2024].sort_values(["pitcher_id", "asof_pitcher_n"]).copy()
    counts = d24.groupby("pitcher_id").size()
    eligible = counts[counts >= MIN_2024_PITCHES].index
    print(f"2024 {MIN_2024_PITCHES}구 이상 던진 투수: {len(eligible):,}명")
    d24e = d24[d24["pitcher_id"].isin(eligible)]

    hist = df[df["season"] <= 2023]
    season_pn = hist.groupby(["pitcher_id", "season"]).agg(
        n=("control_success", "count"), avg=("control_success", "mean"))

    rows = []
    for pid, g in d24e.groupby("pitcher_id"):
        half = len(g) // 2
        first, second = g.iloc[:half], g.iloc[half:]

        career_n = hist[hist["pitcher_id"] == pid]
        career_rate = career_n["control_success"].mean() if len(career_n) else np.nan

        avgs = pd.Series(index=PRIOR_SEASONS, dtype=float)
        for s in PRIOR_SEASONS:
            if (pid, s) in season_pn.index and season_pn.loc[(pid, s), "n"] >= MIN_N_PER_SEASON:
                avgs[s] = season_pn.loc[(pid, s), "avg"]
        slope = season_trend_slope(avgs)

        rows.append(dict(
            pitcher_id=pid, n_first=len(first), n_second=len(second),
            first_half_rate=first["control_success"].mean(),
            second_half_rate=second["control_success"].mean(),
            career_rate=career_rate,
            n_seasons_avail=avgs.notna().sum(),
            trend_slope=slope,
        ))
    perf = pd.DataFrame(rows)
    print(f"대상 투수 수: {len(perf):,}  (기울기 계산 가능한 투수: "
          f"{perf['trend_slope'].notna().sum():,})")

    has_slope = perf[perf["trend_slope"].notna()].copy()
    print(f"\n기울기 있는 투수만으로 진행: {len(has_slope):,}명")
    if len(has_slope) < 10:
        print("⚠️ 표본이 너무 적어 회귀 비교가 불안정할 수 있음 — 참고만 할 것")

    y = has_slope["second_half_rate"].values
    career = has_slope["career_rate"].fillna(has_slope["career_rate"].mean()).values
    first = has_slope["first_half_rate"].values
    slope = has_slope["trend_slope"].values

    print("\n=== (a) 통산평균 대비 추가 정보 ===")
    r2_career, _ = fit_r2([career], y)
    r2_career_slope, coef = fit_r2([career, slope], y)
    print(f" R²(통산평균만): {r2_career:.4f}")
    print(f" R²(통산평균+추세기울기): {r2_career_slope:.4f}  "
          f"(기울기 계수: {coef[2]:+.4f})")

    print("\n=== (b) 이번 시즌 전반(이미 강한 축) 대비 추가 정보 — 더 엄격한 기준 ===")
    r2_first, _ = fit_r2([first], y)
    r2_first_slope, coef2 = fit_r2([first, slope], y)
    print(f" R²(이번시즌 전반만): {r2_first:.4f}")
    print(f" R²(이번시즌 전반+추세기울기): {r2_first_slope:.4f}  "
          f"(기울기 계수: {coef2[2]:+.4f})")

    print("\n판정: (b)에서 R²가 뚜렷하게 안 오르면(예: <+0.01), 이미 도입된")
    print("      시즌내 분해가 추세 정보 대부분을 이미 흡수하고 있다는 뜻 —")
    print("      본 실험(6~12컬럼 실제 학습) 비용 대비 기대값이 낮다고 판단.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
    run(path)
