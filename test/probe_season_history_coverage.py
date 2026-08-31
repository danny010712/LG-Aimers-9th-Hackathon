"""1단계 사전 점검 — 시즌별 개별 컬럼(2019~2023) 아이디어의 전제 조건인
"충분한 과거 시즌 커버리지"가 실제로 있는지 확인한다.

이게 낮으면(가능성 높음) 컬럼을 6~12개 만들어도 대부분 결측이라 애초에
승산이 낮다는 뜻 — 본 실험(코드 학습)으로 넘어가기 전에 걸러낸다.

사용: python probe_season_history_coverage.py <train.csv 경로>
"""
import sys
import numpy as np
import pandas as pd

MIN_N_PER_SEASON = 30   # 이 정도는 던져야 "그 시즌 평균"이 의미 있다고 봄
PRIOR_SEASONS = [2019, 2020, 2021, 2022, 2023]   # 2024 예측 기준


def run(train_path):
    df = pd.read_csv(train_path)
    print(f"train.csv 로드: {len(df):,}행")

    # 2024에 등판한 투수들 (검증 fold의 실제 대상)
    p2024 = df[df["season"] == 2024]
    pitchers_2024 = p2024["pitcher_id"].unique()
    n2024_by_pitcher = p2024.groupby("pitcher_id").size()
    print(f"2024 등판 투수 수: {len(pitchers_2024):,}  (2024 총 투구수 {len(p2024):,})")

    # 투수별 시즌별 투구 수
    counts = df[df["pitcher_id"].isin(pitchers_2024)].groupby(
        ["pitcher_id", "season"]).size().unstack(fill_value=0)
    for s in PRIOR_SEASONS:
        if s not in counts.columns:
            counts[s] = 0
    counts = counts[PRIOR_SEASONS]

    has_enough = counts >= MIN_N_PER_SEASON
    n_seasons_covered = has_enough.sum(axis=1)

    print(f"\n=== (투수 수 기준) {MIN_N_PER_SEASON}구 이상 확보한 과거 시즌 개수 분포 ===")
    dist = n_seasons_covered.value_counts().sort_index()
    for k, v in dist.items():
        print(f" {k}/5개 시즌 확보: {v:,}명 ({v/len(counts):.1%})")

    # 투수 수가 아니라 '2024 행 수'로 가중 — 실제 검증/학습에 얼마나 영향 주는지
    weighted = n_seasons_covered.reindex(n2024_by_pitcher.index).fillna(0)
    w = n2024_by_pitcher.values
    print(f"\n=== (2024 행 수 가중) 과거 시즌 확보 개수별 행 비율 ===")
    for k in range(6):
        mask = weighted == k
        n_rows = w[mask.values].sum() if mask.any() else 0
        print(f" {k}/5개 시즌 확보: {n_rows:,}행 ({n_rows/w.sum():.1%})")

    full5 = (n_seasons_covered == 5).sum()
    print(f"\n5개 시즌 전부 확보한 투수: {full5:,} / {len(counts):,} "
          f"({full5/len(counts):.1%})")
    print(f"\n판정 기준: 이 비율이 낮으면(예: <30%) 개별 시즌 컬럼 6~12개 대부분이")
    print(f"결측 처리될 행이 많다는 뜻 — 본 실험 전에 신중히 볼 것.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
    run(path)
