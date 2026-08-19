"""팀 단위 baseline 세분화 — 전제 확인 스크립트 (09문서 §0 규칙).

모델을 만들기 전에 pandas만으로 세 가지를 확인한다:
  1. season×game_type 안에서 팀 간 평균 성공률 차이가 실제로 있는가
     (있어도 너무 작으면 baseline으로 쓸 이유가 없음)
  2. 그 차이가 "우연한 노이즈"가 아니라 "팀의 지속적인 특성"인가
     → 한 시즌에 평균보다 높았던 팀이, 다음 시즌에도 높은 경향이 있는지
     (순위 상관계수로 확인 — 상관 없으면 그냥 매 시즌 랜덤하게 흔들리는 것)
  3. season×game_type×team 셀마다 표본 수가 얼마나 되는지
     (너무 적으면 베이지안 스무딩이 필수 — 그 강도를 얼마로 할지 감 잡기 위함)

실행: test/ 폴더에서 python precheck_team_baseline.py
"""
import pandas as pd
import numpy as np

DATA = "data/train.csv"
TARGET = "control_success"


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    print(f"전체 행 수: {len(df):,}")
    # print(f"고유 pitcher_team_id 수: {df['pitcher_team_id'].nunique()}")
    print(f"고유 batter_team_id 수: {df['batter_team_id'].nunique()}")
    print()

    # ---- 1. season x game_type x team 별 평균 ----
    # g = df.groupby(["season", "game_type", "pitcher_team_id"])
    g = df.groupby(["season", "game_type", "batter_team_id"])
    table = g[TARGET].agg(["mean", "count"]).reset_index()
    table.columns = ["season", "game_type", "team", "rate", "n"]

    print("=" * 70)
    print("1) season x game_type 안에서 팀 간 편차 (팀 평균 - 그 그룹 전체 평균)")
    print("=" * 70)
    for (season, gt), sub in table.groupby(["season", "game_type"]):
        grp_overall = df[(df["season"] == season) & (df["game_type"] == gt)][TARGET].mean()
        dev = sub["rate"] - grp_overall
        print(f" {season} {gt}: 그룹평균={grp_overall:.4f}  "
              f"팀별 편차 범위=[{dev.min():+.4f}, {dev.max():+.4f}]  "
              f"편차 표준편차={dev.std():.4f}  (팀 수={len(sub)})")

    print()
    print("=" * 70)
    print("2) 팀 효과의 시즌 간 지속성 (순위 상관계수, Spearman)")
    print("   -> 1에 가까울수록 '팀 고유 특성', 0에 가까우면 '그냥 노이즈'")
    print("=" * 70)
    for gt in df["game_type"].unique():
        pivot = table[table["game_type"] == gt].pivot(
            index="team", columns="season", values="rate")
        seasons = sorted(pivot.columns)
        for i in range(len(seasons) - 1):
            s1, s2 = seasons[i], seasons[i + 1]
            pair = pivot[[s1, s2]].dropna()
            if len(pair) >= 3:
                corr = pair[s1].rank().corr(pair[s2].rank())
                print(f" [{gt}] {s1}->{s2} 순위상관: {corr:+.3f}  (팀 수={len(pair)})")

    print()
    print("=" * 70)
    print("3) 셀별 표본 수 분포 (season x game_type x team)")
    print("=" * 70)
    print(table["n"].describe())
    print()
    print(f" 표본 100 미만인 셀 비율: {(table['n'] < 100).mean()*100:.1f}%")
    print(f" 표본 500 미만인 셀 비율: {(table['n'] < 500).mean()*100:.1f}%")

    print()
    print("=" * 70)
    print("참고: batter_team_id 기준도 동일하게 보고 싶으면 아래 주석 해제")
    print("=" * 70)
    # (pitcher_team_id를 batter_team_id로 바꿔서 위 로직 재사용하면 됨)


if __name__ == "__main__":
    main()
