"""pitch_of_pa 기반 투수 작업량(fatigue) 피처 — 전제 확인 (09문서 §0 규칙).

pitcher_id 매핑과 무관하게, trackman_history.csv 안에서만 먼저 확인한다:
  1. 투수별 "경기당 평균 투구수"가 실제로 유의미하게 다른가
  2. 투수별 "타자 한 명당 평균 투구수(PA당)"가 실제로 다른가
  3. 그 차이가 시즌이 바뀌어도 유지되는 "투수 고유 특성"인가, 아니면 노이즈인가
     (팀 baseline 전제확인 때와 같은 방식 — 순위상관)

pitch_no(경기 전체 투구수 누적, 진짜 순차적)를 정렬 기준으로 쓰고,
pitch_of_pa==1인 지점을 "새 타석 시작"으로 보고 타석 경계를 잡는다.

실행: trackman_history.csv를 test/ 폴더(또는 경로 지정)에 두고
python precheck_pitch_workload.py
"""
import pandas as pd
import numpy as np

DATA = "data/trackman_history.csv"   # 실제 경로에 맞게 수정


def compute_pa_lengths(g):
    """한 (pitcher_trackman_id, trackman_game_id) 그룹 안에서, pitch_no로
    정렬한 뒤 pitch_of_pa==1을 타석 시작으로 보고 각 타석의 길이(투구수)를 센다."""
    g = g.sort_values("pitch_no")
    pa_id = (g["pitch_of_pa"] == 1).cumsum()
    lengths = g.groupby(pa_id).size()
    return lengths


def main():
    tm = pd.read_csv(DATA, encoding="utf-8-sig")
    print(f"전체 행 수: {len(tm):,}")
    print(f"고유 pitcher_trackman_id 수: {tm['pitcher_trackman_id'].nunique()}")
    print(f"고유 (pitcher, game) 조합 수: "
          f"{tm[['pitcher_trackman_id','trackman_game_id']].drop_duplicates().shape[0]:,}")
    print()

    # ---- 1) 경기당 투구수 (game workload) ----
    # ⚠️ pitch_no는 "이 투수의" 투구 번호가 아니라 "이 경기 전체(양 팀 합산)"
    # 투구 번호다(data_description.md: "경기 내 투구 번호"). max(pitch_no)를
    # 쓰면 그 투수의 투구수가 아니라 그 투수가 등판할 때까지 경기 전체에서
    # 이미 던져진 누적 투구수가 섞여 나온다(실측: 평균 227, 최대 475 —
    # 실제 투수 1명의 경기당 투구수로는 비현실적으로 높음). 올바른 방법은
    # 그 (투수,경기) 조합에 속한 "행 개수"를 세는 것 — 각 행이 그 투수가
    # 던진 공 하나씩이므로.
    game_pitch = tm.groupby(["pitcher_trackman_id", "trackman_game_id", "season"]).size()
    game_pitch = game_pitch.reset_index(name="pitches_in_game")

    print("=" * 70)
    print("1) 경기당 투구수 분포")
    print("=" * 70)
    print(game_pitch["pitches_in_game"].describe())
    print()

    # 투수별 시즌 평균
    pitcher_season = game_pitch.groupby(["pitcher_trackman_id", "season"])["pitches_in_game"].agg(["mean", "count"])
    pitcher_season = pitcher_season.reset_index()
    overall_std = pitcher_season["mean"].std()
    print(f"투수x시즌 평균 투구수의 전체 표준편차: {overall_std:.2f}")
    print()

    # ---- 2) 순위상관 (투수 고유 특성인지) ----
    print("=" * 70)
    print("2) 경기당 투구수 - 투수 효과의 시즌 간 지속성 (순위 상관계수)")
    print("=" * 70)
    pivot = pitcher_season.pivot(index="pitcher_trackman_id", columns="season", values="mean")
    seasons = sorted(pivot.columns)
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        pair = pivot[[s1, s2]].dropna()
        if len(pair) >= 5:
            corr = pair[s1].rank().corr(pair[s2].rank())
            print(f" {s1}->{s2} 순위상관: {corr:+.3f}  (투수 수={len(pair)})")

    # ---- 3) 타석당 투구수 (효율성/지구력) ----
    print()
    print("=" * 70)
    print("3) 타석당 투구수(PA당) 분포 및 지속성")
    print("=" * 70)
    pa_records = []
    for (pid, gid), g in tm.groupby(["pitcher_trackman_id", "trackman_game_id"]):
        lengths = compute_pa_lengths(g)
        season = g["season"].iloc[0]
        for length in lengths:
            pa_records.append({"pitcher_trackman_id": pid, "season": season,
                              "pa_length": length})
    pa_df = pd.DataFrame(pa_records)
    print(pa_df["pa_length"].describe())
    print()

    pitcher_pa_season = pa_df.groupby(["pitcher_trackman_id", "season"])["pa_length"].agg(["mean", "count"])
    pitcher_pa_season = pitcher_pa_season.reset_index()
    pivot2 = pitcher_pa_season.pivot(index="pitcher_trackman_id", columns="season", values="mean")
    seasons2 = sorted(pivot2.columns)
    for i in range(len(seasons2) - 1):
        s1, s2 = seasons2[i], seasons2[i + 1]
        pair = pivot2[[s1, s2]].dropna()
        if len(pair) >= 5:
            corr = pair[s1].rank().corr(pair[s2].rank())
            print(f" [PA당] {s1}->{s2} 순위상관: {corr:+.3f}  (투수 수={len(pair)})")

    print()
    print("=" * 70)
    print("참고: pitcher_id <-> pitcher_trackman_id 매핑이 있으면, 이 결과가")
    print("      실제로 asof_pitcher_success_rate 등과 상관 있는지도 추가로")
    print("      확인 가능. 매핑 파일 있으면 알려줄 것.")
    print("=" * 70)


if __name__ == "__main__":
    main()
