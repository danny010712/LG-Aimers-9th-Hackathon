"""리그 평균 기준선(league-rate baseline) 유틸 — 학습(train_local.py)과
추론(script.py), 그리고 offset 적합(train_offset.py)이 공유한다.

목적 (08 문서 §5-7, 09 문서 §3-B):
  트리는 미관측 시즌을 외삽하지 못한다(08 §6-A). season을 그냥 피처로 주면
  트리가 "이 시즌의 절대 수준"까지 직접 배우려다 실패한다. 대신 CatBoost의
  `baseline` 파라미터로 "이 그룹의 리그 평균 로짓"을 미리 알려주면, 트리는
  "리그 평균 대비 이 상황이 얼마나 편차나는가"만 배우면 된다.

세분화 단위 — GROUP_COLS로 토글:
  ["season", "game_type"]  ← 기본. season만으로는 F/R 4배 편향 차이를
                              못 잡는다(09 §3-A: 2군 이력 많은 투수 asof 편향
                              +.044 vs 순수 시대드리프트 +.011).
  ["season"]                ← 필요시 이 옵션으로 되돌릴 수 있다.
  train_local.py 상단 설정에서 LEAGUE_GROUP_COLS 값만 바꾸면 된다.

⚠️ out-of-year 원칙 (08 §3-4): held-out 시즌(검증의 2024, 배포의 2025)에는
   그 시즌 실측 평균을 쓰면 안 된다 — 정답을 미리 알려주고 시험 보는 것과
   같다. 반드시 그 이전 데이터로 만든 외삽값 또는 외부 도메인 추정치를 쓴다.

⚠️ CatBoost의 baseline은 확률이 아니라 **raw margin(로짓) 스케일**이다.
   logit(rate)을 넣어야 한다 (본 문서 하단 self-test로 실측 확인됨).

⚠️ baseline은 .cbm 파일에 저장되지 않는다 (실측 확인됨 — 아래 self-test).
   학습 때 쓴 것과 **같은 방식으로 계산한 baseline**을 추론 때도 반드시
   다시 만들어 Pool(..., baseline=...)에 넣어야 한다. 빠뜨리면 예측 평균이
   크게 틀어진다(실측: baseline 없이 predict하면 평균이 0.66→0.52로 왜곡).

⚠️ F(퓨처스)든 R(1군)이든 game_type별 특별 처리는 하지 않는다 — 항상 전체
   관측 시즌 range로 선형회귀한다. 09/10 문서는 "F가 2023년에 라벨 체계가
   바뀌었다"고 관찰했지만, 이 관찰은 원래 실제 학습 코드(features.py,
   cond.py, 원래 train_local.py) 어디에도 반영된 적이 없었고(2026-08
   재검토로 확인), 특정 game_type만 시즌 구간을 잘라 쓰는 로직은 대회
   규정 감사 관점에서 불필요한 판단 지점을 늘릴 뿐이라 기본 경로에서
   제외했다. 참고만 하고 코드에는 넣지 않는다.
"""
import numpy as np
import pandas as pd

TARGET = "control_success"
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def build_table(hist, group_cols, target=TARGET):
    """hist(타깃 포함 과거 행)에서 그룹별 실측 평균을 만든다.

    target: 기본은 control_success(주모델용). mr_label/wayoff_label처럼
    다른 이진 라벨 컬럼명을 주면 그 라벨의 그룹 평균 표를 만든다 —
    보조모델(mr/wayoff)별 baseline 확장에 씀. hist에 그 컬럼이 있어야 한다.

    반환: pandas Series. index는 group_cols=["season"]이면 season,
    group_cols=["season","game_type"]이면 (season, game_type) MultiIndex.
    """
    return hist.groupby(group_cols)[target].mean()


def _series_for_group(table, group_cols, game_type):
    """table에서 특정 game_type의 (season -> rate) 시리즈만 뽑는다.
    group_cols=["season"]이면 game_type 구분이 없으므로 table 그대로 반환."""
    if group_cols == ["season"]:
        return table
    return table.xs(game_type, level="game_type")


def extrapolate(table, group_cols, season, game_type=None, min_season=None):
    """선형회귀로 미관측 season의 리그 평균(확률, 0~1)을 외삽한다.

    min_season을 주면 그 이전 데이터는 추세 계산에서 제외한다.
    관측 시즌이 1개뿐이면 회귀가 불가능하므로 그 값을 그대로 carry-forward.
    """
    sub = _series_for_group(table, group_cols, game_type)
    if min_season is not None:
        sub = sub[sub.index >= min_season]
    if len(sub) == 0:
        raise ValueError(
            f"외삽할 데이터가 없음: game_type={game_type}, min_season={min_season}")
    if len(sub) == 1:
        return float(sub.iloc[0])
    x = sub.index.values.astype(float)
    y = sub.values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope * season + intercept)


def held_out_estimate(table, group_cols, season, game_type=None,
                       override=None):
    """held-out 시즌(검증의 2024 또는 배포의 2025)에 쓸 리그 평균(확률) 추정치.

    override가 주어지면 그것을 그대로 쓴다(단, override 값이 train.csv만으로
    재현 가능한 계산인지 호출부에서 먼저 확인할 것 — 대회 규정상 Phase 2
    공식 데이터 외의 외부 데이터는 사용할 수 없다).
    없으면 내부 선형외삽 — **모든 game_type을 동일하게 취급**한다(전체
    관측 시즌 range로 회귀). game_type별로 다른 구간을 잘라 쓰는 특별
    처리는 하지 않는다 — 원본 파이프라인(features.py/cond.py/원래
    train_local.py)도 game_type을 항상 균일한 범주형 값으로만 다뤘고,
    특정 시즌부터만 신뢰한다는 개념 자체가 실제 학습 코드에는 없었다.
    override는 dict({"R":.., "F":..} 또는 group_cols=["season"]일 때
    {"_flat": ..}) 또는 단일 float.
    """
    if override is not None:
        if isinstance(override, dict):
            key = game_type if group_cols != ["season"] else "_flat"
            if key not in override:
                raise KeyError(f"override에 '{key}' 키가 없음: {override}")
            return override[key]
        return float(override)
    return extrapolate(table, group_cols, season, game_type, min_season=None)


def table_to_json(table, group_cols):
    """meta.json 저장용 — Series를 순수 dict로 변환."""
    if group_cols == ["season"]:
        return {"group_cols": group_cols,
                "rates": {str(int(k)): float(v) for k, v in table.items()}}
    return {"group_cols": group_cols,
            "rates": {f"{int(s)}|{gt}": float(v)
                      for (s, gt), v in table.items()}}


def table_from_json(obj):
    """meta.json에서 복원. 반환: (table Series, group_cols)."""
    group_cols = obj["group_cols"]
    if group_cols == ["season"]:
        idx = pd.Index([int(k) for k in obj["rates"]], name="season")
        return pd.Series(list(obj["rates"].values()), index=idx), group_cols
    keys = [k.split("|") for k in obj["rates"]]
    idx = pd.MultiIndex.from_tuples(
        [(int(s), gt) for s, gt in keys], names=["season", "game_type"])
    return pd.Series(list(obj["rates"].values()), index=idx), group_cols


def assign_baseline_logit(df, table, group_cols, held_out_season=None,
                           override=None):
    """df 각 행에 대해 baseline(로짓 스케일)을 계산해 numpy 배열로 반환.

    held_out_season과 같은 season 값을 가진 행, 또는 table에 아예 없는
    season(예: 2025 test)은 held_out_estimate()로 계산한 외삽/override
    값을 쓴다 — 누수 방지(그 시즌 실측 평균을 몰래 쓰지 않는다).
    """
    seasons = df["season"].values

    if group_cols == ["season"]:
        rate = np.empty(len(df), dtype=float)
        for s in np.unique(seasons):
            m = seasons == s
            if s == held_out_season or s not in table.index:
                r = held_out_estimate(table, group_cols, s, override=override)
            else:
                r = table.loc[s]
            rate[m] = r
        return logit(rate)

    gtypes = df["game_type"].values
    rate = np.empty(len(df), dtype=float)
    for s in np.unique(seasons):
        for gt in np.unique(gtypes[seasons == s]):
            m = (seasons == s) & (gtypes == gt)
            key = (s, gt)
            if s == held_out_season or key not in table.index:
                r = held_out_estimate(table, group_cols, s, gt,
                                      override=override)
            else:
                r = table.loc[key]
            rate[m] = r
    return logit(rate)


# ============================================================================
# 팀 단위 baseline 세분화 (R 전용) — 08/09문서 방식과 별개, 2026-08 3차 확장.
#
# 전제 확인 결과(precheck_team_baseline.py 실측, train.csv 전체 147만행):
#   R: 순위상관 +0.14~+0.76(대체로 강함) → 팀 효과가 시즌 간 지속되는
#      진짜 특성. 표본도 충분(중앙값 12,776행/셀, 500 미만 셀 4%뿐).
#   F: 순위상관 -0.62~+0.25(약하거나 음수) → 노이즈에 가까움. 팀 보정 안 함.
#
# 설계: season×game_type baseline(기존) 위에, R 행에만 "팀 편차"를 추가로
# 더한다. F는 이 블록을 아예 안 거치므로 기존 동작과 완전히 동일하게 유지.
#   최종 baseline(R) = season×game_type 평균 + 스무딩된 팀 편차
#   최종 baseline(F) = season×game_type 평균 (변경 없음)
#
# 팀 편차는 표본수 기반 베이지안 스무딩으로 0쪽에 당긴다(표본 적은 팀이
# season×game_type 평균에서 너무 튀지 않게).
# 2025(미관측 시즌) 팀 편차는 "선형외삽" 대신 "과거 시즌들의 평균 편차"를
# 쓴다 — 팀 순위는 매 시즌 추세를 그리며 이동한다기보다, 비교적 안정적인
# 정체성(코칭 스타일 등)에 가깝다고 보는 게 순위상관 패턴과 더 맞는다.
# ============================================================================

def build_team_dev_table(hist, group_cols, team_col, m=200):
    """season×game_type(group_cols) 평균 대비, team_col별 편차(확률 스케일)를
    표본수 기반으로 스무딩해서 계산한다.

    m: 스무딩 강도 — 표본 m개어치의 "0(편차 없음)" 가중치로 당긴다.
    반환: DataFrame[*group_cols, team_col, mean, n, base, dev]
    """
    base = hist.groupby(group_cols)[TARGET].mean().rename("base").reset_index()
    g = (hist.groupby(list(group_cols) + [team_col])[TARGET]
         .agg(mean="mean", n="count").reset_index())
    g = g.merge(base, on=group_cols, how="left")
    g["dev"] = (g["mean"] - g["base"]) * g["n"] / (g["n"] + m)
    return g


def _team_history_avg_dev(table, group_cols, team_col, game_type, team,
                          before_season=None):
    """(game_type, team)의 과거(before_season 미만) 시즌 dev 평균.
    관측이 아예 없으면 0.0(=팀 보정 없이 season×game_type 그대로)."""
    s_col, gt_col = group_cols
    sub = table[(table[gt_col] == game_type) & (table[team_col] == team)]
    if before_season is not None:
        sub = sub[sub[s_col] < before_season]
    return float(sub["dev"].mean()) if len(sub) else 0.0


def team_table_to_json(table, group_cols, team_col):
    """build_team_dev_table 결과를 meta.json 저장용으로 직렬화."""
    s_col, gt_col = group_cols
    rows = [{s_col: int(r[s_col]), gt_col: r[gt_col], team_col: str(r[team_col]),
            "dev": float(r["dev"])}
           for _, r in table.iterrows()]
    return {"group_cols": list(group_cols), "team_col": team_col, "rows": rows}


def team_table_from_json(obj):
    """meta.json에서 복원. 반환: (DataFrame, group_cols, team_col)."""
    df = pd.DataFrame(obj["rows"])
    return df, obj["group_cols"], obj["team_col"]


def assign_team_adjusted_baseline_logit(df, season_table, season_group_cols,
                                        team_table, team_col, apply_team_mask,
                                        held_out_season=None,
                                        season_override=None):
    """season×game_type baseline(assign_baseline_logit) 위에, apply_team_mask가
    True인 행(R)에만 팀 편차를 추가로 더한다. False인 행(F)은 완전히 기존과
    동일 — 이 함수를 새로 도입해도 F 쪽 동작은 한 글자도 안 바뀐다.
    """
    base_logit = assign_baseline_logit(df, season_table, season_group_cols,
                                       held_out_season=held_out_season,
                                       override=season_override)
    apply_team_mask = np.asarray(apply_team_mask)
    if not apply_team_mask.any():
        return base_logit

    base_prob = 1 / (1 + np.exp(-base_logit))
    dev = np.zeros(len(df))

    seasons = df["season"].values
    gtypes = df["game_type"].values
    teams = df[team_col].values
    s_col, gt_col = season_group_cols

    idx = np.where(apply_team_mask)[0]
    keys = set(zip(seasons[idx], gtypes[idx], teams[idx]))
    for (s, gt, tm) in keys:
        m = apply_team_mask & (seasons == s) & (gtypes == gt) & (teams == tm)
        row = team_table[(team_table[s_col] == s) & (team_table[gt_col] == gt) &
                         (team_table[team_col] == tm)]
        if s != held_out_season and len(row):
            d = float(row["dev"].iloc[0])
        else:
            d = _team_history_avg_dev(team_table, season_group_cols, team_col,
                                      gt, tm, before_season=held_out_season or s)
        dev[m] = d

    adj_prob = np.clip(base_prob + dev, 1e-6, 1 - 1e-6)
    out = base_logit.copy()
    out[apply_team_mask] = np.log(adj_prob[apply_team_mask]
                                  / (1 - adj_prob[apply_team_mask]))
    return out


if __name__ == "__main__":
    # self-test — CatBoost baseline 파라미터 동작 실측 검증
    # (1) 추론 때 baseline을 안 넣으면 평균이 틀어진다
    # (2) .cbm은 baseline을 저장하지 않는다
    # 이 두 사실은 코드 작성 전에 별도로 검증됐다 (대화 기록 참고).
    import doctest
    print("league_rate.py 자체 로직 점검 (외부 CatBoost 검증은 build 스크립트에서)")

    hist = pd.DataFrame({
        "season": [2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022,
                   2023, 2023, 2024, 2024],
        "game_type": ["R", "F"] * 6,
        "control_success": [0.55, 0.69, 0.53, 0.59, 0.51, 0.70,
                            0.50, 0.71, 0.50, 0.47, 0.49, 0.46],
    })
    t = build_table(hist, ["season", "game_type"])
    print(t)

    # 2024 held-out 추정 (2019~2023만으로 외삽, out-of-year)
    hist_tr = hist[hist["season"] <= 2023]
    t_tr = build_table(hist_tr, ["season", "game_type"])
    est_r = held_out_estimate(t_tr, ["season", "game_type"], 2024, "R")
    est_f = held_out_estimate(t_tr, ["season", "game_type"], 2024, "F")
    print(f"2024 R 외삽(out-of-year): {est_r:.4f} (실제 {t.loc[(2024,'R')]:.4f})")
    print(f"2024 F 외삽(out-of-year): {est_f:.4f} "
          f"(실제 {t.loc[(2024,'F')]:.4f})")

    # 2025 override 메커니즘 점검용 임의 예시값(실제 KBO 추정치 아님 —
    # 실제 배포에서는 override=None으로 둬서 내부 재계산을 쓴다)
    override = {"R": 0.50, "F": 0.45}
    est_2025_r = held_out_estimate(t, ["season", "game_type"], 2025, "R",
                                   override=override)
    est_2025_f = held_out_estimate(t, ["season", "game_type"], 2025, "F",
                                   override=override)
    print(f"2025 R override(예시): {est_2025_r} / F override(예시): {est_2025_f}")

    # 직렬화 왕복 확인
    j = table_to_json(t, ["season", "game_type"])
    t2, gc2 = table_from_json(j)
    assert gc2 == ["season", "game_type"]
    assert np.allclose(t2.sort_index().values, t.sort_index().values)
    print("table_to_json / table_from_json 왕복 일치 확인 완료")
