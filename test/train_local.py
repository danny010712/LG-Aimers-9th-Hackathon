"""로컬 학습 스크립트 (submit.zip에 포함되지 않음).

train.csv 학습 → runs/<RUN>/ 에 모델·메타·zip·결과 저장.
검증: 2019~2023 학습 → 2024 홀드아웃 (2025 분포이동 모사, 08 문서 §3-B).
최종: 검증에서 얻은 best_iter 만큼 전체 데이터(2024 포함)로 재학습해 저장.

실행마다 RUN 이름을 바꿀 것. 기존 RUN이 있으면 실행을 거부한다.
(run 001/002는 LightGBM판. 설정은 08 문서 §3-C, 모델은 runs/에 보존됨.)

=== league-rate baseline 추가 (2026-08, 08 §5-7 / 09 §3-B) ===
season을 그냥 피처로 주면 트리가 "이 시즌의 절대 수준"까지 직접 배우려다
실패한다(미관측 시즌 외삽 불가, §6-A). 대신 CatBoost의 `baseline` 파라미터로
"이 그룹(season×game_type)의 리그 평균 로짓"을 미리 알려주고, 트리는
"평균 대비 편차"만 배우게 한다. 표현력 증가가 아니다(§3 용량 제약 통과).

⚠️ CatBoost baseline은 .cbm에 저장되지 않는다(실측 확인). 추론 때도 반드시
   같은 방식으로 baseline을 다시 계산해서 넣어야 한다 → script.py, meta.json
   참고. 이 원칙이 지금 코드 전체에서 가장 신경 써야 할 부분이다.
⚠️ out-of-year 원칙(§3-4): 검증(2024)에는 2024 실측 평균을 쓰지 않고
   2019~2023만으로 외삽한다. 최종 배포 모델은 2019~2024 실측을 쓰되
   2025는 여전히 외삽/외부추정치(EST_2025_OVERRIDE)를 쓴다.
⚠️ 이 축은 언제든 끌 수 있다 — USE_LEAGUE_BASELINE=False로 두면 이전과
   완전히 같은 동작(순수 season 피처만).
"""
import io
import json
import os
import sys
import time
import zipfile

# Windows 기본 stdout(cp949)은 일부 기호를 못 써서 print에서 죽는다.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "common")
from features import engineer, CAT_COLS, compute_rate_means  # noqa: E402
import cond  # noqa: E402
import league_rate as lr  # noqa: E402

# ===== 이번 실행 설정 =====================================================
RUN = "028_teamfight"
NOTE = ("새로운 시도: season x game_type x team_id 별 baseline")
SEEDS = [42, 7, 2024, 99, 1, 123, 777]   # 007과 동일
POLICIES = ["SymmetricTree"]
USE_COND = False

# --- 추가 피처(EXTRA_FE) 설정 ---
# True/False 토글 하나로 완전히 켜고 끌 수 있다 — features.py의 extra_fe
# 분기와 연결됨. 끄면 이전(015)과 완전히 동일한 동작.
USE_EXTRA_FE = True

# --- league-rate baseline 설정 ---
USE_LEAGUE_BASELINE = True
# ["season", "game_type"] ↔ ["season"] 로 언제든 토글 가능 (09 §3-A 근거로 기본은 분리).
LEAGUE_GROUP_COLS = ["season", "game_type"]
# F/R 모두 동일하게 취급한다 — 특정 시즌부터만 신뢰한다는 식의 특별 처리
# 없음. 원본 파이프라인(features.py, cond.py, 원래 train_local.py)도
# game_type을 항상 균일한 범주형 값으로만 다뤘을 뿐, 특정 game_type만
# 시즌 구간을 잘라 쓰는 로직은 실제 학습 코드 어디에도 없었다(2026-08
# 재검토로 확인). 09/10 문서의 "F는 2023년에 라벨 체계가 바뀌었다"는
# 관찰은 참고만 하고, 코드에는 반영하지 않는다.
# ⚠️ 2025 test용 override. None이면 아래서 league_rate.extrapolate()가
#    **이 스크립트 안에서 train.csv만으로** 직접 선형외삽한다 — 외부 문서를
#    전혀 참조하지 않는다. 대회 규칙 3번 "외부 데이터 사용 금지" 조항
#    때문에 하드코딩 값 대신 이 방식을 기본으로 둔다.
LEAGUE_EST_2025_OVERRIDE = None
# artifacts/auxpred/*.npy는 baseline 없는 구모델 캐시라 재사용 불가.
# league-baseline 계열은 별도 폴더에 새로 쌓는다 (train_offset.py가 여기서 읽음).
LEAGUE_CACHE = "artifacts/auxpred_league"

# --- 팀 단위 baseline 세분화 (R 전용) ---
# precheck_team_baseline.py 실측(train.csv 전체): R은 팀 효과가 시즌 간
# 지속되는 진짜 특성(순위상관 +0.14~+0.76), F는 노이즈에 가까움(-0.62~+0.25)
# → R에만 적용. F는 이 토글을 켜도 동작이 전혀 안 바뀐다.
USE_TEAM_BASELINE = True
TEAM_COL = "pitcher_team_id"
TEAM_SMOOTH_M = 200   # 표본 m개어치 무게로 팀 편차를 0쪽에 당김

PARAMS = dict(
    iterations=2000, learning_rate=0.05, depth=6,
    thread_count=-1, verbose=0, eval_metric="Logloss",   # CatBoost는 -1 (0은 크래시)
    early_stopping_rounds=100,
)
# =========================================================================

DATA = "data/train.csv"
COMMON = "common"
ID, TARGET = "row_id", "control_success"


def fmt_secs(s):
    if s < 90:
        return f"{s:.0f}초"
    return f"{s/60:.1f}분"


class Timer:
    """시드/이터레이션마다 걸린 시간을 누적해서 남은 예상 시간을 계산한다."""
    def __init__(self, total_steps, label):
        self.total = total_steps
        self.label = label
        self.times = []
        self.t0 = time.time()
        print(f" [{label}] 총 {total_steps}개 예정, 시작 {time.strftime('%H:%M:%S')}")

    def tick(self, step_label, extra=""):
        now = time.time()
        elapsed = now - (self.t0 if not self.times else self._last)
        self._last = now
        self.times.append(elapsed)
        done = len(self.times)
        avg = sum(self.times) / done
        remain = avg * (self.total - done)
        total_elapsed = now - self.t0
        print(f"   ↳ {step_label} 소요 {fmt_secs(elapsed)}  "
              f"(누적 {fmt_secs(total_elapsed)}, 평균 {fmt_secs(avg)}/개, "
              f"남은 예상 {fmt_secs(remain)}, 완료예정 "
              f"{time.strftime('%H:%M:%S', time.localtime(now + remain))})"
              f"{extra}")

    def done(self):
        total = time.time() - self.t0
        print(f" [{self.label}] 전체 소요: {fmt_secs(total)}")
        return total


def bss(p, y):
    r = y.mean()
    brier = float(np.mean((p - y) ** 2))
    skill = 1 - brier / (r * (1 - r))
    return brier, skill, max(0.0, 100000 * skill)


def build_zip(out_dir):
    """common/ + model/ → submit<번호>.zip.
    최상위에 model/ + script.py + requirements.txt + features.py (여분 폴더 없음)."""
    path = os.path.join(out_dir, f"submit{RUN.split('_')[0]}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        files = ["script.py", "requirements.txt", "features.py", "cond.py"]
        if USE_LEAGUE_BASELINE:
            files.append("league_rate.py")   # zip에 반드시 포함 — 빠지면 ModuleNotFoundError
        for f in files:
            z.write(os.path.join(COMMON, f), f)
        model_dir = os.path.join(out_dir, "model")
        for f in sorted(os.listdir(model_dir)):
            z.write(os.path.join(model_dir, f), "model/" + f)
    return path


def build_baseline(df, table, group_cols, held_out_season, override,
                   team_table=None):
    """df 전체 행에 대한 baseline(로짓) 배열. USE_LEAGUE_BASELINE=False면 None.

    team_table이 주어지면(USE_TEAM_BASELINE=True), game_type=='R'인 행에만
    팀 편차를 추가로 얹는다. F는 완전히 기존과 동일하게 동작한다.
    """
    if not USE_LEAGUE_BASELINE:
        return None
    if team_table is None:
        return lr.assign_baseline_logit(
            df, table, group_cols, held_out_season=held_out_season,
            override=override)
    team_mask = (df["game_type"] == "R").values
    return lr.assign_team_adjusted_baseline_logit(
        df, table, group_cols, team_table, TEAM_COL, team_mask,
        held_out_season=held_out_season, season_override=override)


def main():
    out_dir = os.path.join("runs", RUN)
    if os.path.exists(os.path.join(out_dir, "model")):
        raise SystemExit(f"이미 존재함: {out_dir} — RUN 이름을 바꿀 것")
    os.makedirs(os.path.join(out_dir, "model"))
    os.makedirs(LEAGUE_CACHE, exist_ok=True)

    print(f"[{RUN}] Load train...")
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y = df[TARGET].astype(int).values
    tr = (df["season"] <= 2023).values
    va = (df["season"] == 2024).values

    # global_mean은 학습 구간에서만 계산 (검증 누수 방지).
    global_mean = float(y[tr].mean())
    # rate_means도 같은 원칙 — 학습 구간(2019~2023)에서만 계산해 고정하고,
    # 검증·최종재학습·추론 전부 이 값을 그대로 재사용한다(global_mean과 동일한
    # 성격). USE_EXTRA_FE=False면 그냥 None(안 씀).
    rate_means = compute_rate_means(df[tr]) if USE_EXTRA_FE else None
    X = engineer(df.drop(columns=[ID, TARGET]), global_mean,
                extra_fe=USE_EXTRA_FE, rate_means=rate_means)

    if USE_COND:
        print(" 조건부 표 생성 (시즌별 과거만)...", flush=True)
        C = cond.build_training_columns(df)
        for c in cond.COND_COLS:
            X[c] = C[c].values
        print(f" cond 결측률 {X[cond.COND_COLS].isna().mean().mean()*100:.1f}%")

    feature_cols = list(X.columns)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in CAT_COLS]
    print(f" rows={len(df)}  feats={len(feature_cols)}  "
          f"global_mean={global_mean:.4f}  league_baseline={USE_LEAGUE_BASELINE}"
          f"({LEAGUE_GROUP_COLS if USE_LEAGUE_BASELINE else '-'})")

    # ---- league-rate baseline: 검증 단계 (2019~2023 -> 2024) ----
    # 표는 학습 구간(2019~2023)에서만 만든다. 2024는 out-of-year 외삽으로 대체
    # (override 없이 — 내부 추세 외삽만 써서 "정말로 안 본 해"를 시험한다).
    # game_type별 특별 처리 없음 — F/R 모두 같은 방식(전체 관측 구간 회귀).
    if USE_LEAGUE_BASELINE:
        table_val = lr.build_table(df[tr], LEAGUE_GROUP_COLS)
        team_table_val = None
        if USE_TEAM_BASELINE:
            team_table_val = lr.build_team_dev_table(
                df[tr], LEAGUE_GROUP_COLS, TEAM_COL, m=TEAM_SMOOTH_M)
            print(f" 팀 편차표(검증, tr만) 행 수={len(team_table_val)}")
        base_tr = build_baseline(df[tr], table_val, LEAGUE_GROUP_COLS,
                                 held_out_season=None, override=None,
                                 team_table=team_table_val)
        base_va = build_baseline(df[va], table_val, LEAGUE_GROUP_COLS,
                                 held_out_season=2024, override=None,
                                 team_table=team_table_val)
        print(f" league_rate 검증표 그룹 수={len(table_val)}  "
              f"baseline(로짓) tr 평균={base_tr.mean():+.4f}  "
              f"va 평균={base_va.mean():+.4f}")
        # train_offset.py의 fit_offset()이 쓰는 t(=y[have & season==2024])와
        # 길이를 맞추기 위해, 캐시로 저장할 예측도 같은 have 필터를 적용한다.
        # (mr/wayoff 기존 캐시가 이미 이 필터로 저장돼 있어서, success 쪽만
        # 필터 없이 저장하면 길이가 어긋나 nll()에서 broadcast 에러가 난다.)
        L = pd.read_csv("recovered_labels.csv.gz")
        have_all = df[[ID]].merge(L, on=ID, how="left")["middle"].notna().values
        have_va = have_all[va]
        print(f" recovered_labels 복원 커버리지(2024): "
              f"{have_va.sum():,}/{len(have_va):,}")
    else:
        base_tr = base_va = have_va = None

    print(f"\n--- 검증 (2019-2023 -> 2024), {POLICIES} x {SEEDS} ---")
    val_preds, best_iters, tags = [], [], []
    pool_tr = Pool(X[tr], y[tr], cat_features=ci, baseline=base_tr)
    pool_va = Pool(X[va], y[va], cat_features=ci, baseline=base_va)
    timer = Timer(len(POLICIES) * len(SEEDS), "검증 학습")
    for gp in POLICIES:
        for sd in SEEDS:
            m = CatBoostClassifier(**dict(PARAMS, grow_policy=gp,
                                          random_seed=sd)).fit(
                pool_tr, eval_set=pool_va, use_best_model=True)
            p = m.predict_proba(pool_va)[:, 1]
            val_preds.append(p)
            it = m.get_best_iteration()
            if it < 1:
                # 극단적으로 이른 조기종료(0번째가 최적) — CatBoost는
                # iterations=0을 거부해서 최종 재학습이 크래시한다(실측
                # 확인, 2026-08). 표본이 아주 적거나 baseline이 대부분을
                # 이미 설명하는 경우 드물게 발생할 수 있다. 최소 1로 방어.
                print(f" ⚠️ {gp[:3]}_{sd} best_iteration={it} → 1로 보정")
                it = 1
            best_iters.append(it)
            tags.append(f"{gp[:3]}_{sd}")
            print(f" {tags[-1]:<10} iter={best_iters[-1]:<5} "
                  f"score~{bss(p, y[va])[2]:.1f}", flush=True)
            timer.tick(tags[-1])
            if USE_LEAGUE_BASELINE:
                # train_offset.py의 b,c 적합용 out-of-sample 캐시.
                # 반드시 have_va로 필터링해서 저장 — mr/wayoff 기존 캐시와
                # 길이를 맞춰야 fit_offset()의 nll()에서 배열 길이가 어긋나지
                # 않는다 (season==2024 전체가 아니라, 그중 라벨 복원된 행만).
                np.save(os.path.join(LEAGUE_CACHE, f"success_2024_{sd}.npy"),
                        p[have_va])
    timer.done()

    brier, skill, score = bss(np.mean(val_preds, axis=0), y[va])
    print(f"\n[VAL 2024 · {len(SEEDS)}시드 평균] Brier={brier:.6f} "
          f"skill={skill:.5f} score~{score:.1f}")
    print(" ⚠️ league-rate baseline 사용 시 이 검증 점수는 §4 원칙대로 "
          "절대값으로 채택 판정하지 말 것 — 007(baseline 없음)과 out-of-year "
          "방식(2022 등 다른 연도 기준)으로 다시 비교할 것.")

    # ---- 최종: 2019~2024 전체 재학습 ----
    # 표는 전체(2019~2024)로 다시 만든다 — 2019~2024는 이제 전부 '학습에 쓰이는
    # 과거'이므로 실측 평균을 그대로 쓴다 (global_mean과 같은 성격).
    # 2025(test)는 여기 없으므로 baseline은 이 재학습 자체엔 필요 없다 —
    # 2025용 baseline은 추론(script.py)에서 LEAGUE_EST_2025_OVERRIDE로 계산한다.
    print("\n--- 전체 데이터(2019-2024) 재학습 ---")
    if USE_LEAGUE_BASELINE:
        table_full = lr.build_table(df, LEAGUE_GROUP_COLS)
        team_table_full = None
        if USE_TEAM_BASELINE:
            team_table_full = lr.build_team_dev_table(
                df, LEAGUE_GROUP_COLS, TEAM_COL, m=TEAM_SMOOTH_M)
            print(f" 팀 편차표(전체) 행 수={len(team_table_full)}")
        base_full = build_baseline(df, table_full, LEAGUE_GROUP_COLS,
                                   held_out_season=None, override=None,
                                   team_table=team_table_full)
        print(f" league_rate 전체표 그룹 수={len(table_full)}  "
              f"baseline(로짓) 평균={base_full.mean():+.4f}")
    else:
        table_full = base_full = team_table_full = None

    pool_all = Pool(X, y, cat_features=ci, baseline=base_full)
    final = dict(PARAMS)
    final.pop("early_stopping_rounds")
    timer2 = Timer(len(tags), "최종 재학습")
    fold_models = []
    for tag, it in zip(tags, best_iters):
        gp = "SymmetricTree" if tag.startswith("Sym") else "Depthwise"
        sd = int(tag.split("_")[1])
        m = CatBoostClassifier(**dict(final, grow_policy=gp, random_seed=sd,
                                      iterations=it)).fit(pool_all)
        m.save_model(os.path.join(out_dir, "model", f"model_{tag}.cbm"))
        fold_models.append(m)
        print(f" {tag} iter={it} 저장")
        timer2.tick(tag)
    timer2.done()

    # ---- feature importance (시드 평균) ----
    # PredictionValuesChange(기본값) — pool 없이도 계산 가능, 학습 중 분기
    # 기여도를 누적한 값. baseline을 쓴 모델도 그대로 지원된다.
    imp_df = None
    try:
        imps = [m.get_feature_importance(prettified=True) for m in fold_models]
        imp_avg = (pd.concat(imps).groupby("Feature Id")["Importances"]
                  .mean().sort_values(ascending=False))
        print(f"\n--- Feature Importance (시드 {len(fold_models)}개 평균, 상위 20개) ---")
        for i, (feat, val) in enumerate(imp_avg.head(20).items(), 1):
            bar = "█" * int(val / imp_avg.max() * 30)
            print(f" {i:>2}. {feat:<40} {val:6.2f}  {bar}")
        imp_path = os.path.join(out_dir, "feature_importance.csv")
        imp_avg.to_csv(imp_path, header=["importance"])
        print(f" 전체 {len(imp_avg)}개 -> {imp_path} 저장")
        imp_df = imp_avg
    except Exception as e:
        print(f" ⚠️ feature importance 계산 실패(무시하고 진행): {e}")

    if USE_COND:
        for name, t in cond.build_tables(df).items():
            t.to_csv(os.path.join(out_dir, "model", f"cond_{name}.csv"),
                     index=False, encoding="utf-8")
        print(f" cond 표 {len(cond.SPECS)}개 저장")

    meta = {"seeds": tags, "feature_cols": feature_cols,
            "cat_cols": CAT_COLS, "global_mean": global_mean,
            "use_cond": USE_COND,
            "extra_fe": USE_EXTRA_FE, "rate_means": rate_means,
            "kitchen_sink": USE_KITCHEN_SINK}
    if USE_LEAGUE_BASELINE:
        meta["league_baseline"] = {
            "enabled": True,
            **lr.table_to_json(table_full, LEAGUE_GROUP_COLS),
            "test_override": LEAGUE_EST_2025_OVERRIDE,
        }
        if USE_TEAM_BASELINE:
            meta["league_baseline"]["team_adjustment"] = lr.team_table_to_json(
                team_table_full, LEAGUE_GROUP_COLS, TEAM_COL)
    else:
        meta["league_baseline"] = {"enabled": False}
    json.dump(meta, open(os.path.join(out_dir, "model", "meta.json"), "w",
                         encoding="utf-8"))
    json.dump({"run": RUN, "note": NOTE, "model": "catboost", "seeds": tags,
               "params": PARAMS, "n_features": len(feature_cols),
               "best_iters": best_iters,
               "league_baseline": USE_LEAGUE_BASELINE,
               "league_group_cols": LEAGUE_GROUP_COLS if USE_LEAGUE_BASELINE else None,
               "val_2024": {"brier": brier, "skill": skill, "score": score},
               "lb_2025": None},
              open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nSaved {build_zip(out_dir)}")


if __name__ == "__main__":
    main()
