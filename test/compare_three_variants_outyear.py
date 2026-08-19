"""보조모델 설정 세 가지를 out-of-year(2019-2023 -> 2024)로 비교한다.

비교 대상 (모두 주모델은 018_extra_fe로 동일 — 이미 확인된 p_matchup 포함):
  A) 019류: 보조모델 옛 피처(57개, extra_fe 없음, p_matchup 없음), baseline 없음
  B) 022류: 보조모델 새 피처(58개, p_matchup 있음), baseline 없음
  C) 021/024류: 보조모델 새 피처(58개, p_matchup 있음), baseline 있음

세 개 다 "2019~2023만 학습 -> 2024로 평가"라는 같은 out-of-sample 조건에서
새로 학습해서 비교한다 — 이미 만들어진 019/022/024 런의 최종모델(전체
2019~2024로 학습됨)을 그대로 2024에 평가하면 2024가 이미 학습에 포함돼
있어서 out-of-year가 아니게 된다. 그래서 세 조합 다 이 스크립트 안에서
2019~2023 -> 2024로 다시(따로) 학습한다 — 009_offset의 b,c를 검증했던
방식(08문서 §3-4)과 동일한 원칙.

⚠️ 채택 도구가 아니라 진단 도구. 최종 판단은 LB로.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize

sys.path.insert(0, "common")
from features import engineer, CAT_COLS  # noqa: E402
import league_rate as lr  # noqa: E402

BASE_RUN = "018_extra_fe"          # 주모델 (세 조합 다 동일 — p_matchup 포함)
AUX_SEEDS = [42, 7, 2024]
AUX_LEAGUE_GROUP_COLS = ["season", "game_type"]
CACHE = "artifacts/auxpred_league"
DATA = "data/train.csv"
ID, TARGET = "row_id", "control_success"
PARAMS = dict(iterations=2000, learning_rate=0.05, depth=6,
              thread_count=-1, verbose=0, eval_metric="Logloss")

# 비교할 세 조합: (이름, 보조모델 extra_fe 사용 여부, 보조모델 baseline 사용 여부)
VARIANTS = [
    ("A: 옛 피처(57), baseline 없음", False, False),
    ("B: 새 피처(58,p_matchup), baseline 없음", True, False),
    ("C: 새 피처(58,p_matchup), baseline 있음", True, True),
]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bss(p, y):
    r = y.mean()
    brier = float(np.mean((p - y) ** 2))
    return brier, max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def run_variant(name, use_extra_fe_aux, use_aux_baseline,
                df, y, tr_mask, va_mask, have, lab, base_meta,
                X_old, X_new, ci_old, ci_new):
    print(f"\n=== {name} ===")
    X = X_new if use_extra_fe_aux else X_old
    ci = ci_new if use_extra_fe_aux else ci_old

    tr_have = tr_mask & have
    va_have = va_mask & have

    val_tables = {}
    if use_aux_baseline:
        df_tr = df[tr_have].copy()
        for lname in ("mr", "wayoff"):
            df_tr[f"_lab_{lname}"] = lab[lname][tr_have]
            val_tables[lname] = lr.build_table(
                df_tr, AUX_LEAGUE_GROUP_COLS, target=f"_lab_{lname}")

    preds_va = {}
    for lname in ("mr", "wayoff"):
        base_tr = base_va = None
        if use_aux_baseline:
            base_tr = lr.assign_baseline_logit(
                df[tr_have], val_tables[lname], AUX_LEAGUE_GROUP_COLS,
                held_out_season=None, override=None)
            base_va = lr.assign_baseline_logit(
                df[va_have], val_tables[lname], AUX_LEAGUE_GROUP_COLS,
                held_out_season=2024, override=None)
        pool_tr = Pool(X[tr_have], lab[lname][tr_have], cat_features=ci,
                      baseline=base_tr)
        pool_va_eval = Pool(X[va_have], lab[lname][va_have], cat_features=ci,
                            baseline=base_va)
        pool_va_pred = Pool(X[va_have], cat_features=ci, baseline=base_va)

        ps = []
        for sd in AUX_SEEDS:
            m = CatBoostClassifier(**dict(
                PARAMS, random_seed=sd, early_stopping_rounds=100)).fit(
                pool_tr, eval_set=pool_va_eval, use_best_model=True)
            ps.append(m.predict_proba(pool_va_pred)[:, 1])
            print(f"  {lname}_{sd} best_iter={m.get_best_iteration()}")
        preds_va[lname] = np.mean(ps, axis=0)

    raw_seeds = [int(tag.split("_")[1]) for tag in base_meta["seeds"]]
    P_success = np.mean([np.load(os.path.join(CACHE, f"success_2024_{s}.npy"))
                         for s in raw_seeds], axis=0)

    z = logit(P_success)
    u = logit(preds_va["mr"]) - logit(preds_va["mr"]).mean()
    v = logit(preds_va["wayoff"]) - logit(preds_va["wayoff"]).mean()
    t = y[va_have]

    def nll(w):
        p = np.clip(sigmoid(z + w[0] * u + w[1] * v), 1e-9, 1 - 1e-9)
        return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

    b, c = minimize(nll, [0.0, 0.0], method="Nelder-Mead").x
    brier_b, bss_b = bss(sigmoid(z), t)
    brier_a, bss_a = bss(sigmoid(z + b * u + c * v), t)
    print(f" b={b:.4f} c={c:.4f}")
    print(f" 2024 out-of-year BSS: 주모델만 {bss_b:.1f} -> +offset {bss_a:.1f}"
          f"  (델타 {bss_a-bss_b:+.1f})")
    return bss_a


def main():
    print("[compare_three_variants_outyear] 주모델 =", BASE_RUN)
    base_meta = json.load(open(os.path.join("runs", BASE_RUN, "model",
                                            "meta.json"), encoding="utf-8"))
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y = df[TARGET].astype(int).values
    tr_mask = (df["season"] <= 2023).values
    va_mask = (df["season"] == 2024).values

    # 보조모델용 X 두 벌 — 옛 피처(57, extra_fe 없음) / 새 피처(58, p_matchup 포함)
    raw = df.drop(columns=[ID, TARGET])
    X_old = engineer(raw, base_meta["global_mean"], extra_fe=False, rate_means=None)
    X_new = engineer(raw, base_meta["global_mean"],
                     extra_fe=base_meta.get("extra_fe", False),
                     rate_means=base_meta.get("rate_means"))
    X_new = X_new[base_meta["feature_cols"]]
    for c in CAT_COLS:
        X_old[c] = X_old[c].astype(str)
        X_new[c] = X_new[c].astype(str)
    # ⚠️ X_old(57개)와 X_new(58개)는 스무딩 단계에서 추가되는 컬럼 수가 달라서
    # (2개 vs 10개), 그 뒤에 나오는 count_state 등의 절대 위치가 서로 다르다
    # (실측 확인: count_state old=51 new=59). ci를 반드시 각각 따로 계산할 것 —
    # 하나를 공유하면 CatBoost가 엉뚱한 컬럼을 범주형으로 취급한다.
    ci_old = [X_old.columns.get_loc(c) for c in CAT_COLS]
    ci_new = [X_new.columns.get_loc(c) for c in CAT_COLS]

    L = pd.read_csv("recovered_labels.csv.gz")
    L = df[[ID]].merge(L, on=ID, how="left")
    have = L["middle"].notna().values
    mr = ((L["middle"] == 1) | (L["reverse"] == 1)).values
    lab = {"mr": mr.astype(int), "wayoff": ((y == 0) & ~mr).astype(int)}

    results = {}
    for name, use_fe, use_bl in VARIANTS:
        results[name] = run_variant(name, use_fe, use_bl, df, y, tr_mask,
                                    va_mask, have, lab, base_meta,
                                    X_old, X_new, ci_old, ci_new)

    print("\n=== 결론 (2024 out-of-year BSS, 높을수록 좋음) ===")
    for name, score in results.items():
        print(f" {name}: {score:.1f}")
    names = list(results.keys())
    print(f"\n B-A(새 피처 자체 효과): {results[names[1]]-results[names[0]]:+.1f}")
    print(f" C-B(baseline 추가 효과): {results[names[2]]-results[names[1]]:+.1f}")
    print(f" C-A(전체 효과): {results[names[2]]-results[names[0]]:+.1f}")


if __name__ == "__main__":
    main()
