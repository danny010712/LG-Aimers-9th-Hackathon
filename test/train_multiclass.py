"""조인트 3-클래스 모델 — success/mr/wayoff를 CatBoost MultiClass 하나로 동시학습.
기존(train_local.py 성공모델 + train_offset.py mr/wayoff 보조모델 + logit 오프셋)을
모델 1개로 대체한다. mr=middle|reverse, wayoff=(y=0)&~mr, success=(y=1) —
정확히 3분할(예외 0건, 09 §1-D)이라 4클래스(overlap 있는 middle/reverse 분리) 대신 이걸 쓴다.

검증(2019-2023->2024, 2세트x3시드): 순수 P(success) 단독조차 216최종블렌드(883.6)에
근접, 자체 오프셋(logit(P_success)+b*(logit(P_mr)-mu)+c*(logit(P_wayoff)-mu)) 적용시
원시드 Δ+10.3 / 새시드 Δ+2.4 (둘 다 양성, 채택 기준 통과).

⚠️ have(~86%, 복원라벨 존재)로만 학습 가능 — 주모델(기존엔 100%) 대비 학습행이 준다.
   그런데도 이겼다(순수 아키텍처 효과 자체가 +20 안팎, test_multiclass_joint.py).
"""
import io
import json
import os
import sys
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize

sys.path.insert(0, "common")
from features import engineer, build_anchor, rate_priors, CAT_COLS  # noqa: E402
import cond  # noqa: E402

# ===== 이번 실행 설정 =====================================================
RUN = "221_multiclass_joint_bcwayoff"
NOTE = ("220(조인트 3클래스, cond_ph+cond_bh) + cond_bc_wayoff(타자x볼카운트->wayoff) 추가. "
        "이 축은 로컬(구 아키텍처, 3승3패 노이즈)만으로 기각했었는데, cond_bh가 로컬 노이즈였음에도 "
        "LB+12.6(219->220)이었던 걸 보고 재검토 대상으로 선정 — LB로 직접 확인.")
COND_ONLY = ["ph", "bh"]
# EXTRA_COND: cond.py SPECS를 안 건드리고(다른 run 영향 없음) 독립적으로 추가하는
# leak-safe 시즌롤링 조건부표. (name, keys, prior_entity_col, target_col, M)
EXTRA_COND = [("bc_wayoff", ["batter_id", "count_state"], "batter_id", "wayoff", 50)]
SEEDS = [42, 7, 2024]
PARAMS = dict(iterations=2000, learning_rate=0.05, depth=6,
              thread_count=-1, verbose=0, eval_metric="MultiClass",
              loss_function="MultiClass", early_stopping_rounds=100)
# =========================================================================


def build_extra_table(hist, keys, prior_col, target_col, m):
    g = hist.groupby(keys)[target_col].agg(["sum", "count"])
    pr = hist.groupby(prior_col)[target_col].mean().rename("prior")
    g = g.join(pr, on=prior_col)
    gmean = float(hist[target_col].mean())
    g["prior"] = g["prior"].fillna(gmean)
    v = (g["sum"] + m * g["prior"]) / (g["count"] + m)
    return v.rename("val").reset_index()


def build_extra_training_columns(df, keys, prior_col, target_col, m):
    out = pd.Series(index=df.index, dtype=float)
    for s in sorted(df["season"].unique()):
        hist = df[df["season"] < s]
        if len(hist) == 0:
            continue
        t = build_extra_table(hist, keys, prior_col, target_col, m)
        mrow = (df["season"] == s).values
        merged = df.loc[mrow, keys].merge(t, on=keys, how="left")
        out.loc[mrow] = merged["val"].values
    return out

DATA = "data/train.csv"
COMMON = "common"
ID, TARGET = "row_id", "control_success"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bss(p, y):
    r = y.mean()
    brier = float(np.mean((p - y) ** 2))
    skill = 1 - brier / (r * (1 - r))
    return brier, skill, max(0.0, 100000 * skill)


def build_zip(out_dir):
    path = os.path.join(out_dir, f"submit{RUN.split('_')[0]}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["script.py", "requirements.txt", "features.py", "cond.py"]:
            z.write(os.path.join(COMMON, f), f)
        model_dir = os.path.join(out_dir, "model")
        for f in sorted(os.listdir(model_dir)):
            z.write(os.path.join(model_dir, f), "model/" + f)
    return path


def main():
    out_dir = os.path.join("runs", RUN)
    if os.path.exists(os.path.join(out_dir, "model")):
        raise SystemExit(f"이미 존재함: {out_dir} — RUN 이름을 바꿀 것")
    os.makedirs(os.path.join(out_dir, "model"))

    print(f"[{RUN}] Load train...")
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y = df[TARGET].astype(int).values
    tr = (df["season"] <= 2023).values
    va = (df["season"] == 2024).values
    global_mean = float(y[tr].mean())

    anchor = build_anchor(df)
    priors = rate_priors(df[tr])
    X = engineer(df.drop(columns=[ID, TARGET]), global_mean, anchor=anchor,
                 priors=priors)
    X = X.drop(columns=["p_matchup"])

    print(" 조건부 표 생성 (시즌별 과거만)...", flush=True)
    _dfl = df.merge(pd.read_csv("recovered_labels.csv.gz"), on=ID, how="left")
    Ccols = cond.build_training_columns(_dfl)
    use = [c for c in cond.COND_COLS
           if any(c == "cond_" + n or c == "cond_" + n + "_dev" for n in COND_ONLY)]
    for c in use:
        X[c] = Ccols[c].values
    print(f" cond 열 {use}", flush=True)

    L = pd.read_csv("recovered_labels.csv.gz")
    L = df[[ID]].merge(L, on=ID, how="left")
    have = L["middle"].notna().values
    mr = ((L["middle"] == 1) | (L["reverse"] == 1)).values
    cls = np.where(y == 1, 2, np.where(mr, 0, 1)).astype(int)  # 0=mr 1=wayoff 2=success

    if EXTRA_COND:
        _dfl["wayoff"] = ((y == 0) & ~mr).astype(float)
        _dfl.loc[_dfl["middle"].isna(), "wayoff"] = np.nan
        # count_state는 raw df에 없는 파생열(cond.add_keys()와 동일 공식) — EXTRA_COND
        # 키에 필요하면 여기서 만든다.
        if "count_state" not in _dfl.columns:
            _dfl["count_state"] = (_dfl["balls_before"].astype(str) + "-"
                                   + _dfl["strikes_before"].astype(str))
        for name, keys, prior_col, target_col, m in EXTRA_COND:
            col = f"cond_{name}"
            X[col] = build_extra_training_columns(_dfl, keys, prior_col, target_col, m).values
            use.append(col)
            print(f" extra_cond 열 {col} 결측률 {X[col].isna().mean()*100:.1f}%", flush=True)
    print(f" 라벨: 복원 {have.sum():,}/{len(df):,}  "
          f"mr={np.mean(cls[have]==0):.4f} wayoff={np.mean(cls[have]==1):.4f} "
          f"success={np.mean(cls[have]==2):.4f}", flush=True)

    feature_cols = list(X.columns)
    cat_cols_here = [c for c in CAT_COLS if c in X.columns]
    for c in cat_cols_here:
        X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in cat_cols_here]
    print(f" rows={len(df)}  feats={len(feature_cols)}  global_mean={global_mean:.4f}")

    tr_h, va_h = tr & have, va & have

    print(f"\n--- 검증 (2019-2023 -> 2024), {SEEDS} ---")
    pool_tr = Pool(X[tr_h], cls[tr_h], cat_features=ci)
    pool_va = Pool(X[va_h], cls[va_h], cat_features=ci)
    val_probs, best_iters, tags = [], [], []
    for sd in SEEDS:
        m = CatBoostClassifier(**dict(PARAMS, random_seed=sd)).fit(
            pool_tr, eval_set=pool_va, use_best_model=True)
        proba = m.predict_proba(X[va])  # 전체 2024(have 아닌 행 포함)에 대해 예측
        val_probs.append(proba)
        best_iters.append(m.get_best_iteration())
        tags.append(f"Sym_{sd}")
        print(f" {tags[-1]:<10} iter={best_iters[-1]:<5} "
              f"단독P(success) score~{bss(proba[:,2], y[va])[2]:.1f}", flush=True)

    P = np.mean(val_probs, axis=0)  # [:,0]=mr [:,1]=wayoff [:,2]=success, 전체 2024행
    y_va_all = y[va]
    have_va = have[va]

    # 오프셋 계수는 have 행만으로 적합 (mr/wayoff 라벨이 있는 행만 의미 있음)
    z = logit(P[have_va, 2]); u = logit(P[have_va, 0]); v = logit(P[have_va, 1])
    mu_u, mu_v = float(u.mean()), float(v.mean())
    uu, vv = u - mu_u, v - mu_v
    t = y_va_all[have_va]

    def nll(w):
        p = np.clip(1 / (1 + np.exp(-(z + w[0] * uu + w[1] * vv))), 1e-9, 1 - 1e-9)
        return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

    b, c = minimize(nll, [0.0, 0.0], method="Nelder-Mead").x
    before = bss(1 / (1 + np.exp(-z)), t)[2]
    after = bss(1 / (1 + np.exp(-(z + b * uu + c * vv))), t)[2]
    print(f"\n 오프셋 계수 b={b:.4f} c={c:.4f}  mu_mr={mu_u:.4f} mu_wayoff={mu_v:.4f}")
    print(f" 2024(have) 자기적합: P(success)단독={before:.1f} -> 오프셋적용후={after:.1f}")

    # 참고용: 2024 전체 행(have 아닌 행은 오프셋 없이 success단독으로) 최종 점수
    z_all = logit(P[:, 2]); u_all = logit(P[:, 0]); v_all = logit(P[:, 1])
    z_final = np.where(have_va, z_all + b * (u_all - mu_u) + c * (v_all - mu_v), z_all)
    p_final = 1 / (1 + np.exp(-z_final))
    _, _, score_full = bss(p_final, y_va_all)
    print(f" 참고: 2024 전체({len(y_va_all):,}행, have아닌행은 오프셋 미적용) "
          f"최종 score~{score_full:.1f}")

    # build_shift.py 추정자②(검증편향역산)가 쓸 캐시. have행만, 오프셋까지 적용된
    # 최종 OOS 예측 — 209_base044류의 success단독 캐시와는 성격이 달라 별도 디렉토리.
    VALPRED_DIR = f"artifacts/auxpred_mc_{RUN.split('_')[0]}"
    os.makedirs(VALPRED_DIR, exist_ok=True)
    np.save(os.path.join(VALPRED_DIR, "final_2024_have.npy"),
            (1 / (1 + np.exp(-z_final)))[have_va])
    print(f" OOS 최종예측(have행) 저장 -> {VALPRED_DIR}/final_2024_have.npy", flush=True)

    print("\n--- 전체 데이터(2019-2024, have행만) 재학습 ---")
    pool_all = Pool(X[have], cls[have], cat_features=ci)
    final = dict(PARAMS)
    final.pop("early_stopping_rounds")
    for tag, it, sd in zip(tags, best_iters, SEEDS):
        m = CatBoostClassifier(**dict(final, random_seed=sd, iterations=it)).fit(pool_all)
        m.save_model(os.path.join(out_dir, "model", f"model_{tag}.cbm"))
        print(f" {tag} iter={it} 저장")

    for name, t_ in cond.build_tables(_dfl).items():
        t_.to_csv(os.path.join(out_dir, "model", f"cond_{name}.csv"),
                  index=False, encoding="utf-8")
    print(f" cond 표 {len(cond.SPECS)}개 저장")

    extra_cond_meta = []
    for name, keys, prior_col, target_col, m in EXTRA_COND:
        t_ = build_extra_table(_dfl, keys, prior_col, target_col, m)
        t_ = t_.rename(columns={"val": f"cond_{name}"})
        t_.to_csv(os.path.join(out_dir, "model", f"cond_{name}.csv"),
                  index=False, encoding="utf-8")
        extra_cond_meta.append({"name": name, "keys": keys})
        print(f" extra_cond 표 {name} 저장 ({len(t_)}행)", flush=True)

    last = int(df["season"].max()) + 1
    anchor[anchor["apply_season"] == last].to_csv(
        os.path.join(out_dir, "model", "anchor.csv"), index=False, encoding="utf-8")
    print(f" 기준점 표 저장 (apply_season={last})")

    json.dump({"model_type": "multiclass_joint", "seeds": tags,
               "feature_cols": feature_cols, "cat_cols": cat_cols_here,
               "global_mean": global_mean, "use_cond": True, "use_inseason": True,
               "use_role": False, "rate_priors": priors, "extra_cond": extra_cond_meta,
               "offset_mc": {"b": float(b), "c": float(c),
                             "mu_mr": mu_u, "mu_wayoff": mu_v}},
              open(os.path.join(out_dir, "model", "meta.json"), "w", encoding="utf-8"))
    json.dump({"run": RUN, "note": NOTE, "model": "catboost_multiclass", "seeds": tags,
               "params": PARAMS, "n_features": len(feature_cols),
               "best_iters": best_iters,
               "val_2024_success_only": before,
               "val_2024_with_offset": after,
               "val_2024_full_2024rows": score_full,
               "lb_2025": None},
              open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nSaved {build_zip(out_dir)}")


if __name__ == "__main__":
    main()
