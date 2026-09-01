"""222 긴급 배포용 — 검증 스킵, 220의 best_iter/offset_mc(b,c,mu)를 그대로 재사용해서
team13_transition 추가 피처로 바로 전체 재학습만 수행. 시간 압박(마감 임박) 대응.
전제: team13_transition은 65->66열 중 1개 바이너리 플래그일 뿐이라 best_iter/offset이
220과 크게 다르지 않을 것이라는 근사 — 검증 없이 재사용은 이번 1회성 긴급조치.
"""
import io, json, os, sys, zipfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "common")
from features import engineer, build_anchor, rate_priors, CAT_COLS  # noqa: E402
import cond  # noqa: E402

RUN = "222_multiclass_joint_team13"
NOTE = ("220(조인트 3클래스, cond_ph+cond_bh) + team13_transition(체제전환 지시자, "
        "features.py engineer()에 추가된 raw-column 파생 플래그). 마감 임박으로 검증 스킵 — "
        "team13 사전검증(원시드Δ-1.1/새시드Δ+4.5, 노이즈 안, 심한 손해 아님)만 확인 후 "
        "220의 best_iters=[511,580,402] 및 offset_mc(b,c,mu_mr,mu_wayoff)를 그대로 재사용해 "
        "전체 재학습만 수행. 오프셋 재적합 안 함(긴급조치).")
COND_ONLY = ["ph", "bh"]
SEEDS = [42, 7, 2024]
BEST_ITERS = [511, 580, 402]
OFFSET_MC = {"b": -0.04431488871029191, "c": 0.0030295141808389598,
             "mu_mr": -0.5142127742542337, "mu_wayoff": -1.982972506813531}
PARAMS = dict(learning_rate=0.05, depth=6, thread_count=-1, verbose=0,
              eval_metric="MultiClass", loss_function="MultiClass")

DATA = "data/train.csv"
ID, TARGET = "row_id", "control_success"


def build_zip(out_dir):
    path = os.path.join(out_dir, f"submit{RUN.split('_')[0]}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["script.py", "requirements.txt", "features.py", "cond.py"]:
            z.write(os.path.join("common", f), f)
        model_dir = os.path.join(out_dir, "model")
        for f in sorted(os.listdir(model_dir)):
            z.write(os.path.join(model_dir, f), "model/" + f)
    return path


def main():
    out_dir = os.path.join("runs", RUN)
    if os.path.exists(os.path.join(out_dir, "model")):
        raise SystemExit(f"이미 존재함: {out_dir}")
    os.makedirs(os.path.join(out_dir, "model"))

    print(f"[{RUN}] Load train...", flush=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y = df[TARGET].astype(int).values
    tr = (df["season"] <= 2023).values
    global_mean = float(y[tr].mean())

    anchor = build_anchor(df)
    priors = rate_priors(df[tr])
    X = engineer(df.drop(columns=[ID, TARGET]), global_mean, anchor=anchor, priors=priors)
    X = X.drop(columns=["p_matchup"])

    print(" 조건부 표 생성...", flush=True)
    _dfl = df.merge(pd.read_csv("recovered_labels.csv.gz"), on=ID, how="left")
    Ccols = cond.build_training_columns(_dfl)
    use = [c for c in cond.COND_COLS
           if any(c == "cond_" + n or c == "cond_" + n + "_dev" for n in COND_ONLY)]
    for c in use:
        X[c] = Ccols[c].values

    L = pd.read_csv("recovered_labels.csv.gz")
    L = df[[ID]].merge(L, on=ID, how="left")
    have = L["middle"].notna().values
    mr = ((L["middle"] == 1) | (L["reverse"] == 1)).values
    cls = np.where(y == 1, 2, np.where(mr, 0, 1)).astype(int)

    feature_cols = list(X.columns)
    cat_cols_here = [c for c in CAT_COLS if c in X.columns]
    for c in cat_cols_here:
        X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in cat_cols_here]
    assert "team13_transition" in feature_cols
    print(f" rows={len(df)}  feats={len(feature_cols)}  have={have.sum():,}", flush=True)

    print("\n--- 전체 데이터(2019-2024, have행만) 재학습 (검증 스킵, 220 best_iter 재사용) ---",
          flush=True)
    pool_all = Pool(X[have], cls[have], cat_features=ci)
    for sd, it in zip(SEEDS, BEST_ITERS):
        m = CatBoostClassifier(**dict(PARAMS, random_seed=sd, iterations=it)).fit(pool_all)
        m.save_model(os.path.join(out_dir, "model", f"model_Sym_{sd}.cbm"))
        print(f" Sym_{sd} iter={it} 저장", flush=True)

    for name, t_ in cond.build_tables(_dfl).items():
        t_.to_csv(os.path.join(out_dir, "model", f"cond_{name}.csv"),
                  index=False, encoding="utf-8")
    print(f" cond 표 {len(cond.SPECS)}개 저장", flush=True)

    last = int(df["season"].max()) + 1
    anchor[anchor["apply_season"] == last].to_csv(
        os.path.join(out_dir, "model", "anchor.csv"), index=False, encoding="utf-8")
    print(f" 기준점 표 저장 (apply_season={last})", flush=True)

    json.dump({"model_type": "multiclass_joint", "seeds": [f"Sym_{s}" for s in SEEDS],
               "feature_cols": feature_cols, "cat_cols": cat_cols_here,
               "global_mean": global_mean, "use_cond": True, "use_inseason": True,
               "use_role": False, "rate_priors": priors, "extra_cond": [],
               "offset_mc": OFFSET_MC},
              open(os.path.join(out_dir, "model", "meta.json"), "w", encoding="utf-8"))
    json.dump({"run": RUN, "note": NOTE, "model": "catboost_multiclass",
               "seeds": [f"Sym_{s}" for s in SEEDS], "params": PARAMS,
               "n_features": len(feature_cols), "best_iters": BEST_ITERS,
               "val_2024_with_offset": None, "lb_2025": None,
               "warning": "검증 스킵 — 220 best_iter/offset 재사용한 긴급배포"},
              open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nSaved {build_zip(out_dir)}", flush=True)


if __name__ == "__main__":
    main()
