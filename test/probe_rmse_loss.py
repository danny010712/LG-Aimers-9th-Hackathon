"""RMSE loss(Brier 직접 최소화) vs Logloss — 08 §7 "미측정 학습 노브".

지표가 Brier인데 우리는 Logloss로 학습해 왔다. 둘 다 proper이지만 가중이 다르다
(Logloss는 극단 확률에 훨씬 큰 벌점). RMSE는 평가지표를 직접 민다.

베이스는 **013 구성**(= 021의 성공모델). USE_COND/USE_ROLE 없음, ins 4열.
Logloss 팔이 859.0 부근을 재현하는지가 자기검증이다 (09 §0-12).

⚠️ 손실함수는 외삽 제어(depth류)가 아니다 — 효과가 연도 넘어 측정된다.
   그래서 제출 슬롯 없이 로컬로 판정 가능. 여기서 죽으면 끝.

산출: artifacts/rmse_probe/{loss}_2024_{seed}.npy (하류 offset 재적합용)
"""
import io
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

sys.path.insert(0, "common")
from features import engineer, build_anchor, rate_priors, CAT_COLS  # noqa: E402

ID, TARGET = "row_id", "control_success"
SEEDS = [42, 7, 2024]
OUT = "artifacts/rmse_probe"
BASE_PARAMS = dict(iterations=2000, learning_rate=0.05, depth=6,
                   thread_count=-1, verbose=0, early_stopping_rounds=100)


def bss(p, y):
    r = y.mean()
    return max(0.0, 100000 * (1 - np.mean((p - y) ** 2) / (r * (1 - r))))


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[load]", flush=True)
    df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    y = df[TARGET].astype(int).values
    tr = (df["season"] <= 2023).values
    va = (df["season"] == 2024).values

    gm = float(y[tr].mean())
    X = engineer(df.drop(columns=[ID, TARGET]), gm,
                 anchor=build_anchor(df), priors=rate_priors(df[tr]))
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in CAT_COLS]
    print(f" rows={len(df)} feats={X.shape[1]} gm={gm:.4f}", flush=True)

    pool_tr = Pool(X[tr], y[tr], cat_features=ci)
    pool_va = Pool(X[va], y[va], cat_features=ci)
    yv = y[va].astype(float)

    res = {}
    for loss in ("Logloss", "RMSE"):
        preds = []
        for sd in SEEDS:
            t0 = time.time()
            if loss == "Logloss":
                m = CatBoostClassifier(**BASE_PARAMS, eval_metric="Logloss",
                                       random_seed=sd)
                m.fit(pool_tr, eval_set=pool_va, use_best_model=True)
                p = m.predict_proba(pool_va)[:, 1]
            else:
                m = CatBoostRegressor(**BASE_PARAMS, loss_function="RMSE",
                                      eval_metric="RMSE", random_seed=sd)
                m.fit(pool_tr, eval_set=pool_va, use_best_model=True)
                p = m.predict(pool_va)
            n_out = int(((p < 0) | (p > 1)).sum())
            p = np.clip(p, 1e-6, 1 - 1e-6)
            preds.append(p)
            np.save(os.path.join(OUT, f"{loss}_2024_{sd}.npy"), p)
            print(f" {loss:<8} seed={sd:<5} iter={m.get_best_iteration():<5} "
                  f"score~{bss(p, yv):.1f}  범위밖 {n_out}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        pm = np.mean(preds, axis=0)
        res[loss] = pm
        print(f"[{loss}] {len(SEEDS)}시드 평균 score~{bss(pm, yv):.1f}  "
              f"pred mean={pm.mean():.4f} 범위 {pm.min():.4f}~{pm.max():.4f}\n",
              flush=True)

    a, b = bss(res["Logloss"], yv), bss(res["RMSE"], yv)
    print(f"=== Δ(RMSE − Logloss) = {b - a:+.1f} BSS  (기준 013 = 859.0) ===")
    print(f"    두 예측 corr = {np.corrcoef(res['Logloss'], res['RMSE'])[0,1]:.5f}")
    # 블렌드도 함께 — 구조가 다르면 평균이 이길 수 있다
    for w in (0.25, 0.5, 0.75):
        print(f"    blend w_rmse={w:.2f}  "
              f"score~{bss((1-w)*res['Logloss'] + w*res['RMSE'], yv):.1f}")


if __name__ == "__main__":
    main()
