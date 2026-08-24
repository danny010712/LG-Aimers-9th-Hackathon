"""offset 계수 적합용 검증 예측 캐시 생성 (2019~23 학습 → 2024 예측).

`train_offset.py`의 `fit_offset()`은 성공모델의 **out-of-sample 2024 예측**에서
b·c·mu를 적합한다. 그런데 그 캐시는 피처 구성이 바뀔 때마다 새로 만들어야 한다 —
003 설정으로 만든 캐시를 013(시즌내 분해 4열 추가) 모델에 쓰면 계수가 어긋난다.

`train_local.py`는 검증 예측을 저장하지 않으므로 여기서 같은 조건으로 다시 뽑는다.
(같은 시드·같은 파라미터·같은 데이터 → train_local의 검증 단계와 동일한 모델)

⚠️ 저장 길이는 `season==2024 & 라벨 복원됨`으로 맞춘다. mr/wayoff 기존 캐시가
   그 필터로 저장돼 있어서(253,116행), 안 맞추면 fit의 nll()에서 길이가 어긋난다.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "common")
from features import engineer, build_anchor, rate_priors, CAT_COLS  # noqa: E402

OUT = "artifacts/auxpred_ins"
SEEDS = [42, 7, 2024]
USE_INSEASON = True                 # features.py의 anchor 경로를 켤지
PARAMS = dict(iterations=2000, learning_rate=0.05, depth=6,
              thread_count=-1, verbose=0, eval_metric="Logloss",
              early_stopping_rounds=100)
ID, TARGET = "row_id", "control_success"


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Load train...", flush=True)
    df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    y = df[TARGET].astype(int).values
    tr = (df["season"] <= 2023).values
    va = (df["season"] == 2024).values

    global_mean = float(y[tr].mean())
    anchor = build_anchor(df) if USE_INSEASON else None
    priors = rate_priors(df[tr]) if USE_INSEASON else None
    X = engineer(df.drop(columns=[ID, TARGET]), global_mean, anchor=anchor,
                 priors=priors)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in CAT_COLS]

    L = pd.read_csv("recovered_labels.csv.gz")
    have = df[[ID]].merge(L, on=ID, how="left")["middle"].notna().values
    have_va = have[va]
    print(f" feats={X.shape[1]}  global_mean={global_mean:.4f}  "
          f"va={va.sum():,}  저장 길이={have_va.sum():,}", flush=True)

    pool_tr = Pool(X[tr], y[tr], cat_features=ci)
    pool_va = Pool(X[va], y[va], cat_features=ci)
    for sd in SEEDS:
        m = CatBoostClassifier(**dict(PARAMS, random_seed=sd)).fit(
            pool_tr, eval_set=pool_va, use_best_model=True)
        p = m.predict_proba(pool_va)[:, 1]
        np.save(os.path.join(OUT, f"success_2024_{sd}.npy"), p[have_va])
        r = y[va].mean()
        score = 100000 * (1 - np.mean((p - y[va]) ** 2) / (r * (1 - r)))
        print(f" seed {sd}: iter={m.get_best_iteration()} score~{score:.1f} 저장",
              flush=True)


if __name__ == "__main__":
    main()
