"""보조 라벨 탐침 — 복원 라벨(§1-C)이 실제로 학습되는지 확인한다.

질문 1: 보조 라벨은 타깃보다 신호가 세다(η² 10~140배, 09 §1-F). 그게 실제
        모델 예측력으로 이어지는가?
질문 2: 보조 예측 3~4개만으로 control_success를 얼마나 설명하는가?
        (스태킹 안 (b)가 성립할지 가르는 숫자)

방법: 기존 피처 파이프라인을 그대로 쓰고 **y만 바꿔서** CatBoost를 각각 학습.
      2019~2023 학습 → 2024 홀드아웃. 1시드. 탐침이므로 lr을 올려 가볍게.

⚠️ 로컬 2024 점수는 LB를 예측하지 못한다(08 §0-0). 여기서 보는 건 절대 점수가 아니라
   **같은 조건에서 라벨끼리의 상대 비교**다.
"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "common")
from features import engineer, CAT_COLS  # noqa: E402

DATA = "data/train.csv"
LABELS = "recovered_labels.csv.gz"
OUT = "probe_aux_result.json"

PARAMS = dict(iterations=1500, learning_rate=0.08, depth=6, thread_count=-1,
              verbose=0, eval_metric="Logloss", early_stopping_rounds=80,
              random_seed=42)

# 학습할 보조 라벨. mr = middle 또는 reverse (실패 = mr ∪ wayoff, 서로소)
TASKS = ["success", "middle", "reverse", "wayoff", "mr", "fastball", "breaking"]


def bss(p, y):
    """라벨마다 base rate가 다르므로 자기 base rate 기준으로 정규화해 비교한다."""
    r = y.mean()
    brier = float(np.mean((p - y) ** 2))
    return brier, max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def auc(p, y):
    o = np.argsort(p)
    rank = np.empty(len(p), float)
    rank[o] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (rank[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    print("Load...", flush=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    lab = pd.read_csv(LABELS)
    df = df.merge(lab, on="row_id", how="left")

    keep = df["middle"].notna().values          # 투수당 마지막 1투구(792행) 제외
    df = df[keep].reset_index(drop=True)
    df["wayoff"] = ((df["control_success"] == 0) & (df["middle"] == 0)
                    & (df["reverse"] == 0)).astype(int)
    df["mr"] = ((df["middle"] == 1) | (df["reverse"] == 1)).astype(int)
    df["success"] = df["control_success"]

    tr = (df["season"] <= 2023).values
    va = (df["season"] == 2024).values
    global_mean = float(df.loc[tr, "control_success"].mean())

    drop = ["row_id", "control_success"] + TASKS + ["ball", "strike", "offspeed"]
    X = engineer(df.drop(columns=[c for c in drop if c in df.columns]), global_mean)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in CAT_COLS]
    print(f" rows={len(df):,}  feats={X.shape[1]}  "
          f"train={tr.sum():,}  val2024={va.sum():,}", flush=True)

    pool_tr_X = X[tr]
    pool_va = Pool(X[va], cat_features=ci)

    res, preds = {}, {}
    for t in TASKS:
        y = df[t].astype(int).values
        m = CatBoostClassifier(**PARAMS).fit(
            Pool(pool_tr_X, y[tr], cat_features=ci),
            eval_set=Pool(X[va], y[va], cat_features=ci), use_best_model=True)
        p = m.predict_proba(pool_va)[:, 1]
        preds[t] = p
        br, sc = bss(p, y[va])
        res[t] = dict(base_rate=float(y[va].mean()), brier=br, bss=sc,
                      auc=float(auc(p, y[va])), best_iter=m.get_best_iteration())
        print(f" {t:9s} base={res[t]['base_rate']:.4f}  BSS={sc:8.1f}  "
              f"AUC={res[t]['auc']:.4f}  iter={res[t]['best_iter']}", flush=True)

    # 예측 저장 — 이후 분석을 재학습 없이 하기 위해
    pd.DataFrame({"row_id": df.loc[va, "row_id"].values,
                  "y": df.loc[va, "control_success"].values,
                  **{f"p_{t}": preds[t] for t in TASKS}}).to_csv(
        "probe_aux_val2024.csv.gz", index=False, compression="gzip")

    # --- 보조 예측이 success 예측 위에 무언가를 더하는가 (= 안 (b)의 핵심) ---
    # 2024를 절반으로 갈라 한쪽에서 결합기를 학습하고 다른 쪽에서 평가 (누수 방지).
    print("\n--- 결합 (2024 절반 학습 / 절반 평가) ---", flush=True)
    yv = df.loc[va, "control_success"].astype(int).values
    h = np.arange(len(yv)) % 2 == 0
    from sklearn.linear_model import LogisticRegression

    def logit(z):
        z = np.clip(z, 1e-6, 1 - 1e-6)
        return np.log(z / (1 - z))

    def combo(names, gbdt=False):
        Z = logit(np.column_stack([preds[n] for n in names]))
        if gbdt:   # 교호작용 허용 — 실제 (b)는 메인 모델이 보조 예측을 자유롭게 쓴다
            m = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=4,
                                   thread_count=-1, verbose=0,
                                   random_seed=42).fit(Z[h], yv[h])
            return bss(m.predict_proba(Z[~h])[:, 1], yv[~h])
        lr = LogisticRegression(max_iter=1000).fit(Z[h], yv[h])
        return bss(lr.predict_proba(Z[~h])[:, 1], yv[~h])

    combos = {
        "[기준] success 단독": ["success"],
        "보조만: mr+wayoff": ["mr", "wayoff"],
        "보조만: m+r+wayoff": ["middle", "reverse", "wayoff"],
        "보조만: m+r+wayoff+구종": ["middle", "reverse", "wayoff",
                                    "fastball", "breaking"],
        "success + mr+wayoff": ["success", "mr", "wayoff"],
        "success + 구종2": ["success", "fastball", "breaking"],
        "success + 전부": ["success", "middle", "reverse", "wayoff", "mr",
                           "fastball", "breaking"],
    }
    res["combo"] = {}
    for k, names in combos.items():
        br, sc = combo(names)
        _, sg = combo(names, gbdt=True)
        res["combo"][k] = dict(brier=br, bss_linear=sc, bss_gbdt=sg)
        print(f" {k:26s} 선형={sc:8.1f}   GBDT={sg:8.1f}", flush=True)

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
