"""C2 — 4번째 숨은 라벨 `inplay` 추출·검정.

`ball + strike ∈ {0,1}` 이고 18.7%가 둘 다 0이다(09 §1-C).
그 여집합 = 타자가 스윙해 맞힌 투구(인플레이/파울)로 추정된다.

라벨 수준에선 `inplay = 1 − ball − strike`로 결정적이지만, offset은 **로짓의 선형결합**이라
`logit(p_inplay)`이 `logit(p_ball)`·`logit(p_strike)`의 선형결합이 아니다.
→ 08 §5-4가 `ball`/`strike`를 기각했다고 해서 자동으로 죽지 않는다.

핵심 검정: `p_inplay`가 `p_success`와 직교하는가.
(`wayoff`는 −0.014로 거의 직교해서 offset이 작동했다.)
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "common")
from features import engineer, CAT_COLS  # noqa: E402

PARAMS = dict(iterations=1500, learning_rate=0.08, depth=6, thread_count=-1,
              verbose=0, eval_metric="Logloss", early_stopping_rounds=80,
              random_seed=42)


def bss(p, y):
    r = y.mean()
    return max(0.0, 100000 * (1 - float(np.mean((p - y) ** 2)) / (r * (1 - r))))


def main():
    lab = pd.read_csv("recovered_labels.csv.gz")
    df = pd.read_csv("data/train.csv", encoding="utf-8-sig").merge(lab, on="row_id")
    df = df[df["middle"].notna()].reset_index(drop=True)

    df["inplay"] = 1 - df["ball"] - df["strike"]
    v = df["inplay"]
    print(f"inplay 값 분포: {v.value_counts().to_dict()}", flush=True)
    assert v.isin([0, 1]).all(), "inplay가 0/1이 아니다"
    print(f"inplay 비율 {v.mean():.4f}", flush=True)
    print("시즌별 inplay 비율 (R만):", flush=True)
    print(df[df.game_type == "R"].groupby("season").inplay.mean().round(4).to_string(),
          flush=True)

    tr = (df["season"] <= 2023).values
    va = (df["season"] == 2024).values
    gm = float(df.loc[tr, "control_success"].mean())

    drop = ["row_id", "control_success", "inplay", "success", "middle", "reverse",
            "ball", "strike", "fastball", "breaking", "offspeed"]
    X = engineer(df.drop(columns=[c for c in drop if c in df.columns]), gm)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in CAT_COLS]

    y = df["inplay"].astype(int).values
    m = CatBoostClassifier(**PARAMS).fit(
        Pool(X[tr], y[tr], cat_features=ci),
        eval_set=Pool(X[va], y[va], cat_features=ci), use_best_model=True)
    p = m.predict_proba(Pool(X[va], cat_features=ci))[:, 1]
    print(f"\ninplay 모델: base={y[va].mean():.4f}  BSS={bss(p, y[va]):.1f}  "
          f"iter={m.get_best_iteration()}", flush=True)

    out = pd.read_csv("probe_aux_val2024.csv.gz")
    add = pd.DataFrame({"row_id": df.loc[va, "row_id"].values, "p_inplay": p})
    out = out.merge(add, on="row_id", how="left")
    out.to_csv("probe_aux_val2024.csv.gz", index=False, compression="gzip")
    print(f"저장: probe_aux_val2024.csv.gz (+p_inplay, 결측 {out.p_inplay.isna().sum()})",
          flush=True)


if __name__ == "__main__":
    main()
