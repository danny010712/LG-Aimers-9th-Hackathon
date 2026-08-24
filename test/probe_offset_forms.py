"""O1~O3 — offset **형태** 축 검정 (09 §3-I).

질문: 현행 2파라미터(`mr`+`wayoff`)보다 1파라미터 형태가 out-of-year에서 나은가?
  O1  contrast = logit(p_mr) − logit(p_wayoff)      (1 파라미터)
  O2a mr 단독 / O2b wayoff 단독                      (1 파라미터, 08 §5-4가 안 해본 "빼기")
  O3  w·p_success + (1−w)·(1 − p_mr − p_wayoff)      (확률공간 항등식, 1 파라미터)
  V0  mr + wayoff                                    (현행 2 파라미터, 기준)

⚠️ 전이를 **여러 개**로 잰다. CLAUDE.md 채택기준 5항 — 전이 1개짜리 순위는 증거가 아니다
   (246조합을 2022→2024 하나로 줄 세워 `both`에 속은 사건이 그 규칙의 유래).

프로토콜: 연도 T에 대해 `season ≤ T−1` 학습 → T 예측. (success/mr/wayoff 각각)
  계수·mu는 **출처 연도 S**에서 적합 → **목표 연도 T**에 적용. S ≠ T 인 조합만 본다.
  a=1·d=0 고정 (08 §0-5: 자유 계수는 시즌 전이가 깨진다).

📎 보조모델 시드 수는 델타에 무관(08 §5-4: (2,2) +25.7 / (4,4) +26.6) → 1시드로 비교한다.
채점은 **R만**과 전체 둘 다 (08 §0-4: F가 만든 공짜 점수 제거).
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "common")
from features import engineer, CAT_COLS  # noqa: E402

YEARS = [2021, 2022, 2023, 2024]
LABELS = ["success", "mr", "wayoff"]
OUT = "probe_offset_forms_preds.csv.gz"
PARAMS = dict(iterations=2000, learning_rate=0.05, depth=6, thread_count=-1,
              verbose=0, eval_metric="Logloss", early_stopping_rounds=100,
              random_seed=42)


def main():
    lab = pd.read_csv("recovered_labels.csv.gz")
    df = pd.read_csv("data/train.csv", encoding="utf-8-sig").merge(lab, on="row_id")
    df = df[df["middle"].notna()].reset_index(drop=True)
    df["wayoff"] = ((df["control_success"] == 0) & (df["middle"] == 0)
                    & (df["reverse"] == 0)).astype(int)
    df["mr"] = ((df["middle"] == 1) | (df["reverse"] == 1)).astype(int)
    df["success"] = df["control_success"]
    print(f"rows {len(df):,}", flush=True)

    drop = (["row_id", "control_success"] + LABELS
            + ["middle", "reverse", "ball", "strike", "fastball", "breaking", "offspeed"])
    keep = [c for c in df.columns if c not in drop]

    out = []
    for T in YEARS:
        tr = (df["season"] <= T - 1).values
        va = (df["season"] == T).values
        gm = float(df.loc[tr, "control_success"].mean())
        X = engineer(df[keep], gm)          # global_mean은 학습 구간에서만
        for c in CAT_COLS:
            X[c] = X[c].astype(str)
        ci = [X.columns.get_loc(c) for c in CAT_COLS]
        pool_va = Pool(X[va], cat_features=ci)
        res = {"row_id": df.loc[va, "row_id"].values,
               "season": T,
               "game_type": df.loc[va, "game_type"].values,
               "y": df.loc[va, "control_success"].values}
        for lb in LABELS:
            y = df[lb].astype(int).values
            m = CatBoostClassifier(**PARAMS).fit(
                Pool(X[tr], y[tr], cat_features=ci),
                eval_set=Pool(X[va], y[va], cat_features=ci), use_best_model=True)
            res["p_" + lb] = m.predict_proba(pool_va)[:, 1]
            print(f"  T={T} {lb:8s} iter={m.get_best_iteration():<5} "
                  f"train={tr.sum():,}", flush=True)
        out.append(pd.DataFrame(res))

    pd.concat(out, ignore_index=True).to_csv(OUT, index=False, compression="gzip")
    print(f"\n저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()
