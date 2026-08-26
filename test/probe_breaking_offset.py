"""`breaking`을 offset 항으로 — out-of-year 판정 (09 §3-K, 2026-08-25).

배경: 09 §2-D 표는 `ball`/`strike`/`inplay`/**구종 3종**을 한 줄로 묶어 "+0.2~1.4"로
기록했다. 그러나 실측된 것은 `ball`(+0.2)·`strike`(+0.8)·`inplay`(+1.0~1.4)뿐이고
구종의 "+0.6"은 **2단 혼합모델**(§3-E)이지 offset 항이 아니다. → **미측정 가능성.**

2024 캐시로 잰 예비 결과(013 + 배포 shift):
    전역 breaking 항  Δ +6.23   레벨 제거 후 **+6.22**   예측평균 Δ +0.000003
    021 풀스택 위     Δ +3.91   레벨 제거 후 +3.90
    K-fold OOF(2024 내부) +5.2 (K=5·10 × 3시드, 계수 −0.014~−0.024로 안정)
→ §3-J(c 2S분리)를 죽인 **레벨 테스트를 통과한다.** 2S 분리는 불필요(+6.44 vs +6.23)
  → 전역 계수만 쓰면 `d` 적합 함정도 회피.

남은 것은 out-of-year 하나. 이 스크립트가 `breaking` 보조모델을 연도별로 학습해
`probe_offset_forms_preds.csv.gz`(success/mr/wayoff)에 붙인다.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "common")
from features import engineer, CAT_COLS  # noqa: E402

YEARS = [2022, 2023, 2024]
OUT = "probe_breaking_preds.csv.gz"
PARAMS = dict(iterations=2000, learning_rate=0.05, depth=6, thread_count=-1,
              verbose=0, eval_metric="Logloss", early_stopping_rounds=100)


def main():
    lab = pd.read_csv("recovered_labels.csv.gz")
    df = pd.read_csv("data/train.csv", encoding="utf-8-sig").merge(lab, on="row_id")
    df = df[df["middle"].notna()].reset_index(drop=True)
    print(f"rows {len(df):,}", flush=True)
    drop = (["row_id", "control_success", "success", "mr", "wayoff"]
            + ["middle", "reverse", "ball", "strike", "fastball", "breaking", "offspeed"])
    keep = [c for c in df.columns if c not in drop]

    out = []
    for T in YEARS:
        tr = (df["season"] <= T - 1).values
        va = (df["season"] == T).values
        gm = float(df.loc[tr, "control_success"].mean())
        X = engineer(df[keep], gm)
        for c in CAT_COLS:
            X[c] = X[c].astype(str)
        ci = [X.columns.get_loc(c) for c in CAT_COLS]
        y = df["breaking"].astype(int).values
        m = CatBoostClassifier(**PARAMS).fit(
            Pool(X[tr], y[tr], cat_features=ci),
            eval_set=Pool(X[va], y[va], cat_features=ci), use_best_model=True)
        p = m.predict_proba(Pool(X[va], cat_features=ci))[:, 1]
        print(f"  T={T} breaking iter={m.get_best_iteration():<5} train={tr.sum():,} "
              f"mean_p={p.mean():.4f}", flush=True)
        out.append(pd.DataFrame({"row_id": df.loc[va, "row_id"].values,
                                 "season": T, "p_breaking": p}))
    pd.concat(out, ignore_index=True).to_csv(OUT, index=False, compression="gzip")
    print(f"저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()
