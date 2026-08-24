"""015 shift 산출 전 진단 — 가짜 test 구성에 따라 예측 평균이 얼마나 달라지나.

시즌내 분해(§5-10)를 넣은 뒤 §5-6의 가짜 test가 깨진다:
2024 행의 `asof`는 이미 2024년 말까지이므로, 2025용 기준점(2024말)을 빼면
`dn=0`이 되어 `ins_*`가 통산값으로 붕괴한다. 실제 2025 test는 `dn>0`이다
(5행 샘플 실측: dn = +380, +399).

두 구성을 재서 차이를 본다.
  A) 배포와 동일 경로: apply_season=2025 기준점  → ins_ 붕괴 (dn=0)
  B) 시즌 진행 중 모사: apply_season=2024 기준점 → ins_ 정상 (dn 중앙 505)
     2024 행을 "미관측 시즌이 절반쯤 진행된 상태"로 보는 것. 실제 2025의
     ins_ 분포와 구조가 같다(레벨만 2024치라 ~0.01 높다).
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "common")
from features import engineer, prepare, build_anchor  # noqa: E402

RUN = "014_offset_inseason"
ID, TARGET = "row_id", "control_success"


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def main():
    mdir = os.path.join("runs", RUN, "model")
    meta = json.load(open(os.path.join(mdir, "meta.json"), encoding="utf-8"))
    df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    anchor_all = build_anchor(df)

    fake = df[df["season"] == 2024].drop(columns=[TARGET]).copy()
    fake["season"] = 2025
    print(f"가짜 test {len(fake):,}행 (2024 행, season->2025)", flush=True)

    for tag, apply_s in [("A 배포경로(2025 기준점)", 2025),
                         ("B 시즌진행 모사(2024 기준점)", 2024)]:
        a = anchor_all[anchor_all["apply_season"] == apply_s].copy()
        a["apply_season"] = 2025          # fake의 season 값에 맞춘다
        fe = engineer(fake.drop(columns=[ID]), meta["global_mean"], anchor=a)
        X = prepare(fe, meta["feature_cols"], meta["cat_cols"])
        ci = [X.columns.get_loc(c) for c in meta["cat_cols"]]
        pool = Pool(X, cat_features=ci)

        off = meta["offset"]
        Xa = prepare(fe, off["aux_feature_cols"], meta["cat_cols"])
        pool_a = Pool(Xa, cat_features=[Xa.columns.get_loc(c)
                                        for c in meta["cat_cols"]])

        def avg(prefix, seeds, pl):
            return np.mean([CatBoostClassifier().load_model(
                os.path.join(mdir, f"{prefix}{s}.cbm")).predict_proba(pl)[:, 1]
                for s in seeds], axis=0)

        p = np.clip(avg("model_", meta["seeds"], pool), 1e-6, 1 - 1e-6)
        base = p.mean()
        z = (logit(p)
             + off["b"] * (logit(avg("mr_", off["seeds"], pool_a)) - off["mu_mr"])
             + off["c"] * (logit(avg("wayoff_", off["seeds"], pool_a))
                           - off["mu_wayoff"]))
        p = 1 / (1 + np.exp(-z))
        print(f"\n[{tag}]")
        print(f"  ins_pitcher_n 중앙 {fe['ins_pitcher_n'].median():.0f}  "
              f"ins_pitcher_success_rate 평균 {fe['ins_pitcher_success_rate'].mean():.4f}")
        print(f"  성공모델만 평균 {base:.4f}   offset 후 평균 {p.mean():.4f}",
              flush=True)


if __name__ == "__main__":
    main()
