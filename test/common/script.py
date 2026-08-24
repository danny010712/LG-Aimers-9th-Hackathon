"""평가 서버 추론 스크립트.

./data/test.csv 로드 → CatBoost 시드 앙상블 추론 → ./output/submission.csv 저장.
각 행 독립 예측 (test 내부 행 간 통계 사용 안 함).
피처 생성은 features.py 를 학습과 공유한다.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from features import engineer, prepare  # noqa: E402
import cond  # noqa: E402

ID, TARGET = "row_id", "control_success"


def main():
    data_dir = "./data"
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), encoding="utf-8-sig")
    sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"),
                      encoding="utf-8-sig")

    meta = json.load(open(os.path.join(BASE, "model", "meta.json"),
                          encoding="utf-8"))
    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]

    # global_mean은 학습 때 쓴 값을 그대로 재사용해야 한다 (test에서 재계산 금지).
    anchor = None
    if meta.get("use_inseason"):
        # 시즌내 분해 기준점(§5-10). 학습 때 train으로 만들어 zip에 실려 있다.
        # 각 행은 자기 asof와 이 표만 쓴다 — test의 다른 행은 보지 않는다.
        anchor = pd.read_csv(os.path.join(BASE, "model", "anchor.csv"),
                             encoding="utf-8")
    fe = engineer(test.drop(columns=[ID]), meta["global_mean"], anchor=anchor,
                  priors=meta.get("rate_priors"))
    if meta.get("use_cond"):
        # 조건부 표는 학습 때 train 전체로 만들어 zip에 실려 있다.
        # test에서 새로 계산하지 않는다 (test 내부 행 간 통계 사용 금지 규정).
        tables = {n: pd.read_csv(os.path.join(BASE, "model", f"cond_{n}.csv"),
                                 encoding="utf-8")
                  for n, _, _ in cond.SPECS}
        fe = cond.apply_tables(fe, tables)
    X = prepare(fe, feature_cols, cat_cols)
    pool = Pool(X, cat_features=[X.columns.get_loc(c) for c in cat_cols])

    def make_pool(cols):
        Z = prepare(fe, cols, cat_cols)
        return Pool(Z, cat_features=[Z.columns.get_loc(c) for c in cat_cols])

    def avg_proba(prefix, seeds, pl=None):
        ps = []
        for sd in seeds:
            m = CatBoostClassifier()
            m.load_model(os.path.join(BASE, "model", f"{prefix}{sd}.cbm"))
            ps.append(m.predict_proba(pl if pl is not None else pool)[:, 1])
        return np.mean(ps, axis=0)

    p = np.clip(avg_proba("model_", meta["seeds"]), 1e-6, 1 - 1e-6)

    off = meta.get("offset")
    if off:
        # 보조모델(mr/wayoff)은 주모델과 다른 피처 집합으로 학습됐을 수 있다.
        # 예: 013은 시즌내 분해 4열을 더 갖는데 보조모델은 009(003 피처)를
        # 그대로 복사해 쓴다. 그때 주모델용 Pool을 재사용하면 CatBoost가
        # 피처 불일치로 죽는다 → 보조 전용 Pool을 따로 만든다.
        # engineer() 출력(fe)은 하나로 두고 열 부분집합만 다르게 뽑는다.
        aux_pool = (make_pool(off["aux_feature_cols"])
                    if off.get("aux_feature_cols") else None)

        # 실패모드 offset (08 문서 §5). y=0 ⟺ (M∪R) ⊎ W 를 이용해
        # 합에서 상쇄되던 성분 정보를 되돌린다.
        # a=1·d=0 고정 — 스케일/절편을 적합하면 그게 calibration이 되어
        # 시즌 간 전이가 깨진다(자기연도 +53.8 → 한 해 건너 −210~−638).
        # mu는 학습 때 저장한 값. test에서 평균을 내면 규정 위반이다.
        def logit(q):
            q = np.clip(q, 1e-6, 1 - 1e-6)
            return np.log(q / (1 - q))

        z = (logit(p)
             + off["b"] * (logit(avg_proba("mr_", off["seeds"], aux_pool))
                           - off["mu_mr"])
             + off["c"] * (logit(avg_proba("wayoff_", off["seeds"], aux_pool))
                           - off["mu_wayoff"]))
        p = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)

    shift = meta.get("logit_shift")
    if shift:
        # 시즌 base rate 하락 보정 (08 §5-6). 트리는 미관측 시즌을 외삽하지 못해
        # 2025 예측이 2024 수준에 갇힌다. 상수는 **학습 때 계산해 저장**한 값이다
        # (test 평균을 보고 정하면 규정 위반).
        p = np.clip(1 / (1 + np.exp(-(np.log(p / (1 - p)) + shift))), 1e-6, 1 - 1e-6)

    pred_map = dict(zip(test[ID], p))
    sub[TARGET] = [pred_map.get(rid, 0.5) for rid in sub[ID]]

    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved ./output/submission.csv rows={len(sub)} "
          f"seeds={len(meta['seeds'])} offset={'Y' if off else 'N'} "
          f"mean={p.mean():.4f}")


if __name__ == "__main__":
    main()
