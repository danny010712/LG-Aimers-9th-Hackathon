"""평가 서버 추론 스크립트.

./data/test.csv 로드 → CatBoost 시드 앙상블 추론 → ./output/submission.csv 저장.
각 행 독립 예측 (test 내부 행 간 통계 사용 안 함).
피처 생성은 features.py 를 학습과 공유한다.

=== league-rate baseline 추가 ===
⚠️ meta["league_baseline"]["enabled"]가 True면, 성공모델(model_*.cbm)은
baseline 위에서 잔차만 학습된 상태다. CatBoost baseline은 .cbm에
저장되지 않으므로(실측 확인), 여기서도 학습 때와 똑같은 방식으로
baseline을 다시 계산해 Pool에 넣어야 한다. 안 넣으면 예측 평균이
크게 틀어진다 — 이 부분이 이번 변경에서 가장 중요한 포인트다.
mr_*.cbm / wayoff_*.cbm(보조모델)은 baseline을 쓰지 않는다(변경 없음).
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
import league_rate as lr  # noqa: E402

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
    # rate_means(EXTRA_FE용)도 마찬가지로 학습 때 저장된 값을 그대로 쓴다.
    fe = engineer(test.drop(columns=[ID]), meta["global_mean"],
                 extra_fe=meta.get("extra_fe", False),
                 rate_means=meta.get("rate_means"))
    if meta.get("use_cond"):
        tables = {n: pd.read_csv(os.path.join(BASE, "model", f"cond_{n}.csv"),
                                 encoding="utf-8")
                  for n, _, _ in cond.SPECS}
        fe = cond.apply_tables(fe, tables)
    X = prepare(fe, feature_cols, cat_cols)
    ci = [X.columns.get_loc(c) for c in cat_cols]

    # ---- league-rate baseline (성공모델 전용) ----
    lb = meta.get("league_baseline", {"enabled": False})
    main_baseline = None
    if lb.get("enabled"):
        # test는 raw test.csv에서 season/game_type을 봐야 한다 (fe/X는 문자열
        # 캐스팅 등으로 원본과 dtype이 달라질 수 있으므로 원본 test에서 가져옴).
        raw = test[["season"] + (["game_type"] if "game_type" in lb["group_cols"]
                                 else [])].copy()
        table, group_cols = lr.table_from_json(lb)
        # 2025(test) 시즌은 table에 없으므로 자동으로 held_out_estimate()가
        # 적용된다 (assign_baseline_logit 내부에서 "not in table.index" 분기).
        # test_override가 None이면(기본값) train.csv만으로 만든 표에서
        # league_rate.extrapolate()가 선형외삽으로 재계산한다 — 외부 문서를
        # 참조하지 않는다. 수동 override를 쓰려면 그 값이 train.csv만으로
        # 재현 가능한지 먼저 확인할 것 (data_description.md §6 "외부 데이터"
        # 조항 참고).
        main_baseline = lr.assign_baseline_logit(
            raw, table, group_cols, held_out_season=None,
            override=lb.get("test_override"))
        print(f" league_rate baseline 적용: 그룹={group_cols} "
              f"평균(로짓)={main_baseline.mean():+.4f} "
              f"(override={lb.get('test_override')})")

    pool_main = Pool(X, cat_features=ci, baseline=main_baseline)

    def avg_proba(prefix, seeds, pool):
        ps = []
        for sd in seeds:
            m = CatBoostClassifier()
            m.load_model(os.path.join(BASE, "model", f"{prefix}{sd}.cbm"))
            ps.append(m.predict_proba(pool)[:, 1])
        return np.mean(ps, axis=0)

    p = np.clip(avg_proba("model_", meta["seeds"], pool_main), 1e-6, 1 - 1e-6)

    off = meta.get("offset")
    if off:
        # ⚠️ mr/wayoff 보조모델은 주모델과 다른 feature_cols/extra_fe로
        # 학습됐을 수 있다(예: 주모델만 EXTRA_FE를 켠 경우). 주모델용 X를
        # 그대로 재사용하면 CatBoost가 "피처 이름이 다르다"고 에러를 낸다
        # (실측 확인된 버그, 2026-08) — 보조모델 전용 X_aux를 따로 만든다.
        af = off.get("aux_features")
        if af is None:
            # 구버전 meta(이 수정 이전에 만든 offset run) 호환용 — 주모델과
            # 같은 스키마였다고 가정한다(당시엔 실제로 항상 같았음).
            af = {"global_mean": meta["global_mean"],
                 "extra_fe": meta.get("extra_fe", False),
                 "rate_means": meta.get("rate_means"),
                 "feature_cols": feature_cols, "cat_cols": cat_cols}
        fe_aux = engineer(test.drop(columns=[ID]), af["global_mean"],
                          extra_fe=af.get("extra_fe", False),
                          rate_means=af.get("rate_means"))
        X_aux = prepare(fe_aux, af["feature_cols"], af["cat_cols"])
        ci_aux = [X_aux.columns.get_loc(c) for c in af["cat_cols"]]

        # mr/wayoff 각각 자기만의 league-rate baseline을 쓸 수 있다(09 §1-E:
        # 드리프트 방향이 서로 달라 하나로 공유하면 안 됨) — 있으면 각각
        # 별도 Pool로 만든다. 없으면(구버전 offset run) baseline 없이 동일.
        alb = off.get("aux_league_baseline", {})
        pools = {}
        for name in ("mr", "wayoff"):
            info = alb.get(name, {"enabled": False})
            base_aux = None
            if info.get("enabled"):
                table_aux, gc_aux = lr.table_from_json(info)
                base_aux = lr.assign_baseline_logit(
                    test, table_aux, gc_aux, held_out_season=None,
                    override=info.get("test_override"))
            pools[name] = Pool(X_aux, cat_features=ci_aux, baseline=base_aux)

        # 실패모드 offset (08 문서 §5). y=0 ⟺ (M∪R) ⊎ W 를 이용해
        # 합에서 상쇄되던 성분 정보를 되돌린다.
        # a=1·d=0 고정 — 스케일/절편을 적합하면 그게 calibration이 되어
        # 시즌 간 전이가 깨진다(자기연도 +53.8 → 한 해 건너 −210~−638).
        # mu는 학습 때 저장한 값. test에서 평균을 내면 규정 위반이다.
        def logit(q):
            q = np.clip(q, 1e-6, 1 - 1e-6)
            return np.log(q / (1 - q))

        z = (logit(p)
             + off["b"] * (logit(avg_proba("mr_", off["seeds"], pools["mr"]))
                           - off["mu_mr"])
             + off["c"] * (logit(avg_proba("wayoff_", off["seeds"], pools["wayoff"]))
                           - off["mu_wayoff"]))
        p = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)

    shift = meta.get("logit_shift")
    if shift:
        # 시즌 base rate 하락 보정 (08 §5-6). league-rate baseline이 이미
        # 평균 수준 대부분을 잡아준다면, 이 값은 아주 작은 잔여 보정이어야
        # 정상이다(0 또는 0에 가까운 값). 큰 폭이면 baseline이 기대만큼
        # 작동하지 않았다는 신호 — build_shift.py의 진단 출력을 볼 것.
        p = np.clip(1 / (1 + np.exp(-(np.log(p / (1 - p)) + shift))), 1e-6, 1 - 1e-6)

    pred_map = dict(zip(test[ID], p))
    sub[TARGET] = [pred_map.get(rid, 0.5) for rid in sub[ID]]

    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved ./output/submission.csv rows={len(sub)} "
          f"seeds={len(meta['seeds'])} offset={'Y' if off else 'N'} "
          f"league_baseline={'Y' if lb.get('enabled') else 'N'} "
          f"mean={p.mean():.4f}")


if __name__ == "__main__":
    main()
