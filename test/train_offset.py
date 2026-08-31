"""실패모드 offset 빌더 — 기존 run 위에 보정항을 얹는다 (08 문서 §5).

`BASE_RUN`의 성공 모델을 **파일째 복사**한다. 재학습하지 않는다.
→ 성공 성분이 그 run의 LB 점수를 낸 모델과 동일 = 진짜 단일 변수.
  009 = 003(881.73) + offset → LB **945.40** (+63.7, 팀 최고 갱신)
  010 = 007(7시드) + offset  ← 현재 설정. 009 대비 시드 수만 다르다.

얹는 것:
  logit(p) = logit(p_success) + b·(logit(p_mr) − mu_mr) + c·(logit(p_wayoff) − mu_wayoff)

  mr     = middle ∪ reverse      (겹침 50,266건이 있어 M,R을 따로 두면 이중계산)
  wayoff = 실패인데 M,R 둘 다 0   (정의상 M∪R과 서로소)
  → y=0 ⟺ (M∪R) ⊎ W, 예외 0건 (09 문서 §1-D, 본 세션 독립 검증)

🔴 a(스케일)·d(절편)는 적합하지 않는다. 그게 곧 calibration이고 시즌 간 전이가 깨진다:
   무제약형 자기연도 +53.8 → 한 해 건너 −210~−638. a=1·d=0 고정이 필수다.
🔴 mu는 **학습 때 계산해 저장**한다. test에서 평균을 내면 규정 위반(test 내부 행간 통계).

계수 b,c와 mu는 artifacts/auxpred/ 의 검증 예측(2019~23 학습 → 2024)에서 적합한다.
배포 유사 실측: 2022 계수를 2024에 적용 → +26.6 (08 §5).

라벨: test/recovered_labels.csv.gz (09 세션 산출, 본 세션 검증 완료).
"""
import io
import json
import os
import shutil
import sys
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize

sys.path.insert(0, "common")
from features import (engineer, build_anchor, rate_priors,  # noqa: E402
                      CAT_COLS)
import cond  # noqa: E402

RUN = "218_offset_base044_mrbh"
BASE_RUN = "209_base044"             # 성공 모델을 가져올 run (재학습하지 않는다) — 순수 044
AUX_SEEDS = [42, 7, 2024]             # 보조모델 시드
# 보조모델 개선 3단계: 216(mr+cond_ph, LB+3.83 확정) 위에 cond_bh(+dev, 타자 플래툰)를
# 추가. 주모델에서는 실패(052, LB-9.33)했지만 주모델에 아예 없는 열이라(COND_ONLY=[ph]만
# 사용) 보조모델엔 안전할 수 있다는 가설 - cond_ph의 성공 패턴과 대칭.
# 사전검증(3시드x2세트, mr+cond_ph 기준 대비): Δ+3.6/+9.0 (둘 다 양수, cond_ph와
# 같은 채택 기준 통과).
# 참고: ins_pitcher_success_rate/n 추가(217)는 LB-1.87로 216보다 나빠서 기각됨 —
# 이번엔 그 축은 빼고 cond_bh만 추가.
AUX_FROM = ""    # 빈 문자열 = falsy -> 재학습 경로
AUX_COPY_FROM = "009_offset"   # AUX_COLS(57열) 기본 목록 + wayoff 복사 출처
RETRAIN_AUX = ["mr"]          # wayoff는 009_offset 그대로 복사(단일변수 유지)
MR_EXTRA_COLS = ["cond_ph", "cond_ph_dev", "cond_bh", "cond_bh_dev"]
AUX_OUT = "artifacts/auxpred_mrbh_aux"
# 계수 적합에 쓸 검증 예측(2019~23 학습 -> 2024)의 시드.
# 성공 쪽은 BASE_RUN의 시드 수와 맞춰야 한다.
FIT_SUCCESS_SEEDS = [42, 7, 2024]  # 209_base044 3시드
# ⚠️ 성공모델 캐시는 BASE_RUN(209_base044, 순정)의 검증예측이어야 한다.
CACHE = "artifacts/auxpred_base044"   # 209_base044의 VALPRED_DIR과 일치
AUX_CACHE = "artifacts/auxpred"  # wayoff(복사)용 009_offset 원본 검증예측
BEST_ITER = {"mr": {42: 360, 7: 480, 2024: 404},
             "wayoff": {42: 351, 7: 354, 2024: 438}}
PARAMS = dict(iterations=2000, learning_rate=0.05, depth=6,
              thread_count=-1, verbose=0, eval_metric="Logloss")

DATA = "data/train.csv"
COMMON = "common"
ID, TARGET = "row_id", "control_success"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bss(p, y):
    r = y.mean()
    brier = float(np.mean((p - y) ** 2))
    return brier, max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def fit_offset(df, y, P_mr=None, P_wayoff=None):
    """2024 out-of-sample 예측에서 b, c, mu 적합.

    mr/wayoff 예측은 두 경로로 온다:
      - AUX_FROM 있음(복사): 그 모델이 실제로 냈던 캐시(AUX_CACHE)를 읽는다
      - AUX_FROM 없음(재학습): 방금 검증 단계에서 나온 예측을 인자로 받는다
        ← 옛 캐시를 쓰면 방금 학습한 모델과 다른 기준으로 계수가 잡힌다
    """
    P = {"success": np.mean(
        [np.load(os.path.join(CACHE, f"success_2024_{s}.npy"))
         for s in FIT_SUCCESS_SEEDS], axis=0)}
    for name, given in (("mr", P_mr), ("wayoff", P_wayoff)):
        P[name] = given if given is not None else np.mean(
            [np.load(os.path.join(AUX_CACHE, f"{name}_2024_{s}.npy"))
             for s in AUX_SEEDS], axis=0)
    L = pd.read_csv("recovered_labels.csv.gz")
    have = df[[ID]].merge(L, on=ID, how="left")["middle"].notna().values
    m = (df["season"] == 2024).values & have
    t = y[m]
    z, u, v = logit(P["success"]), logit(P["mr"]), logit(P["wayoff"])
    mu = (float(u.mean()), float(v.mean()))
    u, v = u - mu[0], v - mu[1]

    def nll(w):
        p = np.clip(1 / (1 + np.exp(-(z + w[0] * u + w[1] * v))), 1e-9, 1 - 1e-9)
        return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

    b, c = minimize(nll, [0.0, 0.0], method="Nelder-Mead").x
    before = bss(1 / (1 + np.exp(-z)), t)[1]
    after = bss(1 / (1 + np.exp(-(z + b * u + c * v))), t)[1]
    print(f" 계수 b={b:.4f} c={c:.4f}  mu_mr={mu[0]:.4f} mu_wayoff={mu[1]:.4f}")
    print(f" 2024 자기적합 참고: {before:.1f} -> {after:.1f} ({after-before:+.1f})")
    return float(b), float(c), mu


def main():
    out_dir = os.path.join("runs", RUN)
    if os.path.exists(os.path.join(out_dir, "model")):
        raise SystemExit(f"이미 존재함: {out_dir} — RUN 이름을 바꿀 것")
    os.makedirs(os.path.join(out_dir, "model"))
    mdir = os.path.join(out_dir, "model")

    base_meta = json.load(open(os.path.join("runs", BASE_RUN, "model",
                                            "meta.json"), encoding="utf-8"))
    print(f"[{RUN}] 기반 {BASE_RUN}: seeds={base_meta['seeds']} "
          f"feats={len(base_meta['feature_cols'])} gm={base_meta['global_mean']}")
    assert len(base_meta["seeds"]) == len(FIT_SUCCESS_SEEDS), \
        "계수 적합 시드 수가 기반 run의 성공모델 개수와 다르다"
    # 003의 meta는 use_cond 키가 생기기 전에 만들어졌다(None). 실제 열로 확인한다.
    # 보조모델은 AUX_COPY_FROM(009=57열)만 쓰므로 주모델에 cond가 있어도 무방하다.
    # 다만 그때는 meta에 use_cond가 서 있어야 script.py가 표를 붙인다.
    _has_cond = any(c.startswith("cond_") for c in base_meta["feature_cols"])
    assert not _has_cond or base_meta.get("use_cond"), (
        "기반 run에 cond 열이 있는데 meta['use_cond']가 꺼져 있다 — 추론 KeyError")

    print(" Load train...", flush=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y = df[TARGET].astype(int).values

    # 피처는 003과 완전히 같아야 한다 -> global_mean도 003의 저장값을 쓴다.
    gm = base_meta["global_mean"]
    # 기반 run이 시즌내 분해를 쓰면 여기서도 같은 기준점을 만들어야 열이 맞는다.
    anchor = build_anchor(df) if base_meta.get("use_inseason") else None
    # ANCHOR_SPECS가 늘면 priors도 그만큼 필요하다. success는 _add_inseason이
    # global_mean을 쓰므로(코드 확인) 다시 계산해도 주모델 값이 안 변한다.
    priors = (rate_priors(df[df["season"] <= 2023])
              if base_meta.get("use_inseason") else None)
    FE = engineer(df.drop(columns=[ID, TARGET]), gm, anchor=anchor,
                  priors=priors)

    # cond_* 열은 engineer()가 아니라 cond.build_training_columns()가 만든다
    # (train_local.py와 동일 경로). AUX_COPY_FROM이 cond를 쓰는 run(211_base044_phpcb)
    # 이면 AUX_COLS에 cond_* 가 섞여 들어오므로 FE에도 붙여줘야 한다.
    # 🔴 cond_* 필요 목록은 base_meta 것만이 아니라 MR_EXTRA_COLS에서 요구하는 것도
    # 포함해야 한다(예: cond_bh — 주모델엔 없지만 보조 mr엔 추가하는 경우, 218 참고).
    _need_cond = {c for c in base_meta["feature_cols"] if c.startswith("cond_")}
    if not AUX_FROM:
        _need_cond |= {c for c in MR_EXTRA_COLS if c.startswith("cond_")}
    if _has_cond or _need_cond:
        _dfl = df.merge(pd.read_csv("recovered_labels.csv.gz"), on=ID, how="left")
        _C = cond.build_training_columns(_dfl)
        for _c in _need_cond:
            FE[_c] = _C[_c].values

    # 보조모델 열 — 기본은 AUX_COPY_FROM(009 = 003 피처 57열).
    # 🔴 주모델 열(61)을 쓰면 안 된다: 017이 그렇게 해서 LB -4.69로 졌다.
    _am = json.load(open(os.path.join("runs", AUX_COPY_FROM, "model",
                                      "meta.json"), encoding="utf-8"))
    AUX_COLS = {"mr": list(_am["feature_cols"]),
                "wayoff": list(_am["feature_cols"])}
    # 추가 열은 **재학습할 때만** 붙인다. AUX_FROM 복사 경로에서는 보조모델이
    # AUX_FROM의 열로 이미 학습돼 있어서 여기 손대면 안 된다.
    if not AUX_FROM:
        for _c in MR_EXTRA_COLS:
            if _c not in AUX_COLS["mr"]:
                AUX_COLS["mr"].append(_c)
    _miss = sorted({c_ for v in AUX_COLS.values() for c_ in v}
                   - set(FE.columns))
    assert not _miss, f"engineer()가 만들지 않은 열: {_miss}"
    print(f" 보조 열: mr {len(AUX_COLS['mr'])}개 / wayoff "
          f"{len(AUX_COLS['wayoff'])}개  (추가 {MR_EXTRA_COLS})", flush=True)

    def make_X(cols):
        Z = FE[cols].copy()
        for c_ in CAT_COLS:
            Z[c_] = Z[c_].astype(str)
        return Z, [Z.columns.get_loc(c_) for c_ in CAT_COLS]

    # 보조 라벨
    L = pd.read_csv("recovered_labels.csv.gz")
    L = df[[ID]].merge(L, on=ID, how="left")
    have = L["middle"].notna().values
    mr = ((L["middle"] == 1) | (L["reverse"] == 1)).values
    lab = {"mr": mr.astype(int),
           "wayoff": ((y == 0) & ~mr).astype(int)}
    print(f" 라벨: 복원 {have.sum():,}/{len(df):,}  "
          f"mr={lab['mr'][have].mean():.4f} wayoff={lab['wayoff'][have].mean():.4f}")

    if AUX_FROM:
        for name in ("mr", "wayoff"):
            for sd in AUX_SEEDS:
                shutil.copy(os.path.join("runs", AUX_FROM, "model",
                                         f"{name}_{sd}.cbm"),
                            os.path.join(mdir, f"{name}_{sd}.cbm"))
        print(f" 보조 모델 {AUX_FROM} 에서 {2*len(AUX_SEEDS)}개 복사")
        b, c, mu = fit_offset(df, y)
        best_iters = BEST_ITER
    else:
        # 재학습 경로. 2단계로 간다 (train_local.py와 같은 구조):
        #  ① 검증(2019~23 -> 2024)으로 best_iter를 **다시 찾는다**.
        #     BEST_ITER 하드코딩은 003 피처로 찾은 값이라, 피처가 바뀌면 손실
        #     지형도 바뀌어 그대로 쓰면 안 된다.
        #  ② 그때 나온 2024 예측으로 b·c를 적합한다. 옛 캐시를 쓰면 방금 학습한
        #     모델과 다른 기준이 된다.
        tr_m = (df["season"] <= 2023).values & have
        va_m = (df["season"] == 2024).values & have
        print(f"\n--- 보조 모델 검증 (2019-2023 -> 2024) tr={tr_m.sum():,} "
              f"va={va_m.sum():,} ---", flush=True)
        for name in [n for n in ("mr", "wayoff") if n not in RETRAIN_AUX]:
            for sd in AUX_SEEDS:
                shutil.copy(os.path.join("runs", AUX_COPY_FROM, "model",
                                         f"{name}_{sd}.cbm"),
                            os.path.join(mdir, f"{name}_{sd}.cbm"))
            print(f" {name}: {AUX_COPY_FROM}에서 {len(AUX_SEEDS)}개 복사 "
                  "(재학습 없음)", flush=True)
        best_iters = {"mr": {}, "wayoff": {}}
        val_preds, seed_preds = {}, {}
        for name in RETRAIN_AUX:
            Xn, ci = make_X(AUX_COLS[name])
            p_tr = Pool(Xn[tr_m], lab[name][tr_m], cat_features=ci)
            p_va = Pool(Xn[va_m], lab[name][va_m], cat_features=ci)
            ps = []
            for sd in AUX_SEEDS:
                mdl = CatBoostClassifier(**dict(
                    PARAMS, random_seed=sd, early_stopping_rounds=100)).fit(
                    p_tr, eval_set=p_va, use_best_model=True)
                best_iters[name][sd] = max(mdl.get_best_iteration(), 1)
                ps.append(mdl.predict_proba(p_va)[:, 1])
                print(f" {name}_{sd} best_iter={best_iters[name][sd]}", flush=True)
            val_preds[name] = np.mean(ps, axis=0)
            seed_preds[name] = dict(zip(AUX_SEEDS, ps))
        # 재학습하지 않은 쪽은 None -> fit_offset이 AUX_CACHE에서 읽는다.
        b, c, mu = fit_offset(df, y, val_preds.get("mr"),
                              val_preds.get("wayoff"))

        os.makedirs(AUX_OUT, exist_ok=True)
        for name in ("mr", "wayoff"):
            for sd in AUX_SEEDS:
                dst = os.path.join(AUX_OUT, f"{name}_2024_{sd}.npy")
                if name in RETRAIN_AUX:
                    np.save(dst, seed_preds[name][sd])
                else:
                    shutil.copy(os.path.join(AUX_CACHE,
                                             f"{name}_2024_{sd}.npy"), dst)
        print(f" 보조 검증예측 저장 -> {AUX_OUT} "
              f"(재학습 {RETRAIN_AUX}, 나머지는 {AUX_CACHE}에서 복사)", flush=True)

        print("\n--- 보조 모델 전체데이터 재학습 ---", flush=True)
        for name in RETRAIN_AUX:
            Xn, ci = make_X(AUX_COLS[name])
            p_all = Pool(Xn[have], lab[name][have], cat_features=ci)
            for sd in AUX_SEEDS:
                it = best_iters[name][sd]
                mdl = CatBoostClassifier(**dict(PARAMS, random_seed=sd,
                                                iterations=it)).fit(p_all)
                mdl.save_model(os.path.join(mdir, f"{name}_{sd}.cbm"))
                print(f" {name}_{sd} iter={it} 저장", flush=True)

    # 기반 run의 성공 모델을 그대로 복사 (재학습 금지 = 단일 변수 보장)
    for sd in base_meta["seeds"]:
        shutil.copy(os.path.join("runs", BASE_RUN, "model", f"model_{sd}.cbm"),
                    os.path.join(mdir, f"model_{sd}.cbm"))
    print(f" {BASE_RUN} 성공 모델 {len(base_meta['seeds'])}개 복사")

    # 시즌내 분해 기준점 표도 같이 옮긴다 — 빠지면 추론에서 FileNotFoundError.
    if base_meta.get("use_inseason"):
        # 🔴 ANCHOR_SPECS를 늘렸으므로 스키마가 바뀐다(s0_middle, s0_reverse 추가).
        #    BASE_RUN 것을 그대로 복사하면 추론이 KeyError로 죽는다.
        #    **주모델이 쓰는 s0_success 값이 그대로인지 반드시 대조한다.**
        _last = int(df["season"].max()) + 1
        _new = anchor[anchor["apply_season"] == _last].reset_index(drop=True)
        _old = pd.read_csv(os.path.join("runs", BASE_RUN, "model", "anchor.csv"),
                           encoding="utf-8")
        _k = ["apply_season", "who", "id"]
        # 옛 run은 열 이름이 `s0`(013 시절), 최근 run은 `s0_success`다. 둘 다 받는다.
        _oc = "s0" if "s0" in _old.columns else "s0_success"
        _chk = (_old[_k + ["n0", _oc]]
                .rename(columns={"n0": "n0_o", _oc: "s0_o"})
                .merge(_new[_k + ["n0", "s0_success"]], on=_k, how="outer"))
        assert len(_chk) == len(_old) == len(_new), (len(_chk), len(_old), len(_new))
        assert (_chk["n0_o"] == _chk["n0"]).all(), "n0 불일치"
        _d = float((_chk["s0_o"] - _chk["s0_success"]).abs().max())
        assert _d < 1e-9, f"s0_success 불일치 max={_d}"
        _new.to_csv(os.path.join(mdir, "anchor.csv"), index=False,
                    encoding="utf-8")
        print(f" 기준점 표 재생성 {len(_new):,}행 — 주모델 값 동일 확인 "
              f"(s0_success 최대차 {_d:.1e})")

    # cond 표는 주모델과 함께 움직인다. 안 옮기면 추론에서 파일이 없다.
    if _has_cond:
        _src = os.path.join("runs", BASE_RUN, "model")
        _n = 0
        for _f in sorted(os.listdir(_src)):
            if _f.startswith("cond_") and _f.endswith(".csv"):
                shutil.copy(os.path.join(_src, _f), os.path.join(mdir, _f))
                _n += 1
        assert _n, f"use_cond인데 {_src}에 cond_*.csv가 없다"
        print(f" cond 표 {_n}개 복사 ({BASE_RUN} -> {RUN})")

    meta = dict(base_meta)
    meta["offset"] = {"seeds": AUX_SEEDS, "b": b, "c": c,
                      "mu_mr": mu[0], "mu_wayoff": mu[1]}
    if not AUX_FROM:
        # 보조 둘이 서로 다른 열을 쓴다 -> script.py가 Pool을 따로 만든다.
        meta["offset"]["mr_feature_cols"] = AUX_COLS["mr"]
        meta["offset"]["wayoff_feature_cols"] = AUX_COLS["wayoff"]
        if base_meta.get("use_inseason"):
            meta["rate_priors"] = priors
    if AUX_FROM:
        # 복사해온 보조모델은 그 run의 피처 집합으로 학습돼 있다. BASE_RUN이
        # 열을 더 갖고 있으면(013 = 003 + 시즌내 4열) 주모델 Pool을 그대로
        # 쓸 수 없다 → script.py가 이 목록으로 보조 전용 Pool을 만든다.
        aux_meta = json.load(open(os.path.join("runs", AUX_FROM, "model",
                                               "meta.json"), encoding="utf-8"))
        if aux_meta["feature_cols"] != base_meta["feature_cols"]:
            # 실제 Pool은 FE(engineer() 전체 출력)에서 뽑으므로, 기준은
            # base_meta["feature_cols"](주모델이 고른 부분집합)가 아니라 FE.columns다.
            # 주모델이 일부 열을 의도적으로 뺀 경우(208_nodelta)에도 FE엔 남아있다.
            missing = [c_ for c_ in aux_meta["feature_cols"] if c_ not in FE.columns]
            assert not missing, f"보조모델 피처가 engineer() 출력에 없다: {missing}"
            meta["offset"]["aux_feature_cols"] = aux_meta["feature_cols"]
            print(f" 보조모델 피처 {len(aux_meta['feature_cols'])}개 "
                  f"(주모델 {len(base_meta['feature_cols'])}개) — 별도 Pool 사용")
        # 🔴 보조모델의 cat_cols는 주모델 것과 다를 수 있다(215: 주모델이 피처를
        # 가지치기하면서 top_bottom/base_state가 빠졌는데 보조는 여전히 그걸 씀).
        # build_shift.py/script.py가 보조 Pool을 만들 때 meta["cat_cols"](주모델
        # 것)을 그대로 재사용하면 CatBoost가 "Feature X is Categorical in model but
        # marked different"로 죽는다 — 실제로 215에서 재현됨. 항상 별도로 저장한다.
        if aux_meta.get("cat_cols") != base_meta.get("cat_cols"):
            meta["offset"]["aux_cat_cols"] = aux_meta["cat_cols"]
            print(f" 보조모델 cat_cols가 주모델과 달라 별도 저장: {aux_meta['cat_cols']}")
    json.dump(meta, open(os.path.join(mdir, "meta.json"), "w", encoding="utf-8"))

    json.dump({"run": RUN, "note": (
        f"{BASE_RUN} 성공모델 그대로 + 실패모드 offset(mr/wayoff). 재학습 없음 = 단일 변수. "
        "a=1,d=0 고정(calibration 성분 제거), mu는 학습때 저장. "
        "배포 유사 실측 2022계수->2024 = +26.6. 08 문서 §5"),
        "base_run": BASE_RUN, "aux_from": AUX_FROM,
        "success_seeds": base_meta["seeds"], "aux_seeds": AUX_SEEDS,
        "params": PARAMS,
        "offset": meta["offset"], "best_iters": best_iters,
        "lb_2025": None},
        open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)

    path = os.path.join(out_dir, f"submit{RUN.split('_')[0]}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("script.py", "requirements.txt", "features.py", "cond.py"):
            z.write(os.path.join(COMMON, f), f)
        for f in sorted(os.listdir(mdir)):
            z.write(os.path.join(mdir, f), "model/" + f)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
