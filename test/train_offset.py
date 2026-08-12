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
from features import engineer, CAT_COLS  # noqa: E402

RUN = "010_offset_seeds7"
BASE_RUN = "007_seeds7"               # 성공 모델을 가져올 run (재학습하지 않는다)
AUX_SEEDS = [42, 7, 2024]             # 보조모델 시드
AUX_FROM = "009_offset"               # 보조모델을 복사해올 run. None이면 학습한다
# 계수 적합에 쓸 검증 예측(2019~23 학습 -> 2024)의 시드.
# 성공 쪽은 BASE_RUN의 시드 수와 맞춰야 한다 (007=7시드).
FIT_SUCCESS_SEEDS = [42, 7, 2024, 99, 1, 123, 777]
CACHE = "artifacts/auxpred"       # "aux"는 Windows 예약 장치명이라 git이 못 연다
# 검증(2019~23 -> 2024)에서 얻은 조기중단 지점. 전체 재학습에 그대로 쓴다.
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


def fit_offset(df, y):
    """캐시된 2024 검증 예측에서 b, c, mu 적합. 추가 학습 없음."""
    use = {"success": FIT_SUCCESS_SEEDS, "mr": AUX_SEEDS, "wayoff": AUX_SEEDS}
    P = {l: np.mean([np.load(os.path.join(CACHE, f"{l}_2024_{s}.npy"))
                     for s in use[l]], axis=0)
         for l in use}
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
    assert not any(c.startswith("cond_") for c in base_meta["feature_cols"]), \
        "기반 run에 cond 열이 있다"

    print(" Load train...", flush=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y = df[TARGET].astype(int).values

    # 피처는 003과 완전히 같아야 한다 -> global_mean도 003의 저장값을 쓴다.
    gm = base_meta["global_mean"]
    X = engineer(df.drop(columns=[ID, TARGET]), gm)
    X = X[base_meta["feature_cols"]]
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in CAT_COLS]

    b, c, mu = fit_offset(df, y)

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
    else:
        print("\n--- 보조 모델 전체데이터 학습 ---", flush=True)
        for name in ("mr", "wayoff"):
            p_all = Pool(X[have], lab[name][have], cat_features=ci)
            for sd in AUX_SEEDS:
                it = BEST_ITER[name][sd]
                m = CatBoostClassifier(**dict(PARAMS, random_seed=sd,
                                              iterations=it)).fit(p_all)
                m.save_model(os.path.join(mdir, f"{name}_{sd}.cbm"))
                print(f" {name}_{sd} iter={it} 저장", flush=True)

    # 기반 run의 성공 모델을 그대로 복사 (재학습 금지 = 단일 변수 보장)
    for sd in base_meta["seeds"]:
        shutil.copy(os.path.join("runs", BASE_RUN, "model", f"model_{sd}.cbm"),
                    os.path.join(mdir, f"model_{sd}.cbm"))
    print(f" {BASE_RUN} 성공 모델 {len(base_meta['seeds'])}개 복사")

    meta = dict(base_meta)
    meta["offset"] = {"seeds": AUX_SEEDS, "b": b, "c": c,
                      "mu_mr": mu[0], "mu_wayoff": mu[1]}
    json.dump(meta, open(os.path.join(mdir, "meta.json"), "w", encoding="utf-8"))

    json.dump({"run": RUN, "note": (
        f"{BASE_RUN} 성공모델 그대로 + 실패모드 offset(mr/wayoff). 재학습 없음 = 단일 변수. "
        "a=1,d=0 고정(calibration 성분 제거), mu는 학습때 저장. "
        "배포 유사 실측 2022계수->2024 = +26.6. 08 문서 §5"),
        "base_run": BASE_RUN, "aux_from": AUX_FROM,
        "success_seeds": base_meta["seeds"], "aux_seeds": AUX_SEEDS,
        "params": PARAMS,
        "offset": meta["offset"], "best_iters": BEST_ITER,
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
