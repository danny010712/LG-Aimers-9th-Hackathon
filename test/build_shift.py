"""전역 로짓 이동 빌더 — 기존 run에 시즌 base rate 보정을 얹는다 (08 §5-6).

011 = 010 + 절반(−0.0208) → LB 985.09 / 012 = 전량(−0.0416) → LB 998.00.

트리는 미관측 시즌을 외삽하지 못한다(§6-A). 예측 평균이 마지막 학습 시즌 수준에
갇히므로, 2025의 리그 수준 추정치까지 로짓을 상수만큼 민다.

=== 2026-08-24 개정 ===
1. **예측을 직접 계산한다.** 예전엔 010이 낸 예측 CSV를 읽었는데, 시즌내 분해
   (§5-10)가 들어간 뒤로는 피처 구성마다 예측이 달라져 캐시를 못 믿는다.
2. **2025 추정치를 train만으로 만든다.** 예전 0.477은 10 문서(KBO 공개자료)에서
   가져온 값이고 그건 규정 2-3 "외부 데이터 사용 금지"의 회색지대다. 대신 train
   안에서 만드는 독립 추정자 2개의 평균을 쓴다:
     ① 연도별 리그 성공률의 선형외삽                          → 0.4747
     ② 가짜test 예측평균 − (검증에서 잰 미관측 시즌 과대예측폭) → 0.4777
   둘이 대상을 위아래로 감싼다. 독립 추정자 평균 = 분산 감소의 표준 처방.
3. **가짜 test의 `ins_*`를 살린다.** 2024 행에 2025용 기준점(2024말)을 붙이면
   `dn=0`이 되어 시즌내 분해가 통산으로 붕괴한다. 실제 2025 test는 `dn>0`이므로
   (5행 샘플 실측 +380/+399), 2024용 기준점을 붙여 "시즌 진행 중" 상태를 모사한다.
   실측 차이는 예측평균 0.0009로 미미하나 구조를 맞추는 쪽을 쓴다.

🔴 이동 상수는 **여기서 계산해 meta.json에 저장**한다. script.py가 test 평균을 보고
   정하면 test 내부 행간 통계 = 규정 위반이다.
🔴 §6-J가 금지한 것은 "사후 이동" 자체가 아니라 **이동량을 fold에서 적합하는 것**이다.
   여기서 하는 건 목표값을 밖에서 정하고 그 차이만큼 미는 것이다.

모델은 BASE_RUN의 것을 그대로 복사한다. 재학습 없음 = 단일 변수.
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
from scipy.optimize import brentq

sys.path.insert(0, "common")
from features import engineer, prepare, build_anchor  # noqa: E402

RUN = "015_shift_inseason"
BASE_RUN = "014_offset_inseason"
FRACTION = 1.0            # 012에서 전량이 예측대로 적중(잔여 여지 +0.12)
# 검증 예측 캐시 — 추정자 ②의 편향을 재는 데 쓴다 (BASE_RUN의 피처 구성으로 만든 것)
VAL_CACHE = "artifacts/auxpred_ins"
VAL_SEEDS = [42, 7, 2024]
DATA = "data/train.csv"
COMMON = "common"
ID, TARGET = "row_id", "control_success"


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(mdir, meta, fe):
    """script.py와 같은 경로로 최종 확률(offset까지). shift는 아직 안 건다."""
    X = prepare(fe, meta["feature_cols"], meta["cat_cols"])
    ci = [X.columns.get_loc(c) for c in meta["cat_cols"]]
    pool = Pool(X, cat_features=ci)

    def avg(prefix, seeds, pl):
        return np.mean([CatBoostClassifier().load_model(
            os.path.join(mdir, f"{prefix}{s}.cbm")).predict_proba(pl)[:, 1]
            for s in seeds], axis=0)

    p = np.clip(avg("model_", meta["seeds"], pool), 1e-6, 1 - 1e-6)
    off = meta.get("offset")
    if off:
        # 보조모델은 피처 집합이 다를 수 있다 (013=61열 vs 009 보조=57열).
        cols = off.get("aux_feature_cols") or meta["feature_cols"]
        Xa = prepare(fe, cols, meta["cat_cols"])
        pa = Pool(Xa, cat_features=[Xa.columns.get_loc(c)
                                    for c in meta["cat_cols"]])
        z = (logit(p)
             + off["b"] * (logit(avg("mr_", off["seeds"], pa)) - off["mu_mr"])
             + off["c"] * (logit(avg("wayoff_", off["seeds"], pa))
                           - off["mu_wayoff"]))
        p = np.clip(sigmoid(z), 1e-6, 1 - 1e-6)
    return p


def main():
    out_dir = os.path.join("runs", RUN)
    if os.path.exists(os.path.join(out_dir, "model")):
        raise SystemExit(f"이미 존재함: {out_dir} — RUN 이름을 바꿀 것")

    mdir_base = os.path.join("runs", BASE_RUN, "model")
    meta = json.load(open(os.path.join(mdir_base, "meta.json"), encoding="utf-8"))
    print(f"[{RUN}] 기반 {BASE_RUN}  use_inseason={meta.get('use_inseason')}",
          flush=True)

    df = pd.read_csv(DATA, encoding="utf-8-sig")
    fake = df[df["season"] == 2024].drop(columns=[TARGET]).copy()
    fake["season"] = 2025

    anchor = None
    if meta.get("use_inseason"):
        # 2024용 기준점(=2023말)을 붙여 '시즌 진행 중'을 모사한다 (모듈 주석 3번).
        a = build_anchor(df)
        anchor = a[a["apply_season"] == 2024].copy()
        anchor["apply_season"] = 2025
    fe = engineer(fake.drop(columns=[ID]), meta["global_mean"], anchor=anchor)
    print(f" 가짜 test {len(fake):,}행 구성 완료", flush=True)

    p = predict(mdir_base, meta, fe)
    cur = float(p.mean())

    # ① 리그 성공률 선형외삽 (train만)
    lg = df.groupby("season")[TARGET].mean()
    sl, ic = np.polyfit(lg.index.values.astype(float), lg.values, 1)
    est1 = float(sl * 2025 + ic)
    # ② 검증에서 잰 '미관측 시즌 과대예측폭'을 가짜test 평균에서 뺀다 (train만)
    L = pd.read_csv("recovered_labels.csv.gz")
    have = df[[ID]].merge(L, on=ID, how="left")["middle"].notna().values
    m = (df["season"] == 2024).values & have
    pv = np.mean([np.load(os.path.join(VAL_CACHE, f"success_2024_{s}.npy"))
                  for s in VAL_SEEDS], axis=0)
    bias = float(pv.mean() - df[TARGET].values[m].mean())
    est2 = cur - bias
    est = (est1 + est2) / 2

    print(f" 가짜test 예측 평균 {cur:.4f}")
    print(f" 2025 추정 ① 선형외삽 {est1:.4f} / ② 검증편향({bias:+.4f}) 역산 "
          f"{est2:.4f}  → 평균 {est:.4f}")

    target = cur - FRACTION * (cur - est)
    lz = logit(p)
    if abs(cur - target) < 1e-9:
        d = 0.0
    else:
        d = float(brentq(lambda x: float(np.mean(sigmoid(lz + x))) - target,
                         -1.0, 1.0, xtol=1e-10))
    after = float(np.mean(sigmoid(lz + d)))
    print(f" logit_shift = {d:+.6f}   적용 후 평균 {after:.4f}")
    s = cur - after
    print(f" 기대(참고): 추정 적중 {100000*(2*s*(cur-est) - s*s)/0.2494:+.1f} / "
          f"하락 멈춤 {100000*(-s*s)/0.2494:+.1f}")

    shutil.copytree(mdir_base, os.path.join(out_dir, "model"))
    mdir = os.path.join(out_dir, "model")
    meta_out = json.load(open(os.path.join(mdir, "meta.json"), encoding="utf-8"))
    meta_out["logit_shift"] = d
    json.dump(meta_out, open(os.path.join(mdir, "meta.json"), "w",
                             encoding="utf-8"))

    json.dump({"run": RUN, "note": (
        f"{BASE_RUN} + 전역 로짓 이동 {d:+.6f}. 모델 재학습 없음 = 단일 변수. "
        f"가짜test 예측평균 {cur:.4f} → {after:.4f}. "
        f"2025 추정 {est:.4f} = (선형외삽 {est1:.4f} + 검증편향역산 {est2:.4f})/2, "
        "둘 다 train만 사용(외부 데이터 없음). 08 §5-6"),
        "base_run": BASE_RUN, "logit_shift": d,
        "mean_before": cur, "mean_after": after,
        "est_2025": est, "est_linear": est1, "est_bias": est2,
        "val_bias": bias, "fraction": FRACTION, "lb_2025": None},
        open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)

    path = os.path.join(out_dir, f"submit{RUN.split('_')[0]}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f_ in ("script.py", "requirements.txt", "features.py", "cond.py"):
            z.write(os.path.join(COMMON, f_), f_)
        for f_ in sorted(os.listdir(mdir)):
            z.write(os.path.join(mdir, f_), "model/" + f_)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
