"""실패모드 offset 빌더 — 기존 run 위에 보정항을 얹는다 (08 문서 §5).

`BASE_RUN`의 성공 모델을 **파일째 복사**한다. 재학습하지 않는다.

얹는 것:
  logit(p) = logit(p_success) + b·(logit(p_mr) − mu_mr) + c·(logit(p_wayoff) − mu_wayoff)

🔴 a(스케일)·d(절편)는 적합하지 않는다. a=1·d=0 고정이 필수다.
🔴 mu는 **학습 때 계산해 저장**한다. test에서 평균을 내면 규정 위반.

=== league-rate baseline 대응 (2026-08) ===
BASE_RUN이 league-rate baseline을 쓰는 모델(015 등)이면, `b`,`c` 적합에
쓰는 out-of-sample 성공모델 예측(2019~23 학습→2024 예측)도 **baseline이
적용된 채로 만들어진 값**이어야 한다. 이 값은 train_local.py의 검증 단계가
이미 `artifacts/auxpred_league/success_2024_{seed}.npy`로 저장해뒀으므로
여기서는 그걸 읽기만 한다 — 이 스크립트 자체는 baseline을 직접 다루지
않는다(성공모델 예측값은 이미 완성된 채로 들어옴).

mr·wayoff 보조모델은 이번 변경 범위 밖이다(baseline 없이, 기존 방식
그대로). 필요하면 이후에 AUX_USE_LEAGUE_BASELINE=True로 확장 가능하도록
자리만 남겨둔다.

⚠️ CACHE는 BASE_RUN이 league-rate baseline을 쓰는지에 따라 자동으로
   artifacts/auxpred_league ↔ artifacts/auxpred 중 고른다. 수동으로
   고정하려면 CACHE_OVERRIDE에 경로를 직접 지정.
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

RUN = "016_offset_league"
BASE_RUN = "015_league_baseline"      # league-rate baseline 성공모델
AUX_SEEDS = [42, 7, 2024]             # 보조모델 시드
AUX_FROM = "009_offset"               # 보조모델을 복사해올 run. None이면 학습한다
FIT_SUCCESS_SEEDS = [42, 7, 2024, 99, 1, 123, 777]   # BASE_RUN 시드 수와 일치해야 함
CACHE_OVERRIDE = None                 # None이면 base_meta로 자동 판단
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


def fit_offset(df, y, cache):
    """캐시된 2024 검증 예측에서 b, c, mu 적합. 추가 학습 없음.

    cache: success 예측을 읽어올 폴더. BASE_RUN이 league-rate baseline을
    쓰면 artifacts/auxpred_league(이 예측값 자체에 이미 baseline이 반영돼
    있음), 아니면 기존 artifacts/auxpred.
    """
    use = {"success": (FIT_SUCCESS_SEEDS, cache),
           "mr": (AUX_SEEDS, "artifacts/auxpred"),
           "wayoff": (AUX_SEEDS, "artifacts/auxpred")}
    P = {l: np.mean([np.load(os.path.join(c, f"{l}_2024_{s}.npy"))
                     for s in seeds], axis=0)
         for l, (seeds, c) in use.items()}
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
    print(f" cache={cache}")
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
    assert not any(c.startswith("cond_") for c in base_meta["feature_cols"]), \
        "기반 run에 cond 열이 있다"

    league = base_meta.get("league_baseline", {"enabled": False})
    if CACHE_OVERRIDE:
        cache = CACHE_OVERRIDE
    else:
        cache = "artifacts/auxpred_league" if league.get("enabled") \
            else "artifacts/auxpred"
    print(f" BASE_RUN league_baseline={league.get('enabled')}  cache={cache}")

    print(" Load train...", flush=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y = df[TARGET].astype(int).values

    gm = base_meta["global_mean"]
    X = engineer(df.drop(columns=[ID, TARGET]), gm)
    X = X[base_meta["feature_cols"]]
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in CAT_COLS]

    b, c, mu = fit_offset(df, y, cache)

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
        print(f" 보조 모델 {AUX_FROM} 에서 {2*len(AUX_SEEDS)}개 복사 "
              f"(baseline 없이 학습된 기존 모델 — 변경 없음)")
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

    # base_meta 전체(feature_cols, cat_cols, global_mean, league_baseline ...)를
    # 그대로 물려받고, offset 계수만 추가한다. league_baseline 키도 자동으로
    # 여기 포함되므로 script.py가 추론 때 baseline을 재계산할 수 있다.
    meta = dict(base_meta)
    meta["offset"] = {"seeds": AUX_SEEDS, "b": b, "c": c,
                      "mu_mr": mu[0], "mu_wayoff": mu[1]}
    json.dump(meta, open(os.path.join(mdir, "meta.json"), "w", encoding="utf-8"))

    json.dump({"run": RUN, "note": (
        f"{BASE_RUN}(league_baseline={league.get('enabled')}) 성공모델 그대로 "
        "+ 실패모드 offset(mr/wayoff, baseline 미적용). 재학습 없음 = 단일 변수. "
        "a=1,d=0 고정, mu는 학습때 저장."),
        "base_run": BASE_RUN, "aux_from": AUX_FROM, "cache": cache,
        "success_seeds": base_meta["seeds"], "aux_seeds": AUX_SEEDS,
        "params": PARAMS,
        "offset": meta["offset"], "best_iters": BEST_ITER,
        "lb_2025": None},
        open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)

    path = os.path.join(out_dir, f"submit{RUN.split('_')[0]}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        files = ["script.py", "requirements.txt", "features.py", "cond.py"]
        if league.get("enabled"):
            files.append("league_rate.py")
        for f in files:
            z.write(os.path.join(COMMON, f), f)
        for f in sorted(os.listdir(mdir)):
            z.write(os.path.join(mdir, f), "model/" + f)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
