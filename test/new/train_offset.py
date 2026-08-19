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

mr·wayoff 보조모델도 이번에 baseline을 확장한다(2026-08 2차 수정).
`USE_AUX_LEAGUE_BASELINE=True`면, mr/wayoff 각각 자기 라벨 기준
season×game_type 표를 따로 만들어(09 §1-E: mr↑ wayoff↓, 드리프트
방향이 서로 다름) 학습 시점에 baseline으로 주입한다. 주모델의
league-rate baseline과 완전히 같은 원리·같은 out-of-year/외부데이터
원칙을 그대로 따른다.

🔴 baseline을 넣으려면 mr/wayoff를 **반드시 새로 학습**해야 한다
   (baseline은 .cbm에 저장 안 되므로, AUX_FROM으로 복사해온 기존
   모델엔 넣을 수가 없다). 즉 `USE_AUX_LEAGUE_BASELINE=True`면
   `AUX_FROM=None`이 강제된다 — 아래 main()에서 assert로 확인.

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
import league_rate as lr  # noqa: E402

RUN = "019_offset_extra_fe_baseline_alsoaux"
BASE_RUN = "018_extra_fe"             # league-rate baseline + EXTRA_FE 성공모델 (재사용)
AUX_SEEDS = [42, 7, 2024]             # 보조모델 시드
AUX_FROM = None                       # baseline 넣으려면 반드시 None (새로 학습)
FIT_SUCCESS_SEEDS = [42, 7, 2024, 99, 1, 123, 777]   # BASE_RUN 시드 수와 일치해야 함
CACHE_OVERRIDE = None                 # None이면 base_meta로 자동 판단
# ⚠️ 예전엔 BEST_ITER를 상수로 박아뒀었다(009_offset을 만들 때 한 번
# 조기종료로 찾은 값). baseline·extra_fe 등 설정이 바뀌면 최적 반복 횟수도
# 같이 바뀔 수 있어서, 지금은 AUX_FROM=None으로 새로 학습할 때마다
# 검증 단계(2019~2023→2024)에서 매번 직접 탐색한다(아래 main() 참고).
PARAMS = dict(iterations=2000, learning_rate=0.05, depth=6,
              thread_count=-1, verbose=0, eval_metric="Logloss")

# --- 보조모델 league-rate baseline 설정 (주모델과 별개 토글) ---
USE_AUX_LEAGUE_BASELINE = True
AUX_LEAGUE_GROUP_COLS = ["season", "game_type"]
# 2025 test용 override. None이면 train.csv 내부 선형외삽(주모델과 동일 원칙).
AUX_LEAGUE_EST_OVERRIDE = None

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


def fit_from_arrays(P_success, P_mr, P_wayoff, t):
    """이미 계산된 예측 배열(전부 같은 길이, out-of-sample)로 b,c,mu 적합.
    출처가 캐시 파일이든 방금 학습에서 나온 값이든 상관없이 공유하는 핵심 계산."""
    z, u, v = logit(P_success), logit(P_mr), logit(P_wayoff)
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


def fit_offset(df, y, cache):
    """캐시된 2024 검증 예측에서 b, c, mu 적합. 추가 학습 없음.

    ⚠️ AUX_FROM이 설정된 경우(mr/wayoff를 복사해오는 경우) 전용이다 —
    복사해온 모델이 어떤 예측을 냈었는지는 artifacts/auxpred의 캐시로만
    알 수 있기 때문. AUX_FROM=None(새로 학습)인 경우는 이 함수를 쓰지 않고
    main()이 검증 단계에서 직접 뽑은 예측으로 fit_from_arrays()를 호출한다
    — 그래야 지금 막 학습한 모델과 계수가 서로 어긋나지 않는다(실측 확인된
    불일치 버그, 2026-08: 예전엔 AUX_FROM=None이어도 이 함수가 무조건
    artifacts/auxpred의 옛 캐시를 읽어서, 방금 학습한 새 모델과 다른
    기준으로 계수가 나왔었다).

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
    print(f" cache={cache} (mr/wayoff는 항상 artifacts/auxpred 고정 — "
          f"AUX_FROM 전용 경로)")
    b, c, mu = fit_from_arrays(P["success"], P["mr"], P["wayoff"], t)
    return b, c, mu
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
    if USE_AUX_LEAGUE_BASELINE:
        assert AUX_FROM is None, (
            "USE_AUX_LEAGUE_BASELINE=True면 mr/wayoff를 baseline과 함께 "
            "새로 학습해야 한다 — AUX_FROM을 None으로 둘 것 "
            "(baseline은 .cbm에 저장 안 되므로 복사해온 모델엔 넣을 수 없음)")
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
    X = engineer(df.drop(columns=[ID, TARGET]), gm,
                extra_fe=base_meta.get("extra_fe", False),
                rate_means=base_meta.get("rate_means"))
    X = X[base_meta["feature_cols"]]
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in CAT_COLS]

    L = pd.read_csv("recovered_labels.csv.gz")
    L = df[[ID]].merge(L, on=ID, how="left")
    have = L["middle"].notna().values
    mr = ((L["middle"] == 1) | (L["reverse"] == 1)).values
    lab = {"mr": mr.astype(int),
           "wayoff": ((y == 0) & ~mr).astype(int)}
    print(f" 라벨: 복원 {have.sum():,}/{len(df):,}  "
          f"mr={lab['mr'][have].mean():.4f} wayoff={lab['wayoff'][have].mean():.4f}")

    if AUX_FROM:
        # 복사해오는 경우만 옛 캐시(artifacts/auxpred) 기준으로 b,c를 적합한다
        # — 복사해온 모델이 실제로 낸 예측이 그 캐시이기 때문에 앞뒤가 맞는다.
        b, c, mu = fit_offset(df, y, cache)
        aux_src_meta = json.load(open(os.path.join(
            "runs", AUX_FROM, "model", "meta.json"), encoding="utf-8"))
        best_iters = None   # 복사해온 모델이라 이 스크립트는 반복 횟수를 모름
        for name in ("mr", "wayoff"):
            for sd in AUX_SEEDS:
                shutil.copy(os.path.join("runs", AUX_FROM, "model",
                                         f"{name}_{sd}.cbm"),
                            os.path.join(mdir, f"{name}_{sd}.cbm"))
        print(f" 보조 모델 {AUX_FROM} 에서 {2*len(AUX_SEEDS)}개 복사 "
              f"(feature_cols={len(aux_src_meta['feature_cols'])}개, "
              f"extra_fe={aux_src_meta.get('extra_fe', False)})")
    else:
        aux_src_meta = base_meta   # 새로 학습 = 주모델과 같은 X를 그대로 씀
        print("\n--- 보조 모델: 검증(2019-2023 -> 2024)으로 best_iter 탐색 ---",
              flush=True)

        # train_local.py와 동일한 2단계 구조. 하드코딩된 BEST_ITER를 매번
        # 재사용하지 않는다 — baseline·extra_fe 등 어떤 변경이든 손실
        # 지형이 바뀌면 최적 반복 횟수도 같이 바뀔 수 있어서, 재학습할
        # 때마다 그 시점 설정 기준으로 직접 찾는다.
        tr_mask = (df["season"] <= 2023).values & have
        va_mask = (df["season"] == 2024).values & have
        print(f" tr={tr_mask.sum():,}  va={va_mask.sum():,}")

        # --- 검증 단계: mr/wayoff 각각 (있다면) baseline 표를 tr만으로 생성 ---
        val_tables = {}
        if USE_AUX_LEAGUE_BASELINE:
            df_tr = df[tr_mask].copy()
            for name in ("mr", "wayoff"):
                df_tr[f"_lab_{name}"] = lab[name][tr_mask]
                val_tables[name] = lr.build_table(
                    df_tr, AUX_LEAGUE_GROUP_COLS, target=f"_lab_{name}")

        best_iters = {"mr": {}, "wayoff": {}}
        val_preds = {"mr": [], "wayoff": []}   # b,c 적합에 재사용 (옛 캐시 대신)
        for name in ("mr", "wayoff"):
            base_tr = base_va = None
            if USE_AUX_LEAGUE_BASELINE:
                base_tr = lr.assign_baseline_logit(
                    df[tr_mask], val_tables[name], AUX_LEAGUE_GROUP_COLS,
                    held_out_season=None, override=None)
                # 2024는 out-of-year 외삽(tr만으로 만든 표 기준) — 실측 평균
                # 쓰면 검증 누수.
                base_va = lr.assign_baseline_logit(
                    df[va_mask], val_tables[name], AUX_LEAGUE_GROUP_COLS,
                    held_out_season=2024, override=None)
            pool_tr = Pool(X[tr_mask], lab[name][tr_mask], cat_features=ci,
                          baseline=base_tr)
            pool_va = Pool(X[va_mask], lab[name][va_mask], cat_features=ci,
                          baseline=base_va)
            for sd in AUX_SEEDS:
                m = CatBoostClassifier(**dict(
                    PARAMS, random_seed=sd,
                    early_stopping_rounds=100)).fit(
                    pool_tr, eval_set=pool_va, use_best_model=True)
                it = m.get_best_iteration()
                if it < 1:
                    # 극단적으로 이른 조기종료(0번째가 최적) — CatBoost는
                    # iterations=0을 거부한다. 표본이 아주 적은 라벨(wayoff
                    # 등)에서 드물게 발생할 수 있다. 최소 1로 방어.
                    print(f" ⚠️ {name}_{sd} best_iteration={it} → 1로 보정 "
                          f"(라벨 표본이 너무 적거나 baseline이 대부분을 "
                          f"이미 설명하는 경우일 수 있음, 결과 해석시 참고)")
                    it = 1
                best_iters[name][sd] = it
                print(f" {name}_{sd} best_iter={best_iters[name][sd]}",
                      flush=True)
                # b,c 적합용 — 이 시드가 예측한 va 확률(baseline까지 반영된
                # 최종 확률). 옛 artifacts/auxpred 캐시를 안 쓰고 이 자리에서
                # 나온 값을 직접 쓴다 — 지금 막 학습한 모델과 어긋나지 않게.
                val_preds[name].append(m.predict_proba(pool_va)[:, 1])

        # --- b, c, mu 적합: 주모델(cache) + 방금 나온 mr/wayoff va 예측 ---
        # 주모델은 train_local.py가 이미 저장해둔 out-of-sample 캐시를 그대로
        # 쓴다(그쪽은 최신 상태가 보장됨 — 매번 train_local.py가 새로 씀).
        raw_seeds = FIT_SUCCESS_SEEDS
        P_success = np.mean([np.load(os.path.join(cache, f"success_2024_{s}.npy"))
                             for s in raw_seeds], axis=0)
        P_mr = np.mean(val_preds["mr"], axis=0)
        P_wayoff = np.mean(val_preds["wayoff"], axis=0)
        t_va = y[va_mask]
        b, c, mu = fit_from_arrays(P_success, P_mr, P_wayoff, t_va)

        # --- 최종 재학습: have 전체(2019~2024)로, baseline은 전체표 사용 ---
        print("\n--- 보조 모델 전체데이터(2019-2024) 재학습 ---", flush=True)
        aux_tables = {}
        if USE_AUX_LEAGUE_BASELINE:
            df_have = df[have].copy()
            for name in ("mr", "wayoff"):
                df_have[f"_lab_{name}"] = lab[name][have]
                aux_tables[name] = lr.build_table(
                    df_have, AUX_LEAGUE_GROUP_COLS, target=f"_lab_{name}")
                print(f" {name} league_rate 표 그룹 수={len(aux_tables[name])}")

        for name in ("mr", "wayoff"):
            baseline_aux = None
            if USE_AUX_LEAGUE_BASELINE:
                baseline_aux = lr.assign_baseline_logit(
                    df[have], aux_tables[name], AUX_LEAGUE_GROUP_COLS,
                    held_out_season=None, override=None)
                print(f" {name} baseline(로짓) 평균={baseline_aux.mean():+.4f}")
            p_all = Pool(X[have], lab[name][have], cat_features=ci,
                        baseline=baseline_aux)
            for sd in AUX_SEEDS:
                it = best_iters[name][sd]
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
    #
    # ⚠️ 주모델과 보조모델(mr/wayoff)이 서로 다른 feature_cols/extra_fe로
    # 학습됐을 수 있다(예: AUX_FROM이 extra_fe 없던 옛날 run을 가리키는
    # 경우). 그래서 보조모델 전용 피처 스키마를 "aux_features"로 따로
    # 저장한다 — script.py/build_shift.py가 주모델용 X와 보조모델용
    # X_aux를 각각 따로 만들어야 CatBoost가 "피처 이름이 다르다"고
    # 터지지 않는다 (실측 확인된 버그, 2026-08).
    meta = dict(base_meta)
    meta["offset"] = {
        "seeds": AUX_SEEDS, "b": b, "c": c,
        "mu_mr": mu[0], "mu_wayoff": mu[1],
        "aux_features": {
            "global_mean": aux_src_meta["global_mean"],
            "extra_fe": aux_src_meta.get("extra_fe", False),
            "rate_means": aux_src_meta.get("rate_means"),
            "feature_cols": aux_src_meta["feature_cols"],
            "cat_cols": aux_src_meta.get("cat_cols", CAT_COLS),
        },
    }
    if USE_AUX_LEAGUE_BASELINE:
        meta["offset"]["aux_league_baseline"] = {
            name: {"enabled": True,
                  **lr.table_to_json(aux_tables[name], AUX_LEAGUE_GROUP_COLS),
                  "test_override": AUX_LEAGUE_EST_OVERRIDE}
            for name in ("mr", "wayoff")
        }
    else:
        meta["offset"]["aux_league_baseline"] = {
            "mr": {"enabled": False}, "wayoff": {"enabled": False}}
    json.dump(meta, open(os.path.join(mdir, "meta.json"), "w", encoding="utf-8"))

    json.dump({"run": RUN, "note": (
        f"{BASE_RUN}(league_baseline={league.get('enabled')}) 성공모델 그대로 "
        f"+ 실패모드 offset(mr/wayoff, aux_league_baseline={USE_AUX_LEAGUE_BASELINE}). "
        "재학습 없음(성공모델) = 단일 변수. "
        "a=1,d=0 고정, mu는 학습때 저장."),
        "base_run": BASE_RUN, "aux_from": AUX_FROM, "cache": cache,
        "success_seeds": base_meta["seeds"], "aux_seeds": AUX_SEEDS,
        "params": PARAMS,
        "offset": meta["offset"], "best_iters": best_iters,
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
