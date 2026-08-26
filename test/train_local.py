"""로컬 학습 스크립트 (submit.zip에 포함되지 않음).

train.csv 학습 → runs/<RUN>/ 에 모델·메타·zip·결과 저장.
검증: 2019~2023 학습 → 2024 홀드아웃 (2025 분포이동 모사, 08 문서 §3-B).
최종: 검증에서 얻은 best_iter 만큼 전체 데이터(2024 포함)로 재학습해 저장.

실행마다 RUN 이름을 바꿀 것. 기존 RUN이 있으면 실행을 거부한다.
(run 001/002는 LightGBM판. 설정은 08 문서 §3-C, 모델은 runs/에 보존됨.)
"""
import io
import json
import os
import sys
import zipfile

# Windows 기본 stdout(cp949)은 일부 기호를 못 써서 print에서 죽는다.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "common")
from features import engineer, build_anchor, rate_priors, CAT_COLS  # noqa: E402
import cond  # noqa: E402
import role  # noqa: E402

# ===== 이번 실행 설정 =====================================================
RUN = "040_role"
NOTE = ("013 대비 단일 변수: 투수 역할(등판당 투구수) 표를 트리 피처로 추가. "
        "asof는 통산 누적만 주고 경기당 소화량은 train 행을 봐야 나온다 = 모델이 "
        "도달 못 하는 값(시즌내 분해와 같은 구조). 013 주모델 859.0이 비교 기준.")
SEEDS = [42, 7, 2024]
POLICIES = ["SymmetricTree"]              # grow_policy 혼합은 개수 맞추니 +0.8 (§3-L)
# cond는 교정된 채택기준에서 탈락 — 합계 +8.2는 죽은 fold 2023(+11.0)이 만든 것이고
# 유효 fold만 세면 −2.8이다 (08 §3). depth도 6 유지 (d8 이득은 2024 단독).
# 🔥 cond 재개 — 단, **살아있는 ph 표 하나만**.
# 004는 4표 번들(pc·ph·bc·pi)로 줘서 LB -5.4였고, 08 §5-12가 그 패배를
# "죽은 표 3개가 살아있는 1개를 희석"으로 설명했다. 그런데 같은 정보를
# offset으로 주니 021에서 +5.27이었다. **번들링이 원인인지 파라미터화가
# 원인인지 분리된 적이 없다.** ph만 주면 갈린다.
USE_COND = False
COND_ONLY = ["ph"]        # None이면 SPECS 전부
# 🔥 투수 역할(등판당 투구수) 표 — 09 §2-P.
# 레벨 보정 형태로는 전이가 갈렸다(2023->2024 -8.08 / 2022->2024 +10.86, 진폭 부호 1/3).
# 여기서는 **트리 피처**로 준다. 039가 보인 대로 형태가 결과를 바꿀 수 있고,
# 트리 피처는 카운트·상황과 교호작용할 수 있어 진폭 문제를 안 탄다.
USE_ROLE = True
USE_INSEASON = True                       # 시즌내 성적 분해 (§5-10)
# offset 계수 적합용 검증 예측을 여기 쌓는다. 피처 구성이 바뀌면 캐시도 바뀌므로
# train_offset.py의 CACHE와 반드시 같은 경로여야 한다.
VALPRED_DIR = "artifacts/auxpred_role"
PARAMS = dict(
    iterations=2000, learning_rate=0.05, depth=6,   # 021과 동일. 단일 변수는 cond_ph
    thread_count=-1, verbose=0, eval_metric="Logloss",   # CatBoost는 -1 (0은 크래시)
    early_stopping_rounds=100,
)
# =========================================================================

DATA = "data/train.csv"
COMMON = "common"
ID, TARGET = "row_id", "control_success"


def bss(p, y):
    r = y.mean()
    brier = float(np.mean((p - y) ** 2))
    skill = 1 - brier / (r * (1 - r))
    return brier, skill, max(0.0, 100000 * skill)


def build_zip(out_dir):
    """common/ + model/ → submit<번호>.zip.
    최상위에 model/ + script.py + requirements.txt + features.py (여분 폴더 없음).
    Python zipfile 사용 — PowerShell Compress-Archive는 경로 구분자로 백슬래시를
    써서 Linux 평가서버에서 깨질 수 있다."""
    path = os.path.join(out_dir, f"submit{RUN.split('_')[0]}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("script.py", "requirements.txt", "features.py", "cond.py",
                  "role.py"):
            z.write(os.path.join(COMMON, f), f)
        model_dir = os.path.join(out_dir, "model")
        for f in sorted(os.listdir(model_dir)):
            z.write(os.path.join(model_dir, f), "model/" + f)
    return path


def main():
    out_dir = os.path.join("runs", RUN)
    if os.path.exists(os.path.join(out_dir, "model")):
        raise SystemExit(f"이미 존재함: {out_dir} — RUN 이름을 바꿀 것")
    os.makedirs(os.path.join(out_dir, "model"))

    print(f"[{RUN}] Load train...")
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y = df[TARGET].astype(int).values
    tr = (df["season"] <= 2023).values
    va = (df["season"] == 2024).values

    # global_mean은 학습 구간에서만 계산 (검증 누수 방지).
    # 추론에서 같은 값을 써야 하므로 meta.json에 저장한다.
    global_mean = float(y[tr].mean())

    # 시즌내 분해용 기준점. season S 행에는 S−1 시즌 말 통산이 붙으므로
    # df 전체로 만들어도 누수가 구조적으로 불가능하다 (build_anchor 주석 참고).
    anchor = build_anchor(df) if USE_INSEASON else None
    # rate prior도 global_mean과 같은 원칙 — 학습 구간에서만 계산해 meta에 저장한다.
    priors = rate_priors(df[tr]) if USE_INSEASON else None
    X = engineer(df.drop(columns=[ID, TARGET]), global_mean, anchor=anchor,
                 priors=priors)

    # 조건부 개인기록: 학습 행에는 '그 시즌 이전'으로 만든 표를 붙인다.
    if USE_COND:
        print(" 조건부 표 생성 (시즌별 과거만)...", flush=True)
        C = cond.build_training_columns(df)
        use = ([c for c in cond.COND_COLS
                if any(c == "cond_" + n or c == "cond_" + n + "_dev"
                       for n in COND_ONLY)] if COND_ONLY else list(cond.COND_COLS))
        assert use, COND_ONLY
        for c in use:
            X[c] = C[c].values
        print(f" cond 열 {use}  결측률 {X[use].isna().mean().mean()*100:.1f}%")

    if USE_ROLE:
        print(" 역할 표 생성 (시즌별 과거만)...", flush=True)
        R = role.build_training_columns(df)
        for c in role.ROLE_COLS:
            X[c] = R[c].values
        print(f" role 열 {role.ROLE_COLS}  결측률 "
              f"{X[role.ROLE_COLS].isna().mean().mean()*100:.1f}%", flush=True)

    feature_cols = list(X.columns)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in CAT_COLS]
    print(f" rows={len(df)}  feats={len(feature_cols)}  "
          f"global_mean={global_mean:.4f}")

    print(f"\n--- 검증 (2019-2023 -> 2024), {POLICIES} x {SEEDS} ---")
    val_preds, best_iters, tags = [], [], []
    pool_tr = Pool(X[tr], y[tr], cat_features=ci)
    pool_va = Pool(X[va], y[va], cat_features=ci)
    for gp in POLICIES:
        for sd in SEEDS:
            m = CatBoostClassifier(**dict(PARAMS, grow_policy=gp,
                                          random_seed=sd)).fit(
                pool_tr, eval_set=pool_va, use_best_model=True)
            p = m.predict_proba(pool_va)[:, 1]
            val_preds.append(p)
            best_iters.append(m.get_best_iteration())
            tags.append(f"{gp[:3]}_{sd}")
            print(f" {tags[-1]:<10} iter={best_iters[-1]:<5} "
                  f"score~{bss(p, y[va])[2]:.1f}", flush=True)

    # offset 계수 적합용 out-of-sample 캐시. 예전엔 make_valpred.py로 따로 뽑느라
    # 같은 모델을 두 번 학습했다. 길이는 mr/wayoff 캐시와 맞춘다
    # (season==2024 & 라벨 복원됨) — 안 맞추면 fit의 nll()에서 배열이 어긋난다.
    if USE_INSEASON:
        os.makedirs(VALPRED_DIR, exist_ok=True)
        L_ = pd.read_csv("recovered_labels.csv.gz")
        have_va = df[[ID]].merge(L_, on=ID, how="left")["middle"].notna().values[va]
        for tag, p_ in zip(tags, val_preds):
            np.save(os.path.join(VALPRED_DIR,
                                 f"success_2024_{tag.split('_')[1]}.npy"),
                    p_[have_va])
        print(f" 검증 예측 {len(tags)}개 저장 -> {VALPRED_DIR} "
              f"({have_va.sum():,}행)", flush=True)

    brier, skill, score = bss(np.mean(val_preds, axis=0), y[va])
    print(f"\n[VAL 2024 · {len(SEEDS)}시드 평균] Brier={brier:.6f} "
          f"skill={skill:.5f} score~{score:.1f}")

    print("\n--- 전체 데이터(2019-2024) 재학습 ---")
    pool_all = Pool(X, y, cat_features=ci)
    final = dict(PARAMS)
    final.pop("early_stopping_rounds")
    for tag, it in zip(tags, best_iters):
        gp = "SymmetricTree" if tag.startswith("Sym") else "Depthwise"
        sd = int(tag.split("_")[1])
        m = CatBoostClassifier(**dict(final, grow_policy=gp, random_seed=sd,
                                      iterations=it)).fit(pool_all)
        m.save_model(os.path.join(out_dir, "model", f"model_{tag}.cbm"))
        print(f" {tag} iter={it} 저장")

    # 추론용 표: train 전체(2019~2024)로 만든다.
    # 학습 행이 '그 행 이전 시즌 전부'를 썼던 것과 같은 규칙 (2025 기준 과거 = 전체).
    if USE_COND:
        for name, t in cond.build_tables(df).items():
            t.to_csv(os.path.join(out_dir, "model", f"cond_{name}.csv"),
                     index=False, encoding="utf-8")
        print(f" cond 표 {len(cond.SPECS)}개 저장")

    if USE_ROLE:
        role.build_table(df).to_csv(os.path.join(out_dir, "model", "role.csv"),
                                    index=False, encoding="utf-8")
        print(" role 표 저장 (train 전체)")

    if USE_INSEASON:
        # 2025 행에 붙일 기준점만 싣는다 (학습용 과거 시즌 행은 추론에 안 쓴다).
        last = int(df["season"].max()) + 1
        anchor[anchor["apply_season"] == last].to_csv(
            os.path.join(out_dir, "model", "anchor.csv"),
            index=False, encoding="utf-8")
        print(f" 기준점 표 저장 (apply_season={last}, "
              f"{(anchor['apply_season'] == last).sum():,}행)")

    json.dump({"seeds": tags, "feature_cols": feature_cols,
               "cat_cols": CAT_COLS, "global_mean": global_mean,
               "use_cond": USE_COND, "use_inseason": USE_INSEASON,
               "use_role": USE_ROLE,
               "rate_priors": priors},
              open(os.path.join(out_dir, "model", "meta.json"), "w",
                   encoding="utf-8"))
    json.dump({"run": RUN, "note": NOTE, "model": "catboost", "seeds": tags,
               "params": PARAMS, "n_features": len(feature_cols),
               "best_iters": best_iters,
               "val_2024": {"brier": brier, "skill": skill, "score": score},
               "lb_2025": None},
              open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nSaved {build_zip(out_dir)}")


if __name__ == "__main__":
    main()
