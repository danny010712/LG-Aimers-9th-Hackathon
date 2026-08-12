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
from features import engineer, CAT_COLS  # noqa: E402
import cond  # noqa: E402

# ===== 이번 실행 설정 =====================================================
RUN = "008_depth5"
NOTE = ("007(d6, 7시드)에서 depth만 6->5. 007과 단일 변수 = 용량 축소 탐침. "
        "근거: 004->005에서 depth 6->8이 로컬 +16 / LB -34.2 (§4-2). "
        "2025는 미관측 시즌이라 얕은 트리가 외삽에 유리하다는 가설. "
        "⚠️ 로컬은 낮게 나오는 게 정상 (d5 2시드 772.6 vs d6 782.9). 로컬은 §4대로 못 믿는다.")
SEEDS = [42, 7, 2024, 99, 1, 123, 777]   # 007과 동일 — depth만 다르게 한다
POLICIES = ["SymmetricTree"]              # grow_policy 혼합은 개수 맞추니 +0.8 (§3-L)
# cond는 교정된 채택기준에서 탈락 — 합계 +8.2는 죽은 fold 2023(+11.0)이 만든 것이고
# 유효 fold만 세면 −2.8이다 (08 §3). depth도 6 유지 (d8 이득은 2024 단독).
USE_COND = False
PARAMS = dict(
    iterations=2000, learning_rate=0.05, depth=5,
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
        for f in ("script.py", "requirements.txt", "features.py", "cond.py"):
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
    X = engineer(df.drop(columns=[ID, TARGET]), global_mean)

    # 조건부 개인기록: 학습 행에는 '그 시즌 이전'으로 만든 표를 붙인다.
    if USE_COND:
        print(" 조건부 표 생성 (시즌별 과거만)...", flush=True)
        C = cond.build_training_columns(df)
        for c in cond.COND_COLS:
            X[c] = C[c].values
        print(f" cond 결측률 {X[cond.COND_COLS].isna().mean().mean()*100:.1f}%")

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

    json.dump({"seeds": tags, "feature_cols": feature_cols,
               "cat_cols": CAT_COLS, "global_mean": global_mean,
               "use_cond": USE_COND},
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
