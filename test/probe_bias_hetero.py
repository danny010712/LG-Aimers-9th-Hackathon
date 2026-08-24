"""편향 균질성 탐침 — 전역 로짓 이동으로 못 지운 잔여 편향이 있는가.

배경: run 012에서 전역 로짓 이동(-0.0416)으로 평균 편향을 0으로 만들었고
LB 두 점으로 역산하니 그 축의 남은 여지는 +0.13뿐(축 종료). 하지만 전역 이동은
'모든 행의 편향이 같다'는 가정 아래 상수 하나만 뺀 것이다. 세그먼트별 편향이
흩어져 있으면 그만큼 회수 가능하다:

    잔여 = Σ w_i (b_i - b_bar)^2 = sigma_b^2   ->   ΔBSS ≈ (1e5 / r(1-r)) * sigma_b^2

절차 (평가연도 T마다):
  1. season <= T-1 학습 -> T 예측 (3시드 평균, 고정 280 iter)
  2. T에서 최적 전역 로짓 이동을 걸어 평균 편향을 0으로 만든다
  3. 세그먼트별 잔여 mean(p-y)의 가중분산 sigma_b^2 을 잰다
     표본 노이즈 Σ w_i * Var(b_i) 를 빼서 보정한다

측정만 한다. 채택 판정은 out-of-year로 따로 해야 한다(08 문서 §4-4, §4-5).
"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "common")
from features import engineer, CAT_COLS  # noqa: E402

DATA = "data/train.csv"
ID, TARGET = "row_id", "control_success"
SEEDS = [42, 7, 2024]
ITERS = 280          # run 003의 best_iter 278/257/278 -> 고정. 죽은 fold 문제 회피
EVAL_YEARS = [2023, 2024]
PARAMS = dict(iterations=ITERS, learning_rate=0.05, depth=6,
              thread_count=-1, verbose=0, eval_metric="Logloss")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def best_global_shift(p, y):
    """평균 편향을 0으로 만드는 로짓 이동량을 정확히 푼다(이분법)."""
    lo, hi = -1.0, 1.0
    z = logit(p)
    for _ in range(80):
        mid = (lo + hi) / 2
        if sigmoid(z - mid).mean() - y.mean() > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def segment_bias(p, y, key, min_n=2000):
    """세그먼트별 편향 b_i, 비중 w_i, 노이즈 보정된 sigma_b."""
    d = pd.DataFrame({"p": p, "y": y, "k": np.asarray(key, dtype=object)})
    g = d.groupby("k", dropna=False)
    tab = pd.DataFrame({
        "n": g.size(),
        "pred": g["p"].mean(),
        "actual": g["y"].mean(),
    })
    tab["bias"] = tab["pred"] - tab["actual"]
    tab = tab[tab["n"] >= min_n]
    w = tab["n"] / tab["n"].sum()
    b = tab["bias"]
    raw = float((w * (b - (w * b).sum()) ** 2).sum())
    # 표본 노이즈: Var(b_i) ≈ mean((p-y)^2) / n_i
    mse = float(np.mean((p - y) ** 2))
    noise = float((w * (mse / tab["n"])).sum())
    return tab, w, raw, noise


def main():
    print("Load train...", flush=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y_all = df[TARGET].astype(int).values

    out = {}
    for T in EVAL_YEARS:
        tr = (df["season"] <= T - 1).values
        va = (df["season"] == T).values
        global_mean = float(y_all[tr].mean())
        X = engineer(df.drop(columns=[ID, TARGET]), global_mean)
        feature_cols = list(X.columns)
        for c in CAT_COLS:
            X[c] = X[c].astype(str)
        ci = [X.columns.get_loc(c) for c in CAT_COLS]

        print(f"\n=== eval {T}: train n={tr.sum()} val n={va.sum()} "
              f"global_mean={global_mean:.4f} feats={len(feature_cols)}",
              flush=True)
        pool_tr = Pool(X[tr], y_all[tr], cat_features=ci)
        pool_va = Pool(X[va], cat_features=ci)
        preds = []
        for sd in SEEDS:
            m = CatBoostClassifier(**dict(PARAMS, random_seed=sd)).fit(pool_tr)
            preds.append(m.predict_proba(pool_va)[:, 1])
            print(f"  seed {sd} done", flush=True)
        p = np.mean(preds, axis=0)
        y = y_all[va]
        sub = df.loc[va].reset_index(drop=True)

        r = float(y.mean())
        A = 1e5 / (r * (1 - r))
        brier0 = float(np.mean((p - y) ** 2))
        raw_bias = float(p.mean() - r)
        s = best_global_shift(p, y)
        p2 = sigmoid(logit(p) - s)
        brier1 = float(np.mean((p2 - y) ** 2))
        bss0 = max(0.0, 1e5 * (1 - brier0 / (r * (1 - r))))
        bss1 = max(0.0, 1e5 * (1 - brier1 / (r * (1 - r))))
        print(f"  base rate={r:.4f}  평균편향={raw_bias:+.5f}  "
              f"최적이동 s={s:+.4f}  BSS {bss0:.1f} -> {bss1:.1f}", flush=True)

        # 세그먼트 정의
        pn = sub["asof_pitcher_n"]
        bn = sub["asof_batter_n"]
        segs = {
            "asof_pitcher_n_decile": np.where(
                pn.isna(), "NaN",
                pd.qcut(pn, 10, labels=False, duplicates="drop").astype("Int64")
                .astype(str)),
            "asof_batter_n_decile": np.where(
                bn.isna(), "NaN",
                pd.qcut(bn, 10, labels=False, duplicates="drop").astype("Int64")
                .astype(str)),
            "game_type": sub["game_type"].astype(str),
            "count_state": (sub["balls_before"].astype(str) + "-"
                            + sub["strikes_before"].astype(str)),
            "pred_decile": pd.qcut(p, 10, labels=False,
                                   duplicates="drop").astype(str),
        }

        out[T] = {"base_rate": r, "raw_bias": raw_bias, "shift": s,
                  "brier_before": brier0, "brier_after": brier1, "segments": {}}
        for name, key in segs.items():
            tab, w, raw, noise = segment_bias(p2, y, key)
            var = max(raw - noise, 0.0)
            sig = var ** 0.5
            print(f"\n  --- {name}  (그룹 {len(tab)}개)")
            t = tab.copy()
            t["w"] = w
            print(t[["n", "pred", "actual", "bias"]]
                  .sort_index().round(5).to_string())
            print(f"  sigma_b(raw)={raw**0.5:.5f}  노이즈={noise**0.5:.5f}  "
                  f"sigma_b(보정)={sig:.5f}  회수가능 ΔBSS≈{A*var:+.1f}",
                  flush=True)
            out[T]["segments"][name] = {
                "sigma_b_raw": raw ** 0.5, "noise": noise ** 0.5,
                "sigma_b": sig, "recoverable_bss": A * var,
                "table": {str(k): {"n": int(tab.loc[k, "n"]),
                                   "bias": float(tab.loc[k, "bias"])}
                          for k in tab.index},
            }

    json.dump(out, open("probe_bias_hetero.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nSaved probe_bias_hetero.json")

    print("\n=== 요약: 세그먼트별 회수가능 ΔBSS (연도별 부호 일치 확인) ===")
    names = list(out[EVAL_YEARS[0]]["segments"])
    for n in names:
        vals = [f"{out[T]['segments'][n]['recoverable_bss']:+7.1f}"
                for T in EVAL_YEARS]
        print(f"  {n:26s} " + "  ".join(
            f"{T}:{v}" for T, v in zip(EVAL_YEARS, vals)))


if __name__ == "__main__":
    main()
