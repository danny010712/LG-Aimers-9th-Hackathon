"""볼카운트별 잔여 편향이 연도 간에 전이되는가 (08 문서 §5-9).

§6-N에서 전역 로짓 이동 후에도 count_state에 잔여 편향이 남았다(2024 상한 +10.8,
chi2_11 = 39.6, p~4e-5). 노이즈는 아니다. 문제는 전이되느냐다.

⚠️ 이건 §5-6 line 532가 금지한 "이동량을 fold에서 적합"과 형태가 같다.
차이는 전역 수준(연간 낙폭 분산이 커서 실패)이 아니라 카운트 간 상대 구조라는 점뿐이고,
그게 안정한지는 재봐야 안다.

절차:
  1. 2021·2022·2024 각각 season<=T-1 학습 -> T 예측 (3시드, 280 iter 고정). 예측값 저장.
  2. 각 연도에서 전역 이동 후, 카운트별 잔여를 0으로 만드는 로짓 오프셋 delta_c(T) 12개
  3. 게이트: delta_c 벡터가 연도 간 상관이 있는가 (가중상관 + 부호일치)
  4. 전이 채점: S에서 뽑은 delta를 T에 적용. 기준선은 0이 아니라 '전역 이동만'.
     양쪽 arm 모두 전역 이동을 재최적화해 구조 성분만 비교한다.
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, "common")
from features import engineer, CAT_COLS  # noqa: E402

DATA = "data/train.csv"
CACHE = "artifacts/valpred"
ID, TARGET = "row_id", "control_success"
SEEDS = [42, 7, 2024]
ITERS = 280
YEARS = [2021, 2022, 2024]          # 2023은 F 체제붕괴로 죽은 fold (§6-N)
PARAMS = dict(iterations=ITERS, learning_rate=0.05, depth=6,
              thread_count=-1, verbose=0, eval_metric="Logloss")


def logit(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def solve_shift(z, target_mean):
    """sigmoid(z - d) 의 평균이 target_mean 이 되는 d."""
    lo, hi = -3.0, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(z - mid).mean() > target_mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def bss(p, y):
    r = y.mean()
    return max(0.0, 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r))))


def get_preds(df, y_all, T):
    """season<=T-1 학습 -> T 예측. 캐시에 저장해 재사용한다."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"val{T}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        print(f"  cache hit {path}", flush=True)
        return d["p"], d["y"], d["count"]

    tr = (df["season"] <= T - 1).values
    va = (df["season"] == T).values
    global_mean = float(y_all[tr].mean())
    X = engineer(df.drop(columns=[ID, TARGET]), global_mean)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in CAT_COLS]
    print(f"  train n={tr.sum()} val n={va.sum()} gm={global_mean:.4f}",
          flush=True)
    pool_tr = Pool(X[tr], y_all[tr], cat_features=ci)
    pool_va = Pool(X[va], cat_features=ci)
    preds = []
    for sd in SEEDS:
        m = CatBoostClassifier(**dict(PARAMS, random_seed=sd)).fit(pool_tr)
        preds.append(m.predict_proba(pool_va)[:, 1])
        print(f"    seed {sd} done", flush=True)
    p = np.mean(preds, axis=0)
    y = y_all[va]
    cnt = (df.loc[va, "balls_before"].astype(str) + "-"
           + df.loc[va, "strikes_before"].astype(str)).values
    np.savez_compressed(path, p=p, y=y, count=cnt)
    return p, y, cnt


def extract_delta(p, y, cnt):
    """전역 이동 후 카운트별 잔여를 0으로 만드는 로짓 오프셋."""
    z = logit(p) - solve_shift(logit(p), y.mean())
    out, w = {}, {}
    for c in np.unique(cnt):
        m = cnt == c
        out[c] = solve_shift(z[m], y[m].mean())
        w[c] = m.mean()
    return out, w


def apply_delta(p, cnt, delta):
    d = np.array([delta.get(c, 0.0) for c in cnt])
    return logit(p) - d


def main():
    print("Load train...", flush=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y_all = df[TARGET].astype(int).values

    data, deltas, weights = {}, {}, {}
    for T in YEARS:
        print(f"\n=== eval {T}", flush=True)
        p, y, cnt = get_preds(df, y_all, T)
        data[T] = (p, y, cnt)
        deltas[T], weights[T] = extract_delta(p, y, cnt)
        print(f"  base rate={y.mean():.4f}  BSS(no shift)={bss(p, y):.1f}",
              flush=True)

    keys = sorted(set().union(*[set(d) for d in deltas.values()]))
    print("\n=== delta_c (로짓 오프셋, 양수 = 그 카운트를 더 내려야 함) ===")
    hdr = "  count  " + "".join(f"{T:>10}" for T in YEARS) + "      w(2024)"
    print(hdr)
    for c in keys:
        row = "".join(f"{deltas[T].get(c, float('nan')):>10.4f}" for T in YEARS)
        print(f"  {c:<7}{row}   {weights[YEARS[-1]].get(c, 0):>10.4f}")

    print("\n=== 게이트: 연도 간 가중상관 / 부호일치 ===")
    gate = {}
    for i, A in enumerate(YEARS):
        for B in YEARS[i + 1:]:
            a = np.array([deltas[A].get(c, 0.0) for c in keys])
            b = np.array([deltas[B].get(c, 0.0) for c in keys])
            w = np.array([(weights[A].get(c, 0) + weights[B].get(c, 0)) / 2
                          for c in keys])
            w = w / w.sum()
            am, bm = (w * a).sum(), (w * b).sum()
            cov = (w * (a - am) * (b - bm)).sum()
            r = cov / np.sqrt((w * (a - am) ** 2).sum()
                             * (w * (b - bm) ** 2).sum())
            sign = float((w * (np.sign(a - am) == np.sign(b - bm))).sum())
            gate[f"{A}~{B}"] = {"weighted_corr": float(r), "sign_agree": sign}
            print(f"  {A} ~ {B}:  corr={r:+.3f}   부호일치(가중)={sign:.2f}")

    print("\n=== 전이 채점: S의 delta 를 T에 적용 (양쪽 arm 전역이동 재최적화) ===")
    print("  기준선 = 전역 이동만. 양수여야 의미 있음.")
    trans = {}
    for S in YEARS:
        for T in YEARS:
            if S == T:
                continue
            p, y, cnt = data[T]
            z0 = logit(p)
            base = bss(sigmoid(z0 - solve_shift(z0, y.mean())), y)
            z1 = apply_delta(p, cnt, deltas[S])
            new = bss(sigmoid(z1 - solve_shift(z1, y.mean())), y)
            trans[f"{S}->{T}"] = new - base
            mark = "self" if S == T else ""
            print(f"  {S} -> {T}:  {base:8.1f} -> {new:8.1f}   "
                  f"델타 {new - base:+7.1f} {mark}", flush=True)

    # 자기연도 상한 (in-sample, 참고용)
    print("\n  [참고] 자기연도 상한 (in-sample, 전이 아님)")
    for T in YEARS:
        p, y, cnt = data[T]
        z0 = logit(p)
        base = bss(sigmoid(z0 - solve_shift(z0, y.mean())), y)
        z1 = apply_delta(p, cnt, deltas[T])
        print(f"  {T} -> {T}:  {base:8.1f} -> {bss(sigmoid(z1 - solve_shift(z1, y.mean())), y):8.1f}"
              f"   델타 {bss(sigmoid(z1 - solve_shift(z1, y.mean())), y) - base:+7.1f}")

    json.dump({"delta": {str(T): deltas[T] for T in YEARS},
               "gate": gate, "transfer": trans},
              open("probe_count_offset.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nSaved probe_count_offset.json")

    pos = sum(1 for v in trans.values() if v > 0)
    print(f"\n=== 요약: 전이 {len(trans)}개 중 양수 {pos}개, "
          f"중앙값 {np.median(list(trans.values())):+.1f} ===")


if __name__ == "__main__":
    main()
