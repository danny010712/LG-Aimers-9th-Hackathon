"""투수 좌우편차(platoon split) 잔차 offset — 두 번째 전이로 재확인 (기준 #5).

2024 측정: ≤2023 표 → 013 잔차, 투수분할 CV → +28.1 BSS (b=0.52~0.55)
여기: ≤2022 표 → 2023 잔차. 베이스 모델을 ≤2022로 새로 학습해 2023을 예측한다.

`cond`의 `ph` 표와 **같은 정보, 다른 전달**이다. cond는 트리 피처로 줬고(004 LB −5.4)
여기는 잔차에 고정계수로 더한다(용량 증가 0·재보정 없음 = offset과 같은 구조).
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
sys.path.insert(0, "common")
import features as F
F.ANCHOR_SPECS = [("pitcher", "success"), ("batter", "success")]   # 013 구성
from features import engineer, build_anchor, rate_priors, CAT_COLS

TARGET = "control_success"
df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
df["plat"] = (df.pitcher_hand == df.batter_hand).astype(int)

def run(tr_end, te_year, seed=42):
    tr = (df.season <= tr_end).values
    te = (df.season == te_year).values
    y = df[TARGET].astype(int).values
    gm = float(y[tr].mean())
    X = engineer(df.drop(columns=["row_id", TARGET, "plat"]), gm,
                 anchor=build_anchor(df), priors=rate_priors(df[tr]))
    for c in CAT_COLS: X[c] = X[c].astype(str)
    ci = [X.columns.get_loc(c) for c in CAT_COLS]
    m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6,
                           thread_count=-1, verbose=0, eval_metric="Logloss",
                           early_stopping_rounds=100, random_seed=seed).fit(
        Pool(X[tr], y[tr], cat_features=ci),
        eval_set=Pool(X[te], y[te], cat_features=ci), use_best_model=True)
    p = m.predict_proba(Pool(X[te], cat_features=ci))[:, 1]
    r = y[te].mean()
    print(f"  베이스 {tr_end}->{te_year}  iter={m.get_best_iteration()}  "
          f"BSS={100000*(1-np.mean((p-y[te])**2)/(r*(1-r))):.1f}", flush=True)
    return df[te].copy().assign(pred=p, res=y[te]-p)

def table(hist, M=200):
    g = hist.groupby(["pitcher_id", "plat"])[TARGET].agg(["sum", "count"])
    pr = hist.groupby("pitcher_id")[TARGET].mean().rename("pr")
    g = g.join(pr, on="pitcher_id")
    g["v"] = (g["sum"] + M * g["pr"]) / (g["count"] + M)
    return (g["v"] - g["pr"]).rename("split").reset_index()

def evaluate(d, tab, tag):
    z = d.merge(tab, on=["pitcher_id", "plat"], how="left")
    z["split"] = z["split"].fillna(0)
    x, r = z["split"].values, z["res"].values
    pids = z.pitcher_id.unique()
    fold = pd.Series(np.random.default_rng(0).integers(0, 5, len(pids)), index=pids)
    f = z.pitcher_id.map(fold).values
    oof, bs = np.zeros(len(z)), []
    for k in range(5):
        m = f != k
        b = float((x[m] * r[m]).sum() / (x[m] * x[m]).sum()); bs.append(b)
        oof[~m] = b * x[~m]
    dB = (r ** 2).mean() - ((r - oof) ** 2).mean()
    print(f"  [{tag}] 커버 {(x!=0).mean()*100:.0f}%  b={['%.3f'%v for v in bs]}  "
          f"→ **OOF ΔBSS {dB/0.25*100000:+.1f}**", flush=True)

if __name__ == "__main__":
    print("=== 전이 2: 2022 표 → 2023 (기준 #6: 출처 2022 · 목표 2023) ===", flush=True)
    d23 = run(2022, 2023)
    evaluate(d23, table(df[df.season <= 2022]), "2022->2023")
    print("\n=== 전이 3: 2023 표 → 2024, 베이스도 ≤2023 (최근접) ===", flush=True)
    d24 = run(2023, 2024)
    evaluate(d24, table(df[df.season <= 2023]), "2023->2024")
