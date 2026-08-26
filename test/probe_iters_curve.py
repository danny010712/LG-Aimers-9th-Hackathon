"""나무 수 곡선 — 조기중단이 과적합이 아니라 분포이동에 반응하는가.

현행: eval_metric=Logloss 로 2024 홀드아웃 조기중단 -> best_iter 249~291.
      그런데 2019-2023 평균 0.5316 vs 2024 실제 0.4861 로 base rate가 다르다.
      모델이 학습 레벨로 날카로워질수록 raw Logloss는 **레벨 오차** 때문에 나빠진다.
      배포에서는 logit_shift가 그 레벨을 따로 고친다 -> 정지점 기준이 어긋나 있다.

여기서는 조기중단 없이 2000그루까지 학습하고 나무 수별로 두 값을 잰다:
  raw BSS        (현행 정지 기준이 보는 것)
  레벨보정 BSS   (배포에서 실제로 보게 되는 것)

두 곡선의 정점이 크게 다르면 **우리는 계속 과소적합해 왔다**.

⚠️ 이 축은 미관측 시즌 외삽 제어 부류다(depth 8: 로컬 +7.6 / LB -34.2).
   로컬 곡선은 근거의 절반일 뿐 — 확정은 제출로만 된다.
   다만 여기 논거는 로컬 점수 추종이 아니라 **정지 기준과 배포 구성의 일치**다.
"""
import io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import brentq

sys.path.insert(0, "common")
from features import engineer, build_anchor, rate_priors, CAT_COLS  # noqa: E402

ID, TARGET = "row_id", "control_success"
SEEDS = [42, 7]
MAXIT = 2000
GRID = list(range(50, MAXIT + 1, 50))

df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
y = df[TARGET].astype(int).values
tr, va = (df.season <= 2023).values, (df.season == 2024).values
gm = float(y[tr].mean())
X = engineer(df.drop(columns=[ID, TARGET]), gm,
             anchor=build_anchor(df), priors=rate_priors(df[tr]))
for c in CAT_COLS:
    X[c] = X[c].astype(str)
ci = [X.columns.get_loc(c) for c in CAT_COLS]
pool_tr, pool_va = Pool(X[tr], y[tr], cat_features=ci), Pool(X[va], cat_features=ci)
yv = y[va].astype(float)
r = yv.mean()
print(f"학습평균 {gm:.4f}  2024 실제 {r:.4f}  차이 {gm-r:+.4f}", flush=True)


def bss(p):
    return max(0.0, 100000 * (1 - np.mean((p - yv) ** 2) / (r * (1 - r))))


def recentered(p):
    z = np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))
    d = brentq(lambda d: (1/(1+np.exp(-(z+d)))).mean() - r, -3, 3)
    return bss(1/(1+np.exp(-(z+d))))


raw = {k: [] for k in GRID}
lvl = {k: [] for k in GRID}
for sd in SEEDS:
    t0 = time.time()
    m = CatBoostClassifier(iterations=MAXIT, learning_rate=0.05, depth=6,
                           thread_count=-1, verbose=0, random_seed=sd).fit(pool_tr)
    print(f"seed {sd} 학습 {time.time()-t0:.0f}s", flush=True)
    for k in GRID:
        p = m.predict_proba(pool_va, ntree_end=k)[:, 1]
        raw[k].append(bss(p)); lvl[k].append(recentered(p))
    print(f"seed {sd} 평가 완료 {time.time()-t0:.0f}s", flush=True)

print(f"\n{'나무':>5} {'raw':>9} {'레벨보정':>10}   (2시드 평균)")
br = bl = -1
for k in GRID:
    a, b = np.mean(raw[k]), np.mean(lvl[k])
    mark = ""
    if a > br: br, kr, mark = a, k, mark + " ←raw최고"
    if b > bl: bl, kl = b, k; mark += " ←보정최고"
    print(f"{k:>5} {a:>9.1f} {b:>10.1f}{mark}", flush=True)

print(f"\nraw      정점 {kr}그루  {br:.1f}   (현행 조기중단 249~291)")
print(f"레벨보정 정점 {kl}그루  {bl:.1f}")
print(f"→ 정점 비 {kl/max(kr,1):.1f}배,  현행 278그루 대비 보정정점에서 "
      f"{bl - np.mean(lvl[GRID[min(range(len(GRID)), key=lambda i: abs(GRID[i]-278))]]):+.1f}")
