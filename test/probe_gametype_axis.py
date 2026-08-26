"""투수×game_type(F/R) offset 여지·지속성 — §2-G 축 분해표의 **안 재본 칸**.

배경: 08 §5-12 축 분해표와 09 §2-G가 투수×{좌우·월·주자·1루·득점권·카운트·이닝·2S}를
전부 쟀는데 **game_type만 빠져 있다.** 그런데
  · 2024 레벨 격차 F .4593 vs R .4897 = **−3.0%p** (좌우 3.1%p와 같은 규모)
  · 투수 **37%가 한 시즌에 두 레벨 다 던진다** → 행마다 값이 바뀐다(판별식 ②)
  · `asof_pitcher_*`는 두 레벨을 **섞어서** 누적한다 → 개인별 오염이 실재할 수 있다
🔴 함정: F는 2023에 체제가 붕괴했다(.709 → .473). ≤2022 이력으로 만든 F−R 격차는
   시대가 다르다 → 지속성을 **≤2023 전체 / 2023만** 두 가지로 잰다.

절차는 probe_situation_headroom.py와 동일(비교 가능성 유지).
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import glob
import numpy as np
import pandas as pd

TARGET = "control_success"; M = 270; MIN_P = 200; MIN_CELL = 50


def load():
    cols = ["row_id", "season", "pitcher_id", "pitcher_hand", "batter_hand",
            "game_type", TARGET]
    df = pd.read_csv("data/train.csv", usecols=cols, encoding="utf-8-sig")
    df["plat"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    df["gt"] = (df["game_type"] == "F").astype(int)
    va = (df["season"] == 2024).values
    L = pd.read_csv("recovered_labels.csv.gz", usecols=["row_id", "middle"])
    have = df[["row_id"]].merge(L, on="row_id", how="left")["middle"].notna().values[va]
    pred = np.mean([np.load(p) for p in
                    sorted(glob.glob("artifacts/auxpred_ins_013_backup/*.npy"))], axis=0)
    d = df[va][have].reset_index(drop=True).copy(); d["pred"] = pred
    return df, d


def headroom(d, key, tag):
    x = d.copy(); x["cell"] = key
    pg = x.groupby("pitcher_id").agg(n_p=(TARGET, "size"),
                                     act_p=(TARGET, "mean"), pr_p=("pred", "mean"))
    pg = pg[pg["n_p"] >= MIN_P]
    x = x[x["pitcher_id"].isin(pg.index)]
    cg = x.groupby(["pitcher_id", "cell"]).agg(n=(TARGET, "size"),
                                               act=(TARGET, "mean"), pr=("pred", "mean"))
    cg = cg[cg["n"] >= MIN_CELL].join(pg, on="pitcher_id")
    if len(cg) == 0:
        return None
    n = cg["n"].values.astype(float); w = n / n.sum()
    b = (cg["act"] - cg["pr"]).values
    b_p = (cg["act_p"] - cg["pr_p"]).values
    var = (cg["act"] * (1 - cg["act"])).values
    raw = float((w * (b - (w * b).sum()) ** 2).sum())
    raw_n = float((w * var / n).sum())
    inc = float((w * (b - b_p) ** 2).sum())
    inc_n = float((w * var * (1 / n - 1 / cg["n_p"].values)).sum())
    A = 1e5 / 0.25
    return dict(cells=len(cg), pitchers=cg.index.get_level_values(0).nunique(),
                cover=n.sum() / len(d), raw=A * max(raw - raw_n, 0),
                inc=A * max(inc - inc_n, 0), raw_pre=A * raw, inc_pre=A * inc)


def build_split(hist, weight_rows, col, m=M):
    g = hist.groupby(["pitcher_id", col])[TARGET].agg(["sum", "count"])
    pr = hist.groupby("pitcher_id")[TARGET].mean().rename("pr")
    g = g.join(pr, on="pitcher_id")
    g["v"] = (g["sum"] + m * g["pr"]) / (g["count"] + m)
    t = (g["v"] - g["pr"]).rename("split").reset_index()
    w = weight_rows.merge(t, on=["pitcher_id", col], how="left")["split"]
    t["split"] = t["split"] - float(w.fillna(0).mean())
    return t


def persistence(df, col, lo, hi, tgt):
    hist = df[(df.season >= lo) & (df.season <= hi)]
    cur = df[df.season == tgt]
    a = build_split(hist, hist, col).rename(columns={"split": "a"})
    bt = build_split(cur, cur, col).rename(columns={"split": "b"})
    n = cur.groupby(["pitcher_id", col])[TARGET].size().rename("n").reset_index()
    z = a.merge(bt, on=["pitcher_id", col]).merge(n, on=["pitcher_id", col])
    z = z[z["n"] >= MIN_CELL]
    if len(z) < 30:
        return None
    c = float(np.corrcoef(z["a"], z["b"])[0, 1])
    tau2 = c * z["a"].std() * z["b"].std()
    return dict(cells=len(z), corr=c, tau=np.sqrt(max(tau2, 0)),
                bss=1e5 / 0.25 * max(tau2, 0))


def main():
    df, d = load()
    y = d[TARGET].values; r = y.mean()
    print(f"013 검증예측 {len(d):,}행  BSS="
          f"{1e5*(1-np.mean((d.pred-y)**2)/(r*(1-r))):.1f}\n", flush=True)

    print(f"{'축':<26}{'셀':>7}{'투수':>6}{'커버':>7}{'raw여지':>10}{'증분여지':>10}")
    print("-" * 70)
    for tag, key in [("투수 단독 [대조군]", np.zeros(len(d), int)),
                     ("투수×좌우 [대조군]", d["plat"].values),
                     ("투수×game_type", d["gt"].values),
                     ("투수×좌우×game_type",
                      d["plat"].values * 2 + d["gt"].values)]:
        h = headroom(d, key, tag)
        if h is None:
            print(f"{tag:<26}  (셀 없음)"); continue
        print(f"{tag:<26}{h['cells']:>7,}{h['pitchers']:>6}{h['cover']*100:>6.0f}%"
              f"{h['raw']:>10.1f}{h['inc']:>10.1f}", flush=True)

    print("\n=== 지속성 (§0-5) — 과거 표가 2024에 이어지는가 ===")
    print(f"{'축 / 이력구간':<30}{'셀':>7}{'corr':>8}{'tau':>9}{'전이가능':>10}")
    print("-" * 66)
    for col, tag, lo, hi in [("plat", "투수×좌우 [대조군] ≤2023", 2019, 2023),
                             ("gt", "투수×game_type  ≤2023", 2019, 2023),
                             ("gt", "투수×game_type  2023만", 2023, 2023),
                             ("gt", "투수×game_type  ≤2022(붕괴전)", 2019, 2022)]:
        p = persistence(df, col, lo, hi, 2024)
        if p is None:
            print(f"{tag:<30}  (셀 부족)"); continue
        print(f"{tag:<30}{p['cells']:>7,}{p['corr']:>8.3f}{p['tau']:>9.4f}"
              f"{p['bss']:>10.1f}", flush=True)

    print("\n=== 참고: 레벨 격차 자체가 투수마다 다른가 ===")
    cur = df[df.season == 2024]
    g = cur.groupby(["pitcher_id", "gt"])[TARGET].agg(["mean", "count"])
    g = g[g["count"] >= MIN_CELL].unstack("gt").dropna()
    gap = g[("mean", 1)] - g[("mean", 0)]
    print(f"  두 레벨 다 {MIN_CELL}구 이상 던진 투수 {len(gap)}명")
    print(f"  F−R 격차  평균 {gap.mean():+.4f}  sd {gap.std():.4f}")
    nse = float(np.mean(0.25 * (1 / g[("count", 1)] + 1 / g[("count", 0)])))
    print(f"  이항노이즈 sd {np.sqrt(nse):.4f}  → 참신호 sd "
          f"{np.sqrt(max(gap.var()-nse,0)):.4f}")


if __name__ == "__main__":
    main()
