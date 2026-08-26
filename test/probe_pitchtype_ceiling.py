"""구종 축의 구조적 상한 — 오라클 160.5 vs 달성가능 10.7 (09 §3-E 승격).

§3-E는 "2단 모델을 얹어보니 +0.6"이라는 **경험적** 기각이었다. 여기서는
"어떻게 해도 얼마 이상은 못 나온다"를 **구조적으로** 못 박는다.

  ① 오라클: 진짜 구종을 안다면 013 잔차에서 회수 가능한 몫
  ② 달성가능: 구종을 모르고 P(구종|x)만 쓸 때의 몫 (조정량 분산의 상한)
  ③ 실측: 잔차 편향을 투수 변화구비율 10분위로 재분해

핵심은 ②/① = 6.7%다. P(변화구|x)의 sd가 0.119뿐인데 라벨 자체는 0.46 —
투수가 이 공에 뭘 던질지는 주어진 컬럼으로 거의 예측되지 않는다.
게다가 ②의 재료(투수 믹스 4.7 + 카운트 4.5)는 **트리가 이미 가진 컬럼**이라
정보가 아니라 표현이다(CLAUDE.md §7-1 판별식 ①).

⚠️ ②는 낙관적 상한이다: 구종별 잔차 DELTA를 목표연도(2024) 자체에서 적합했다.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import glob

import numpy as np
import pandas as pd

TY = ["breaking", "fastball", "offspeed"]
A = 1e5 / 0.25


def load():
    cols = ["row_id", "season", "balls_before", "strikes_before", "outs_before",
            "num_runners_on", "asof_pitcher_breaking_rate", "control_success"]
    d = pd.read_csv("data/train.csv", usecols=cols, encoding="utf-8-sig")
    L = pd.read_csv("recovered_labels.csv.gz",
                    usecols=["row_id", "middle", "fastball", "breaking", "offspeed"])
    d = d.merge(L, on="row_id", how="left")
    d["type"] = np.select([d.fastball == 1, d.breaking == 1, d.offspeed == 1],
                          ["fastball", "breaking", "offspeed"], None)
    va = (d.season == 2024).values
    have = d["middle"].notna().values[va]
    p = np.mean([np.load(x) for x in
                 sorted(glob.glob("artifacts/auxpred_ins_013_backup/*.npy"))], axis=0)
    v = d[va][have].reset_index(drop=True).copy()
    v["pred"] = p
    return d, v


def cellkey(x, use):
    k = pd.Series("", index=x.index)
    if "count" in use:
        k = k + x.balls_before.astype(str) + "-" + x.strikes_before.astype(str)
    if "mix" in use:
        q = pd.qcut(x.asof_pitcher_breaking_rate.fillna(-1), 10,
                    labels=False, duplicates="drop")
        k = k + "|m" + pd.Series(q, index=x.index).astype(str)
    if "sit" in use:
        k = k + "|s" + x.outs_before.astype(str) + np.clip(x.num_runners_on, 0, 2).astype(str)
    return k


def main():
    d, v = load()
    hist = d[(d.season <= 2023) & d.type.notna()].copy()

    print("=== ① 오라클: 진짜 구종을 알 때 ===")
    g = v.groupby("type").agg(n=("control_success", "size"),
                              act=("control_success", "mean"), pr=("pred", "mean"))
    g["잔차"] = g.act - g.pr
    print(g.round(4).to_string())
    n = g.n.values.astype(float); w = n / n.sum(); b = g["잔차"].values
    raw = (w * (b - (w * b).sum()) ** 2).sum()
    noise = (w * (g.act * (1 - g.act)).values / n).sum()
    orc = max(raw - noise, 0)
    print(f"  실제 성공률 최대격차 {g.act.max()-g.act.min():.4f}"
          f"   → 오라클 ΔBSS = {A*orc:+.1f}\n")

    DELTA = dict(zip(g.index, g["잔차"].values))
    print("=== ② 달성가능: P(구종|x)만 쓸 때 (DELTA는 2024 적합 = 낙관) ===")
    print(f"{'조건':<26}{'셀':>7}{'sd(조정량)':>12}{'ΔBSS':>9}{'오라클 대비':>11}")
    print("-" * 66)
    for use, tag in [(["mix"], "투수 믹스만 [기존 컬럼]"), (["count"], "카운트만 [기존 컬럼]"),
                     (["count", "mix"], "카운트×믹스"), (["count", "mix", "sit"], "카운트×믹스×상황")]:
        P = pd.crosstab(cellkey(hist, use), hist.type, normalize="index")
        vk = cellkey(v, use)
        adj = sum(vk.map(P[t]).fillna(hist.type.eq(t).mean()) * DELTA[t] for t in TY).values
        var = float(np.var(adj))
        print(f"{tag:<26}{P.shape[0]:>7,}{np.std(adj):>12.5f}{A*var:>9.1f}"
              f"{var/orc*100:>10.1f}%")

    print("\n=== P(breaking|x) 자체가 얼마나 흔들리는가 ===")
    print(f"  라벨 자체 sd = {v.breaking.std():.4f}  ← 이만큼 알아야 오라클")
    for use, tag in [(["mix"], "투수만"), (["count"], "카운트만"), (["count", "mix"], "둘 다")]:
        P = pd.crosstab(cellkey(hist, use), hist.type, normalize="index")["breaking"]
        p = cellkey(v, use).map(P).fillna(P.mean()).values
        print(f"  {tag:<10} sd(P)={np.std(p):.4f}  범위 {p.min():.3f}~{p.max():.3f}")

    print("\n=== ③ 실측: 013 잔차 편향 by 투수 변화구비율 10분위 ===")
    v["dec"] = pd.qcut(v.asof_pitcher_breaking_rate.fillna(-1), 10,
                       labels=False, duplicates="drop")
    t = v.groupby("dec").agg(n=("control_success", "size"),
                             brk=("asof_pitcher_breaking_rate", "mean"),
                             act=("control_success", "mean"), pr=("pred", "mean"))
    t["편향"] = t.pr - t.act
    print(t.round(4).to_string())
    n = t.n.values.astype(float); w = n / n.sum(); b = t["편향"].values
    r2 = (w * (b - (w * b).sum()) ** 2).sum()
    n2 = (w * (t.act * (1 - t.act)).values / n).sum()
    print(f"  → 남은 여지 {A*max(r2-n2,0):+.1f} BSS  (트리가 이미 가진 컬럼이므로 당연)")


if __name__ == "__main__":
    main()
