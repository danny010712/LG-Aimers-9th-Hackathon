"""투수×상황 잔차 여지 측정 — 08 §5-12 축 분해표의 빈 칸 채우기.

배경: 021(투수×좌우)이 채택되며 축 분해표가 만들어졌다. 거기서
  투수×좌우 126 / 투수×월 93 / 투수 단독 22 / 투수×카운트·이닝·2스트라이크 0
이었다. 살아난 건 '몸에서 오는 축'(좌우)뿐이고 '머리에서 오는 축'(카운트·이닝)은
전부 죽었다. 그렇다면 아직 안 잰 칸이 하나 남는다 — **투수×주자유무**
(와인드업 vs 셋포지션 = 몸이 다르게 움직인다).

측정 대상:
  ① 투수×주자유무 · 투수×1루주자 · 투수×득점권 · 투수×월
  ② 대조군: 투수 단독(≈22) · 투수×좌우(≈126) — 문서값 재현되면 절차 검증됨
  ③ 021(platoon) 적용 후 잔차로도 재서 겹침 확인

측정법 (08 §5-12과 동일):
  셀(투수×레벨)별 실제평균 - 예측평균 = 편향 b_i, 사용량 가중 분산에서
  이항 표본노이즈를 뺀 것이 참신호. ΔBSS = 참신호 / 0.25 * 1e5.

  raw    = Σ w_i (b_i - b_bar)^2                    (투수 단독 성분 포함)
  증분   = Σ w_i (b_i - b_pitcher)^2                 (투수 평균 제거 = offset이 실제로 얻는 몫)
  노이즈: raw  Var(b_i)      = p(1-p)/n_i
          증분 Var(b_i-b_p)  = p(1-p)(1/n_i - 1/n_p)

🔴 이건 오라클 천장이다. 021은 여지 126에서 로컬 +20.5 / LB +5.27을 얻었다(0.16 / 0.04).
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import glob

import numpy as np
import pandas as pd

TARGET = "control_success"
M = 270          # build_platoon.py와 동일한 수축 강도
MIN_P = 200      # 투수 최소 투구수
MIN_CELL = 50    # 셀 최소 투구수


def load_2024():
    cols = ["row_id", "season", "pitcher_id", "batter_id", "pitcher_hand",
            "batter_hand", "num_runners_on", "runner_on_1b", "runner_on_2b",
            "runner_on_3b", "base_state", "game_month", "balls_before",
            "strikes_before", "inning", TARGET]
    df = pd.read_csv("data/train.csv", usecols=cols, encoding="utf-8-sig")
    df["plat"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    va = (df["season"] == 2024).values
    L = pd.read_csv("recovered_labels.csv.gz", usecols=["row_id", "middle"])
    have = df[["row_id"]].merge(L, on="row_id", how="left")["middle"].notna().values[va]
    pred = np.mean([np.load(p) for p in
                    sorted(glob.glob("artifacts/auxpred_ins_013_backup/*.npy"))], axis=0)
    d = df[va][have].reset_index(drop=True).copy()
    d["pred"] = pred
    return df, d


def headroom(d, key, tag):
    """축 하나의 raw / 증분 여지를 BSS로 환산."""
    x = d.copy()
    x["cell"] = key
    pg = x.groupby("pitcher_id").agg(n_p=(TARGET, "size"),
                                     act_p=(TARGET, "mean"), pr_p=("pred", "mean"))
    pg = pg[pg["n_p"] >= MIN_P]
    x = x[x["pitcher_id"].isin(pg.index)]
    cg = x.groupby(["pitcher_id", "cell"]).agg(n=(TARGET, "size"),
                                               act=(TARGET, "mean"), pr=("pred", "mean"))
    cg = cg[cg["n"] >= MIN_CELL].join(pg, on="pitcher_id")
    if len(cg) == 0:
        return None
    n = cg["n"].values.astype(float)
    w = n / n.sum()
    b = (cg["act"] - cg["pr"]).values
    b_p = (cg["act_p"] - cg["pr_p"]).values
    var = (cg["act"] * (1 - cg["act"])).values

    raw = float((w * (b - (w * b).sum()) ** 2).sum())
    raw_noise = float((w * var / n).sum())
    inc = float((w * (b - b_p) ** 2).sum())
    inc_noise = float((w * var * (1 / n - 1 / cg["n_p"].values)).sum())

    A = 1e5 / 0.25
    return dict(tag=tag, cells=len(cg), pitchers=cg.index.get_level_values(0).nunique(),
                cover=n.sum() / len(d),
                raw=A * max(raw - raw_noise, 0), inc=A * max(inc - inc_noise, 0),
                raw_pre=A * raw, inc_pre=A * inc)


def build_split(hist, weight_rows, col, m=M):
    g = hist.groupby(["pitcher_id", col])[TARGET].agg(["sum", "count"])
    pr = hist.groupby("pitcher_id")[TARGET].mean().rename("pr")
    g = g.join(pr, on="pitcher_id")
    g["v"] = (g["sum"] + m * g["pr"]) / (g["count"] + m)
    t = (g["v"] - g["pr"]).rename("split").reset_index()
    w = weight_rows.merge(t, on=["pitcher_id", col], how="left")["split"]
    t["split"] = t["split"] - float(w.fillna(0).mean())
    return t


def persistence(df, col, tag):
    """과거(≤2023) 편차 표가 2024 편차와 얼마나 이어지는가."""
    hist = df[df["season"] <= 2023]
    cur = df[df["season"] == 2024]
    a = build_split(hist, hist, col).rename(columns={"split": "a"})
    bt = build_split(cur, cur, col).rename(columns={"split": "b"})
    n = cur.groupby(["pitcher_id", col])[TARGET].size().rename("n").reset_index()
    z = a.merge(bt, on=["pitcher_id", col]).merge(n, on=["pitcher_id", col])
    z = z[z["n"] >= MIN_CELL]
    if len(z) < 30:
        return None
    c = float(np.corrcoef(z["a"], z["b"])[0, 1])
    tau2 = c * z["a"].std() * z["b"].std()
    return dict(tag=tag, cells=len(z), corr=c, tau=np.sqrt(max(tau2, 0)),
                bss=1e5 / 0.25 * max(tau2, 0))


def main():
    df, d = load_2024()
    y = d[TARGET].values
    r = y.mean()
    print(f"013 검증예측 {len(d):,}행  BSS="
          f"{1e5*(1-np.mean((d.pred-y)**2)/(r*(1-r))):.1f}  base rate={r:.4f}\n",
          flush=True)

    rc = df["num_runners_on"].values
    axes = [
        ("투수 단독 [대조군]", np.zeros(len(d), int)),
        ("투수×좌우 [대조군]", d["plat"].values),
        ("투수×주자유무", (d["num_runners_on"] > 0).astype(int).values),
        ("투수×1루주자", d["runner_on_1b"].astype(int).values),
        ("투수×득점권", ((d["runner_on_2b"] | d["runner_on_3b"]) > 0).astype(int).values),
        ("투수×주자수(0/1/2+)", np.clip(d["num_runners_on"].values, 0, 2)),
        ("투수×월", d["game_month"].values),
        ("투수×카운트 [대조군·죽음]",
         d["balls_before"].astype(str).values + "-" + d["strikes_before"].astype(str).values),
    ]
    print(f"{'축':<26}{'셀':>7}{'투수':>6}{'커버':>7}"
          f"{'raw여지':>10}{'증분여지':>10}  (노이즈 빼기 전)")
    print("-" * 88)
    res = {}
    for tag, key in axes:
        h = headroom(d, key, tag)
        if h is None:
            print(f"{tag:<26}  (셀 없음)")
            continue
        res[tag] = h
        print(f"{tag:<26}{h['cells']:>7,}{h['pitchers']:>6}{h['cover']*100:>6.0f}%"
              f"{h['raw']:>10.1f}{h['inc']:>10.1f}"
              f"   ({h['raw_pre']:.0f} / {h['inc_pre']:.0f})", flush=True)

    print("\n=== 지속성: 과거(≤2023) 편차가 2024에 이어지는가 ===")
    print(f"{'축':<26}{'셀':>7}{'corr':>8}{'tau':>9}{'전이가능 BSS':>14}")
    print("-" * 66)
    df["runners"] = (df["num_runners_on"] > 0).astype(int)
    df["r1b"] = df["runner_on_1b"].astype(int)
    df["risp"] = ((df["runner_on_2b"] | df["runner_on_3b"]) > 0).astype(int)
    for col, tag in [("plat", "투수×좌우 [대조군]"), ("runners", "투수×주자유무"),
                     ("r1b", "투수×1루주자"), ("risp", "투수×득점권"),
                     ("game_month", "투수×월")]:
        p = persistence(df, col, tag)
        if p is None:
            print(f"{tag:<26}  (셀 부족)")
            continue
        print(f"{tag:<26}{p['cells']:>7,}{p['corr']:>8.3f}{p['tau']:>9.4f}"
              f"{p['bss']:>14.1f}", flush=True)


if __name__ == "__main__":
    main()
