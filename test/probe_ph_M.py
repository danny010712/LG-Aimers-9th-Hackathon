"""ph 표의 스무딩 M — 044(LB 1062.55)가 쓰는 cond.M=50이 최적인가.

지속 진폭 = corr x 가중sd  (다음 해로 실제 전달되는 신호 크기).
M을 올리면 corr는 오르고 sd는 준다. 곱이 어디서 최대인지 본다.
platoon offset은 M=270을 쓴다(tau 역산값).
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd

df = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                 usecols=["season", "pitcher_id", "batter_id",
                          "pitcher_hand", "batter_hand", "control_success"])
lg = lambda q: np.log(np.clip(q, 1e-9, 1-1e-9) / (1 - np.clip(q, 1e-9, 1-1e-9)))


def dev(d, keys, prior, m):
    g = d.groupby(keys)["control_success"].agg(["sum", "size"])
    b = d.groupby(prior)["control_success"].agg(["sum", "size"])
    pb = (b["sum"] / b["size"]).reindex(g.index.get_level_values(prior)).values
    p = (g["sum"] + m * pb) / (g["size"] + m)
    return pd.DataFrame({"dev": lg(p) - lg(pb), "n": g["size"].values},
                        index=g.index).reset_index()


wsd = lambda t: float(np.sqrt(np.average(t.dev**2, weights=t.n)
                              - np.average(t.dev, weights=t.n)**2))

for name, keys, prior in [("ph 투수x타자손", ["pitcher_id", "batter_hand"], "pitcher_id"),
                          ("bh 타자x투수손", ["batter_id", "pitcher_hand"], "batter_id")]:
    print(f"\n[{name}]   {'M':>5} {'corr':>9} {'가중sd':>9} {'**지속진폭**':>13}")
    best = (None, -1)
    for m in (20, 35, 50, 80, 120, 180, 270, 400, 600):
        a, b = dev(df[df.season <= 2023], keys, prior, m), dev(df[df.season == 2024], keys, prior, m)
        mm = a.merge(b, on=keys, suffixes=("_a", "_b"))
        mm = mm[(mm.n_a >= 30) & (mm.n_b >= 30)]
        c = np.corrcoef(mm.dev_a, mm.dev_b)[0, 1]
        amp = c * wsd(a)
        mark = ""
        if amp > best[1]:
            best = (m, amp); mark = "  <- 최고"
        if m == 50:
            mark += "   <- 현행 044"
        print(f"        {m:>5} {c:>+9.4f} {wsd(a):>9.4f} {amp:>13.4f}{mark}")
    print(f"   최적 M={best[0]}  진폭 {best[1]:.4f}  "
          f"(현행 대비 {100*best[1]/[x for x in [None]][0] if False else 0:.0f})")
