"""투수 역할을 train 정적 표로 만들어 조인 — §2-O의 진폭 문제를 피할 수 있나 (2026-08-25).

사용자 제안: '그 시즌 총 투구수'는 미래 정보라 못 쓰지만, **train에서 투수별 역할을
계산해 표로 만들어 pitcher_id로 조인**하면 된다. platoon.csv·anchor.csv와 같은 형태.

§2-O가 죽은 이유는 '모양은 이어지는데 **진폭**이 10배 흔들려서'였다. 그건 행 단위
파생(ins_n/경과개월)이라 시즌 내에서 계속 변한다. **정적 표는 안 변한다.**

역할 정의: 등판당 투구수 = 그 투수가 한 경기(gkey)에서 던진 투구 수의 중앙값.
  선발은 80~100, 불펜은 15~25 → 고전적 판별자.
"""
import io
import sys

if getattr(sys.stdout, "encoding", "") != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

GK = ["season", "game_month", "game_dayofweek", "pitcher_team_id", "batter_team_id"]
lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
sg = lambda t: 1 / (1 + np.exp(-t))


def load():
    c = ["row_id", "season", "pitcher_id"] + GK[1:]
    return pd.read_csv("data/train.csv", usecols=list(dict.fromkeys(c)))


def role_table(d, upto):
    """<=upto 시즌으로 투수별 등판당 투구수 중앙값. 배포 시에는 train 전체로 만든다."""
    h = d[d.season <= upto].copy()
    h["gkey"] = h[GK].astype(str).agg("|".join, axis=1)
    per = h.groupby(["pitcher_id", "gkey"]).size().rename("n").reset_index()
    t = per.groupby("pitcher_id").agg(ppg=("n", "median"), games=("n", "size"))
    return t[t.games >= 5]


def main():
    d = load()
    print("=== ① 역할 표가 해마다 안정적인가 (진폭 문제 회피 여부) ===")
    a = role_table(d, 2022).rename(columns={"ppg": "a"})
    b = role_table(d, 2023).rename(columns={"ppg": "b"})
    c = role_table(d[d.season == 2024], 2024).rename(columns={"ppg": "c"})
    j = a[["a"]].join(b[["b"]], how="inner").join(c[["c"]], how="inner")
    print(f"  공통 투수 {len(j)}명")
    print(f"  ≤2022 vs ≤2023  corr {np.corrcoef(j.a, j.b)[0,1]:+.3f}")
    print(f"  ≤2023 vs 2024만  corr {np.corrcoef(j.b, j.c)[0,1]:+.3f}   "
          f"← 표 자체의 지속성")
    print(f"  등판당 투구수 분포: 중앙 {j.b.median():.0f}  "
          f"25% {j.b.quantile(.25):.0f}  75% {j.b.quantile(.75):.0f}  "
          f"최대 {j.b.max():.0f}")

    print("\n=== ② 그 표로 나눈 잔차 편향이 해마다 이어지는가 (모양 + 진폭) ===")
    p = pd.read_csv("probe_offset_forms_preds.csv.gz")
    p = p.merge(d[["row_id", "pitcher_id"]], on="row_id")
    K = 3
    prof, amp = {}, {}
    for yv in sorted(p.season.unique()):
        tab = role_table(d, yv - 1)          # 그 해 **이전**까지로 만든 표 = 배포 규칙
        g = p[p.season == yv].copy()
        g["ppg"] = g.pitcher_id.map(tab.ppg)
        g = g[g.ppg.notna()]
        z = lg(g.p_success.values)
        m, w = lg(g.p_mr.values), lg(g.p_wayoff.values)
        q0 = sg(z - 0.099 * (m - m.mean()) + 0.0074 * (w - w.mean()))
        y = g.y.values.astype(float)
        q0 = q0 + (y.mean() - q0.mean())
        k = pd.qcut(g.ppg.rank(method="first"), K, labels=False).values
        prof[yv] = [float(y[k == i].mean() - q0[k == i].mean()) for i in range(K)]
        amp[yv] = prof[yv][-1] - prof[yv][0]
        print(f"  {yv}  n={len(g):>7,}  " + "".join(f"{x:>+10.4f}" for x in prof[yv])
              + f"   진폭 {amp[yv]:>+8.4f}")
    ys = [v for v in prof if v >= 2022]
    A = np.array([prof[v] for v in ys])
    print(f"\n  진폭 부호 {sum(np.sign(amp[v]) == np.sign(np.mean([amp[v] for v in ys])) for v in ys)}/{len(ys)}"
          f"   진폭 변동폭 {max(abs(amp[v]) for v in ys)/max(min(abs(amp[v]) for v in ys),1e-9):.1f}배")
    if len(ys) >= 2:
        print(f"  ≤{ys[-2]} 평균 프로파일 vs {ys[-1]}  corr "
              f"{np.corrcoef(A[:-1].mean(0), A[-1])[0,1]:+.3f}")

    print("\n=== ③ 배포 조건 전이 — S에서 적합 → T에 적용 ===")
    print(f"{'전이':>12}{'기준(전역이동)':>16}{'+역할보정':>11}{'Δ':>9}")

    def bss(pr, y):
        r = y.mean()
        return 1e5 * (1 - np.mean((pr - y) ** 2) / (r * (1 - r)))

    for S, T in [(2023, 2024), (2022, 2024), (2021, 2022)]:
        tabS = role_table(d, S - 1)
        gs, gt = p[p.season == S].copy(), p[p.season == T].copy()
        for g in (gs, gt):
            g["ppg"] = g.pitcher_id.map(tabS.ppg)
        gs, gt = gs[gs.ppg.notna()], gt[gt.ppg.notna()]
        edge = np.quantile(gs.ppg, np.linspace(0, 1, K + 1)[1:-1])
        qs, qt = np.digitize(gs.ppg, edge), np.digitize(gt.ppg, edge)

        def bl(g):
            z = lg(g.p_success.values)
            m, w = lg(g.p_mr.values), lg(g.p_wayoff.values)
            return z - 0.099 * (m - m.mean()) + 0.0074 * (w - w.mean())

        zs, zt = bl(gs), bl(gt)
        ys_, yt = gs.y.values.astype(float), gt.y.values.astype(float)
        s0 = minimize_scalar(lambda s: np.mean((sg(zs + s) - ys_) ** 2),
                             bounds=(-.5, .5), method="bounded").x
        adj = np.full(K, s0)
        for i in range(K):
            mm = qs == i
            if mm.sum() >= 500:
                adj[i] = minimize_scalar(
                    lambda s: np.mean((sg(zs[mm] + s) - ys_[mm]) ** 2),
                    bounds=(-.5, .5), method="bounded").x
        A_, B_ = bss(sg(zt + s0), yt), bss(sg(zt + adj[qt]), yt)
        note = "  <- 배포 최근접" if (S, T) == (2023, 2024) else (
            "  (출처 F 아티팩트)" if S == 2021 else "")
        print(f"{S}->{T:>6}{A_:>16.1f}{B_:>11.1f}{B_-A_:>+9.2f}{note}", flush=True)


if __name__ == "__main__":
    main()
