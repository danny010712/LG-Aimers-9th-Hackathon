"""0-2·1-2 잔차에 남은 구조가 있는가 — 마지막 실적자의 접근 경로 측정 (2026-08-25).

배경: `gate.G5`로 재니 진짜 적자는 **0-2(+51.4) · 1-2(+42.0)** 둘뿐이다(F·3-2는 이득원).
레벨은 아니다(c분리 §3-J 기각). 원인은 해상도이고 "포수 요구가 미관측이라 경로가 없다"고
**추측**해왔다. 여기서 그 추측을 측정으로 바꾼다.

논리: **같은 해 오라클이 상한이다.** 2024 안에서 어떤 컬럼으로 갈라도 잔차 구조가 안 나오면,
해마다 이어지는 구조는 더더욱 없다. 상한이 0이면 축이 닫힌다.

절차: 021 풀스택 예측 → 0-2/1-2 행만 → 컬럼별로 나눠 셀 편향의 분산에서 이항노이즈를 뺀
참신호 → 전역 BSS 기여로 환산(규칙 2-b).
"""
import io
import sys

if getattr(sys.stdout, "encoding", "") != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

import gate

RAW = ["season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
       "balls_before", "strikes_before", "outs_before", "run_total_before", "score_diff_pitcher_team",
       "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
       "li", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
       "pitcher_team_id", "batter_team_id",
       "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
       "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
       "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev5_game_success_rate",
       "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
       "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
       "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
       "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
MIN_CELL = 30


def cells(s, k=10):
    """수치형은 10분위, 범주형은 그대로."""
    if s.dtype.kind in "if" and s.nunique() > 12:
        return pd.qcut(s.rank(method="first"), k, labels=False, duplicates="drop")
    return s.astype(str)


def signal(y, p, key, n_all, u_all):
    """셀 편향의 참신호를 전역 BSS 기여로 환산."""
    d = pd.DataFrame({"y": y, "p": p, "k": np.asarray(key)})
    g = d.groupby("k").agg(n=("y", "size"), yb=("y", "mean"), pb=("p", "mean"))
    g = g[g["n"] >= MIN_CELL]
    if len(g) < 2:
        return None
    n = g["n"].values.astype(float)
    w = n / n.sum()
    b = (g["yb"] - g["pb"]).values
    var = (g["yb"] * (1 - g["yb"])).values
    raw = float((w * (b - (w * b).sum()) ** 2).sum())
    noise = float((w * var / n).sum())
    # 이 세그먼트가 전체에서 차지하는 몫만큼만 전역에 기여한다
    return dict(cells=len(g), raw=raw, noise=noise,
                bss=1e5 * (len(d) / n_all) * max(raw - noise, 0) / u_all)


def main():
    v, y, z = gate.deploy_base(cols=RAW)
    p = gate.sigmoid(z)
    N = len(y)
    U = y.mean() * (1 - y.mean())
    v = v.copy()
    v["cnt"] = v["balls_before"].astype(str) + "-" + v["strikes_before"].astype(str)
    print(f"021 풀스택 2024  BSS={gate.bss(p, y):.1f}  n={N:,}\n", flush=True)

    print("=== ① 카운트별 Murphy 분해 — 0-2가 왜 나쁜가 ===")
    print(f"{'카운트':>6}{'n':>8}{'r':>8}{'pred평균':>10}{'pred sd':>9}"
          f"{'REL':>10}{'RES':>10}{'전역이득':>10}")
    B = np.mean((p - y) ** 2)
    for k in sorted(v["cnt"].unique()):
        i = np.flatnonzero((v["cnt"] == k).values)
        yy, pp = y[i], p[i]
        r = yy.mean()
        q = pd.qcut(pp, 20, labels=False, duplicates="drop")
        g = pd.DataFrame({"q": q, "y": yy, "p": pp}).groupby("q").agg(
            n=("y", "size"), yb=("y", "mean"), pb=("p", "mean"))
        rel = float((g.n * (g.pb - g.yb) ** 2).sum() / len(i))
        res = float((g.n * (g.yb - r) ** 2).sum() / len(i))
        gain = 1e5 * (len(i) / N) * (np.mean((pp - yy) ** 2) - B) / U
        print(f"{k:>6}{len(i):>8,}{r:>8.4f}{pp.mean():>10.4f}{pp.std():>9.4f}"
              f"{rel:>10.6f}{res:>10.6f}{gain:>+10.1f}", flush=True)

    for tag in ("0-2", "1-2"):
        m = (v["cnt"] == tag).values
        yy, pp = y[m], p[m]
        print(f"\n=== ② {tag} 안에서 잔차를 가르는 컬럼이 있는가  (n={m.sum():,}) ===")
        print("   같은 해 오라클이다. 여기서 0이면 해마다 이어지는 구조는 없다.")
        out = []
        for c in RAW:
            if c not in v.columns:
                continue
            r = signal(yy, pp, cells(v.loc[m, c]), N, U)
            if r:
                out.append((r["bss"], c, r["cells"], r["raw"], r["noise"]))
        out.sort(reverse=True)
        print(f"   {'컬럼':<36}{'셀':>5}{'raw':>11}{'노이즈':>11}{'전역BSS':>9}")
        for b, c, k, raw, noi in out[:8]:
            print(f"   {c:<36}{k:>5}{raw:>11.7f}{noi:>11.7f}{b:>9.2f}", flush=True)
        print(f"   ... 나머지 {max(len(out)-8,0)}개는 그 아래. "
              f"**합계 상한 {sum(o[0] for o in out):.1f}** (겹침 무시 = 과대평가)")


if __name__ == "__main__":
    main()
