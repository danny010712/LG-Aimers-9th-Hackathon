"""아이디어 1 — 인접 행 차분 + 복원 라벨로 세이버 지표를 만들어 잔차에 걸어본다.

사용자 착상: train은 다른 행을 봐도 되니 인접 행 차분으로 투구 결과(아웃/득점/출루)를
복원하고, 투수별 ERA·WHIP·FIP류 지표를 만들어 trackman처럼 매칭한다.

🔴 이건 **투수 상수** 피처다 — §7-1의 그 클래스이고 trackman 17열이 같은 자리에서
실모델 잔차 **−10.3**이었다. 그러니 판정은 "만들 수 있나"가 아니라
**"021 잔차를 설명하나"** 다. 프로브 base가 아니라 배포 베이스에서 잰다(§0-8).

K·BB는 복원 라벨로 바로 나온다(2스트라이크+strike=삼진, 3볼+ball=볼넷).
HR·실점만 행 차분이 필요하다.
"""
import io
import sys

if getattr(sys.stdout, "encoding", "") != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

import gate

RAW = ["row_id", "season", "pitcher_id", "batter_id", "balls_before",
       "strikes_before", "outs_before", "run_top_before", "run_bot_before",
       "runner_on_1b", "runner_on_2b", "runner_on_3b", "inning",
       "pitcher_team_id", "batter_team_id", "asof_pitcher_n"]


def build_stats(upto):
    """<=upto 시즌으로 투수별 세이버 지표. 배포 시에는 train 전체로 만들어 zip에 싣는다."""
    d = pd.read_csv("data/train.csv", usecols=RAW).sort_values("row_id")
    L = pd.read_csv("recovered_labels.csv.gz",
                    usecols=["row_id", "ball", "strike"])
    d = d.merge(L, on="row_id", how="left")
    # inplay 열은 없다. 문서상 ball+strike in {0,1} 이므로 둘 다 0 = 인플레이.
    d["inplay"] = ((d.ball == 0) & (d.strike == 0)).astype(float)
    d.loc[d.ball.isna(), "inplay"] = np.nan
    d = d[d.season <= upto].reset_index(drop=True)

    g = ["season", "pitcher_team_id", "batter_team_id"]
    d["gkey"] = d[g].astype(str).agg("|".join, axis=1) + "|" + d.inning.astype(str)
    nx = d.shift(-1)
    same = (d.gkey == nx.gkey).values
    d_out = np.where(same, nx.outs_before.values - d.outs_before.values, np.nan)
    run = d.run_top_before.values + d.run_bot_before.values
    d_run = np.where(same, (nx.run_top_before.values
                            + nx.run_bot_before.values) - run, np.nan)
    d_r1 = np.where(same, nx.runner_on_1b.values - d.runner_on_1b.values, np.nan)

    # 이벤트 라벨 (투구 단위)
    d["K"] = ((d.strikes_before == 2) & (d.strike == 1) & (d.inplay != 1)).astype(int)
    d["BB"] = ((d.balls_before == 3) & (d.ball == 1)).astype(int)
    d["OUT"] = np.nan_to_num(d_out > 0).astype(int)
    d["RUN"] = np.nan_to_num(np.clip(d_run, 0, None))
    d["ON"] = np.nan_to_num(d_r1 > 0).astype(int)
    # 홈런 대용: 주자가 정리되며 득점 (인플레이 + 득점 >= 1)
    d["HRish"] = ((d.inplay == 1) & (np.nan_to_num(d_run) >= 1)).astype(int)

    a = d.groupby("pitcher_id").agg(
        n=("K", "size"), K=("K", "sum"), BB=("BB", "sum"), OUT=("OUT", "sum"),
        RUN=("RUN", "sum"), ON=("ON", "sum"), HR=("HRish", "sum"))
    a = a[a.n >= 300]
    ip = a.OUT / 3.0                                   # 이닝 대용
    a["K_rate"] = a.K / a.n
    a["BB_rate"] = a.BB / a.n
    a["KBB"] = a.K_rate - a.BB_rate
    a["ERAish"] = 9 * a.RUN / ip.clip(lower=1)
    a["WHIPish"] = (a.BB + a.ON) / ip.clip(lower=1)
    a["FIPish"] = (13 * a.HR + 3 * a.BB - 2 * a.K) / ip.clip(lower=1)
    a["HR_rate"] = a.HR / a.n
    return a


COLS = ["K_rate", "BB_rate", "KBB", "ERAish", "WHIPish", "FIPish", "HR_rate"]


def main():
    print("표 생성 (<=2023, 배포 조건과 동일한 시점 규칙)...", flush=True)
    st = build_stats(2023)
    print(f"  투수 {len(st):,}명  (300구 이상)\n")
    print(f"{'지표':<10}{'평균':>10}{'sd':>10}{'최소':>10}{'최대':>10}")
    for c in COLS:
        print(f"{c:<10}{st[c].mean():>10.4f}{st[c].std():>10.4f}"
              f"{st[c].min():>10.4f}{st[c].max():>10.4f}")

    v, y, z = gate.deploy_base(cols=["asof_pitcher_n"])
    p = gate.sigmoid(z)
    N = len(y); U = y.mean() * (1 - y.mean())
    res = y - p
    j = v[["pitcher_id"]].merge(st[COLS], left_on="pitcher_id",
                                right_index=True, how="left")
    cov = j[COLS[0]].notna().mean()
    print(f"\n=== 021 잔차와의 관계 (배포 베이스 BSS {gate.bss(p,y):.1f}) ===")
    print(f"  검증행 커버리지 {100*cov:.1f}%")
    print(f"{'지표':<10}{'잔차 corr':>12}{'10분위 여지':>13}{'귀무 최대':>11}")
    rng = np.random.default_rng(0)
    for c in COLS:
        x = j[c].values
        m = ~np.isnan(x)
        r = float(np.corrcoef(x[m], res[m])[0, 1])
        q = pd.qcut(pd.Series(x[m]).rank(method="first"), 10,
                    labels=False, duplicates="drop").values
        def sig(k):
            df = pd.DataFrame({"y": y[m], "p": p[m], "k": k})
            gg = df.groupby("k").agg(n=("y", "size"), yb=("y", "mean"),
                                     pb=("p", "mean"))
            w = gg.n / gg.n.sum(); b = (gg.yb - gg.pb).values
            raw = float((w * (b - (w * b).sum()) ** 2).sum())
            noise = float((w * (gg.yb * (1 - gg.yb)).values / gg.n.values).sum())
            return 1e5 * (m.sum() / N) * max(raw - noise, 0) / U
        real = sig(q)
        null = max(sig(q[rng.permutation(len(q))]) for _ in range(20))
        print(f"{c:<10}{r:>+12.4f}{real:>13.2f}{null:>11.2f}"
              f"{'   ← 귀무 초과' if real > null else ''}", flush=True)


if __name__ == "__main__":
    main()
