"""RES 적자가 소수에 몰려 있는가 — additive 접근이 틀렸는지 판별 (2026-08-25).

사용자 논지: "소수 상황(platoon 극단, 특정 역할 투수)에 몰려 있다면 additive가
틀린 게 확정되고 분리 모델/가중치 방향이 열린다."

측정: 전역 Brier 기여로만 본다(세그BSS 금지, §0-7). 각 그룹에 대해
  전역이득 = 1e5 * (n_g/N) * (Brier_g − Brier_전체) / U_전역
양수 = 그 그룹이 평균보다 나쁘다(적자). 집중돼 있으면 소수 그룹이 큰 양수를 가진다.
🔴 귀무 대조를 함께 돌린다 — 무작위 분할도 산포는 생긴다.
"""
import io, sys
if getattr(sys.stdout, "encoding", "") != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd, gate

COLS = ["balls_before", "strikes_before", "inning", "num_runners_on", "li",
        "asof_pitcher_n", "asof_batter_n", "asof_pitcher_success_rate",
        "asof_pitcher_breaking_rate", "batter_id"]


def contrib(y, p, key, B, U, N):
    d = pd.DataFrame({"y": y, "p": p, "k": np.asarray(key)})
    g = d.groupby("k").agg(n=("y", "size"))
    g["br"] = d.groupby("k").apply(lambda t: np.mean((t.p - t.y) ** 2),
                                   include_groups=False)
    g = g[g["n"] >= 200]
    g["gain"] = 1e5 * (g["n"] / N) * (g["br"] - B) / U
    return g.sort_values("gain", ascending=False)


def main():
    v, y, z = gate.deploy_base(cols=COLS)
    p = gate.sigmoid(z)
    N = len(y); U = y.mean() * (1 - y.mean()); B = np.mean((p - y) ** 2)
    print(f"021 풀스택 2024  BSS={gate.bss(p,y):.1f}  n={N:,}\n", flush=True)

    # 투수 역할 대용: 그 시즌 투구수
    pc = v.groupby("pitcher_id")["pitcher_id"].transform("size")
    v = v.assign(role=pd.cut(pc, [0, 300, 1000, 99999],
                             labels=["불펜(<300)", "중간(300~1k)", "선발(1k+)"]))
    v["plat"] = (v.pitcher_hand == v.batter_hand).astype(int)

    print("=== ① 축별 적자 집중도 (전역기여, 양수 = 적자) ===")
    print(f"{'축':<22}{'그룹':>5}{'최대적자':>10}{'상위3합':>10}"
          f"{'표준편차':>10}{'귀무 sd':>10}")
    rng = np.random.default_rng(0)
    axes = [("투수 개인", v.pitcher_id.values),
            ("타자 개인", v.batter_id.values),
            ("투수 역할", v.role.astype(str).values),
            ("좌우(plat)", v.plat.values),
            ("카운트", (v.balls_before.astype(str) + "-" +
                       v.strikes_before.astype(str)).values),
            ("이닝", np.clip(v.inning.values, 1, 10)),
            ("투수경력10분위", pd.qcut(v.asof_pitcher_n.rank(method="first"),
                                  10, labels=False).values),
            ("예측10분위", pd.qcut(pd.Series(p).rank(method="first"),
                               10, labels=False).values)]
    for tag, k in axes:
        g = contrib(y, p, k, B, U, N)
        if len(g) < 2:
            continue
        # 귀무: 같은 크기 분포를 유지한 무작위 배정
        kk = np.asarray(k).copy(); rng.shuffle(kk)
        g0 = contrib(y, p, kk, B, U, N)
        print(f"{tag:<22}{len(g):>5}{g.gain.iloc[0]:>+10.1f}"
              f"{g.gain.iloc[:3].sum():>+10.1f}{g.gain.std():>10.2f}"
              f"{g0.gain.std():>10.2f}", flush=True)

    print("\n=== ② 투수 개인 적자의 집중도 — 상위 몇 명이 전체 적자의 몇 %인가 ===")
    g = contrib(y, p, v.pitcher_id.values, B, U, N)
    pos = g[g.gain > 0]
    tot = pos.gain.sum()
    for k in (5, 10, 25, 50):
        print(f"  상위 {k:>3}명  누적 {pos.gain.iloc[:k].sum():>7.1f}"
              f"  ({100*pos.gain.iloc[:k].sum()/tot:>5.1f}%)   "
              f"행 비중 {100*pos.n.iloc[:k].sum()/N:>5.2f}%")
    print(f"  적자 투수 {len(pos)}명 / 전체 {len(g)}명   적자 총합 {tot:.1f}")

    print("\n=== ③ 그 투수들이 공통점을 갖는가 ===")
    top = pos.index[:25]
    ist = v.pitcher_id.isin(top).values
    for c in ["asof_pitcher_n", "asof_pitcher_success_rate",
              "asof_pitcher_breaking_rate", "li"]:
        print(f"  {c:<32} 상위25 중앙 {np.median(v[c][ist]):>9.4f}"
              f"   나머지 {np.median(v[c][~ist]):>9.4f}")
    print(f"  {'plat 비율':<32} 상위25 {v.plat[ist].mean():>9.4f}"
          f"   나머지 {v.plat[~ist].mean():>9.4f}")
    print(f"  {'F 비율':<32} 상위25 {(v.game_type[ist]=='F').mean():>9.4f}"
          f"   나머지 {(v.game_type[~ist]=='F').mean():>9.4f}")


if __name__ == "__main__":
    main()
