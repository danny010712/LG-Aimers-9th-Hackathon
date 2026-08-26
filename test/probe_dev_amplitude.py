"""지속성 corr는 통과했는데 모델이 못 쓴다 — 진폭을 안 봤다 (09 §2-O).

corr는 프로파일을 표준화한다. 같은 corr라도 편차의 **절대 크기**가 작으면
트리가 쓸 신호가 없다. ph(+14.1)와 bh(-2.1)의 차이가 여기 있는지 본다.

지표: 편차(로짓)의 sd를 **투구수 가중**으로 잰다 = 실제로 예측을 얼마나 미는가.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd

M = 50
df = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                 usecols=["season", "pitcher_id", "batter_id", "pitcher_hand",
                          "batter_hand", "balls_before", "strikes_before",
                          "inning", "control_success"])
df["count_state"] = df.balls_before.astype(str) + "-" + df.strikes_before.astype(str)
df["xinn"] = np.clip(df.inning, 1, 9)
lg = lambda q: np.log(np.clip(q, 1e-9, 1-1e-9) / (1 - np.clip(q, 1e-9, 1-1e-9)))


def dev(d, keys, prior, m=M):
    g = d.groupby(keys)["control_success"].agg(["sum", "size"])
    b = d.groupby(prior)["control_success"].agg(["sum", "size"])
    pb = (b["sum"] / b["size"]).reindex(g.index.get_level_values(prior)).values
    p = (g["sum"] + m * pb) / (g["size"] + m)
    return pd.DataFrame({"dev": lg(p) - lg(pb), "n": g["size"].values},
                        index=g.index).reset_index()


CASES = [("ph 투수x타자손", ["pitcher_id", "batter_hand"], "pitcher_id"),
         ("bh 타자x투수손", ["batter_id", "pitcher_hand"], "batter_id"),
         ("pc 투수x카운트", ["pitcher_id", "count_state"], "pitcher_id"),
         ("pi 투수x이닝",   ["pitcher_id", "xinn"], "pitcher_id")]

print(f"{'표':<16} {'가중sd(<=2023)':>14} {'가중sd(2024)':>13} "
      f"{'corr':>8} {'**지속 진폭**':>13}   해석")
print("-" * 82)
for name, keys, prior in CASES:
    a = dev(df[df.season <= 2023], keys, prior)
    b = dev(df[df.season == 2024], keys, prior)
    wsd = lambda t: float(np.sqrt(np.average(t.dev**2, weights=t.n)
                                  - np.average(t.dev, weights=t.n)**2))
    m_ = a.merge(b, on=keys, suffixes=("_a", "_b"))
    m_ = m_[(m_.n_a >= 30) & (m_.n_b >= 30)]
    c = np.corrcoef(m_.dev_a, m_.dev_b)[0, 1]
    # 다음 해로 실제 전달되는 신호 진폭 = corr x sd  (회귀계수 관점)
    keep = c * wsd(a)
    print(f"{name:<16} {wsd(a):>14.4f} {wsd(b):>13.4f} {c:>+8.4f} "
          f"{keep:>13.4f}   {'로짓 %.3f 만큼 민다' % keep}")

print("\n[참고] 배포 platoon offset은 b=1.38을 곱한다 → 실효 이동 = b x 편차")
print("       cond는 트리 피처라 계수가 없다 — 편차 자체가 신호 크기다")
