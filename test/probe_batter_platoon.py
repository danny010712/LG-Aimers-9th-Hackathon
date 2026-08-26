"""타자 x 투수손 (타자 플래툰) 지속성 — ph의 거울상. 지금까지 안 쟀다.

지금까지 잰 16개 축이 전부 `투수 x 상황`이다. 표의 주체를 타자로 바꾼 적이 없다.
야구 도메인상 타자 플래툰 스플릿은 투수 것보다 크고 안정적이다(라인업을 그걸로 짠다).
`asof_batter_*`는 n/success_rate/middle_rate 3열뿐이라 이 정보가 어디에도 없다.

기준: ph +0.3854 = 채택(LB +5.55) / <=0.106 = 기존 기각선.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd

M = 50
df = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                 usecols=["season", "pitcher_id", "batter_id", "pitcher_hand",
                          "batter_hand", "balls_before", "strikes_before",
                          "control_success"])
df["count_state"] = (df.balls_before.astype(str) + "-"
                     + df.strikes_before.astype(str))
lg = lambda q: np.log(np.clip(q, 1e-9, 1-1e-9) / (1 - np.clip(q, 1e-9, 1-1e-9)))


def dev(d, keys, prior, m=M):
    g = d.groupby(keys)["control_success"].agg(["sum", "size"])
    b = d.groupby(prior)["control_success"].agg(["sum", "size"])
    pb = (b["sum"] / b["size"]).reindex(g.index.get_level_values(prior)).values
    p = (g["sum"] + m * pb) / (g["size"] + m)
    return pd.DataFrame({"dev": lg(p) - lg(pb), "n": g["size"].values},
                        index=g.index).reset_index()


CASES = [
    ("bh  타자x투수손",   ["batter_id", "pitcher_hand"],  "batter_id"),
    ("ph  투수x타자손",   ["pitcher_id", "batter_hand"],  "pitcher_id"),   # 대조
    ("bhc 타자x투수손x카운트", ["batter_id", "pitcher_hand", "count_state"], "batter_id"),
    ("phc 투수x타자손x카운트", ["pitcher_id", "batter_hand", "count_state"], "pitcher_id"),
]
print(f"{'표':<26} {'MINN':>5} {'corr':>9} {'진폭비':>8} {'겹칩':>8}")
for name, keys, prior in CASES:
    for MINN in (30, 100):
        a, b = dev(df[df.season <= 2023], keys, prior), dev(df[df.season == 2024], keys, prior)
        mm = a.merge(b, on=keys, suffixes=("_a", "_b"))
        mm = mm[(mm.n_a >= MINN) & (mm.n_b >= MINN)]
        if len(mm) < 30:
            print(f"{name:<26} {MINN:>5}   겹치는 칸 {len(mm)} — 판정 불가"); continue
        c = np.corrcoef(mm.dev_a, mm.dev_b)[0, 1]
        print(f"{name:<26} {MINN:>5} {c:>+9.4f} "
              f"{mm.dev_b.std()/mm.dev_a.std():>8.3f} {len(mm):>8,}")

# 스무딩 강도 민감도 (cond M=50 vs platoon M=270)
print("\n[스무딩 M 민감도  MINN=30]")
for name, keys, prior in CASES[:2]:
    row = []
    for m_ in (20, 50, 120, 270):
        a, b = dev(df[df.season <= 2023], keys, prior, m_), dev(df[df.season == 2024], keys, prior, m_)
        mm = a.merge(b, on=keys, suffixes=("_a", "_b"))
        mm = mm[(mm.n_a >= 30) & (mm.n_b >= 30)]
        row.append(f"M={m_}: {np.corrcoef(mm.dev_a, mm.dev_b)[0,1]:+.4f}")
    print(f"  {name:<26} " + "  ".join(row))

# 전역 크기: 타자 플래툰이 실제로 큰가
print("\n[전역 스플릿 크기 (2024)]")
d24 = df[df.season == 2024]
print(d24.groupby(["batter_hand", "pitcher_hand"])["control_success"].agg(["mean", "size"]).to_string())
