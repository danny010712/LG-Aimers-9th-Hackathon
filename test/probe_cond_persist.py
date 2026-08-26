"""cond 4표의 연도 간 지속성 — 1시간짜리 체인 태우기 전 사전 선별.

044(cond_ph)가 LB +5.55로 살아났다. ph가 산 이유가 좌우 지속성(0.412)이라면
나머지 표도 같은 잣대로 미리 갈린다.

기준선: gate.G1 관측점 — 투수x좌우 0.412 채택 / 나머지 <=0.106 미채택.
⚠️ 09 §2-O: corr는 프로파일을 표준화해 **진폭을 버린다**. 진폭비도 같이 본다.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd

sys.path.insert(0, "common")
import cond  # noqa: E402

df = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                 usecols=["season", "pitcher_id", "batter_id", "batter_hand",
                          "balls_before", "strikes_before", "inning",
                          "control_success"])
df = cond.add_keys(df)
print(f"rows={len(df):,}  M={cond.M}", flush=True)

MINN = 30


def dev_table(d, keys, prior):
    """칸 성공률 − 그 선수 전체 성공률 (로짓). cond.build_tables와 같은 형태."""
    g = d.groupby(keys)["control_success"].agg(["sum", "size"])
    b = d.groupby(prior)["control_success"].agg(["sum", "size"])
    pb = (b["sum"] / b["size"]).reindex(g.index.get_level_values(prior)).values
    p = (g["sum"] + cond.M * pb) / (g["size"] + cond.M)
    lg = lambda q: np.log(q / (1 - q))
    return pd.DataFrame({"dev": lg(p) - lg(pb), "n": g["size"].values},
                        index=g.index).reset_index()


print(f"\n{'표':<6} {'키':<28} {'corr':>8} {'진폭비':>8} {'겹칩':>8}  판정")
print(f"{'-'*6} {'-'*28} {'-'*8} {'-'*8} {'-'*8}")
for name, keys, prior in cond.SPECS:
    a = dev_table(df[df.season <= 2023], keys, prior)
    b = dev_table(df[df.season == 2024], keys, prior)
    m = a.merge(b, on=keys, suffixes=("_a", "_b"))
    m = m[(m.n_a >= MINN) & (m.n_b >= MINN)]
    c = np.corrcoef(m.dev_a, m.dev_b)[0, 1]
    amp = m.dev_b.std() / m.dev_a.std()
    verdict = ("✅ ph급" if c >= 0.30 else
               "⚠️ 애매" if c >= 0.15 else "❌ 기존 기각선(<=0.106) 자리")
    print(f"{name:<6} {'x'.join(keys):<28} {c:>+8.4f} {amp:>8.3f} "
          f"{len(m):>8,}  {verdict}")

# 대조군: 배포에서 검증된 좌우 offset(platoon) 표
sp = df.copy()
sp["plat"] = sp.pitcher_id.astype(str) + "_" + sp.batter_hand.astype(str)
a = dev_table(sp[sp.season <= 2023], ["pitcher_id", "batter_hand"], "pitcher_id")
b = dev_table(sp[sp.season == 2024], ["pitcher_id", "batter_hand"], "pitcher_id")
m = a.merge(b, on=["pitcher_id", "batter_hand"], suffixes=("_a", "_b"))
m = m[(m.n_a >= MINN) & (m.n_b >= MINN)]
print(f"\n[대조] 투수x좌우 (=ph와 동일 키)  corr {np.corrcoef(m.dev_a, m.dev_b)[0,1]:+.4f}  "
      f"n={len(m):,}   ← gate 기준 0.412")
