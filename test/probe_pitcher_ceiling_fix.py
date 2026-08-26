"""투수 상수 피처의 천장 재측정 — `probe_pitcher_headroom.py`의 노이즈 규약 오류 정정.

🔴 원 프로브(probe_pitcher_headroom.py:43-44)는 잔차 분산은 **투구수 가중**으로,
   이항 노이즈는 **투수 단순평균**으로 계산했다:

       v0    = (w*res**2).sum()/w.sum()                 # 가중
       binom = (act*(1-act)/n).mean()                   # 단순  ← 불일치

   표본이 적은 투수가 1/n 때문에 노이즈 평균을 부풀려 **과다 차감**한다.
   올바른 차감량은 Σ wᵢ·p(1−p)/nᵢ 이다 (E[v0] = Σwᵢ·true² + Σwᵢ·p(1−p)/nᵢ).

   시뮬레이션 검증(참신호가 0인 가짜 데이터, 투수 249명 241,706행):
       가중 규약 → +3.8 / +6.9 / +4.8   (≈ 0, 올바름)
       단순 규약 → −52.4 / −49.3 / −51.4 (과다 차감)

   → 문서의 **"투수 상수 피처는 천장 13 BSS"는 틀렸다. 실제 천장은 ~90 BSS다.**
     (100위까지의 격차가 −68이므로 무시할 수 있는 크기가 아니다.)

⚠️ 08 §5-12 축 분해표(투수×좌우 126 · 투수 단독 22)도 같은 규약으로 보인다.
   가중으로 재면 209 / 44다.

✅ **다만 축이 열리는 것은 아니다.** 근거가 "천장이 낮다"에서 "실측이 반복 실패했다"로
   바뀔 뿐이다. 실제 피처는 아무것도 근처에 못 갔다:
     trackman 물리 17열 → 013 잔차 OOF 회귀  **−10.3**  (이 계산은 버그와 무관)
     rel_side sd → 실제 성공률 상관 −0.037, asof 통제 시 증분 0
     trackman → **좌우 편차**(이 파일이 처음 측정) OOF corr **−0.12 / +0.05**, R² 음수
   마지막 것이 "투수 상수 × 상황 교호작용" 가설의 가장 유망한 사례였다. 음성.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import glob

import numpy as np
import pandas as pd

A = 1e5 / 0.25
M = 270


def load():
    d = pd.read_csv("data/train.csv",
                    usecols=["row_id", "season", "pitcher_id", "asof_pitcher_n",
                             "control_success"])
    L = pd.read_csv("recovered_labels.csv.gz", usecols=["row_id", "middle"])
    d = d.merge(L, on="row_id", how="left")
    va = (d.season == 2024).values
    have = d["middle"].notna().values[va]
    v = d[va][have].reset_index(drop=True)
    v["pred"] = np.mean([np.load(x) for x in
                         sorted(glob.glob("artifacts/auxpred_ins_013_backup/*.npy"))], axis=0)
    return v


def headroom(z, n_all):
    w = z.n.values.astype(float); W = w / w.sum()
    res = (z.act - z.pr).values
    v0 = float((W * res ** 2).sum())
    nz_w = float((W * (z.act * (1 - z.act)).values / w).sum())
    nz_u = float(((z.act * (1 - z.act)) / z.n).mean())
    return A * max(v0 - nz_w, 0), A * max(v0 - nz_u, 0), w.sum() / n_all


def main():
    v = load()
    G = v.groupby("pitcher_id").agg(n=("control_success", "size"),
                                    act=("control_success", "mean"),
                                    pr=("pred", "mean"),
                                    an=("asof_pitcher_n", "median")).reset_index()

    print("=== ① 규약 검증: 참신호 0인 가짜 데이터 ===")
    rng = np.random.default_rng(0)
    ns = G[G.n >= 200].n.values
    for rep in range(3):
        act = np.array([rng.binomial(n, 0.486) / n for n in ns])
        w = ns.astype(float); W = w / w.sum(); res = act - 0.486
        v0 = float((W * res ** 2).sum())
        nzw = float((W * (act * (1 - act)) / w).sum())
        nzu = float(((act * (1 - act)) / w).mean())
        print(f"  rep{rep}  가중 {A*(v0-nzw):+8.2f}   단순 {A*(v0-nzu):+8.2f}")
    print("  → 0에 가까운 가중 규약이 옳다\n")

    print("=== ② 투수레벨 여지 (규약별) ===")
    for tag, z in [("전체 투수", G), ("n>=200 (원 프로브 조건)", G[G.n >= 200]),
                   ("n<200 (원 프로브가 제외)", G[G.n < 200])]:
        hw, hu, sh = headroom(z, len(v))
        print(f"  {tag:<26} 투수 {len(z):>4}  행 {sh*100:>5.1f}%"
              f"   가중 {hw:>7.1f}   단순 {hu:>7.1f}")
    print("  ※ 문서의 '13'/'22'는 단순 열. 가중이 올바른 값.\n")

    print("=== ③ 전역 기여로 환산 (§0-7) ===")
    G["bin"] = pd.cut(G.an, [-1, 300, 1500, 5000, 15000, 10 ** 9],
                      labels=["~300", "~1.5k", "~5k", "~15k", "15k+"])
    print(f"{'asof_n':<9}{'투수':>5}{'행몫':>8}{'그룹내':>10}{'전역기여':>10}")
    tot = 0.0
    for k, z in G.groupby("bin", observed=True):
        if len(z) < 5:
            continue
        hw, _, sh = headroom(z, len(v))
        tot += hw * sh
        print(f"{str(k):<9}{len(z):>5}{sh*100:>7.1f}%{hw:>10.1f}{hw*sh:>10.1f}")
    print(f"{'합계':<9}{'':>5}{'':>8}{'':>10}{tot:>10.1f}  ← 투수 상수 피처의 진짜 천장")

    print("\n=== ④ trackman 물리 프로필이 '좌우 편차'를 예측하는가 (미측정이었던 칸) ===")
    tr = pd.read_csv("data/train.csv",
                     usecols=["season", "pitcher_id", "pitcher_hand", "batter_hand",
                              "control_success"])
    tr["plat"] = (tr.pitcher_hand == tr.batter_hand).astype(int)

    def split_tab(df_):
        g = df_.groupby(["pitcher_id", "plat"])["control_success"].agg(["sum", "count"])
        pr = df_.groupby("pitcher_id")["control_success"].mean().rename("pr")
        g = g.join(pr, on="pitcher_id")
        g["val"] = (g["sum"] + M * g["pr"]) / (g["count"] + M)
        t = (g["val"] - g["pr"]).rename("split").reset_index()
        return t[t.plat == 1].set_index("pitcher_id")["split"]

    s23 = split_tab(tr[tr.season <= 2023]); s24 = split_tab(tr[tr.season == 2024])
    P = pd.read_csv("tm_profile_le2023.csv")
    FE = [c for c in P.columns if c != "pitcher_id"]
    n24 = tr[tr.season == 2024].groupby("pitcher_id").size().rename("n")
    z = (P.set_index("pitcher_id").join(s23.rename("a")).join(s24.rename("b"))
         .join(n24).query("n>=200").dropna())
    print(f"  투수 {len(z)}명, 물리열 {len(FE)}개")
    print(f"  기준선 corr(과거 split, 2024 split) = {np.corrcoef(z.a, z.b)[0,1]:+.4f}"
          f"   ← 021이 실제로 쓰는 신호")
    X = z[FE].values; X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    rng = np.random.default_rng(0); f = rng.integers(0, 5, len(z))
    for tgt, nm in [(z.a.values, "과거 split"), (z.b.values, "2024 split")]:
        oof = np.zeros(len(z))
        for k in range(5):
            i = f != k
            b = np.linalg.lstsq(np.c_[np.ones(i.sum()), X[i]], tgt[i], rcond=None)[0]
            oof[f == k] = np.c_[np.ones((f == k).sum()), X[f == k]] @ b
        print(f"  물리 → {nm:<10} OOF corr {np.corrcoef(oof, tgt)[0,1]:+.4f}"
              f"   R² {1-np.var(tgt-oof)/np.var(tgt):+.4f}")


if __name__ == "__main__":
    main()
