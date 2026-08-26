"""후보 **계측기** — 아이디어를 볼 때 빠뜨리지 말아야 할 측정들.

🔴🔴 **이 파일은 기각 판정을 내리지 않는다. 숫자와 기존 관측점만 준다.**

    위양성(틀린 걸 채택)의 비용 = 제출 1회 = 남은 예산의 1/40.
    위음성(맞는 걸 버림)의 비용 = 그 축을 영영 잃음. 필요한 게 +85인데
    가장 큰 두 축(shift +47.04 · 외삽 제어)이 판별력 축이 아니었다.
    **비대칭이 극단적이라 자동 기각은 손해다.** → 09 §0-11.

실측된 위음성 (2026-08-25, 이 파일 자신에 대한 검사):
    구 G2("최적이동 후 잔존 ≥50%")에 **shift 축(LB +47.04)을 걸면 잔존 0% = FAIL**.
    전제가 틀렸다 — 배포 연도(2025)는 미관측이라 오라클 이동이 존재하지 않는다.
    그리고 G2의 쓸모 있는 몫은 G3에 이미 있다(c분리가 죽은 진짜 이유는
    "레벨이라서"가 아니라 "021 스택에 shift가 이미 있어 자리가 없어서"다).
    → G2는 게이트에서 빼고 `level_share()` 진단으로만 남긴다.

계측기 (전부 값만 낸다):
    G1 persistence   <=T-1 표 vs T 표 corr        기존 관측점 옆에 놓아준다
    G3 deploy_delta  021 풀스택에서 Δ             포화를 보려면 여기서 재야 한다
    G4 out_of_year   출처/목표별 Δ와 부호         적합 요소가 있으면 필수
    G5 contribution  raw Brier 전역기여           세그BSS는 손실 지도가 아니다
    null_control     같은 구조를 섞어서 재기       스캔했으면 필수
    level_share      이득 중 레벨 몫 (진단)       기각 근거로 쓰지 말 것

🔴 판정 불가 영역 — 계측기가 답할 수 없는 것:
    depth · 정규화 · 시드 수 등 **미관측 시즌 외삽 제어**는 관측연도(2024)로
    원리적 판정이 불가능하다. G3가 음수여도 그것은 FAIL이 아니라 **무정보**다.
    (034 depth5 = 로컬 −2.8. depth 8은 로컬 +7.6인데 LB −34.2였다.)
    이 부류는 **제출로만 확인된다.**

사용:
    import gate
    v, y, z = gate.deploy_base()
    gate.G3(z_cand, y, z, "후보")
    gate.G1(gate.hist(), "plat")
    gate.G5(z, y, v.game_type)
    python gate.py            # 기존 관측점 재현
"""
import io
import sys

if getattr(sys.stdout, "encoding", "") != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import glob
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

TARGET = "control_success"
BASE_RUN = "021_platoon"     # 배포 베이스. 013에서 재면 포화를 못 본다
# 🔴 021의 성공모델은 **013 구성**(ins 4열)이다. `artifacts/auxpred_ins`는 018(ins 7열)
#    캐시로 덮여 있어서 그걸 읽으면 주모델이 859.0이 아니라 866.2가 나온다.
#    2026-08-25에 실제로 이 버그로 하루치 측정을 잘못된 베이스에서 했다.
SUCCESS_CACHE = "artifacts/auxpred_ins_013_backup"
BASE_CHECK = 912.41          # 021 result.json의 platoon 직전 값. 아래에서 대조한다
M = 270                      # build_platoon.py와 동일한 수축 강도
MIN_CELL = 50

# 🔴 통과선이 아니라 **기존 관측점**이다. 새 후보를 이 옆에 놓고 사람이 판단한다.
#    좌우 0.412 하나만 채택됐다는 사실은 "0.4가 경계"라는 뜻이 아니다 — 관측 1개다.
REF_G1 = [(0.412, "투수×좌우", "LB +5.27 채택"), (0.106, "투수×주자", "미채택"),
          (0.056, "투수×game_type", "미채택"), (0.049, "투수×월", "미채택")]
# 베이스별 out-of-year → LB 전이 배율 (CLAUDE.md §5). 크기 예측용이 아니라 감각용.
REF_MULT = [(881.73, 2.40), (945.40, 1.47), (998.00, 0.79),
            (1051.73, 0.26), (1057.00, -0.5)]


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bss(p, y):
    r = y.mean()
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def _avg(pat):
    f = sorted(glob.glob(pat))
    assert f, f"예측 캐시 없음: {pat}"
    return np.mean([np.load(x) for x in f], axis=0)


def _best_shift(z, y):
    f = lambda s: np.mean((sigmoid(z + s) - y) ** 2)
    return float(minimize_scalar(f, bounds=(-.5, .5), method="bounded",
                                 options={"xatol": 1e-9}).x)


def hist(cols=()):
    """전 시즌 train (G1용). 필요한 열만 추가로 받는다."""
    need = ["row_id", "season", "pitcher_id", "pitcher_hand", "batter_hand",
            "game_type", TARGET]
    df = pd.read_csv("data/train.csv", usecols=list(dict.fromkeys(need + list(cols))))
    df["plat"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


def _deployed_split(v):
    """배포된 platoon.csv를 검증행에 붙인다."""
    tab = pd.read_csv(f"runs/{BASE_RUN}/model/platoon.csv", encoding="utf-8")
    return pd.DataFrame({"pitcher_id": v["pitcher_id"].values,
                         "plat": v["plat"].values}).merge(
        tab, on=["pitcher_id", "plat"], how="left")["split"].fillna(0.0).values


def deploy_base(cols=()):
    """배포 베이스(021 풀스택)의 2024 검증 로짓. → (v, y, z)

    🔴 platoon 항은 **<=2023 표**로 만든다. `runs/*/platoon.csv`는 2024를 포함한
    배포용이라 그대로 2024에서 채점하면 자기적합으로 +77이 붙는다(1010 vs 932.9).
    배포는 미관측 연도에 적용하므로 <=2023 표가 그 상황의 유비다.
    """
    df = hist(cols)
    L = pd.read_csv("recovered_labels.csv.gz", usecols=["row_id", "middle"])
    va = (df["season"] == 2024).values
    have = df[["row_id"]].merge(L, on="row_id", how="left")["middle"].notna().values[va]
    v = df[va][have].reset_index(drop=True)
    m = json.load(open(f"runs/{BASE_RUN}/model/meta.json", encoding="utf-8"))
    off, pl = m["offset"], m["platoon"]
    spl = pd.DataFrame({"pitcher_id": v["pitcher_id"].values,
                        "plat": v["plat"].values}).merge(
        _split_table(df[df.season <= 2023], "plat"),
        on=["pitcher_id", "plat"], how="left")["split"].fillna(0.0).values
    z = (logit(_avg(SUCCESS_CACHE + "/success_2024_*.npy"))
         + off["b"] * (logit(_avg("artifacts/auxpred/mr_2024_*.npy")) - off["mu_mr"])
         + off["c"] * (logit(_avg("artifacts/auxpred/wayoff_2024_*.npy"))
                       - off["mu_wayoff"])
         + m["logit_shift"])
    yv = v[TARGET].values.astype(float)
    got = bss(sigmoid(z), yv)
    assert abs(got - BASE_CHECK) < 1.0, (
        f"베이스 재구성 불일치: {got:.1f} vs 기록 {BASE_CHECK} "
        f"— SUCCESS_CACHE({SUCCESS_CACHE})가 이 run의 것이 맞는지 확인할 것")
    return v, yv, z + pl["b"] * spl


# --------------------------- G1 지속성 ---------------------------
def _split_table(h, col, m=M):
    g = h.groupby(["pitcher_id", col])[TARGET].agg(["sum", "count"])
    pr = h.groupby("pitcher_id")[TARGET].mean().rename("pr")
    g = g.join(pr, on="pitcher_id")
    g["v"] = (g["sum"] + m * g["pr"]) / (g["count"] + m)
    t = (g["v"] - g["pr"]).rename("split").reset_index()
    w = h.merge(t, on=["pitcher_id", col], how="left")["split"]
    t["split"] = t["split"] - float(w.fillna(0).mean())
    return t


def G1(df, col, lo=2019, hi=2023, tgt=2024, verbose=True):
    """지속성 — 과거 편차 표가 목표 연도에 이어지는가. offset 표 후보의 1차 관문.

    여지(headroom)로 판정하지 말 것. 죽은 것이 확정된 투수×카운트가 75.5로 나온다.
    """
    a = _split_table(df[(df.season >= lo) & (df.season <= hi)], col).rename(
        columns={"split": "a"})
    cur = df[df.season == tgt]
    b = _split_table(cur, col).rename(columns={"split": "b"})
    n = cur.groupby(["pitcher_id", col])[TARGET].size().rename("n").reset_index()
    z = a.merge(b, on=["pitcher_id", col]).merge(n, on=["pitcher_id", col])
    z = z[z["n"] >= MIN_CELL]
    if len(z) < 30:
        if verbose:
            print(f"  G1 {col:<14} 셀 부족({len(z)}) — 판정 불가", flush=True)
        return None
    c = float(np.corrcoef(z["a"], z["b"])[0, 1])
    tau2 = c * z["a"].std() * z["b"].std()
    r = dict(cells=len(z), corr=c, tau=np.sqrt(max(tau2, 0)),
             bss=1e5 / 0.25 * max(tau2, 0))
    if verbose:
        near = min(REF_G1, key=lambda t: abs(t[0] - c))
        print(f"  G1 {col:<14} 셀 {r['cells']:>5,}  corr {c:>6.3f}"
              f"  전이가능 {r['bss']:>6.1f} BSS"
              f"   | 가장 가까운 관측점 {near[1]} {near[0]:.3f} ({near[2]})", flush=True)
    return r


# ------------------- level_share (진단 — 옛 G2) -------------------
def level_share(z_a, z_b, y, tag="후보", verbose=True):
    """이득 중 레벨(평균 이동)이 차지하는 몫. **진단이지 판정이 아니다.**

    🔴 기각 근거로 쓰지 말 것. 이 검사가 옛 G2였고, **shift 축(LB +47.04)에
    걸면 잔존 0%가 나온다** — 배포 연도는 미관측이라 오라클 이동이 없기 때문이다.
    레벨 몫이 크다는 것은 "**이미 스택에 있는 것과 겹치는지 G3로 확인하라**"는 신호일 뿐이다.
    """
    ba, bb = bss(sigmoid(z_a), y), bss(sigmoid(z_b), y)
    ba2 = bss(sigmoid(z_a + _best_shift(z_a, y)), y)
    bb2 = bss(sigmoid(z_b + _best_shift(z_b, y)), y)
    d0, d1 = bb - ba, bb2 - ba2
    keep = (d1 / d0) if abs(d0) > 1e-9 else 0.0
    if verbose:
        print(f"  진단 {tag:<14} Δ {d0:>+7.2f} → 레벨제거 후 {d1:>+7.2f}"
              f"  판별력 몫 {keep*100:>5.0f}%"
              f" (예측평균 Δ{sigmoid(z_b).mean()-sigmoid(z_a).mean():+.5f})"
              f"  ※판정 아님", flush=True)
    return dict(a=ba, b=bb, delta=d0, delta_level_free=d1, keep=keep)


# --------------------- G3 배포 베이스 재측정 ---------------------
def G3(z_cand, y, z_base, tag="후보", extrapolation=False, verbose=True):
    """배포 베이스(021 풀스택)에서 잰 Δ.

    013·003 같은 약한 베이스에서 잰 이득은 그대로 오지 않는다 — 관측 배율
    2.40 → 1.47 → 0.79 → 0.26 → ~0으로 베이스 강도에 단조 감소한다.

    🔴 `extrapolation=True` 는 **미관측 시즌 외삽을 제어하는 후보**(depth, 정규화,
    시드 수)에 반드시 준다. 그 부류는 관측연도(2024)로 원리적 판정이 불가능하다 —
    depth 8은 로컬 **+7.6**인데 LB **−34.2**였다. 음수를 기각 근거로 쓰면
    034(depth 5, 로컬 −2.8)를 여기서 잃는다.
    """
    a, b = bss(sigmoid(z_base), y), bss(sigmoid(z_cand), y)
    if verbose:
        if extrapolation:
            note = "  ← 🔴 외삽 제어 = 관측연도로 판정 불가. 제출로만 확인된다"
        else:
            m = min(REF_MULT, key=lambda t: abs(t[0] - a))
            note = f"  | 베이스 {m[0]:.0f} 근처의 관측 배율 {m[1]:+.2f}배"
        print(f"  G3 {tag:<14} 배포베이스 {a:>7.1f} → {b:>7.1f}   Δ {b-a:>+7.2f}{note}",
              flush=True)
    return b - a


# -------------------------- G4 out-of-year --------------------------
CACHE_MULTIYEAR = "probe_offset_forms_preds.csv.gz"
VALID_SRC, VALID_TGT = (2022, 2023, 2024), (2023, 2024)


def G4(base_fit, base_apply, cand_fit, cand_apply, tag="후보", verbose=True,
       extra_cols=()):
    """연도 S에서 적합해 연도 T에 적용 — **기준선과 후보를 함께** 재서 Δ와 부호를 낸다.

    fit(dfS) -> params ;  apply(dfT, params) -> 확률
    출처 2021 제외(F 아티팩트 시대), 목표 2023 제외(죽은 fold). CLAUDE.md §4-1·4-5.

    ⚠️ 캐시는 003 계보다. **부호만 참고하고 크기는 믿지 말 것** — 배포 베이스가
    훨씬 강하므로 배율이 다르다(G3의 REF_MULT 참조).
    ⚠️ 유효 전이가 2~3개뿐이라 부호 하나가 뒤집혀도 그것만으로 기각하지 말 것.
    """
    d = pd.read_csv(CACHE_MULTIYEAR)
    if extra_cols:
        cx = pd.read_csv("data/train.csv", usecols=["row_id"] + list(extra_cols))
        d = d.merge(cx, on="row_id")
    out = []
    for S in VALID_SRC:
        for T in VALID_TGT:
            if S == T or T == 2023:
                continue
            gs, gt = d[d.season == S], d[d.season == T]
            if len(gs) == 0 or len(gt) == 0:
                continue
            yT = gt.y.values.astype(float)
            a = bss(base_apply(gt, base_fit(gs)), yT)
            b = bss(cand_apply(gt, cand_fit(gs)), yT)
            out.append((S, T, a, b, b - a))
    if verbose:
        for S, T, a, b, dd in out:
            near = "  <- 배포 최근접(2024 출처 -> 2025 목표의 유비)" if (S, T) == (2023, 2024) else ""
            print(f"  G4 {tag:<14} {S}->{T}  기준 {a:>8.1f}  후보 {b:>8.1f}"
                  f"  Δ {dd:>+7.2f}{near}", flush=True)
        sg_ = [np.sign(o[4]) for o in out]
        print(f"  G4 {tag:<14} 부호 {sum(s > 0 for s in sg_)}/{len(sg_)} 양수"
              f"   (전이가 {len(sg_)}개뿐 — 하나 뒤집힌 것만으로 기각하지 말 것)",
              flush=True)
    return out


def null_control(y, p, key, n_all=None, u_all=None, reps=5, seed=0,
                 tag="후보", verbose=True):
    """귀무 대조 — 같은 셀 구조를 유지하고 **배정만 섞어서** 같은 값을 재본다.

    🔥 오늘(2026-08-25) 0-2 잔차 축을 죽인 검사다. 실제 최고 3.04인데 **섞은 것이
    5.74**였다. 안 돌렸으면 `game_month` 3.04를 발견으로 착각했다.

    ⚠️ 두 가지 모드가 있고 섞으면 안 된다:
      - 컬럼 여럿을 **스캔**했으면 `max(null)`과 비교한다 (family-wise).
      - **사전에 하나를 지목**했으면 그 컬럼 자신의 null 분포와 비교한다.
        스캔 기준을 사전등록 가설에 적용하면 진짜 효과를 죽인다(위음성).
    """
    y = np.asarray(y); p = np.asarray(p); key = np.asarray(key)
    if n_all is None:
        n_all = len(y)
    if u_all is None:
        u_all = y.mean() * (1 - y.mean())

    def _sig(k):
        df = pd.DataFrame({"y": y, "p": p, "k": k})
        g = df.groupby("k").agg(n=("y", "size"), yb=("y", "mean"), pb=("p", "mean"))
        g = g[g["n"] >= 30]
        if len(g) < 2:
            return None
        n = g["n"].values.astype(float); w = n / n.sum()
        b = (g["yb"] - g["pb"]).values
        raw = float((w * (b - (w * b).sum()) ** 2).sum())
        noise = float((w * (g["yb"] * (1 - g["yb"])).values / n).sum())
        return 1e5 * (len(y) / n_all) * max(raw - noise, 0) / u_all

    real = _sig(key)
    rng = np.random.default_rng(seed)
    null = [v for v in (_sig(key[rng.permutation(len(key))]) for _ in range(reps))
            if v is not None]
    if verbose and real is not None:
        print(f"  귀무 {tag:<14} 실제 {real:>6.2f}   섞음 최고 {max(null):>6.2f}"
              f"  중앙 {float(np.median(null)):>6.2f}"
              f"   {'실제가 더 큼' if real > max(null) else '🔴 섞은 것이 더 크다 = 노이즈'}",
              flush=True)
    return dict(real=real, null=null)


# --------------------------- G5 전역 기여 ---------------------------
def G5(z, y, seg, tag="세그먼트", verbose=True):
    """손실이 어디 있나 — raw Brier의 전역 기여로만 본다.

    세그먼트 BSS는 그 세그먼트 자신의 r(1−r)로 나눈다. 전역 점수와 무관하다.
    그걸 손실 지도로 읽어서 F(실제로는 이득원)를 손실원으로 오진했다.
    '전역이득' 열이 양수여야 진짜 적자다.
    """
    p = sigmoid(z)
    seg = np.asarray(seg)
    n, u, b = len(y), y.mean() * (1 - y.mean()), np.mean((p - y) ** 2)
    rows = []
    for k in pd.unique(seg):
        i = np.flatnonzero(seg == k)
        yy, pp = y[i], p[i]
        r = yy.mean()
        bs = np.mean((pp - yy) ** 2)
        rows.append((k, len(i), r, bs, 1e5 * (1 - bs / (r * (1 - r))),
                     1e5 * (len(i) / n) * (bs - b) / u))
    rows.sort(key=lambda t: -t[5])
    if verbose:
        print(f"  G5 {tag} — 전체 BSS {bss(p, y):.1f}")
        print(f"      {'값':>10}{'n':>9}{'r':>8}{'Brier':>11}"
              f"{'세그BSS(무시)':>14}{'전역이득':>10}")
        for k, m_, r, bs, sb, g in rows:
            print(f"      {str(k):>10}{m_:>9,}{r:>8.4f}{bs:>11.6f}"
                  f"{sb:>14.1f}{g:>+10.1f}", flush=True)
    return rows


# ------------------------------ 기준선 ------------------------------
def main():
    print("계측기 기준선 — 값과 기존 관측점. **판정은 사람이 한다.**\n", flush=True)
    df = hist()
    df["gt"] = (df["game_type"] == "F").astype(int)
    print("[G1 지속성]", flush=True)
    G1(df, "plat")
    G1(df, "gt")
    v, y, z = deploy_base()
    print("\n[진단] level_share — 🔴 기각 근거로 쓰지 말 것", flush=True)
    m0 = json.load(open(f"runs/{BASE_RUN}/model/meta.json", encoding="utf-8"))
    level_share(z - m0["logit_shift"], z, y, "shift(LB+47.04)")
    print("     ↑ 이미 채택돼 +47.04를 낸 축이 판별력 몫 0%다. 옛 G2는 이걸 FAIL로 냈다.",
          flush=True)
    print("\n[G3 배포 베이스]", flush=True)
    # 표는 <=2023으로만 만든다. runs/*/platoon.csv는 2024를 포함한 배포용이라
    # 그대로 2024에서 채점하면 자기적합으로 +99가 나온다.
    tab = _split_table(df[df.season <= 2023], "plat")
    spl = pd.DataFrame({"pitcher_id": v["pitcher_id"].values,
                        "plat": v["plat"].values}).merge(
        tab, on=["pitcher_id", "plat"], how="left")["split"].fillna(0.0).values
    z0 = z - m0["platoon"]["b"] * _deployed_split(v)
    G3(z0 + m0["platoon"]["b"] * spl, y, z0, "platoon(021)")
    G3(z, y, z, "depth 류 예시", extrapolation=True)

    print("\n[귀무 대조]  대조군: 완전 무작위 열 — 실제도 섞음도 같아야 한다", flush=True)
    rng = np.random.default_rng(1)
    null_control(y, sigmoid(z), rng.integers(0, 10, len(y)), tag="무작위열")

    print("\n[G5 전역 기여]", flush=True)
    G5(z, y, v["game_type"].values, "game_type")
    print("\n🔴 어느 줄도 '기각'이 아니다. 값을 보고 사람이 판단한다.", flush=True)


if __name__ == "__main__":
    main()
