"""투수 좌우편차(platoon split) offset 빌더 — 기존 run에 얹는다 (08 §5-12).

`asof_*`는 투수의 **전체** 성공률만 준다. 같은 투수라도 좌/우 타자를 상대할 때
제구 성공률이 다르고, 그 **개인차의 크기**는 어느 컬럼에도 없다. 학습 데이터로
투수별 좌우 편차 표를 만들어 최종 예측에 고정계수로 더한다.

    logit(p) ← logit(p) + b · split[pitcher_id, plat]      plat = (투수손 == 타자손)
    split = 스무딩(그 투수의 plat별 성공률) − 그 투수의 전체 성공률

=== 왜 `cond`(004, LB −5.4)와 다른가 ===
같은 정보를 cond는 **트리 피처**로 줬다. 여기는 **잔차 offset**이다.
  · 용량 증가 0 · 재보정 없음 · 메인 재학습 없음  (offset·shift와 같은 구조)
  · cond는 죽은 표 3개와 묶여 있었다 — `pc`(투수×카운트)·`bc`(타자×카운트)·
    `pi`(투수×이닝)는 013 잔차 기준 남은 여지가 **전부 0**으로 측정됐다.

=== 채택 근거 (전부 013/015 잔차 기준, 투수분할 CV) ===
  ① 지속성 corr(2019-23 → 2024) = **+0.373**
  ② 013 잔차 +28.3 (b 0.504~0.554) │ ③ 새로 학습한 독립 베이스 +29.1 (b 0.517~0.568)
  ④ 베이스 교체: 003 +28.9 / 013 +28.3 / 018 +29.2 — **강해져도 안 줄어든다**
     (offset·shift는 같은 조건에서 73% 줄었다 = 부분 대체재. 이건 아니다.)
  ⑤ 부분집합: R +22.1 / F +39.0 / 전반기 +36.4 / 후반기 +16.8 — 전부 양수
  ⑥ **015 파이프라인 위 +24.7** ← 배포 조건. 이 값을 기준으로 본다.

=== 수축 강도 M ===
지속성 상관에서 모집단 표준편차를 역산한다(empirical Bayes):
    tau^2 = corr x sd_a x sd_b = 0.373 x 0.0414 x 0.0604 → tau = 0.0305
    M = 0.25 / tau^2 ≈ 270
스윕 최고점과 일치했다. **외부 상수는 쓰지 않았다** (규정 2-3).

🔴 split 표는 사용량 가중 평균이 정확히 0이 되도록 중심화한다. 레벨을 못 건드리므로
   §3의 calibration 금지와 충돌하지 않고, 앞단 `logit_shift`도 흐트러뜨리지 않는다.
🔴 표와 계수 b는 **학습 때 만들어 zip에 싣는다.** test에서 만들면 규정 위반.
"""
import io
import json
import os
import shutil
import sys
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.insert(0, "common")
from features import build_anchor  # noqa: E402

RUN = "056_platoon_phb"
BASE_RUN = "055_shift_phb"
M = 270                       # 수축 강도 (tau 역산값. 스윕 최고점과 일치)
# 계수 b 적합용 검증 예측 (2019~23 학습 → 2024). BASE_RUN의 성공모델 = 013.
VAL_SUCCESS = "artifacts/auxpred_condphb"
VAL_AUX = "artifacts/auxpred"
VAL_SEEDS = [42, 7, 2024]   # 성공모델 캐시용. 보조는 off["seeds"]
DATA = "data/train.csv"
COMMON = "common"
ID, TARGET = "row_id", "control_success"


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def build_table(hist, weight_rows, m=M):
    """hist(과거 행)로 투수별 좌우편차 표를 만든다.

    weight_rows = 중심화에 쓸 행 집합(사용량 가중). 표 자체는 hist에서만 만든다.
    """
    g = hist.groupby(["pitcher_id", "plat"])[TARGET].agg(["sum", "count"])
    pr = hist.groupby("pitcher_id")[TARGET].mean().rename("pr")
    g = g.join(pr, on="pitcher_id")
    g["v"] = (g["sum"] + m * g["pr"]) / (g["count"] + m)
    t = (g["v"] - g["pr"]).rename("split").reset_index()
    # 사용량 가중 평균을 0으로 — 레벨을 건드리지 않게 한다.
    w = weight_rows.merge(t, on=["pitcher_id", "plat"], how="left")["split"]
    t["split"] = t["split"] - float(w.fillna(0).mean())
    return t


def apply_split(d, t):
    s = d.merge(t, on=["pitcher_id", "plat"], how="left")["split"]
    return s.fillna(0.0).values


def main():
    out_dir = os.path.join("runs", RUN)
    if os.path.exists(os.path.join(out_dir, "model")):
        raise SystemExit(f"이미 존재함: {out_dir} — RUN 이름을 바꿀 것")

    mdir_base = os.path.join("runs", BASE_RUN, "model")
    meta = json.load(open(os.path.join(mdir_base, "meta.json"), encoding="utf-8"))
    off, shift = meta["offset"], meta["logit_shift"]
    print(f"[{RUN}] 기반 {BASE_RUN}  offset b={off['b']:.5f} c={off['c']:.5f} "
          f"shift={shift:+.6f}", flush=True)

    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df["plat"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)

    # --- 계수 b 적합: ≤2023 표를 2024 out-of-sample 예측의 잔차에 맞춘다 ---
    L = pd.read_csv("recovered_labels.csv.gz")
    have = df[[ID]].merge(L, on=ID, how="left")["middle"].notna().values
    m24 = (df["season"] == 2024).values
    d = df[m24][have[m24]].copy()
    y = d[TARGET].values.astype(float)
    # 🔴 성공모델과 보조모델은 시드 수가 다를 수 있다(025: 성공 7 / 보조 3).
    #    보조는 offset meta에 적힌 시드를 그대로 쓴다.
    def load(d, s, seeds):
        return np.mean([np.load(os.path.join(d, f"{s}_2024_{k}.npy"))
                        for k in seeds], axis=0)

    P = sigmoid(logit(load(VAL_SUCCESS, "success", VAL_SEEDS))
                + off["b"] * (logit(load(VAL_AUX, "mr", off["seeds"]))
                              - off["mu_mr"])
                + off["c"] * (logit(load(VAL_AUX, "wayoff", off["seeds"]))
                              - off["mu_wayoff"])
                + shift)
    bss = lambda p: 100000 * (1 - np.mean((p - y) ** 2) / (y.mean() * (1 - y.mean())))

    fit_tab = build_table(df[df["season"] <= 2023], d)
    x = apply_split(d, fit_tab)

    # 투수 분할 CV — b도 out-of-sample로 재서 낙관 편향을 없앤다.
    pid = d["pitcher_id"].values
    fold = pd.Series(np.random.default_rng(0).integers(0, 5, len(np.unique(pid))),
                     index=np.unique(pid)).reindex(pid).values

    def newton(mask):
        b = 0.0
        for _ in range(50):
            q = sigmoid(logit(P[mask]) + b * x[mask])
            step = (x[mask] * (y[mask] - q)).sum() / max(
                (x[mask] ** 2 * q * (1 - q)).sum(), 1e-9)
            b += step
            if abs(step) < 1e-10:
                break
        return float(b)

    q_oof = np.empty(len(d))
    bs = []
    for k in range(5):
        bk = newton(fold != k)
        bs.append(bk)
        q_oof[fold == k] = sigmoid(logit(P[fold == k]) + bk * x[fold == k])
    b = newton(np.ones(len(d), dtype=bool))
    print(f" fold별 b = {['%.3f' % v for v in bs]}   전체 b = {b:.4f}")
    print(f" 2024 검증: {bss(P):.1f} → {bss(q_oof):.1f}  "
          f"(**OOF ΔBSS {bss(q_oof) - bss(P):+.1f}**)", flush=True)

    # --- 배포용 표: train 전체(2019~2024). 학습 행이 '그 행 이전 시즌 전부'를 쓴 것과 같은 규칙 ---
    tab = build_table(df, df)
    cover = (apply_split(df, tab) != 0).mean()
    print(f" 배포 표 {len(tab):,}행 (투수 {tab.pitcher_id.nunique():,}명), "
          f"train 커버 {cover*100:.1f}%, 중심화 후 평균 "
          f"{apply_split(df, tab).mean():+.2e}")

    shutil.copytree(mdir_base, os.path.join(out_dir, "model"))
    mdir = os.path.join(out_dir, "model")

    # 🔴 anchor.csv 스키마 갱신. BASE_RUN(015)은 013 시절 스키마(`s0` 한 열)로
    #    저장돼 있는데 지금 features.py는 `s0_{key}`를 찾는다 → 그대로 두면
    #    추론이 KeyError로 죽는다(제출 1회 차감). **값이 같은지 대조하고** 덮어쓴다.
    last = int(df["season"].max()) + 1
    new_a = build_anchor(df)
    new_a = new_a[new_a["apply_season"] == last].reset_index(drop=True)
    old_a = pd.read_csv(os.path.join(mdir, "anchor.csv"), encoding="utf-8")
    key = ["apply_season", "who", "id"]
    # 옛 run은 열 이름이 `s0`(013 시절), 최근 run은 `s0_success`다. 둘 다 받는다.
    ocol = "s0" if "s0" in old_a.columns else "s0_success"
    chk = (old_a[key + ["n0", ocol]]
           .rename(columns={"n0": "n0_o", ocol: "s0_o"})
           .merge(new_a[key + ["n0", "s0_success"]], on=key, how="outer"))
    assert len(chk) == len(old_a) == len(new_a), (len(chk), len(old_a), len(new_a))
    assert (chk["n0_o"] == chk["n0"]).all(), "n0 불일치"
    dmax = float((chk["s0_o"] - chk["s0_success"]).abs().max())
    assert dmax < 1e-9, f"s0 불일치 max={dmax}"
    new_a.to_csv(os.path.join(mdir, "anchor.csv"), index=False, encoding="utf-8")
    print(f" anchor.csv 재생성 {len(new_a):,}행 — 015 원본과 값 동일 확인 "
          f"(n0 전부 일치, s0 최대차 {dmax:.1e})", flush=True)
    tab.to_csv(os.path.join(mdir, "platoon.csv"), index=False, encoding="utf-8")
    meta_out = json.load(open(os.path.join(mdir, "meta.json"), encoding="utf-8"))
    meta_out["platoon"] = {"b": b, "m": M}
    json.dump(meta_out, open(os.path.join(mdir, "meta.json"), "w",
                             encoding="utf-8"))

    json.dump({"run": RUN, "note": (
        f"{BASE_RUN} + 투수 좌우편차 offset (b={b:.4f}, M={M}). 재학습 없음 = 단일 변수. "
        f"2024 검증 투수분할 CV {bss(P):.1f} -> {bss(q_oof):.1f} ({bss(q_oof)-bss(P):+.1f}). "
        f"fold별 b {['%.3f' % v for v in bs]}. cond(004, LB -5.4)와 같은 정보 다른 전달 "
        "(트리 피처 -> 잔차 offset, 용량 증가 0). 08 §5-12"),
        "base_run": BASE_RUN, "b": b, "m": M, "fold_b": bs,
        "val_2024": {"before": bss(P), "after_oof": bss(q_oof),
                     "delta": bss(q_oof) - bss(P)},
        "coverage_train": float(cover), "lb_2025": None},
        open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)

    path = os.path.join(out_dir, f"submit{RUN.split('_')[0]}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f_ in ("script.py", "requirements.txt", "features.py", "cond.py"):
            z.write(os.path.join(COMMON, f_), f_)
        for f_ in sorted(os.listdir(mdir)):
            z.write(os.path.join(mdir, f_), "model/" + f_)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
