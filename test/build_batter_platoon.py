"""타자 좌우편차(batter platoon split) offset 빌더 — build_platoon.py의 타자판.

`cond_bh`(타자×투수손, **트리 피처**)는 지속성이 좌우매치급(+0.366/+0.456)이었는데도
LB −9.33이었다(closed.tsv). §7-2가 `cond_ph`에서 이미 증명한 대로 "번들링·파라미터화"가
결과를 뒤집을 수 있다 — cond_bh는 049_condphbh처럼 ph와 번들+트리피처로만 시도됐고,
**잔차 offset 형태**는 한 번도 안 해봤다.

검증(2026-08-29):
  지속성 corr +0.460 (투수판 0.412보다 높음)
  out-of-year(003계보) 2/2 양수: <=2022->2023 +43.3 / <=2023->2024 +4.1
  044 배포조건(투수 platoon 이미 적용된 상태 위에 추가) 5-fold x 5시드 전부 양수:
    +0.12 ~ +1.81 (평균 ~+1)
  두 번째 전이(<=2022->2023, 044급 재학습) fold별 b = 2.63~3.26(전부 안정적 양수), Δ=+27.1
    (2023은 F체제개편 참고용 fold라 절대크기는 신뢰 낮음, 방향만 참고)

메커니즘은 build_platoon.py와 완전히 동일 — 주체만 pitcher_id -> batter_id.
투수 platoon **위에 추가로** 얹는다(단일 변수: 이 스텝만 새로 생김).
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

RUN = "210_batterplatoon_noPP"
BASE_RUN = "209_shift_base044"    # 투수platoon(build_platoon) 건너뜀 — LB상 043>044 확인됨
M = 270                       # 투수판과 동일값 재사용 (신규 하이퍼파라미터 도입 금지)
VAL_SUCCESS = "artifacts/auxpred_base044"
VAL_AUX = "artifacts/auxpred"
VAL_SEEDS = [42, 7, 2024]
DATA = "data/train.csv"
COMMON = "common"
ID, TARGET = "row_id", "control_success"


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def build_table(hist, weight_rows, m=M):
    """hist(과거 행)로 타자별 좌우편차 표를 만든다. build_platoon.py의 build_table과 동일 로직."""
    g = hist.groupby(["batter_id", "plat"])[TARGET].agg(["sum", "count"])
    pr = hist.groupby("batter_id")[TARGET].mean().rename("pr")
    g = g.join(pr, on="batter_id")
    g["v"] = (g["sum"] + m * g["pr"]) / (g["count"] + m)
    t = (g["v"] - g["pr"]).rename("split").reset_index()
    w = weight_rows.merge(t, on=["batter_id", "plat"], how="left")["split"]
    t["split"] = t["split"] - float(w.fillna(0).mean())
    return t


def apply_split(d, t):
    s = d.merge(t, on=["batter_id", "plat"], how="left")["split"]
    return s.fillna(0.0).values


def main():
    out_dir = os.path.join("runs", RUN)
    if os.path.exists(os.path.join(out_dir, "model")):
        raise SystemExit(f"이미 존재함: {out_dir} — RUN 이름을 바꿀 것")

    mdir_base = os.path.join("runs", BASE_RUN, "model")
    meta = json.load(open(os.path.join(mdir_base, "meta.json"), encoding="utf-8"))
    off, shift, plat = meta["offset"], meta["logit_shift"], meta.get("platoon")
    print(f"[{RUN}] 기반 {BASE_RUN}  offset b={off['b']:.5f} c={off['c']:.5f} "
          f"shift={shift:+.6f}  투수platoon={'b=%.4f' % plat['b'] if plat else '없음(건너뜀)'}",
          flush=True)

    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df["plat"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)

    # --- 투수 platoon까지 적용한 상태(z_ref)를 2024 검증셋에서 재구성 ---
    L = pd.read_csv("recovered_labels.csv.gz")
    have = df[[ID]].merge(L, on=ID, how="left")["middle"].notna().values
    m24 = (df["season"] == 2024).values
    d = df[m24][have[m24]].copy()
    y = d[TARGET].values.astype(float)

    def load(dd, s, seeds):
        return np.mean([np.load(os.path.join(dd, f"{s}_2024_{k}.npy"))
                        for k in seeds], axis=0)

    z_off = (logit(load(VAL_SUCCESS, "success", VAL_SEEDS))
             + off["b"] * (logit(load(VAL_AUX, "mr", off["seeds"])) - off["mu_mr"])
             + off["c"] * (logit(load(VAL_AUX, "wayoff", off["seeds"])) - off["mu_wayoff"])
             + shift)
    bssf = lambda p: 100000 * (1 - np.mean((p - y) ** 2) / (y.mean() * (1 - y.mean())))
    if plat:
        def p_build_table(hist, weight_rows, m):
            # build_platoon.py의 build_table과 동일 로직(pitcher_id 기준) — 모듈 import 시
            # 그쪽 파일 최상단의 stdout 재할당과 충돌해서 여기 인라인으로 복제한다.
            g = hist.groupby(["pitcher_id", "plat"])[TARGET].agg(["sum", "count"])
            pr = hist.groupby("pitcher_id")[TARGET].mean().rename("pr")
            g = g.join(pr, on="pitcher_id")
            g["v"] = (g["sum"] + m * g["pr"]) / (g["count"] + m)
            t = (g["v"] - g["pr"]).rename("split").reset_index()
            w = weight_rows.merge(t, on=["pitcher_id", "plat"], how="left")["split"]
            t["split"] = t["split"] - float(w.fillna(0).mean())
            return t
        ptab_fit = p_build_table(df[df.season <= 2023], d, m=plat["m"])
        xp = d.merge(ptab_fit, on=["pitcher_id", "plat"], how="left")["split"].fillna(0.0).values
        z_ref = z_off + plat["b"] * xp
    else:
        z_ref = z_off
    print(f" z_ref({'투수platoon까지' if plat else '투수platoon 없음, shift까지만'}) "
          f"BSS = {bssf(sigmoid(z_ref)):.1f}", flush=True)

    # --- 타자 platoon 표(<=2023) + b 적합 ---
    fit_tab = build_table(df[df.season <= 2023], d)
    x = apply_split(d, fit_tab)

    bid = d["batter_id"].values
    fold = pd.Series(np.random.default_rng(0).integers(0, 5, len(np.unique(bid))),
                     index=np.unique(bid)).reindex(bid).values

    def newton(mask):
        b = 0.0
        for _ in range(50):
            q = sigmoid(z_ref[mask] + b * x[mask])
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
        q_oof[fold == k] = sigmoid(z_ref[fold == k] + bk * x[fold == k])
    b = newton(np.ones(len(d), dtype=bool))
    print(f" fold별 b = {['%.3f' % v for v in bs]}   전체 b = {b:.4f}")
    print(f" 2024 검증: {bssf(sigmoid(z_ref)):.1f} → {bssf(q_oof):.1f}  "
          f"(**OOF ΔBSS {bssf(q_oof) - bssf(sigmoid(z_ref)):+.1f}**)", flush=True)

    # --- 배포용 표: train 전체(2019~2024) ---
    tab = build_table(df, df)
    cover = (apply_split(df, tab) != 0).mean()
    print(f" 배포 표 {len(tab):,}행 (타자 {tab.batter_id.nunique():,}명), "
          f"train 커버 {cover*100:.1f}%, 중심화 후 평균 "
          f"{apply_split(df, tab).mean():+.2e}")

    shutil.copytree(mdir_base, os.path.join(out_dir, "model"))
    mdir = os.path.join(out_dir, "model")
    tab.to_csv(os.path.join(mdir, "platoon_batter.csv"), index=False, encoding="utf-8")
    meta_out = json.load(open(os.path.join(mdir, "meta.json"), encoding="utf-8"))
    meta_out["batter_platoon"] = {"b": b, "m": M}
    json.dump(meta_out, open(os.path.join(mdir, "meta.json"), "w",
                             encoding="utf-8"))

    json.dump({"run": RUN, "note": (
        f"{BASE_RUN} + 타자 좌우편차 offset(b={b:.4f}, M={M}). 재학습 없음 = 단일 변수. "
        f"2024 검증 {bssf(sigmoid(z_ref)):.1f} -> {bssf(q_oof):.1f} ({bssf(q_oof)-bssf(sigmoid(z_ref)):+.1f}). "
        f"fold별 b {['%.3f' % v for v in bs]}. cond_bh(트리피처+ph번들, LB -9.33)와 같은 정보를 "
        "잔차 offset으로(용량증가 0, 투수platoon과 동일 메커니즘)."),
        "base_run": BASE_RUN, "b": b, "m": M, "fold_b": bs,
        "val_2024": {"before": bssf(sigmoid(z_ref)), "after_oof": bssf(q_oof),
                     "delta": bssf(q_oof) - bssf(sigmoid(z_ref))},
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
