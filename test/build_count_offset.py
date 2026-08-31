"""볼카운트(count_state)별 잔여 편향 offset — platoon과 같은 구조로 배포 (08 §5-9).

전역 로짓 이동(shift) 이후에도 count_state(balls-strikes)별로 잔차가 남는다
(2024 in-sample 상한 +10.8, chi2_11 p~4e-5, §5-9). §5-9는 "이동량을 fold에서 적합하는"
형태(§6-J가 금지한 형태)와 닮았다고 경계했으나, 여기서 쓰는 건 **전역 수준(level)이
아니라 카운트 간 상대 구조**이고 §6-J가 실패한 건 연간 낙폭 분산이 큰 level이었다 —
다른 양이라 안정성을 별도로 쟀다(probe_count_offset.py + 강한 base 재검증, 2026-08-30):
지속성 corr +0.39~+0.60(2021~2022~2024 세 쌍), 배포방향 전이 3개 전부 양수(+1.0~+3.7,
평균 +2.28) — platoon 채택 당시 지속성(+0.373)보다 오히려 높다.

    logit(p) ← logit(p) + b · cdev[count_state]
    cdev = EB스무딩(그 카운트의 전체 성공률) − 전체 성공률   (전역 표, 투수 무관)

🔴 표와 계수 b는 학습 때 만들어 zip에 싣는다. platoon과 동일 원칙.
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

RUN = "209_countoff_base044"
BASE_RUN = "209_platoon_base044"
M = 200                       # EB 스무딩 강도 (표본이 커서 platoon의 M=270보다 가볍게)
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


def solve_shift(z, target_mean):
    """sigmoid(z - d)의 평균이 target_mean이 되는 d (probe_count_offset.py와 동일)."""
    lo, hi = -3.0, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(z - mid).mean() > target_mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_table(cnt, z, y, m=M):
    """🔴 raw target이 아니라 **모델 예측(z)의 잔차**로 표를 만든다.
    count_state는 이미 CAT_COLS의 직접 관측 피처라, raw success rate 편차를
    그대로 쓰면 트리가 이미 아는 정보를 이중으로 준다(004 cond 번들과 같은 함정).
    여기서 재는 건 '이미 count_state를 아는 모델이 그래도 못 잡은 것'이다.
    """
    overall = solve_shift(z, y.mean())
    zc = z - overall
    out, w = [], {}
    for c in np.unique(cnt):
        mask = cnt == c
        n = int(mask.sum())
        raw = solve_shift(zc[mask], y[mask].mean())
        shrunk = raw * n / (n + m)     # EB 스무딩: 표본 작을수록 0으로 당김
        out.append((c, shrunk, mask.mean()))
    t = pd.DataFrame(out, columns=["count_state", "cdev", "w"])
    t["cdev"] = t["cdev"] - float((t["cdev"] * t["w"]).sum())   # 사용량 가중 평균 0
    return t[["count_state", "cdev"]]


def apply_dev(d, t):
    s = d.merge(t, on="count_state", how="left")["cdev"]
    return s.fillna(0.0).values


def main():
    out_dir = os.path.join("runs", RUN)
    if os.path.exists(os.path.join(out_dir, "model")):
        raise SystemExit(f"이미 존재함: {out_dir} — RUN 이름을 바꿀 것")

    mdir_base = os.path.join("runs", BASE_RUN, "model")
    meta = json.load(open(os.path.join(mdir_base, "meta.json"), encoding="utf-8"))
    off, shift, plat = meta["offset"], meta["logit_shift"], meta["platoon"]
    print(f"[{RUN}] 기반 {BASE_RUN}  offset b={off['b']:.5f} c={off['c']:.5f} "
          f"shift={shift:+.6f}  platoon b={plat['b']:.4f}", flush=True)

    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df["count_state"] = df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)
    df["plat"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)

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
    P0 = sigmoid(z_off)

    # platoon까지 적용 (배포 표 그대로 사용 — platoon 자체는 재현하지 않고 저장된 표를 쓴다)
    plat_tab = pd.read_csv(os.path.join(mdir_base, "platoon.csv"), encoding="utf-8")
    x_plat = d.merge(plat_tab, on=["pitcher_id", "plat"], how="left")["split"].fillna(0.0).values
    P = sigmoid(logit(P0) + plat["b"] * x_plat)

    bss = lambda p: 100000 * (1 - np.mean((p - y) ** 2) / (y.mean() * (1 - y.mean())))

    # 🔴 표는 raw target이 아니라 **이 파이프라인 자신의 2024 잔차(P vs y)**로 만든다.
    # count_state는 이미 CAT_COLS 직접 피처라, raw rate 편차를 쓰면 트리가 이미 아는
    # 정보를 이중으로 주는 함정(004 cond 번들)에 빠진다. 여기서 재는 '모델이 count_state를
    # 이미 보고도 못 잡은 잔차'만 진짜 새 정보다.
    #
    # 표 자체를 2024로 만들고 같은 2024로 평가하면 순환논리라, 5-fold로 쪼개
    # 4/5로 표를 만들고 나머지 1/5에 적용하는 방식으로 OOF를 낸다(platoon의
    # b 적합과 같은 원리 — 엔티티가 아니라 표 자체를 fold별로 다시 만드는 차이만 있다).
    # 진짜 배포 근거(out-of-year 지속성)는 probe_count_offset.py +
    # count_offset_strongbase_transfer.py에 이미 있다(2021→2024 +3.7, 2022→2024 +1.0,
    # 지속성 corr +0.39~+0.60) — 여긴 그 표를 배포용으로 만드는 스크립트다.
    z = logit(P)
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(d))
    cnt = d["count_state"].values

    q_oof = np.empty(len(d))
    for k in range(5):
        tr_mask = fold != k
        t_k = build_table(cnt[tr_mask], z[tr_mask], y[tr_mask])
        x_k = apply_dev(d.loc[fold == k], t_k)
        q_oof[fold == k] = sigmoid(z[fold == k] + x_k)
    print(f" 2024 검증(5-fold OOF, platoon까지 적용된 상태 기준): {bss(P):.1f} -> {bss(q_oof):.1f}  "
          f"(ΔBSS {bss(q_oof) - bss(P):+.1f})", flush=True)

    # 배포용 표: 2024 전체(가장 최근 holdout)의 잔차로 만든다.
    # shift의 val_bias·offset의 mu와 같은 관례 — 2024가 이 파이프라인의 유일한
    # 정직한 out-of-sample 연도라 그걸로 계산해 그대로 2025에 쓴다. b=1 고정
    # (표 자체가 이미 잔차 적합이라 별도 스케일 계수가 불필요 — probe에서도
    # delta_c를 그대로(스케일 없이) 적용해 전이가 양수였다).
    b = 1.0
    tab = build_table(cnt, z, y)
    print(" 배포 표(2024 잔차 기준):")
    print(tab.to_string(index=False))
    cover = (apply_dev(df, tab) != 0).mean()
    print(f" train 커버 {cover*100:.1f}%")

    shutil.copytree(mdir_base, os.path.join(out_dir, "model"))
    mdir = os.path.join(out_dir, "model")

    tab.to_csv(os.path.join(mdir, "count_offset.csv"), index=False, encoding="utf-8")
    meta_out = json.load(open(os.path.join(mdir, "meta.json"), encoding="utf-8"))
    meta_out["count_offset"] = {"b": b}
    json.dump(meta_out, open(os.path.join(mdir, "meta.json"), "w", encoding="utf-8"))

    json.dump({"run": RUN, "note": (
        f"{BASE_RUN} + 볼카운트 잔여편향 offset (b={b:.1f} 고정, M={M}). 재학습 없음 = 단일 변수. "
        f"표는 raw rate가 아니라 이 파이프라인 자신의 2024 잔차로 적합(004식 이중정보 함정 회피). "
        f"2024 5-fold OOF(platoon 적용후) {bss(P):.1f} -> {bss(q_oof):.1f} ({bss(q_oof)-bss(P):+.1f}). "
        "사전 out-of-year 지속성 corr +0.39~+0.60, 배포방향 전이 3개 전부 양수(+1.0~+3.7, "
        "probe_count_offset.py) 확인 후 진행. 08 §5-9"),
        "base_run": BASE_RUN, "b": b, "m": M,
        "val_2024_oof": {"before": bss(P), "after_oof": bss(q_oof),
                         "delta": bss(q_oof) - bss(P)},
        "coverage_train": float(cover), "lb_2025": None},
        open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)

    path = os.path.join(out_dir, f"submit{RUN.split('_')[0]}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f_ in ("script.py", "requirements.txt", "features.py", "cond.py"):
            zf.write(os.path.join(COMMON, f_), f_)
        for f_ in sorted(os.listdir(mdir)):
            zf.write(os.path.join(mdir, f_), "model/" + f_)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
