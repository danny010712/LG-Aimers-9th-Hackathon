"""전역 로짓 이동 빌더 — 기존 run에 시즌 base rate 보정을 얹는다 (08 §5-6).

011 = 010 + 절반(−0.0208) → LB 985.09 / 012 = 전량(−0.0416) → LB 998.00.

트리는 미관측 시즌을 외삽하지 못한다(§6-A). 그래서 2025 예측 평균이 2024 수준
(0.4873)에 갇혀 있는데, base rate는 6년 연속 하락 중이고 10 문서가 KBO 자료로
추정한 2025 값은 **0.473~0.481 (중앙 0.477)** 이다.

Brier는 평균 편향 δ에 제곱으로 반응한다: 이동량 s를 걸면 ΔBrier = −2sδ + s².
    전량 (s = δ_추정 = 0.0103):  맞으면 +43 / 하락이 멈췄으면 −43
    절반 (s = 0.00515):          맞으면 +32 / 하락이 멈췄으면 **−11**
→ **절반을 건다.** 상승 여력 75%를 가져가면서 하방은 25%만 진다(제곱항 비대칭).

🔴 이동 상수는 **여기서 계산해 meta.json에 저장**한다. script.py가 test 평균을 보고
정하면 test 내부 행간 통계 = 규정 위반이다.
🔴 기준 평균 0.4873은 **2024 행으로 만든 가짜 test**에서 잰 값이다(실제 2025 test는 못 본다).

모델은 010의 것을 그대로 복사한다. 재학습 없음 = 010 대비 단일 변수.
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
from scipy.optimize import brentq

RUN = "012_shift_full"
BASE_RUN = "010_offset_seeds7"
# 010이 245,789행 가짜 test(2024 행)에 낸 예측. 이동량 산출의 기준.
PRED = ("C:/Users/ssy84/AppData/Local/Temp/claude/"
        "c--Users-ssy84-MyProjects-10-LGAimers/"
        "b8c5623b-34db-44ed-9743-5d9ac909853b/scratchpad/sub010.csv")
EST_2025 = 0.477          # 10 문서 §6-C 중앙 추정
FRACTION = 1.0            # 전량. 011(절반)이 예측대로 나와 계획대로 완성한다
COMMON = "common"


def main():
    out_dir = os.path.join("runs", RUN)
    if os.path.exists(os.path.join(out_dir, "model")):
        raise SystemExit(f"이미 존재함: {out_dir} — RUN 이름을 바꿀 것")

    p = pd.read_csv(PRED)["control_success"].values
    cur = float(p.mean())
    target = cur - FRACTION * (cur - EST_2025)

    lg = np.log(p / (1 - p))
    f = lambda d: float(np.mean(1 / (1 + np.exp(-(lg + d))))) - target
    d = float(brentq(f, -1.0, 1.0, xtol=1e-10))
    after = float(np.mean(1 / (1 + np.exp(-(lg + d)))))

    print(f"[{RUN}] 기반 {BASE_RUN}")
    print(f" 현재 평균 {cur:.4f}  2025 추정 {EST_2025}  →  목표 {target:.4f} (추정까지의 {FRACTION:.0%})")
    print(f" logit_shift = {d:+.6f}   적용 후 평균 {after:.4f}")
    # 참고: 추정이 맞을 때 / 하락이 멈췄을 때 기대 점수 변화
    delta_true = cur - EST_2025
    s = cur - after
    base = 0.2498
    print(f" 기대: 추정 적중 {100000*(2*s*delta_true - s*s)/base:+.1f} / "
          f"하락 멈춤 {100000*(-s*s)/base:+.1f}")

    shutil.copytree(os.path.join("runs", BASE_RUN, "model"),
                    os.path.join(out_dir, "model"))
    mdir = os.path.join(out_dir, "model")
    meta = json.load(open(os.path.join(mdir, "meta.json"), encoding="utf-8"))
    meta["logit_shift"] = d
    json.dump(meta, open(os.path.join(mdir, "meta.json"), "w", encoding="utf-8"))

    json.dump({"run": RUN, "note": (
        f"{BASE_RUN} + 전역 로짓 이동 {d:+.6f}. 모델 재학습 없음 = 단일 변수. "
        f"예측 평균 {cur:.4f} → {after:.4f} (2025 추정 {EST_2025}까지의 {FRACTION:.0%}). "
        "트리가 미관측 시즌을 외삽 못 하는 것에 대한 보정. 08 §5-6"),
        "base_run": BASE_RUN, "logit_shift": d,
        "mean_before": cur, "mean_after": after, "est_2025": EST_2025,
        "fraction": FRACTION, "lb_2025": None},
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
