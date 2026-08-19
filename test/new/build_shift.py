"""전역 로짓 이동 빌더 — 기존 run에 시즌 base rate 잔여 보정을 얹는다 (08 §5-6).

=== league-rate baseline 도입 후 변경점 (2026-08) ===
기존엔 010(baseline 없음)이 245,789행 가짜 test에 낸 예측 캐시
(artifacts/sub010.csv.gz, 평균 0.4873)를 그대로 읽어서 이동량을 계산했다.

이제 BASE_RUN(016 등)은 이미 학습 단계에서 league-rate baseline으로
season×game_type 평균 수준을 상당 부분 반영했으므로, **이 스크립트가 직접
가짜 test에 대해 추론을 다시 돌려서** 현재 예측 평균을 구한다(과거처럼
캐시를 그대로 믿을 수 없다 — baseline 유무에 따라 평균이 달라지므로).

가짜 test = train.csv의 season==2024 행에서 `season`만 2025로 바꿔치기한 것
(08 §5-6과 동일한 방식 — 진짜 2025 test는 볼 수 없으므로, "2024 행의 실력
분포 + season=2025라는 라벨"로 미관측 시즌을 흉내낸다). league-rate
baseline이 있으면 이 season=2025 라벨 덕분에 baseline이 자동으로
test_override(기본 None → train.csv 내부 선형외삽)를 쓰게 된다 —
script.py 추론과 완전히 같은 경로.

🔴 shift는 여전히 "이동량을 fold/test에서 적합"하면 안 된다(§6-J 실패
사례). 여기서 하는 건 "현재 예측 평균 vs 외부 추정치(EST_2025)"를 **재는
것**뿐이고, 이동량 자체는 그 차이를 그대로 쓴다(§5-6과 동일한 논리 —
목표값은 외부에서 옴, fold에서 fit한 게 아님).

⚠️ league-rate baseline이 제 역할을 했다면, 여기서 계산되는 잔여 shift는
   0에 가까워야 정상이다(§5-6이 이미 평균 축을 거의 소진했다고 확인함,
   남은 여지 +0.12). 잔여폭이 여전히 크면 baseline이 기대만큼 작동하지
   않았다는 신호이므로, 먼저 그 원인을 봐야지 shift로 계속 덮으면 안 된다.
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
from catboost import CatBoostClassifier, Pool
from scipy.optimize import brentq

sys.path.insert(0, "common")
from features import engineer, prepare  # noqa: E402
import league_rate as lr  # noqa: E402

RUN = "020_shift_extra_fe_baseline_alsoaux"
BASE_RUN = "019_offset_extra_fe_baseline_alsoaux"
# ⚠️ 이전엔 0.477을 10문서에서 그대로 하드코딩했었다. 그런데 10문서 자체가
# "여기서 나온 어떤 값도 피처가 되지 않는다"(외부데이터=규정위반)고 명시하고
# 있어서, League-rate baseline과 같은 원칙으로 바꿨다 — None으로 두면 아래
# main()에서 train.csv(season별 실측 성공률)만으로 직접 선형외삽해서 구한다.
# 외부 문서를 전혀 참조하지 않는다. game_type 구분 없이 전체(season만)
# 기준이라는 점에서 league-rate baseline의 GROUP_COLS=["season","game_type"]
# 표와는 별개로 계산된다(이쪽은 원래도 R/F 구분 없는 "전역" 이동량이었음).
EST_2025_FLAT_OVERRIDE = None
FRACTION = 1.0             # 잔여 shift에 적용할 비율. baseline이 이미 대부분
                           # 잡아줬다면 이 값을 낮춰(예: 0.5) 위험을 더 줄일 수 있다.
DATA = "data/train.csv"
COMMON = "common"
ID, TARGET = "row_id", "control_success"


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_full(mdir, meta, X, ci, fake):
    """script.py와 완전히 같은 로직으로, mdir의 모델들로 X에 대한 최종 확률을
    계산한다 (offset까지 반영, shift는 아직 반영 안 함 — 그건 이 스크립트가
    새로 계산할 대상이므로).

    fake: engineer() 이전의 원본 가짜 test df(row_id, season 포함) —
    주모델용 baseline 계산과 보조모델용 X_aux 재구성에 둘 다 필요하다.
    """
    lb = meta.get("league_baseline", {"enabled": False})
    main_baseline = None
    if lb.get("enabled"):
        table, group_cols = lr.table_from_json(lb)
        main_baseline = lr.assign_baseline_logit(
            fake, table, group_cols, held_out_season=None,
            override=lb.get("test_override"))

    pool_main = Pool(X, cat_features=ci, baseline=main_baseline)

    def avg_proba(prefix, seeds, pool):
        ps = []
        for sd in seeds:
            m = CatBoostClassifier()
            m.load_model(os.path.join(mdir, f"{prefix}{sd}.cbm"))
            ps.append(m.predict_proba(pool)[:, 1])
        return np.mean(ps, axis=0)

    p = np.clip(avg_proba("model_", meta["seeds"], pool_main), 1e-6, 1 - 1e-6)

    off = meta.get("offset")
    if off:
        # ⚠️ mr/wayoff는 주모델과 다른 feature_cols/extra_fe로 학습됐을 수
        # 있다 — 주모델용 X를 그대로 재사용하면 CatBoost가 "피처 이름이
        # 다르다"고 에러를 낸다(실측 확인된 버그, 2026-08). 보조모델 전용
        # X_aux를 따로 만든다.
        af = off.get("aux_features")
        if af is None:
            af = {"global_mean": meta["global_mean"],
                 "extra_fe": meta.get("extra_fe", False),
                 "rate_means": meta.get("rate_means"),
                 "feature_cols": meta["feature_cols"],
                 "cat_cols": meta["cat_cols"]}
        fe_aux = engineer(fake.drop(columns=[ID]), af["global_mean"],
                          extra_fe=af.get("extra_fe", False),
                          rate_means=af.get("rate_means"))
        X_aux = prepare(fe_aux, af["feature_cols"], af["cat_cols"])
        ci_aux = [X_aux.columns.get_loc(c) for c in af["cat_cols"]]

        alb = off.get("aux_league_baseline", {})
        pools = {}
        for name in ("mr", "wayoff"):
            info = alb.get(name, {"enabled": False})
            base_aux = None
            if info.get("enabled"):
                table_aux, gc_aux = lr.table_from_json(info)
                base_aux = lr.assign_baseline_logit(
                    fake, table_aux, gc_aux, held_out_season=None,
                    override=info.get("test_override"))
            pools[name] = Pool(X_aux, cat_features=ci_aux, baseline=base_aux)

        z = (logit(p)
             + off["b"] * (logit(avg_proba("mr_", off["seeds"], pools["mr"]))
                           - off["mu_mr"])
             + off["c"] * (logit(avg_proba("wayoff_", off["seeds"], pools["wayoff"]))
                           - off["mu_wayoff"]))
        p = np.clip(sigmoid(z), 1e-6, 1 - 1e-6)
    return p


def main():
    out_dir = os.path.join("runs", RUN)
    if os.path.exists(os.path.join(out_dir, "model")):
        raise SystemExit(f"이미 존재함: {out_dir} — RUN 이름을 바꿀 것")

    mdir_base = os.path.join("runs", BASE_RUN, "model")
    meta = json.load(open(os.path.join(mdir_base, "meta.json"), encoding="utf-8"))

    print(f"[{RUN}] 기반 {BASE_RUN}  league_baseline="
          f"{meta.get('league_baseline', {}).get('enabled')}")

    # ---- 가짜 test 구성: 2024 행 + season만 2025로 치환 (08 §5-6과 동일 방식) ----
    print(" 가짜 test 구성 (2024 행, season→2025)...", flush=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    fake = df[df["season"] == 2024].drop(columns=[TARGET]).copy()
    fake["season"] = 2025

    # ---- EST_2025_FLAT: train.csv(season별 실측 성공률)만으로 직접 선형외삽 ----
    # game_type 구분 없이 season 하나로만 묶는다 — 원래 shift가 R/F 안 가리고
    # 전체 평균 하나만 이동시키던 것과 같은 성격을 유지한다.
    if EST_2025_FLAT_OVERRIDE is None:
        table_flat = lr.build_table(df, ["season"])
        EST_2025_FLAT = lr.extrapolate(table_flat, ["season"], 2025)
        print(f" EST_2025_FLAT 자동 재계산(선형외삽, train.csv만 사용) = "
              f"{EST_2025_FLAT:.4f}  (연도별 실측: "
              + ", ".join(f"{int(s)}={r:.4f}" for s, r in table_flat.items())
              + ")")
    else:
        EST_2025_FLAT = EST_2025_FLAT_OVERRIDE
        print(f" EST_2025_FLAT 수동 override 사용 = {EST_2025_FLAT:.4f} "
              f"— train.csv만으로 재현 가능한 값인지 먼저 확인했는지 재점검할 것")

    fe = engineer(fake.drop(columns=[ID]), meta["global_mean"],
                 extra_fe=meta.get("extra_fe", False),
                 rate_means=meta.get("rate_means"))
    X = prepare(fe, meta["feature_cols"], meta["cat_cols"])
    ci = [X.columns.get_loc(c) for c in meta["cat_cols"]]

    p = predict_full(mdir_base, meta, X, ci, fake)
    cur = float(p.mean())
    print(f" 가짜 test({len(fake):,}행) 현재 예측 평균 = {cur:.4f}  "
          f"(2025 추정 {EST_2025_FLAT})")

    target = cur - FRACTION * (cur - EST_2025_FLAT)
    lg = logit(p)
    f = lambda d: float(np.mean(sigmoid(lg + d))) - target
    # 이미 baseline이 대부분 맞춰놨다면 cur가 target에 아주 가까워
    # brentq 초기 구간에서 해가 안 잡힐 수 있다 — 그 경우는 "이동이
    # 거의 필요 없다"는 신호이므로 d≈0으로 처리한다.
    if abs(cur - target) < 1e-9:
        d = 0.0
    else:
        d = float(brentq(f, -1.0, 1.0, xtol=1e-10))
    after = float(np.mean(sigmoid(lg + d)))

    print(f" logit_shift = {d:+.6f}   적용 후 평균 {after:.4f}")
    if abs(d) < 0.005:
        print(" → 잔여폭이 작다. league-rate baseline이 평균 수준을 대부분 "
              "잡아준 것으로 보인다 (기대했던 시나리오).")
    else:
        print(" ⚠️ 잔여폭이 여전히 크다. baseline이 기대만큼 작동하지 않았을 "
              "수 있다 — season×game_type 표(meta['league_baseline']) 값과 "
              "override를 먼저 점검할 것.")

    delta_true = cur - EST_2025_FLAT
    s = cur - after
    base = 0.2498
    print(f" 기대(참고용): 추정 적중 {100000*(2*s*delta_true - s*s)/base:+.1f} / "
          f"하락 멈춤 {100000*(-s*s)/base:+.1f}")

    os.makedirs(os.path.join(out_dir, "model"), exist_ok=True)
    shutil.copytree(mdir_base, os.path.join(out_dir, "model"), dirs_exist_ok=True)
    mdir = os.path.join(out_dir, "model")
    meta_out = json.load(open(os.path.join(mdir, "meta.json"), encoding="utf-8"))
    meta_out["logit_shift"] = d
    json.dump(meta_out, open(os.path.join(mdir, "meta.json"), "w",
                             encoding="utf-8"))

    json.dump({"run": RUN, "note": (
        f"{BASE_RUN} + 잔여 전역 로짓 이동 {d:+.6f}. league-rate baseline "
        "적용 후 남은 평균 편차만 보정. 모델 재학습 없음 = 단일 변수. "
        f"가짜test 예측 평균 {cur:.4f} → {after:.4f} (2025 추정 {EST_2025_FLAT})."),
        "base_run": BASE_RUN, "logit_shift": d,
        "mean_before": cur, "mean_after": after, "est_2025": EST_2025_FLAT,
        "fraction": FRACTION, "lb_2025": None},
        open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)

    path = os.path.join(out_dir, f"submit{RUN.split('_')[0]}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        files = ["script.py", "requirements.txt", "features.py", "cond.py"]
        if meta.get("league_baseline", {}).get("enabled"):
            files.append("league_rate.py")
        for f_ in files:
            z.write(os.path.join(COMMON, f_), f_)
        for f_ in sorted(os.listdir(mdir)):
            z.write(os.path.join(mdir, f_), "model/" + f_)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
