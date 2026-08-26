"""제출 zip 전체규모 검증 (08 §8-2). 제출 전 필수.

규정상 script 실행 실패는 **제출 횟수를 차감한다**. 실제 평가 서버와 같은 규모
(245,789행)의 가짜 test로 zip을 통째로 실행해 구조·시간·결측·범위를 확인한다.

과거 이 절차가 실제로 사고를 막았다 — script.py를 고치며 `preds` 변수를 없앴는데
마지막 print가 남아 `NameError`. 그대로 냈으면 1회 차감이었다.

사용: python verify_submit.py runs/015_shift_inseason/submit015.zip
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

N_TEST = 245_789          # 실제 2025 test 행 수
ID, TARGET = "row_id", "control_success"
TOP_ALLOWED = {"script.py", "requirements.txt", "features.py", "cond.py",
               "league_rate.py", "model"}


def main():
    zip_path = sys.argv[1]
    work = tempfile.mkdtemp(prefix="verify_")
    ok = True
    try:
        # --- 1. zip 구조 ---
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            tops = {n.split("/")[0] for n in names}
            bad_sep = [n for n in names if "\\" in n]
            extra = tops - TOP_ALLOWED
            print(f"[구조] 항목 {len(names)}개  최상위 {sorted(tops)}")
            if extra:
                print(f"  ❌ 허용되지 않은 최상위: {extra} (설치오류 위험)")
                ok = False
            if bad_sep:
                print(f"  ❌ 백슬래시 경로 {len(bad_sep)}개 — Linux에서 깨진다")
                ok = False
            for need in ("script.py", "requirements.txt", "features.py"):
                if need not in names:
                    print(f"  ❌ {need} 누락")
                    ok = False
            z.extractall(work)
        print(f"  model/ 파일 {len(os.listdir(os.path.join(work, 'model')))}개")

        # --- 2. script.py가 import하는 모듈이 zip에 다 있나 ---
        src = open(os.path.join(work, "script.py"), encoding="utf-8").read()
        for mod in ("features", "cond", "league_rate"):
            if f"import {mod}" in src and f"{mod}.py" not in names:
                print(f"  ❌ script.py가 {mod}를 import하는데 zip에 없다 "
                      f"= ModuleNotFoundError = 제출 차감")
                ok = False

        # --- 3. 실제 규모 가짜 test 생성 (2024 행 → season 2025) ---
        print(f"[데이터] 가짜 test {N_TEST:,}행 생성...", flush=True)
        df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
        t = df[df["season"] == 2024].head(N_TEST).drop(columns=[TARGET]).copy()
        t["season"] = 2025
        # 🔴 asof_*_n 을 그 선수의 2024 총 투구수만큼 올린다.
        #    안 그러면 anchor(2024 말 누적)보다 작아서 dn = max(n1-n0, 0) = 0 이 되고
        #    **시즌내 분해(최대 축 +53.7)가 100% 폴백으로 떨어진다.**
        #    2026-08-25 감사에서 발견: 015 이후 모든 제출이 그 경로를 미검증으로 내보냈다.
        for who, key in (("pitcher", "pitcher_id"), ("batter", "batter_id")):
            tot = df[df["season"] == 2024].groupby(key).size()
            add = t[key].map(tot).fillna(0).values
            for c in (f"asof_{who}_n", f"asof_{who}_pitchmix_n"):
                if c in t.columns:
                    t[c] = t[c].values + add
        t[ID] = [f"TEST_{i:06d}" for i in range(len(t))]
        d = os.path.join(work, "data")
        os.makedirs(d)
        t.to_csv(os.path.join(d, "test.csv"), index=False, encoding="utf-8")
        pd.DataFrame({ID: t[ID], TARGET: 0.5}).to_csv(
            os.path.join(d, "sample_submission.csv"), index=False,
            encoding="utf-8")

        # --- 4. 실행 ---
        print("[실행] script.py ...", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "script.py"], cwd=work,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        el = time.time() - t0
        print((r.stdout or "").strip())
        if r.returncode != 0:
            print(f"  ❌ 종료코드 {r.returncode} = 제출오류 = 차감")
            print((r.stderr or "")[-3000:])
            return 1
        print(f"  추론 {el:.1f}초  (제한 600초, 여유 {600-el:.0f}초)")
        if el > 480:
            print("  ⚠️ 8분 초과 — 서버가 더 느릴 수 있다")
            ok = False

        # --- 4b. 시즌내 분해 경로가 실제로 돌았는지 ---
        anc = os.path.join(work, "model", "anchor.csv")
        if os.path.exists(anc):
            A = pd.read_csv(anc, encoding="utf-8")
            for who, key in (("pitcher", "pitcher_id"), ("batter", "batter_id")):
                n0 = A[A.who == who].set_index("id").n0
                dn = np.maximum(t[f"asof_{who}_n"].values
                                - t[key].map(n0).fillna(0).values, 0)
                zero = float((dn == 0).mean())
                mark = "❌" if zero > 0.5 else "✅"
                print(f"  {mark} 시즌내 분해 {who}: dn>0 인 행 {100*(1-zero):.1f}% "
                      f"(중앙 {np.median(dn):,.0f})")
                if zero > 0.5:
                    print(f"     🔴 절반 넘게 폴백이다 — 이 검증은 분해 경로를 안 태운다")
                    ok = False

            print("  ⚠️ asof_n을 인위적으로 올렸으므로 **예측 평균은 base rate 검사로"
                  " 쓸 수 없다**. 여기서 보는 것은 '분해 경로가 돌았나'와 결측·범위뿐이다.")

        # --- 5. 산출물 ---
        s = pd.read_csv(os.path.join(work, "output", "submission.csv"))
        p = s[TARGET].values
        print(f"[산출] {len(s):,}행  결측 {int(np.isnan(p).sum())}  "
              f"범위 {p.min():.4f}~{p.max():.4f}  평균 {p.mean():.4f}")
        if len(s) != len(t):
            print(f"  ❌ 행 수 불일치 {len(s):,} vs {len(t):,}")
            ok = False
        if list(s[ID]) != list(t[ID]):
            print("  ❌ row_id 순서/값 불일치")
            ok = False
        if np.isnan(p).any() or p.min() < 0 or p.max() > 1:
            print("  ❌ 결측 또는 범위 이탈")
            ok = False
        if len(np.unique(np.round(p, 6))) < 1000:
            print(f"  ❌ 예측이 사실상 상수 (고유값 {len(np.unique(p))})")
            ok = False
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\n" + ("✅ 통과 — 제출 가능" if ok else "❌ 실패 — 제출 금지"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
