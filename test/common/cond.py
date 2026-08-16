"""조건부 개인기록 — 학습(train_local.py)과 추론(script.py)이 공유한다.

asof_*는 '이 투수의 전체 성공률'만 준다. 여기서는 '이 투수가 이 상황에서'를 만든다.

⚠️ 시점 규칙 (leak 방지):
  학습 행: 시즌 S의 행에는 **S 미만 시즌**으로 만든 표를 붙인다.
  추론 행: 2025 test에는 **train 전체(2019~2024)** 로 만든 표를 붙인다.
  둘 다 "그 행보다 앞선 시즌 전부"라는 같은 규칙이다.

표본이 적은 칸은 그 투수(타자)의 전체 성공률 쪽으로 스무딩한다.
검증: 2024 홀드아웃 780.0 -> 803.9 (08 문서 §3-K).
"""
import numpy as np
import pandas as pd

TARGET = "control_success"
M = 50                                  # 스무딩 강도 (표본 M개에서 반반)

# (이름, 그룹키, 사전분포키)
SPECS = [
    ("pc", ["pitcher_id", "count_state"], "pitcher_id"),
    ("ph", ["pitcher_id", "batter_hand"], "pitcher_id"),
    ("bc", ["batter_id", "count_state"], "batter_id"),
    ("pi", ["pitcher_id", "xinn"], "pitcher_id"),
]
COND_COLS = ["cond_" + n for n, _, _ in SPECS] + ["cond_pc_dev", "cond_ph_dev"]


def add_keys(d):
    """표 조회에 쓰는 파생 키. train/test 양쪽에서 동일하게 만든다."""
    d = d.copy()
    d["count_state"] = (d["balls_before"].astype(str) + "-"
                        + d["strikes_before"].astype(str))
    d["xinn"] = np.clip(d["inning"], 1, 9)
    return d


def build_tables(hist):
    """hist(타깃 포함 과거 행)에서 SPECS별 스무딩 성공률 표를 만든다."""
    hist = add_keys(hist)
    gm = hist[TARGET].mean()
    tables = {}
    for name, keys, prior in SPECS:
        g = hist.groupby(keys)[TARGET].agg(["sum", "count"])
        pr = hist.groupby(prior)[TARGET].mean().rename("prior")
        g = g.join(pr, on=prior)
        g["prior"] = g["prior"].fillna(gm)
        v = (g["sum"] + M * g["prior"]) / (g["count"] + M)
        tables[name] = v.rename("cond_" + name).reset_index()
    return tables


def apply_tables(d, tables):
    """표를 조회해 cond_* 열을 붙인다. 없는 칸은 NaN."""
    d = add_keys(d)
    for name, keys, _ in SPECS:
        t = tables[name]
        d = d.merge(t, on=keys, how="left")
    d["cond_pc_dev"] = d["cond_pc"] - d["asof_pitcher_success_rate"]
    d["cond_ph_dev"] = d["cond_ph"] - d["asof_pitcher_success_rate"]
    return d.drop(columns=["count_state_x"], errors="ignore")


def build_training_columns(df):
    """학습용: 시즌마다 '그 이전 시즌들'로 만든 표를 적용한 cond_* 열만 반환."""
    out = pd.DataFrame(index=df.index, columns=COND_COLS, dtype=float)
    for s in sorted(df["season"].unique()):
        hist = df[df["season"] < s]
        if len(hist) == 0:
            continue                     # 첫 시즌은 참조할 과거가 없다 -> NaN
        tables = build_tables(hist)
        m = (df["season"] == s).values
        got = apply_tables(df.loc[m], tables)
        for c in COND_COLS:
            out.loc[m, c] = got[c].values
    return out
