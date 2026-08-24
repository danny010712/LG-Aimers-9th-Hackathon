"""trackman 투수 물리 프로필 — pitcher_map_seq.csv 기반.

문헌: 제구 예측 최강 물리 지표 = **릴리스포인트 변동성(특히 수평)**.
  - 릴리스포인트 변동성 ↔ 투구 성적 (PMC11608975)
  - 릴리스 각도 최대 영향 / 회전수 최소 (Frontiers 2020, J Sports Sci 2021)
구종마다 릴리스가 다르므로 **구종 내 변동성**을 구해 가중평균한다(레퍼토리 혼합 효과 제거).

시점 분리: cutoff 이하 시즌만 사용. 2024 검증엔 2023, 2025 제출엔 2024.
열은 최소로 (§3 용량 제약).
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd

def build(cutoff):
    m = pd.read_csv("pitcher_map_seq.csv")
    tm = pd.read_csv("data/trackman_history.csv",
                     usecols=["season", "pitcher_trackman_id", "pitch_type_group",
                              "rel_speed", "rel_height", "rel_side", "extension"])
    tm = tm[tm.season <= cutoff]
    tm = tm.merge(m[["pitcher_id", "trackman_id"]],
                  left_on="pitcher_trackman_id", right_on="trackman_id")

    g = tm.groupby(["pitcher_id", "pitch_type_group"])
    w = g.size().rename("w")
    within = g[["rel_side", "rel_height", "rel_speed"]].std()
    within = within.join(w)
    within = within[within.w >= 30]                    # 구종 표본 30개 미만은 sd가 못 믿는다

    def wavg(col):
        num = (within[col] * within.w).groupby(level=0).sum()
        den = within.w.groupby(level=0).sum()
        return num / den

    p = pd.DataFrame({
        "tm_relside_sd": wavg("rel_side"),
        "tm_relheight_sd": wavg("rel_height"),
        "tm_speed_sd": wavg("rel_speed"),
    })
    p["tm_extension"] = tm.groupby("pitcher_id").extension.mean()
    p["tm_n"] = tm.groupby("pitcher_id").size()
    return p.reset_index()

if __name__ == "__main__":
    for cut in (2023, 2024):
        p = build(cut)
        p.to_csv(f"tm_profile_le{cut}.csv", index=False)
        print(f"le{cut}: 투수 {len(p):,}명  "
              f"relside_sd 중앙 {p.tm_relside_sd.median():.4f} "
              f"[{p.tm_relside_sd.quantile(.05):.4f}, {p.tm_relside_sd.quantile(.95):.4f}]",
              flush=True)
