"""trackman ↔ train 투수 ID 대조표 — 게임 시퀀스 정렬 방식.

기존 방식(08 §3-G): 투수별 (시즌×월×요일) 투구수 지문 + 헝가리안 → **정확도 46%**
이 방식: `row_id`가 시간순(09 §1-B)이므로 train에서 경기 경계·투구 순서를 복원할 수 있다.
  게임 서명 = 첫 N투구의 (이닝, 초/말, 볼, 스트라이크, 아웃) 시퀀스
  → 두 경기가 같은 서명을 가질 확률은 사실상 0. 지문이 아니라 식별자다.
  게임이 짝지어지면 투구를 나란히 놓고 투수 ID를 직접 투표시킨다.

출력: pitcher_map_seq.csv  (pitcher_id, trackman_id, purity, n_pitch)
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from collections import Counter, defaultdict
import numpy as np
import pandas as pd

SIG_N = 40          # 서명에 쓸 투구 수
MIN_PITCH = 30      # 투표 집계 최소 투구 수

def sig(inn, tb, b, s, o, n=SIG_N):
    return "".join(f"{a}{c}{d}{e}{f}" for a, c, d, e, f
                   in zip(inn[:n], tb[:n], b[:n], s[:n], o[:n]))

def main():
    tm = pd.read_csv("data/trackman_history.csv",
                     usecols=["season", "game_month", "game_dayofweek", "trackman_game_id",
                              "pitch_no", "inning", "top_bottom", "balls_before",
                              "strikes_before", "outs_before", "pitcher_trackman_id"])
    tm["tb"] = tm.top_bottom.str[0]
    tm = tm.sort_values(["trackman_game_id", "pitch_no"])

    tr = pd.read_csv("data/train.csv",
                     usecols=["row_id", "season", "game_month", "game_dayofweek", "inning",
                              "top_bottom", "balls_before", "strikes_before", "outs_before",
                              "pitcher_team_id", "batter_team_id", "pitcher_id"])
    pair = (np.minimum(tr.pitcher_team_id, tr.batter_team_id) * 100
            + np.maximum(tr.pitcher_team_id, tr.batter_team_id))
    tr["gid"] = ((tr.inning.diff() < 0) | (pair.diff() != 0)
                 | (tr.season.diff() != 0)).cumsum()
    print(f"train 추정 게임 {tr.gid.nunique():,} / trackman 게임 {tm.trackman_game_id.nunique():,}",
          flush=True)

    idx, store = defaultdict(list), {}
    for gid, z in tm.groupby("trackman_game_id"):
        key = (z.season.iloc[0], z.game_month.iloc[0], z.game_dayofweek.iloc[0])
        idx[key].append((gid, sig(z.inning.values, z.tb.values, z.balls_before.values,
                                  z.strikes_before.values, z.outs_before.values)))
        store[gid] = z.pitcher_trackman_id.values
    print(f"trackman 서명 인덱스 완료 (키 {len(idx)})", flush=True)

    votes = defaultdict(Counter)
    matched = ambiguous = miss = 0
    for gid, z in tr.groupby("gid"):
        key = (z.season.iloc[0], z.game_month.iloc[0], z.game_dayofweek.iloc[0])
        s = sig(z.inning.values, z.top_bottom.values, z.balls_before.values,
                z.strikes_before.values, z.outs_before.values)
        hit = [g for g, ss in idx.get(key, []) if ss == s]
        if len(hit) != 1:
            ambiguous += len(hit) > 1
            miss += len(hit) == 0
            continue
        matched += 1
        tids = store[hit[0]]
        k = min(len(z), len(tids))
        for pid, tid in zip(z.pitcher_id.values[:k], tids[:k]):
            votes[pid][tid] += 1

    print(f"게임 매칭: 유일 {matched:,} / 중복 {ambiguous:,} / 불일치 {miss:,} "
          f"({matched/tr.gid.nunique()*100:.1f}%)", flush=True)

    rows = []
    for pid, c in votes.items():
        tid, top = c.most_common(1)[0]
        n = sum(c.values())
        if n >= MIN_PITCH:
            rows.append(dict(pitcher_id=pid, trackman_id=tid, purity=top / n, n_pitch=n))
    m = pd.DataFrame(rows).sort_values("n_pitch", ascending=False)

    # 1:1 강제 — 같은 trackman_id를 두 투수가 주장하면 투구 수가 많은 쪽이 가져간다
    m["dup"] = m.trackman_id.duplicated(keep="first")
    print(f"\n투수 {len(m):,}명 (train 전체 {tr.pitcher_id.nunique():,}명 중 "
          f"{len(m)/tr.pitcher_id.nunique()*100:.1f}%)")
    print(f"  순도 평균 {m.purity.mean():.4f}  중앙 {m.purity.median():.4f}")
    for t in (0.95, 0.90, 0.80):
        print(f"  순도 >{t:.2f} : {(m.purity > t).mean()*100:.1f}%")
    print(f"  trackman_id 충돌 {int(m.dup.sum())}건")
    print(f"  매핑된 투수의 train 투구 커버리지 = "
          f"{tr.pitcher_id.isin(m.pitcher_id).mean()*100:.1f}%")

    m[~m.dup].drop(columns="dup").to_csv("pitcher_map_seq.csv", index=False)
    print("\n저장: pitcher_map_seq.csv", flush=True)

if __name__ == "__main__":
    main()
