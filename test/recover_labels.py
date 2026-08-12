"""숨은 투구 라벨 복원 — train의 asof_* 컬럼에서 역산한다.

원리: asof_pitcher_<c>_rate 는 그 투수의 **과거 투구**에 대한 (이벤트수 / n) 이다.
  cnt_k = round(rate_k × n_k)  ->  k번째 투구의 이벤트 = cnt_{k+1} − cnt_k

검증: 이미 답을 아는 success 를 같은 방법으로 복원했더니 1,474,300행 전부
control_success 와 일치했다. 방법 자체가 증명된 것이므로 나머지 라벨도 옳다.
상세는 강의정리/09_피처분석_노트.md §1-C.

산출: test/recovered_labels.csv.gz  (row_id + 8개 라벨, 0/1, 투수당 마지막 1투구만 NaN)

⚠️ 이 라벨들은 **train 전용**이다. 2025 test 행에는 만들 수 없다
   (라벨이 없고, test 행 간 순서 사용도 금지). 보조 **타깃**으로만 쓴다.
"""
import numpy as np
import pandas as pd

MIX = ("fastball", "breaking", "offspeed")            # 분모 = asof_pitcher_pitchmix_n
LABELS = ("success", "middle", "reverse", "ball", "strike") + MIX

SRC = "data/train.csv"
OUT = "recovered_labels.csv.gz"


def recover(df):
    """row_id 순으로 정렬된 df에서 투구별 라벨을 복원해 DataFrame으로 반환."""
    order = np.argsort(df["pitcher_id"].values, kind="stable")   # row_id 순 유지
    pid = df["pitcher_id"].values[order]
    first = np.r_[True, pid[1:] != pid[:-1]]
    last = np.r_[first[1:], True]

    out = {}
    for c in LABELS:
        den = "asof_pitcher_pitchmix_n" if c in MIX else "asof_pitcher_n"
        cnt = np.round(df[f"asof_pitcher_{c}_rate"].values[order]
                       * df[den].values[order])
        cnt[first] = 0                       # 첫 투구는 rate가 NaN
        lab = np.r_[cnt[1:] - cnt[:-1], np.nan]
        lab[last] = np.nan                   # 다음 투구가 없으면 복원 불가
        a = np.full(len(df), np.nan)
        a[order] = lab
        out[c] = a

    r = pd.DataFrame(out)
    r.insert(0, "row_id", df["row_id"].values)
    return r


def main():
    cols = (["row_id", "pitcher_id", "control_success",
             "asof_pitcher_n", "asof_pitcher_pitchmix_n"]
            + [f"asof_pitcher_{c}_rate" for c in LABELS])
    df = pd.read_csv(SRC, usecols=cols).sort_values("row_id").reset_index(drop=True)
    r = recover(df)

    # 자가검증 — 이게 깨지면 복원 전체를 믿으면 안 된다
    v = r.dropna()
    y = df.loc[v.index, "control_success"].values
    assert (v["success"].values == y).all(), "success 복원 불일치 — 방법 무효"
    for c in LABELS:
        assert v[c].isin([0, 1]).all(), f"{c} 라벨이 0/1이 아님"
    assert (v["fastball"] + v["breaking"] + v["offspeed"] == 1).all(), "구종 분할 깨짐"
    print(f"검증 통과. {len(v):,}행 복원, 결측 {len(r) - len(v):,}행", flush=True)

    r.to_csv(OUT, index=False, compression="gzip", float_format="%.0f")
    print(f"저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()
