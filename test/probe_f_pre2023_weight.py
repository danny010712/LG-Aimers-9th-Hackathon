"""pre-2023 F 행에 CatBoost 표본가중치(0.3, 09 §1-I dilution 비율)를 주는 축.
행 삭제(3종, closed.tsv)와 다르게 정보를 지우지 않고 신뢰도만 낮춘다. 044 기준선
(COND_ONLY=[ph], inseason on, p_matchup 제외, 63열) 위에서 단일변수로 검증.

결과 (2026-08-30):
  val_2024(원래시드[42,7,2024]) Δ=+10.1 → 실제 LB **-4.07**(213_shift_base044_fw03=1059.174
  vs 043_shift_condph=1063.239, 동일구성에서 가중치만 뺀 버전). **로컬 대폭 개선이 LB 악화로 감.**
  사후 진단: 시드[99,1,777]로는 Δ=+3.7(노이즈 안), 연도전이 2021→2022는 Δ=-24.9(역전).
  09 §2-O(월당소화량, 진폭 10배 불안정)와 같은 실패 모양 — 제출 전에 이미 의심스러웠고
  실측으로 확정 기각됐다. → closed.tsv, audit.py PAIRS 참고.

이 스크립트는 3시드x3전이 배포조건 검증 재현용 (044 기준, cond_ph 포함).
"""
import io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
sys.path.insert(0, 'common')
from features import engineer, build_anchor, rate_priors, CAT_COLS
import cond

WEIGHT = 0.3
SEEDS3 = [42, 7, 2024]


def bss(p, yy):
    r = yy.mean()
    return max(0.0, 100000 * (1 - np.mean((p - yy) ** 2) / (r * (1 - r))))


def main():
    t0 = time.time()
    df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
    y = df['control_success'].astype(int).values
    anchor = build_anchor(df)
    is_pre_f = ((df['season'] <= 2022) & (df['game_type'] == 'F')).values
    dfl_full = df.merge(pd.read_csv('recovered_labels.csv.gz'), on='row_id', how='left')

    def transfer(tr_end, te_year):
        trm = (df.season <= tr_end).values
        vam = (df.season == te_year).values
        gm = float(y[trm].mean())
        priors = rate_priors(df[trm])
        X = engineer(df.drop(columns=['row_id', 'control_success']), gm, anchor=anchor, priors=priors)
        X = X.drop(columns=['p_matchup'])
        C = cond.build_training_columns(dfl_full)
        for c in ['cond_ph', 'cond_ph_dev']:
            X[c] = C[c].values
        for c in CAT_COLS:
            X[c] = X[c].astype(str)
        ci = [X.columns.get_loc(c) for c in CAT_COLS]
        w = np.where(is_pre_f, WEIGHT, 1.0)

        def fit(use_weight):
            ps = []
            for sd in SEEDS3:
                p_tr = Pool(X[trm], y[trm], cat_features=ci, weight=(w[trm] if use_weight else None))
                m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, thread_count=-1,
                                       verbose=0, eval_metric='Logloss', early_stopping_rounds=100,
                                       random_seed=sd).fit(
                    p_tr, eval_set=Pool(X[vam], y[vam], cat_features=ci), use_best_model=True)
                ps.append(m.predict_proba(Pool(X[vam], cat_features=ci))[:, 1])
            return bss(np.mean(ps, axis=0), y[vam])

        a = fit(False)
        b = fit(True)
        print(f' {tr_end}->{te_year}: 기준(3시드)={a:.1f}  +F가중치{WEIGHT}(3시드)={b:.1f}  Δ={b-a:+.1f}', flush=True)
        return b - a

    d1 = transfer(2023, 2024)
    d2 = transfer(2022, 2023)
    d3 = transfer(2021, 2022)
    print(f"\n부호: {['+' if dd > 0 else '-' for dd in (d1, d2, d3)]}   소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
