"""09 §1-J 관측(pre-2023 F 이력 비중이 클수록 asof_pitcher_success_rate가 부풀려짐,
편향 최대 +.076)을 실제로 교정 피처로 써봤을 때 신호가 있는지 검증.

f_pre2023_share = 그 투수의 이 투구 이전 pre-2023(season<=2022) F 투구 수 / asof_pitcher_n
(asof_pitcher_n 오름차순 = 커리어 순서, build_anchor()와 동일 관례. test에서는 정적 상수.)

결과 (5시드 x 3전이, 2026-08-30):
 2023->2024(주기준) Δ=-1.4   2022->2023(죽은fold) Δ=-13.0   2021->2022(참고) Δ=+12.3
 주기준 fold가 노이즈(σ≈9.3, 5시드 SE≈5.9) 안. 참고 fold 둘도 부호가 갈림(role_ppg와 같은 패턴).
 → 신호 없음, 행 삭제(3종)와는 다른 접근이었으나 역시 기각.
"""
import io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
sys.path.insert(0, 'common')
from features import engineer, build_anchor, rate_priors, CAT_COLS

t0 = time.time()
df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y = df['control_success'].astype(int).values
anchor = build_anchor(df)


def bss(p, yy):
    r = yy.mean()
    return max(0.0, 100000 * (1 - np.mean((p - yy) ** 2) / (r * (1 - r))))


d = df.sort_values(['pitcher_id', 'asof_pitcher_n']).reset_index()
isf = ((d['season'] <= 2022) & (d['game_type'] == 'F')).astype(int)
d['_isf'] = isf.values
cs = d.groupby('pitcher_id')['_isf'].cumsum()
d['f_pre2023_n_before'] = (cs - d['_isf']).values
d['f_pre2023_share'] = np.where(d['asof_pitcher_n'] > 0,
                                 d['f_pre2023_n_before'] / d['asof_pitcher_n'].clip(lower=1), 0.0)
d['f_pre2023_share'] = d['f_pre2023_share'].clip(0, 1)
d = d.sort_values('index').reset_index(drop=True)
f_share_col = d['f_pre2023_share'].values

SEEDS5 = [42, 7, 2024, 99, 1]


def transfer(tr_end, te_year):
    trm = (df.season <= tr_end).values
    vam = (df.season == te_year).values
    gm = float(y[trm].mean())
    priors = rate_priors(df[trm])
    X = engineer(df.drop(columns=['row_id', 'control_success']), gm, anchor=anchor, priors=priors)
    X = X.drop(columns=['p_matchup'])
    X_new = X.copy()
    X_new['f_pre2023_share'] = f_share_col
    for X_ in (X, X_new):
        for c in CAT_COLS:
            X_[c] = X_[c].astype(str)
    ci = [X.columns.get_loc(c) for c in CAT_COLS]
    ci2 = [X_new.columns.get_loc(c) for c in CAT_COLS]

    def fit(XX, ci_):
        ps = []
        for sd in SEEDS5:
            m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, thread_count=-1,
                                   verbose=0, eval_metric='Logloss', early_stopping_rounds=100,
                                   random_seed=sd).fit(
                Pool(XX[trm], y[trm], cat_features=ci_), eval_set=Pool(XX[vam], y[vam], cat_features=ci_),
                use_best_model=True)
            ps.append(m.predict_proba(Pool(XX[vam], cat_features=ci_))[:, 1])
        return bss(np.mean(ps, axis=0), y[vam]), ps

    a, pa = fit(X, ci)
    b, pb = fit(X_new, ci2)
    print(f' {tr_end}->{te_year}: 기준(5시드)={a:.1f}  +f_pre2023_share(5시드)={b:.1f}  Δ={b-a:+.1f}', flush=True)
    return b - a


if __name__ == '__main__':
    d1 = transfer(2023, 2024)
    d2 = transfer(2022, 2023)
    d3 = transfer(2021, 2022)
    print(f'\n부호: {["+" if dd>0 else "-" for dd in (d1,d2,d3)]}   소요 {time.time()-t0:.0f}s', flush=True)
