"""투수 고정 피처의 천장 측정 — trackman 재검토(09 §2-F)에서 나온 부산물.

베이스를 `sm0`(시즌시작 통산 스무딩) 1열로 두면 trackman 17열이 out-of-year
+120.7 BSS를 낸다. **실모델(013) 잔차로 다시 재면 −10.3이다.** 프로브 베이스가
약해 트리가 이미 다른 열로 잡던 몫을 이득으로 센 것 (CLAUDE.md §5).

핵심 산출: 013 이후 **투수레벨에 남은 참신호는 13 BSS**다 (통제없음 758 →
시즌시작 통산 676 → 013 13). 즉 **투수마다 상수인 피처는 종류 불문 천장 ~13**.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd, glob

PHYS = ["rel_speed","spin_rate","induced_vert_break","horz_break",
        "extension","rel_height","rel_side","zone_speed"]
m = pd.read_csv("pitcher_map_seq.csv")
RAW = pd.read_csv("data/trackman_history.csv",
                  usecols=["season","pitcher_trackman_id","pitch_type_group"]+PHYS)
RAW = RAW.merge(m[["pitcher_id","trackman_id"]],
                left_on="pitcher_trackman_id", right_on="trackman_id")
tm = RAW[RAW.season<=2023]
g = tm.groupby(["pitcher_id","pitch_type_group"]); w_=g.size().rename("w")
sd=g[PHYS].std().join(w_); sd=sd[sd.w>=30]; mu=g[PHYS].mean().join(w_); mu=mu[mu.w>=30]
f=lambda fr,c:(fr[c]*fr.w).groupby(level=0).sum()/fr.w.groupby(level=0).sum()
P=pd.DataFrame({f"sd_{c}":f(sd,c) for c in PHYS})
for c in PHYS: P[f"mu_{c}"]=f(mu,c)
P["mix"]=tm.pitch_type_group.eq(tm.pitch_type_group.mode()[0]).groupby(tm.pitcher_id).mean()
P=P.reset_index(); FE=[c for c in P.columns if c!="pitcher_id"]

df = pd.read_csv("data/train.csv", usecols=["row_id","season","pitcher_id","control_success"])
va = (df.season==2024).values
L = pd.read_csv("recovered_labels.csv.gz")
have = df[["row_id"]].merge(L, on="row_id", how="left")["middle"].notna().values[va]
pred = np.mean([np.load(p) for p in sorted(glob.glob("artifacts/auxpred_ins_013_backup/*.npy"))],axis=0)
d = df[va][have].copy(); d["pred"]=pred
print(f"013 검증예측 {len(d):,}행  Brier={np.mean((d.pred-d.control_success)**2):.6f}"
      f"  BSS={100000*(1-np.mean((d.pred-d.control_success)**2)/(d.control_success.mean()*(1-d.control_success.mean()))):.1f}")

G = d.groupby("pitcher_id").agg(n=("control_success","size"),
                                act=("control_success","mean"), pr=("pred","mean")).reset_index()
z = G[G.n>=200].merge(P, on="pitcher_id").dropna(subset=FE)
w = z["n"].values.astype(float); res = (z["act"]-z["pr"]).values
binom = float((z["act"]*(1-z["act"])/z["n"]).mean())
v0 = (w*res**2).sum()/w.sum()
print(f"\n투수 {len(z)}명 (2024 행의 {w.sum()/len(d)*100:.0f}%)")
print(f"013 잔차분산 {v0:.6f}  −이항노이즈 {binom:.6f} = 참신호 {max(v0-binom,0):.6f}"
      f"  → 투수레벨 남은 여지 {max(v0-binom,0)/0.25*100000:.0f} BSS")

rng=np.random.default_rng(0); fold=rng.integers(0,5,len(z))
def cv(cols, y):
    oof=np.zeros(len(z))
    for k in range(5):
        A=np.c_[np.ones((fold!=k).sum()), z.loc[fold!=k,cols].values] if cols else np.ones(((fold!=k).sum(),1))
        ww=w[fold!=k]
        b=np.linalg.lstsq(A*np.sqrt(ww)[:,None], y[fold!=k]*np.sqrt(ww), rcond=None)[0]
        B=np.c_[np.ones((fold==k).sum()), z.loc[fold==k,cols].values] if cols else np.ones(((fold==k).sum(),1))
        oof[fold==k]=B@b
    r=y-oof; return (w*r**2).sum()/w.sum()
b0=cv([],res)
print(f"\n5-fold CV (투수 단위, 013 잔차 대상)  절편만 {b0:.6f}")
for tag,cols in [("sd 8열",[f"sd_{c}" for c in PHYS]),("mu 8열",[f"mu_{c}" for c in PHYS]),
                 ("전체 17열",FE),("sd_rel_side 단독",["sd_rel_side"])]:
    v=cv(cols,res)
    print(f"  {tag:<16} {v:.6f}   Δ={(b0-v)/0.25*100000:+7.1f} BSS (OOF)")
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd, glob

PHYS = ["rel_speed","spin_rate","induced_vert_break","horz_break",
        "extension","rel_height","rel_side","zone_speed"]
m = pd.read_csv("pitcher_map_seq.csv")
RAW = pd.read_csv("data/trackman_history.csv",
                  usecols=["season","pitcher_trackman_id","pitch_type_group"]+PHYS)
RAW = RAW.merge(m[["pitcher_id","trackman_id"]],
                left_on="pitcher_trackman_id", right_on="trackman_id")
tm = RAW[RAW.season<=2023]
g = tm.groupby(["pitcher_id","pitch_type_group"]); w_=g.size().rename("w")
sd=g[PHYS].std().join(w_); sd=sd[sd.w>=30]; mu=g[PHYS].mean().join(w_); mu=mu[mu.w>=30]
f=lambda fr,c:(fr[c]*fr.w).groupby(level=0).sum()/fr.w.groupby(level=0).sum()
P=pd.DataFrame({f"sd_{c}":f(sd,c) for c in PHYS})
for c in PHYS: P[f"mu_{c}"]=f(mu,c)
P["mix"]=tm.pitch_type_group.eq(tm.pitch_type_group.mode()[0]).groupby(tm.pitcher_id).mean()
P=P.reset_index(); FE=[c for c in P.columns if c!="pitcher_id"]

df = pd.read_csv("data/train.csv", usecols=["row_id","season","pitcher_id","control_success"])
va = (df.season==2024).values
L = pd.read_csv("recovered_labels.csv.gz")
have = df[["row_id"]].merge(L, on="row_id", how="left")["middle"].notna().values[va]
pred = np.mean([np.load(p) for p in sorted(glob.glob("artifacts/auxpred_ins_013_backup/*.npy"))],axis=0)
d = df[va][have].copy(); d["pred"]=pred
print(f"013 검증예측 {len(d):,}행  Brier={np.mean((d.pred-d.control_success)**2):.6f}"
      f"  BSS={100000*(1-np.mean((d.pred-d.control_success)**2)/(d.control_success.mean()*(1-d.control_success.mean()))):.1f}")

G = d.groupby("pitcher_id").agg(n=("control_success","size"),
                                act=("control_success","mean"), pr=("pred","mean")).reset_index()
z = G[G.n>=200].merge(P, on="pitcher_id").dropna(subset=FE)
w = z["n"].values.astype(float); res = (z["act"]-z["pr"]).values
binom = float((z["act"]*(1-z["act"])/z["n"]).mean())
v0 = (w*res**2).sum()/w.sum()
print(f"\n투수 {len(z)}명 (2024 행의 {w.sum()/len(d)*100:.0f}%)")
print(f"013 잔차분산 {v0:.6f}  −이항노이즈 {binom:.6f} = 참신호 {max(v0-binom,0):.6f}"
      f"  → 투수레벨 남은 여지 {max(v0-binom,0)/0.25*100000:.0f} BSS")

rng=np.random.default_rng(0); fold=rng.integers(0,5,len(z))
def cv(cols, y):
    oof=np.zeros(len(z))
    for k in range(5):
        A=np.c_[np.ones((fold!=k).sum()), z.loc[fold!=k,cols].values] if cols else np.ones(((fold!=k).sum(),1))
        ww=w[fold!=k]
        b=np.linalg.lstsq(A*np.sqrt(ww)[:,None], y[fold!=k]*np.sqrt(ww), rcond=None)[0]
        B=np.c_[np.ones((fold==k).sum()), z.loc[fold==k,cols].values] if cols else np.ones(((fold==k).sum(),1))
        oof[fold==k]=B@b
    r=y-oof; return (w*r**2).sum()/w.sum()
b0=cv([],res)
print(f"\n5-fold CV (투수 단위, 013 잔차 대상)  절편만 {b0:.6f}")
for tag,cols in [("sd 8열",[f"sd_{c}" for c in PHYS]),("mu 8열",[f"mu_{c}" for c in PHYS]),
                 ("전체 17열",FE),("sd_rel_side 단독",["sd_rel_side"])]:
    v=cv(cols,res)
    print(f"  {tag:<16} {v:.6f}   Δ={(b0-v)/0.25*100000:+7.1f} BSS (OOF)")
