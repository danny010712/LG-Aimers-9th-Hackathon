# 강의 1. Tabular ML: From Classical Models to Foundation Models

> 강사: Hankook Lee (성균관대 Efficient Learning Lab)
> 구성: 6개 PDF(Intro / Classical / Deep / Representation / LLM / TabPFN) + 실습 노트북
> **해커톤 관련도 🔥🔥🔥 최상** — 스포츠 데이터는 tabular 성격이 강함. 이 강의가 대회 파이프라인의 뼈대.

**중요도 범례**: 🔥 해커톤 핵심 / ⭐ 중요 / 📎 참고

---

## 0. 한눈에 보는 실전 결론 (먼저 읽기) 🔥

해커톤에서 tabular 문제를 받으면 이 순서로 움직인다:

1. **EDA** — 결측/분포/클래스불균형/타깃상관 파악.
2. **누수(leakage) 차단** — 전처리는 반드시 **train에만 fit**, val/test엔 transform만. 시계열이면 시간순 split.
3. **베이스라인**: LightGBM/XGBoost/CatBoost (GBDT). tabular의 사실상 SOTA·1순위. 🔥
4. **평가지표를 문제에 맞게** 선택 (불균형 → PR-AUC, F1 / 회귀 → RMSE·MAE). CV로 신뢰도 확보.
5. **피처 엔지니어링** — 모델 교체보다 성능 향상이 큰 경우 많음. 🔥
6. **HPO** (Optuna) + **CV 기반** 튜닝.
7. **앙상블** — GBDT 3종 + (데이터 작으면) TabPFN 블렌딩. 🔥
8. 소규모(≤10K행)면 **TabPFN v2**를 강력한 카드로. 딥러닝(TabM 등)은 여력 있을 때.

> **핵심 격언**: "For tabular data, tree-based methods are often the *first* models to try." GBDT 먼저, 딥러닝은 나중.

---

## 1. Tabular 데이터의 특성 (왜 vision/NLP와 다른가) ⭐

| 특성 | 의미 | 해커톤 함의 |
|---|---|---|
| **이질적 피처(heterogeneous)** | 한 행에 수치·범주·이진·날짜·텍스트 혼재 | 타입별 전처리 필수 |
| **공간/순서 구조 없음** | 컬럼 순서 무의미 → permutation-invariant | CNN/RNN inductive bias 안 통함 |
| **소~중 규모** | 보통 수천 행 | data-efficient·과적합 회피 필요 → 트리 강함 🔥 |
| **결측 흔함** | 센서 실패, 미응답 | 결측 처리 전략 필수. 트리는 native 처리 |
| **노이즈·불완전 라벨** | 라벨이 나중에 수집 → **leakage 위험** | 미래 정보 피처 배제, 시간순 평가 🔥 |
| **클래스 불균형** | 소수 클래스 <1%도 | accuracy 함정 → PR-AUC/recall 🔥 |
| **도메인 다양·지식 제한** | 중요 상호작용 미지 | 데이터에서 관계 학습, spurious shortcut 주의 |

## 2. Tabular 태스크 종류 📎

- **예측(지도학습)**: 회귀 `y∈ℝ` / 분류 `y∈{1..K}` ← **해커톤 대부분 여기** 🔥
- **이상탐지**: 대개 비지도, 정상만 풍부 (사기·불량 탐지)
- **군집화**: k-means, GMM, DBSCAN (고객 세분화)
- **Table QA / 합성데이터 생성**: LLM 시대 관련 📎

---

## 3. ML 파이프라인 (해커톤 워크플로우) 🔥

`데이터수집 → EDA → 전처리 → 모델링 → 평가 → 배포·모니터링`

### 3-1. EDA
요약통계(mean/median/quartile/std), 결측 패턴, 분포(histogram/KDE), 피처-타깃 상관, 이상치, 클래스 불균형.

### 3-2. 전처리 — 누수 방지가 생명 🔥🔥
```python
# 틀림 (전체에 fit → test 정보 누수)
X = scaler.fit_transform(X)
X_train, X_test = split(X, ...)

# 맞음
X_train, X_test = split(X, ...)
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)   # transform만!
```

**결측치 처리**
- 삭제 (정보 손실) / 상수 채움("Unknown" → 결측 신호 보존) / 통계 대치(mean·median·mode) / 모델 기반(kNN 대치) / **트리 native 지원** (XGBoost·LightGBM은 결측 그대로 먹음) 🔥

**범주형 인코딩** ⭐
| 방법 | 아이디어 | 언제 |
|---|---|---|
| One-Hot | 카테고리마다 이진 컬럼 | 저cardinality, 순서 없음 |
| Ordinal | 순서 정수 매핑 | 자연 순서 있음(S<M<L) |
| **Target Encoding** | 카테고리를 평균 타깃값으로 | **고cardinality** 🔥 (누수 주의 → CV 안에서 fit) |
| Embedding | 학습 벡터 표현 | 딥러닝 |

**수치 변환** ⭐
- 스케일링: 표준화 `(x-μ)/σ` / min-max. **트리는 스케일 불변** → 불필요. 선형·딥러닝·kNN엔 필수.
- 분포 변환: **log**(오른쪽 꼬리: 소득·가격·카운트), **quantile**(랭크 매핑, 이상치 강건), binning.

### 3-3. 피처 엔지니어링 🔥🔥 (성능의 핵심)
> "Feature engineering can sometimes matter more than model choices."

| 패턴 | 예시 | 스포츠 응용 아이디어 |
|---|---|---|
| 집계(Aggregation) | 최근 30일 평균 구매액 | 최근 N경기 평균 득점·실점 |
| 비율(Ratio) | 부채/소득 | 슈팅 성공률, 점유율 |
| 시간차(Time diff) | 마지막 로그인 이후 일수 | 직전 경기 이후 휴식일 |
| 날짜 분해 | 월·요일·공휴일 플래그 | 홈/원정, 시즌 단계 |
| 상호작용(Interaction) | Price×Quantity | 선수효율×출전시간 |
| 도메인 피처 | BMI, 금융지표 | Elo 레이팅, 최근 폼(form) |

### 3-4. 평가 🔥🔥
- **Held-out test** + **Cross-validation**(데이터 작을 때 특히). 시계열은 **TimeSeriesSplit**.
- **지표 선택이 곧 순위**:
  - 회귀: MAE, MSE/RMSE, R² = 1 − Σ(yᵢ−ŷᵢ)²/Σ(yᵢ−ȳ)²
  - 이진분류: Accuracy, Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=P·R 조화평균
  - 임계값 무관: **AUC-ROC, AUC-PR**. **불균형이면 PR-AUC가 ROC보다 유의미** 🔥
- 문제별 지표: 사기탐지→PR-AUC·고정정밀도에서 recall / 의료→recall 최대화 / 스팸→precision 최대화.
- **가장 좋은 지표 = 오류의 실제 비용을 반영하는 지표.** 대회는 **주최측 지정 지표에 직접 최적화**. 🔥

**CV/HPO 함정** ⚠️
- Train-test 오염(전처리에 test 사용 금지).
- 단일 val split에 수천 조합 튜닝 → val 과적합. → nested CV, 조합 수 절제.

### 3-5. 해석가능성 📎→⭐
- **Permutation Importance**: 피처 셔플 후 성능 하락폭.
- **SHAP**: 예측 기여도 분해. 트리모델에서 특히 효율적. (피처 선택·디버깅에 유용)

### 3-6. 도구 🔥
NumPy/Pandas/scikit-learn, **Optuna**(HPO), PyTorch. 데이터: Kaggle, UCI, OpenML, AI Hub.

---

## 4. Classical ML (베이스라인의 왕) 🔥

> 트리 기반은 tabular에서 여전히 dominant. 단순·빠름·해석가능·이질적 피처 자연 처리.

### 4-1. 선형/로지스틱 회귀 ⭐ (해석가능 베이스라인)
- 선형회귀: MSE 최소화, closed-form `ŵ=(XᵀX)⁻¹Xᵀy`, 비용 `O(nd²+d³)`.
- **규제**: Ridge(L2, weight 부드럽게 축소) / Lasso(L1, 일부 정확히 0 → 피처선택). λ는 CV로.
- 로지스틱: sigmoid로 확률화, 손실=cross-entropy(convex), 다중분류=softmax.
- 한계: 비선형·상호작용 직접 포착 불가, 이상치·전처리 민감. **그래도 강한 sanity-check 베이스라인.**

### 4-2. kNN 📎
- "네 이웃 k개를 닮았다." 학습 없음(데이터=모델). 분류=다수결, 회귀=평균.
- **스케일링 필수**(큰 범위 피처가 거리 지배). 고차원에서 약함. k는 CV로(분류는 홀수).

### 4-3. 결정트리 ⭐
- 입력공간 재귀 분할 + 리프별 지역 예측. `f(x)=Σⱼ ŷⱼ·𝕀(x∈Rⱼ)`.
- 최적 분할은 NP-complete → **greedy**로 한 노드씩 성장. 분할 이득 `p(k)·Δ(k,d,t)` 최대화.
- 불순도: **Gini** `Σ_c ŷ(1−ŷ)` 또는 **entropy** `−Σ ŷ log ŷ`. 회귀는 분산.
- **복잡도 제어**(과적합↔과소적합): pre-pruning(`max_depth`, `min_samples_leaf`, `min_impurity_decrease`), post-pruning(cost-complexity).
- 장점: 규칙 해석 쉬움, 비선형·상호작용, 전처리·스케일 불필요. 단점: 과적합·불안정(작은 변화에 트리 급변).

### 4-4. 앙상블 🔥🔥 (대회 핵심)
서로 다른 모델(데이터/알고리즘/HP)의 예측 결합 → 일반화·강건성 ↑.

**Bagging (병렬)** — bootstrap aggregating
- n개 복원추출로 m개 데이터셋 → 독립학습 → 평균/다수결.
- 편향 유지, **분산 감소**. 각 부트스트랩은 ~63% 고유샘플(나머지 OOB는 검증용).
- 상관 있으면 분산감소 제한: `Var=(1/m)(1−ρ)σ²+ρσ²` → 상관 낮추는 게 관건.

**Random Forest** 🔥
- Bagging + **노드마다 랜덤 피처 부분집합**에서만 최적분할 → 트리 decorrelate.
- 단순한데 잘 됨. Kaggle 단골 베이스라인.

**Boosting (순차)** — 이전 오류 교정
- **GBDT (Gradient Boosted Decision Trees)** 🔥🔥 — 현대 tabular 라이브러리의 심장.
  1. 초기 `F₀(x)=ȳ`
  2. 잔차(=제곱오차의 gradient) `rᵢ=yᵢ−F_{m−1}(xᵢ)`
  3. 얕은 트리 `hₘ`를 잔차에 적합
  4. `Fₘ(x)=F_{m−1}(x)+η·hₘ(x)` (η=학습률)
- 얕은 트리를 많이 쌓아 강한 모델. **핵심 HP**: 트리 수, 학습률 η, 트리 깊이/리프 수, subsample. 튜닝 민감. ⚠️

**현대 GBDT 라이브러리** 🔥🔥
| 라이브러리 | 강점 | 실전 팁 |
|---|---|---|
| **XGBoost** | 견고·확장성 | `tree_method="hist"`, early stopping |
| **LightGBM** | **빠르고 메모리 효율** | 대용량 1순위, leaf-wise |
| **CatBoost** | **범주형 자동 처리** | 범주 많은 데이터에서 인코딩 없이 강력 |

→ 셋 다 GBDT 원리 공유, 실전 챌린지만 다름. **해커톤은 3종 다 돌려 앙상블**하는 게 정석.

### 4-5. Isolation Forest (이상탐지) 📎
- "이상치는 랜덤 분할로 쉽게 고립." 랜덤 피처·분할값으로 트리 → **평균 경로길이 짧으면 이상**. 라벨 불필요.

---

## 5. Deep Learning for Tabular ⭐ (여력 있을 때의 무기)

> **주의**: 트리가 여전히 강함(Grinsztajn 2022, NeurIPS): NN은 과도하게 매끄러운 해로 편향, 무의미 피처에 취약, 회전 불변성. **딥러닝은 표현학습·상호작용·전이·멀티모달이 필요할 때 매력적.**

### 5-1. MLP 계열 ⭐
- **Plain MLP / ResNet-MLP** (Gorishniy 2021): ResNet-MLP이 의외로 강한 베이스라인. **튜닝하면 단순 MLP도 경쟁력.** 범주형은 one-hot/임베딩.
- **수치 피처 임베딩** (Gorishniy 2022) 🔥: 각 수치 피처를 스칼라 대신 **벡터로 임베딩**(PLE=Piecewise Linear Encoding, Periodic activation). → 백본 선택만큼 임베딩 방식이 중요.
- **RealMLP** (Holzmüller 2024): 강건 전처리 + 임베딩 + 학습가능 스케일 + 개선된 init/정규화 + **메타튜닝 기본 HP**(데이터별 HPO 없이도 강함).
- **TabR / ModernNCA** 📎: kNN 검색을 NN에 결합(이웃 정보=강력한 inductive bias).
- **TabM** (Gorishniy ICLR 2025) 🔥: **파라미터 공유 앙상블**. 공유 W에 멤버별 저랭크 스케일(r,s) → 한 MLP 비용으로 앙상블 효과. 단일 MLP·일반 deep ensemble 능가. (실습에서 직접 구현)

### 5-2. Attention 계열 📎→⭐
- 컬럼=토큰, self-attention으로 **피처 상호작용** 학습(permutation-equivariant).
- **TabTransformer**(2020): 범주형만 토큰화. **FT-Transformer**(2021): Feature Tokenizer + [CLS], 수치는 선형투영·범주는 lookup → attention. MLP·GBDT 능가하는 표준 템플릿.
- **NPT/SAINT**: 샘플 간 attention(intersample) → 학습형 kNN. 단, **타깃 누수 주의**·비용 큼.
- 최신: T2G-Former(관계그래프), ExcelFormer, AMFormer(산술 상호작용).

## 6. Representation Learning (SSL) 📎

> 라벨 비쌈 + unlabeled 풍부 → 사전학습. tabular는 보편 augmentation이 없어 어려움.

- **재구성 기반**: VIME(피처 손상 후 복원 + mask 추정).
- **대조학습**: SubTab(컬럼 부분집합=뷰), SCARF(피처 손상), SAINT(CutMix/Mixup).
- **잠재공간/augmentation-free**: RaTab, T-JEPA(잠재공간 예측).
- **전이·few-shot**: STUNT(self-generated task), TransTab·CARTE(컬럼명·스키마 초월 전이).
- **이상탐지 SSL**: ICL, MCM, NPT-AD, Disent-AD, DRL — 각기 "정상"의 정의가 다름.

→ 해커톤에서 unlabeled 대량 제공 시 카드. 일반적으론 우선순위 낮음.

---

## 7. LLMs with Tabular Data 📎

> LLM은 **의미(semantic)·사전지식**을 tabular에 주입. 단, 행을 텍스트로 **직렬화** 필요, 수치·긴테이블·추론비용에 약함.

- **직접 예측**: LIFT, TabLLM, TABLET (행 직렬화 → zero/few-shot 분류). 실용성 낮음(비용·컨텍스트).
- **의미 컴포넌트로 활용** 🔥 (더 실용적): **CAAFE**(LLM이 피처엔지니어링 Python 코드 생성), FeatLLM, OCTree(결정트리 추론으로 피처규칙 최적화), DeLTa.
  → **"모든 행을 LLM에 묻지 말고, LLM으로 피처만 만들고 최종예측은 GBDT."** 비용↓·효율↑.
- **이상탐지**: AnoLLM(정상 likelihood), ReTabAD(메타데이터), AutoAnoEval.
- **파운데이션**: TabuLa-8B(Llama3-8B, 2.1B행 학습).

## 8. TabPFN — 새 패러다임 🔥 (소규모 데이터의 비밀병기)

> **Prior-Fitted Network (PFN)**: 합성 prior에서 뽑은 수많은 태스크로 사전학습 → **fitting 없이 in-context 예측**. 트랜스포머가 베이지안 사후분포를 학습.

- **TabPFN v1** (ICLR 2023): 행=토큰, 학습셋+테스트를 함께 attention. **<1K행·<100컬럼·<10클래스** 제약. 소규모에서 MLP·GBDT 능가 + 초고속.
- **TabPFN v2** (Nature 2025) 🔥: **셀=토큰**, 2D attention(피처·샘플 교대), 풍부한 합성 prior. **≤10K샘플·≤500피처**. 범주형·결측 자동. 소규모 tabular에서 매우 강력.
- **v3 / TabICL / TabDPT / TACO / MITRA / ConTextTab**: 확장성·실데이터 사전학습·의미정보 등 최신 변형. 📎

**실전**: 데이터가 **≤10K행이면 TabPFN v2를 반드시 시도**. 학습 사실상 0, GBDT와 앙상블 시 강력. 🔥

---

## 9. 실습 노트북 핵심 (실전 코드 템플릿) 🔥🔥

데이터: **UCI MAGIC Gamma Telescope** (이진분류, 10개 수치피처, 비선형 강함, 19,020행).
비교 모델: LR → DecisionTree → **XGBoost** → **TabM** → LLM(Qwen) → **TabPFN**.

**전처리 이원화** (트리 vs 선형/딥) 🔥:
```python
# 트리용: 원본 그대로 (스케일 불필요)
X_train_tree = X_train_raw
# 선형/딥러닝용: StandardScaler (train에만 fit!)
scaler = StandardScaler().fit(X_train_raw)
X_train_lin = scaler.transform(X_train_raw)
X_test_lin  = scaler.transform(X_test_raw)
```

**XGBoost + early stopping** (대회 베이스라인 표준) 🔥:
```python
from xgboost import XGBClassifier
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_tree, y_train, test_size=0.2, stratify=y_train, random_state=SEED)
xgb = XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    eval_metric="auc", early_stopping_rounds=50,
    tree_method="hist", random_state=SEED, n_jobs=-1)
xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
```

**TabPFN v2** (fitting 거의 없음, ≤10K로 subsample) 🔥:
```python
from tabpfn import TabPFNClassifier
tabpfn = TabPFNClassifier(device="cuda")
tabpfn.fit(X_tr_pfn, y_tr_pfn)          # 데이터를 메모리에 올리는 수준
y_score = tabpfn.predict_proba(X_test_tree)[:, 1]
```

**LLM 분류** = 행 직렬화 → few-shot 프롬프트 → 다음토큰에서 `gamma`/`hadron` logit 비교. 작은 모델(0.5B)은 약함 → 모델 키우면(3B) 개선.

**실험적 교훈** ⭐:
- **small-data(n=500)일수록 모델 간 차이 뚜렷** → **TabPFN이 소규모에서 특히 강함**.
- 데이터 크기(100/500/2000/15000)별로 돌리면 **data-efficiency 곡선** 확인 가능.
- AUC 기준 정렬로 모델 비교. GBDT·TabPFN이 상위권 경쟁.

---

## 10. 해커톤 체크리스트 (이 강의 압축) 🔥

- [ ] EDA: 결측·불균형·타깃상관·이상치·시간구조 확인
- [ ] Leakage 차단: 전처리 train-fit, 시계열 시간순 split
- [ ] 지표를 주최측 지정에 맞춰 CV로 검증
- [ ] 베이스라인: LightGBM/XGBoost/CatBoost
- [ ] 피처 엔지니어링(집계·비율·시간차·상호작용·도메인) — 최대 투자처
- [ ] 범주형: target encoding(고card, CV내 fit) / CatBoost
- [ ] Optuna로 HPO (CV 기반)
- [ ] ≤10K행이면 TabPFN v2 추가
- [ ] 앙상블/블렌딩 (GBDT 3종 + TabPFN)
- [ ] SHAP로 피처 검증·디버깅

---

> 💬 **Q&A 메모** — (질문 생기면 여기에 요약 기록)
