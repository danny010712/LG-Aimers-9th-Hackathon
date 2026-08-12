# 강의 4. Optimization & Decision-Focused Learning / Time-Series Analysis

> 강사: Yongjae Lee (UNIST 산업공학 / Financial Engineering Lab)
> 구성: 1~3강 Optimization & DFL + 4~6강 Time-Series
> 성격: **해커톤 관련도 🔥🔥 상** — 특히 **Time-Series 파트가 스포츠 시계열(경기 흐름·시즌 추세)에 직결**. Optimization은 배경, DFL은 최신 연구.

**중요도 범례**: 🔥 해커톤 핵심 / ⭐ 중요 / 📎 참고

---

## 0. 실전 압축 요약 🔥

- **시계열 = i.i.d 아님.** 인접 관측이 관계됨 → 일반 통계기법 바로 못 씀. **시간 순서·정상성(stationarity)·계절성 고려 필수.** 🔥
- **정상성 있는 데이터가 예측 잘 됨** (한 입력→한 출력 대응이 안정적).
- 시계열 모델 계보: **Model-driven(ARIMA/GARCH)** vs **Data-driven(RF/NN)**. 데이터 복잡·비선형이면 data-driven. 🔥
- 딥 시계열: **RNN → LSTM(장기의존) → Transformer(TFT/Informer/Autoformer)**. 병렬화로 대규모 학습. 🔥
- **검증은 반드시 시간순 split** (TimeSeriesSplit). 미래→과거 누수 금지. 🔥
- Optimization: **볼록이면 봉우리 하나 → 전역해 보장.** LP⊂QP⊂QCQP⊂SOCP⊂SDP 위계. 📎
- **DFL(Decision-Focused Learning)**: "예측정확도 ≠ 최적결정" → 결정 품질로 학습. 최신 연구 주제. 📎

---

# Part A. Optimization & Decision-Focused Learning

## A-1. 최적화란 ⭐
- 제약하 최선해 찾기: `min f₀(x) s.t. fᵢ(x)≤bᵢ`. 변수=결정변수, f₀=목적함수.
- 예: 포트폴리오 최적화(위험 최소), 회로 설계, **데이터 피팅/ML(파라미터=변수, 오차=목적)**.
- **ML vs 최적화**: 최적화=결정·결과의 수학적 모델링, ML=데이터로 매핑 학습. 단 "**학습은 곧 최적화를 포함**." RL·대규모 샘플링 최적화로 수렴 중.

## A-2. 왜 볼록 최적화? 🔥 (핵심 직관)
- 일반 최적화 = 안대 쓰고 산 오르기(봉우리 여럿, 최고봉 모름) → 어렵다.
- **볼록 최적화 = 봉우리 하나** → 매 스텝 더 높은 곳 가면 결국 정상. **지역최적=전역최적.**
- 위계: `쉬움 [LP] ⊂ [볼록 = 효율적으로 풂] ⊂ [비볼록 = 어려움]`.
- "쉬운 문제와 어려운 문제를 가르는 건 **선형성이 아니라 볼록성**."

## A-3. 효율적으로 풀리는 문제 클래스 ⭐
| 클래스 | 형태 | 특징 |
|---|---|---|
| **Least-squares** | `min ‖Ax−b‖²` | 해석해 `x*=(AᵀA)⁻¹Aᵀb`, 성숙기술. 다항회귀도 이 형태 |
| **LP** (선형계획) | `min cᵀx s.t. Gx⪯h, Ax=b` | feasible=polyhedron. diet/piecewise-linear |
| **QP** (이차계획) | `min ½xᵀPx+qᵀx` | **Markowitz 포트폴리오**(`min wᵀΣw−λμᵀw`) |
| **QCQP** | 이차 목적+이차 제약 | Sharpe ratio 최대화. LP⊂QP⊂QCQP |
| **SOCP** | 2차 원뿔 제약 | **Robust LP**(불확실성: 결정론/확률 모델) |
| **SDP** | LMI(선형행렬부등식) | LP의 행렬 버전 |

- **위계**: LP ⊂ QP ⊂ QCQP ⊂ SOCP ⊂ SDP ⊂ CP(cone program).
- **교훈** 🔥: "**Formulation이 중요**." 같은 문제도 정식화(LS/WLS/LP/convex)에 따라 정확도·난이도 천차만별. 훈련 안 된 직관은 위험(쉬운 문제가 어려워 보이거나 그 반대).
- 볼록 판정: 미분가능 최적성 `∇f₀(x)ᵀ(y−x)≥0 ∀feasible y`.

## A-4. Decision-Focused Learning (DFL) 📎 (최신 연구)
- **문제의식**: 보통 **Predict-then-optimize** (ML 예측 → 최적화 결정).
  - **Issue 1**: 예측 오차 → garbage in garbage out.
  - **Issue 2** 🔥: **"최선의 예측 ≠ 최적의 결정."** MSE 최소 예측이 결정 품질을 반영 못 함. (예: 재고 낮은 시설은 수요>재고라 대응 필요한데 MSE모델은 이를 못 담음.)
- **DFL 아이디어**: 예측손실 `ℒ_pred`(예: `‖y−ŷ‖²`) 대신 **결정손실 `ℒ_dec = c(a*(ŷ), y)`** 로 학습. 즉 예측이 유도한 결정의 실제 비용을 최소화.
- **난점**: `da*(y)/dy`가 미분 불가/비유일 → **surrogate(대리목적)** 사용: SPO+, LODL, 국소근사, Tangent-Space Projection(PEAR).
- **주의**: 결정품질만 과최적화하면 예측품질 붕괴 → 다른 downstream엔 무용 → **규제항 추가**.
- 응용: 포트폴리오(Mean-Variance, GMV 공분산 추정), LLM 통합 예측.

---

# Part B. Time-Series Analysis 🔥🔥 (스포츠 데이터 핵심)

## B-1. 시계열의 본질 🔥
- **시계열 = 여러 시점 관측 데이터.** 기존 통계는 **i.i.d 가정** → 시계열은 인접 관측에 **관계성** 있어 바로 적용 어려움.
- 탐색 질문 예: 전해↔이듬해 관계? (강우량=거의 무관 / 토끼 개체수=관계 있음 / 오일필터 판매=**계절성(seasonality)** 있음).
- **핵심 개념 — 정상성(Stationarity)** 🔥: 한 입력에 한 출력이 안정적으로 대응 = 함수 성질 유지 → **예측 잘 됨**. 비정상(non-stationary)이면 예측 어려움 → 차분·변환으로 정상화 시도.

## B-2. Model-driven vs Data-driven 🔥
| | Model-driven | Data-driven |
|---|---|---|
| 예 | **ARIMA, GARCH** | **Random Forest, Neural Net** |
| 수식/분포 가정 | 있음 (틀 고정) | 없음 |
| 장점 | 도메인지식 활용·일반화 쉬움 | 복잡·비선형 관계 반영 |
| 단점 | 복잡한 환경 부적합 | 많은 데이터 필요·해석 어려움 |

→ **스포츠 데이터가 비선형·다변수면 data-driven(트리/딥)**, 단순 추세·계절성이면 통계모델도 강함. 둘 다 시도·앙상블.

## B-3. 딥러닝 시계열 모델 ⭐→🔥

### RNN 계열
- **RNN**: 순환 구조로 hidden state 업데이트. **Vanishing gradient** 한계 — 뒤로 갈수록 초기 입력 영향 소실(초기값 잊음). ⚠️
- **LSTM** 🔥: gate로 장기의존성 유지 → RNN 한계 보완. 시계열 딥러닝의 기본기.
- **Neural ODE**: hidden state를 연속으로 → **불규칙 간격 관측** 처리(미분값 학습).
- **DeepAR** (Amazon): 확률적 예측(분포 출력) RNN.

### 생성모델 (증강·이상탐지) 📎
- **GAN**: Generator vs Discriminator 경쟁. 시계열: **TimeGAN, QuantGAN**(합성 시계열), **TadGAN**(이상탐지).
- **Diffusion**: 노이즈 추가→역과정 학습으로 생성. 시계열: **TimeGrad**.

### Attention/Transformer 계열 🔥
- **seq2seq** 한계: 긴 입력 요약 어려움 → **Attention**: 입력 중 어디 집중할지 판단.
- **Transformer** (Vaswani 2017): RNN 없이 attention만. **병렬화로 대규모 학습** 가능.
- 시계열 특화: **TFT(Temporal Fusion Transformer)** 🔥(해석가능·다변량·분위수예측), **Informer, Autoformer**(장기예측·효율 attention).

### LLM 기반 📎
- **LLMTime**(수치 토큰화, 대수관계 이해 한계), **TimeLLM**(TS→임베딩 재프로그래밍), **TimesNet**(주파수/진폭 변환), Time-VLM(멀티모달).

## B-4. Case Studies 📎
- 금융 시계열은 불확실성·비정상성 큼 → **외부 컨텍스트(뉴스·이벤트)로 완화**.
- Prediction markets(Kalshi)로 비정형 이벤트를 거래가능 데이터화. Granger causality + LLM 필터로 lead-lag 트레이딩.

---

## 해커톤 적용 포인트 🔥

**스포츠 데이터가 시계열이면:**
- [ ] **시간순 정렬 후 TimeSeriesSplit CV** — 랜덤 split 금지(누수).
- [ ] **정상성 점검** — 추세·계절성 제거(차분, 로그), ADF 검정. 비정상이면 변환.
- [ ] **시계열 피처**: lag(직전 N경기), rolling mean/std(최근 폼), 계절/요일/시즌단계, 시간차(휴식일), 추세.
- [ ] **베이스라인**: 통계(ARIMA/지수평활) + **GBDT에 lag 피처** (실전 강력·빠름) 🔥.
- [ ] **딥러닝 카드**: LSTM / TFT (다변량·장기·확률예측 필요 시). 데이터 충분할 때.
- [ ] **평가**: 예측이 실제 **결정/순위 지표**와 연결되는지 확인 (DFL 통찰: 정확도≠효용).
- [ ] 외부 데이터(날씨·일정·상대전적)로 불확실성 완화.

**Optimization 관점:**
- 자원배분/라인업 최적화 같은 결정 문제면 LP/QP로 정식화 가능. `cvxpy` 활용.
- 볼록성 확보되면 전역해 보장. 정식화가 성능 좌우.

---

> 💬 **Q&A 메모** — (질문 생기면 여기에 요약 기록)
