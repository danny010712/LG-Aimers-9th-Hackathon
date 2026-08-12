# 강의 3. Mathematics for ML

> 강사: Jinwoo Shin (KAIST 김재철AI대학원)
> 구성: 3강 (Matrix Decompositions / Convex Optimization / PCA)
> 성격: ML **수학 기초**. **해커톤 관련도 ⭐ 중** — 대부분 이론이나, **PCA·SVD(차원축소)와 볼록성·GD 직관**은 실전에 직접 닿음.

**중요도 범례**: 🔥 해커톤 핵심 / ⭐ 중요 / 📎 참고
**방침**: 수식은 결론·의미 위주. 증명은 참고로만.

---

## 0. 실전 압축 요약 🔥

- **PCA/SVD = 차원축소·노이즈 제거·시각화·피처압축**. 고차원·다중공선성 데이터에서 카드. `sklearn.decomposition.PCA`. 🔥
- **공분산행렬의 고유벡터 = 주성분**, 고유값 = 그 방향 분산. 큰 고유값 M개만 취하면 최적 저차원 표현.
- **볼록(convex)이면 지역최소=전역최소** → 선형·로지스틱 회귀·SVM이 신뢰성 있는 이유. ⭐
- 최적화 구분: batch/mini-batch/SGD + Momentum·RMSProp·Adam (지도학습 강의와 연결).
- 나머지(det/trace/Cholesky/KKT/duality)는 배경지식 📎.

---

## Lecture 1. Matrix Decompositions (행렬 분해)

### 1-1. Determinant & Trace 📎
- **det(A)**: 2×2는 `a₁₁a₂₂−a₁₂a₂₁`. 일반화는 **Laplace expansion**(여인수 전개).
- **det(A)≠0 ⇔ rank(A)=n ⇔ A 가역**. 핵심 성질: `det(AB)=det(A)det(B)`, `det(Aᵀ)=det(A)`, 삼각행렬은 대각원소 곱, 가우스 소거로 계산.
- **trace(A)** = 대각합 `Σaᵢᵢ`. 선형: `tr(A+B)=tr(A)+tr(B)`.

### 1-2. Eigenvalue & Eigenvector ⭐ (ML의 핵심 도구)
- 정의: `Ax = λx` (x≠0). λ=고유값, x=고유벡터.
- 구하기: `det(A − λI) = 0` (특성방정식) → λ → `(A−λI)x=0`로 고유벡터. **고유벡터는 유일하지 않음**(span).
- 성질: **det(A)=Πλᵢ, tr(A)=Σλᵢ**. 서로 다른 고유값 n개 → 고유벡터 선형독립(basis).

### 1-3. Cholesky Decomposition 📎
- **대칭·양정치 행렬** A = LLᵀ (L=하삼각, 양의 대각). "행렬의 제곱근" 격.
- 응용: 다변량 가우시안 공분산 분해, 랜덤변수 선형변환, 빠른 det 계산(`det(A)=Πlᵢᵢ²`).

### 1-4. Eigendecomposition & Diagonalization ⭐
- **대각화**: `D = P⁻¹AP` (D=대각). **직교대각화**: P가 직교행렬 → `D = PᵀAP`.
- **핵심 정리**: A가 **직교대각화 가능 ⇔ A가 대칭**.
- **Spectral Theorem** (A 대칭이면): (a) 고유값 모두 실수, (b) 다른 고유값의 고유벡터 직교, (c) 직교 고유기저 존재. → P의 열 = A의 고유벡터들.
- **대각화의 힘**: `Aᵏ = PDᵏP⁻¹` (거듭제곱 쉬움), det 계산 간단.

### 1-5. SVD (Singular Value Decomposition) 🔥 (가장 중요)
- **모든 행렬** A∈ℝ^{m×n} = **UΣVᵀ**. U(m×m 직교), V(n×n 직교), Σ(대각에 특이값 σᵢ≥0).
- 배경: AᵀA는 항상 **대칭·양반정치** → 직교대각화 가능.
- **EVD vs SVD** ⭐:
  | | EVD `A=PDP⁻¹` | SVD `A=UΣVᵀ` |
  |---|---|---|
  | 존재 | 정방+고유기저 있을 때 | **항상 존재** |
  | 직교성 | P는 대칭일 때만 직교 | U,V 항상 직교(회전) |
  | 형태 | 정방행렬 | **비정방도 가능** |
  - 관계: A의 좌특이벡터=AAᵀ 고유벡터, 우특이벡터=AᵀA 고유벡터, 특이값=고유값의 제곱근. **A 대칭이면 EVD=SVD**.
- **실전 의미** 🔥: 저랭크 근사(노이즈 제거·압축), 추천시스템(행렬완성), PCA 계산의 기반.

---

## Lecture 2. Convex Optimization (볼록 최적화)

> "ML 모델 훈련 = 좋은 파라미터 찾기 = 최적화 문제 풀기." 최적 필요조건: `∇f=0` (stationary point).

### 2-1. GD 기반 최적화 ⭐ (지도학습 강의와 중복·심화)
- `xₖ₊₁ = xₖ + γₖdₖ`. **Descent direction**: `∇f·d<0`이면 하강방향. **Steepest descent**: `d=−∇f`.
- **Taxonomy** 🔥:
  - 데이터량별: **Batch**(전체) / **Mini-batch**(k<n) / **SGD**(불편 gradient 추정).
  - 적응방식별: Momentum, NAG, **Adagrad, RMSprop, Adam**.
- **Momentum**: `xₖ₊₁ = xₖ − γ∇f + αΔxₖ`. 메모리항 α로 진동 감쇠·수렴 개선. (step 너무 작으면 느림, 크면 overshoot·zigzag).

### 2-2. Constrained Optimization & Lagrange 📎
- 표준형: `min f(x) s.t. gᵢ(x)≤0, hⱼ(x)=0`.
- **Lagrangian**: `L(x,λ,ν) = f(x) + Σλᵢgᵢ(x) + Σνᵢhᵢ(x)`, λ⪰0.
- **Dual function** `D(λ,ν)=inf_x L`. **약 쌍대성(Weak duality)**: `d* ≤ p*` (항상 성립). 쌍대문제는 **항상 볼록**(concave의 sup).

### 2-3. Convex Sets & Functions ⭐
- **볼록집합**: 두 점 잇는 선분이 집합 안. **볼록함수**: `f(θx+(1−θ)y) ≤ θf(x)+(1−θ)f(y)`.
- **Jensen 부등식**: `f(E[X]) ≤ E[f(X)]`.
- **판정**:
  - 1차: `f(y)−f(x) ≥ ∇f(x)ᵀ(y−x)` → **국소정보(접선)가 전역 하한**. `∇f(x)=0`이면 **전역최소**! 🔥
  - 2차: `∇²f(x) ⪰ 0` (Hessian 양반정치, 위로 굽음).
- 볼록 예: eˣ, |x|^p(p≥1), 모든 norm, max, log-sum-exp. concave 예: log x, 기하평균.
- **볼록 보존 연산**: 양수 가중합, 아핀합성 `f(Ax+b)`, max, pointwise sup.

### 2-4. Convex Optimization & KKT ⭐
- **핵심 통찰** 🔥: "쉬운 문제와 어려운 문제의 경계는 **선형성이 아니라 볼록성**." 볼록목적 + 볼록제약 + 아핀 등식제약.
- **강 쌍대성(Strong duality)**: `d* = p*`. 볼록 + 제약자격(Slater) → 성립. → 쌍대 풀면 원문제 풂.
- **KKT 조건**: stationarity(`∇L=0`) + primal/dual feasibility + **complementary slackness**(`λᵢgᵢ=0`). 강쌍대성이면 필요조건, 볼록+Slater면 충분조건.
- **LP**(선형계획) / **QP**(이차계획, Q 대칭 양정치) — SVM이 QP. 📎

---

## Lecture 3. PCA (Principal Component Analysis) 🔥🔥

> **차원축소의 대표. 해커톤에서 고차원·중복 피처 압축, 다중공선성 완화, 시각화, 노이즈 제거에 활용.**

### 3-1. 왜 차원축소?
- 고차원 = 분석·시각화 어렵고 중복(redundant) 많음. 압축처럼 compact 표현 선호.

### 3-2. PCA 알고리즘 🔥 (실전 절차)
1. **Centering**: 평균 빼기.
2. **Standardization**: 차원별 표준편차로 나눔.
3. **고유값/벡터**: **공분산행렬 S = (1/N)XXᵀ**의 M-largest 고유값·고유벡터 계산.
4. **Projection**: 데이터를 고유벡터(주부분공간)에 투영.
5. 표준화·centering 되돌리기.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=10)          # 상위 10개 주성분
X_reduced = pca.fit_transform(X_scaled)   # 반드시 스케일링 후!
print(pca.explained_variance_ratio_.cumsum())  # 누적 설명분산 확인
```

### 3-3. 왜 고유벡터인가 (Maximum Variance 관점) ⭐
- **코드(code)**: `zₙ = Bᵀxₙ` (B=직교기저, encoder=Bᵀ, decoder=B, `x̃=Bz`).
- 1차원 투영 분산 `V₁ = b₁ᵀSb₁`를 `‖b₁‖=1` 제약하에 최대화 → **Lagrange → Sb₁=λ₁b₁**. 즉 **b₁은 S의 고유벡터, 분산 V₁=λ₁**.
- **가장 큰 분산 = 가장 큰 고유값** → 제1주성분. k번째는 앞의 것들과 직교 제약하에 k번째 고유값의 고유벡터.

### 3-4. 계산: EVD vs SVD ⭐
- **Approach 1 (EVD)**: 공분산 S 직접 고유분해.
- **Approach 2 (SVD)** 🔥: 데이터행렬 `X=UΣVᵀ` → **U의 열이 S의 고유벡터**, `λ_d = σ_d²/N`. (수치적으로 더 안정 → 실무 표준).
- **Low-rank 근사** (Eckart-Young): top-M 특이값만 남기면 `X̃_M = Σᵢ₌₁ᴹ σᵢuᵢvᵢᵀ`가 rank-M 최적근사(재구성오차 최소).

### 3-5. 고차원 PCA 트릭 📎
- **N ≪ D**일 때 (D=피처 매우 큼, 예: 이미지 10,000픽셀): S는 D×D로 큼. 대신 **(1/N)XᵀX (N×N)** 의 고유분해로 우회 → 계산 훨씬 쉬움. 고유벡터는 X 곱해 복원.

---

## 해커톤 적용 포인트 🔥

- **PCA**: 피처 많고 상관 높을 때 → 차원축소·시각화(2~3D). 단, **트리모델엔 보통 불필요**(스케일·회전 불변 아니고 해석성 손해). 선형·거리기반·딥러닝 전처리에 유용. 반드시 **스케일링 후** 적용.
- **SVD**: 노이즈 제거, 잠재요인(추천), 텍스트 LSA.
- **explained_variance_ratio_**로 몇 개 주성분 남길지 결정(누적 90~95%).
- **볼록성 직관**: 선형·로지스틱·SVM은 볼록 → 전역최적 보장·재현성↑. 딥러닝은 비볼록 → 초기화·LR 민감.
- 공분산·고유값 개념은 **다중공선성 진단**(상관행렬 고유값)에 활용.

---

> 💬 **Q&A 메모** — (질문 생기면 여기에 요약 기록)
