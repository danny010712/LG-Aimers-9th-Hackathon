# 강의 6. LLM Application & Evaluation

> 강사: Jaehyung Kim (연세대)
> 구성: 3파트 (Decoding / RAG / Evaluation)
> 성격: **해커톤 관련도 📎 하** (스포츠 tabular엔 직접 낮음) — **단 Decoding 파라미터(temperature/top-p), Evaluation 지표(EM/F1/ROUGE/LLM-as-Judge)는 LLM 활용·평가 시 🔥 실전.**

**중요도 범례**: 🔥 해커톤 핵심 / ⭐ 중요 / 📎 참고

---

## 0. 실전 압축 요약 🔥

- **Decoding = 확률분포에서 토큰 뽑는 법.** greedy(최고확률) / beam(다중후보) / **sampling + temperature + top-k/top-p**. 🔥
  - **temperature**: T>1 평탄(다양)·T<1 뾰족(결정적). **top-p(nucleus)**: 누적확률 τ까지. **재현성 원하면 T=0/greedy.**
- **RAG = 검색으로 최신·외부 지식 주입** (LLM 지식 cutoff·hallucination 보완). 검색: **BM25(단어중첩)** vs **Dense(DPR/Contriever, 임베딩 유사도)**. ⭐
- **평가 3요소**: target task / evaluation method / metric. 🔥
  - 정답 있음: 분류=accuracy, QA=**EM·F1**, 요약=**ROUGE**·임베딩유사도(Sentence-BERT).
  - 정답 없음: **perplexity**, **G-Eval / LLM-as-Judge**(확장성·설명력, 단 **편향** 주의).
- **Self-consistency**: 여러 번 샘플 → 다수결로 추론 정확도↑. 🔥

---

# Part A. Decoding of LLMs

## A-1. 기본 개념 ⭐
- **LM** = 토큰열 확률분포. autoregressive: `P(x)=∏P(xₜ|x<ₜ)`. Transformer + self-supervised + 대량 텍스트.
- **Tokenization**: character / **subword(BPE, 실전 표준)** / word. vocab 트레이드오프.
- **학습**: 임베딩 lookup `W∈ℝ^{|V|×d}` → Transformer 출력 → **tied W로 vocab 분류** → cross-entropy.
- **생성**: K개 토큰 = K번 순차 추론. **EOS 토큰**으로 종료.

## A-2. 기본 디코딩 알고리즘 🔥 (파라미터 이해 필수)
| 방법 | 아이디어 | 장단점 |
|---|---|---|
| **Greedy** | 매번 argmax | 쉬움 / 후속 고려 못해 suboptimal |
| **Beam Search** | 상위 num_beams 후보 유지·확장 | 더 나은 출력 기회 / 계산비용↑ |
| **Sampling** | 분포에서 랜덤 추출 `x~p̂` | 탐색·다양성↑ / 품질↓ 가능 |
| **Temperature** | `softmax(o/T)` | **T>1 smoothing(다양), T<1 sharpening(결정적)** |
| **Top-K** | 상위 K개 토큰만 샘플 | 분포 모양 무시(고정 K) |
| **Top-P (Nucleus)** | 누적확률 τ까지 적응적 K | 분포 모양 반영. **실전 기본** 🔥 |

→ 실전: 창의성 필요=sampling+top-p+적당한 T / 재현·정확성=greedy 또는 T낮게.

## A-3. 고급 디코딩 📎
- **Diverse Beam Search**: beam을 그룹으로 나눠 그룹간 **다양성 페널티** 추가 (n-gram/임베딩 유사도).
- **Contrastive Decoding** ⭐: 큰(expert)·작은(amateur) 모델 분포 **대비**로 정제. amateur 나쁜 습성 빼기 + Adaptive Plausibility Constraint(top-p 유사). amateur 선택으로 용도 변경(일반성능/RAG 문맥집중/detoxic). 파생: **CAD**(문맥인지, hallucination↓), **VCD**(비전-언어 hallucination↓).
- **Speculative Decoding** 🔥(추론 가속): **작은 모델이 draft(K토큰) → 큰 모델이 1번 추론으로 검증**. 쉬운 토큰은 작은 모델이 빠르게. **출력 동일성 이론 보장**. draft 클수록 acceptance↑·속도↓ 트레이드오프.
- **Guided/Controlled Decoding**: 디코딩 중 reward/score 반영.

---

# Part B. Retrieval Augmented Generation (RAG) ⭐

## B-1. 왜 RAG? 🔥
- LLM은 **knowledge cutoff**(GPT-4o≈2023-10, LLaMA3≈2023-12) → 최신 지식 부족.
- **Fine-tuning으로 새 지식 주입은 제한적**(오히려 hallucination↑·정확도↓ 가능).
- → **검색해서 관련 지식을 입력에 추가** (retrieve-and-read). QA 표준.

## B-2. Retrieval 방법 ⭐
- **Web/PageRank**: 링크 중요도 (damping factor d).
- **BM25** ⭐: 단어중첩 기반(sparse). query 키워드 `qᵢ`와 문서 중첩↑ → 점수↑. 학습 불필요·강건.
- **Dense Retrieval** 🔥:
  - **DPR**: 두 인코더(BERT)로 query·passage 임베딩 → dot-product 유사도. contrastive loss(positive/in-batch negative).
  - **Contriever**: self-supervised contrastive(같은 문서 crop=positive). 라벨 불필요.
  - → **best 방법 없음**. 학습기반(DPR/Contriever)은 test 분포 민감.

## B-3. RAG 적용 방식 ⭐
- **Inference-base**: 검색문서를 입력에 prepend (간단).
- **Training-base**: LLM이 검색정보 활용하도록 학습.

**추론레벨 개선:**
- **REPLUG** 📎: 블랙박스 LLM에 문서 각각 prepend 후 **출력확률 앙상블**(context window 한계 회피). 검색기를 LM 신호로 fine-tune(LSR, KL 최소화).
- **HyDE** ⭐(query 강화, zero-shot): LLM이 **가상문서 생성 → 그걸로 검색**. Contriever가 BM25 밑돌아도 HyDE 붙이면 능가. **LameR**: 검색으로 초기생성 개선(반복).
- **Noisy-robust RAG** 🔥: **잡음 문서는 오히려 성능 악화**(RAG < 무검색일 수도). 목표=관련시 도움·무관시 무해.
  - **RetRobust**: NLI 모델로 검색문서(전제)가 Q+A(가설) entail하는지 판단, 아니면 무검색 답변. 소량 학습(QLoRA)로 무관 문맥 무시하도록.

**학습레벨 개선:**
- **Self-RAG**: SFT로 적응적 RAG(검색 여부·비평 self-reflection).
- **Search-R1** 📎: **RL(PPO/GRPO)로 적응적 RAG 학습**(DeepSeek-R1 방식). 특수토큰 `<think><search><information><answer>`. CoT+검색 interleaving 자동 발현. outcome reward(정답 기반).

---

# Part C. Evaluation of LLMs 🔥 (평가 설계 = 대회의 본질)

## C-1. 평가 3요소 🔥
1. **Target task**: 무엇을 할 것인가.
2. **Evaluation method**: 어떻게 평가할 것인가.
3. **Evaluation metric**: 성공을 어떻게 측정할 것인가.
- DL 평가는 **test data**(같은 분포, 학습중 미관측) 기반.

## C-2. LLM 평가의 특수성 ⭐
- LLM은 다양한 태스크 수행 → **여러 태스크 함께 평가해야** (어려움).
- 입력 프롬프트 중요: **상세 instruction + 좋고 많은 예시(few-shot) + CoT** → 성능↑.
- **Decoding 영향** ⚠️: T>0이면 같은 질문에 다른 답 → **Self-consistency**(여러 샘플 다수결)로 안정화. 🔥
- 벤치마크: **MMLU**(57과목 객관식, accuracy), 단 LLM 출력이 형식 안 지킴 → instruction으로 완화.

## C-3. 정답 있는 태스크 지표 🔥
| 태스크 | 지표 |
|---|---|
| 객관식 QA | **Accuracy** |
| 단답 QA (NQ) | **EM(Exact Match)** + **F1**(단어단위 precision·recall) |
| 요약 (XSum) | **ROUGE-N**(N-gram 중첩) + **임베딩 유사도**(Sentence-BERT cosine) |

- **EM**: 정답과 완전일치 1/0. **F1**: 예측·정답 단어 겹침. **ROUGE**: 생성·참조 n-gram 중첩(recall 지향).
- **임베딩 거리**(L2/cosine): 의미 유사도. sentence encoder 품질 중요.

## C-4. 정답 없는 태스크 평가 🔥 (실무 핵심)
- 요약·스토리생성 등 고정 정답 없음 → open-ended.
- **Perplexity**: 모델이 단어열 예측하는 정도(낮을수록 좋음). 단 인간 체감품질과 항상 일치 X.
- **G-Eval** ⭐: LLM으로 출력 품질 평가. task정보 + 평가기준 → LLM이 평가단계 생성 → 채점. ROUGE·BERTScore보다 **인간과 상관 높음**.
- **Relative Evaluation**: 쌍대비교(pairwise)로 우열. **Chatbot Arena**(실사용자 투표, 신뢰도 높음).
- **LLM-as-Judge** 🔥: 사람 대신 LLM이 채점. **장점: 확장성·설명력**(GPT-4 vs 인간 일치 85%). **주의 — 편향**:
  1. **Position bias**: 앞 위치 선호 → **위치 바꿔 2회 평가 후 평균**.
  2. **Verbosity bias**: 긴 답 선호 → **길이 통제**(AlpacaEval LC Win Rate).
  3. **Self-preference bias**: 자기 생성물 선호.

---

## 해커톤 적용 포인트 🔥

- **문제가 tabular이면 이 강의는 배경** — 우선순위 낮음.
- **LLM/텍스트 관여 시 실전 유용**:
  - LLM 피처엔지니어링·serialization 실험 시 **temperature/top-p 조정**으로 출력 제어. **재현성 필요하면 T=0(greedy)**. 🔥
  - LLM 추론 여러 번 → **Self-consistency(다수결)** 로 안정적 예측.
  - 최신·외부 지식 필요하면 **RAG**(BM25 간단 or Dense). **잡음 문서 주의**(무관하면 무검색이 나을 수).
  - 추론 속도 문제면 **Speculative Decoding**.
- **평가 관점 일반화** 🔥: 대회 지표(target task·method·metric)를 명확히 하고 **그 지표에 직접 최적화**. 생성/유사도 문제면 EM/F1/ROUGE/임베딩 중 무엇으로 채점되는지 파악.
- LLM으로 산출물 품질 자동채점 필요 시 **LLM-as-Judge**(편향 보정 필수).

---

> 💬 **Q&A 메모** — (질문 생기면 여기에 요약 기록)
