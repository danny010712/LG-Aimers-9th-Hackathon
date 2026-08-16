# 강의 5. 딥러닝 자연어처리 기초와 LLM Agent

> 강사: 이환희 (중앙대 AI학과) · LG AI Research Special Lecture
> 구성: 6강 (ML/DL 기초 / NLP+RNN / Transformer·Attention / LLM 사전학습·RLHF / LLM Agent / AI 하드웨어·GPU)
> 성격: **해커톤 관련도 📎 하** (스포츠 tabular 문제엔 직접 낮음) — **단 GPU/VRAM 사이징은 🔥 실전(Colab에서 모델 돌릴 때), Transformer·Attention은 ⭐ 교양 필수.**

**중요도 범례**: 🔥 해커톤 핵심 / ⭐ 중요 / 📎 참고

---

## 0. 실전 압축 요약 🔥

- **활성함수 비선형성 필수**: 선형 유닛만 쌓으면 단층으로 축약됨(XOR 못 풂). ReLU가 표준. ⭐
- **학습=Backpropagation**: Forward(예측)→Loss→Backward(가중치 조정) 반복.
- **Transformer 핵심 = Self-Attention**: 각 토큰이 다른 모든 토큰 참조 → 문맥표현. 병렬화로 대규모 학습. **max path length O(1)** (RNN은 O(n)). ⭐
- **LLM 학습 진화**: 사전학습(next-token) → Fine-tuning → In-context Learning(zero/few-shot) → **SFT + RLHF**(정렬). 📎
- **RAG**: 벡터DB 검색으로 최신·외부 지식 주입 (LLM 한계 보완). ⭐
- **LLM Agent**: Augmented LLM = 두뇌 + Tools(actuator) + Memory + Planning. **ReAct = Thought→Action→Observation 순환.** 📎
- **GPU VRAM 사이징** 🔥: 파라미터×정밀도. FP16이면 8B≈16GB. **Full fine-tune은 가중치+gradient+optimizer(AdamW 2배)+activation ≈ 64GB+.** → Colab 16GB에선 **QLoRA(4bit)** 필수.

---

## Lec 1. ML/DL 기초 ⭐ (지도학습 강의와 중복 — 요점만)

- **ML 3요소**: 데이터 + 피처 + 알고리즘. 경험(E)으로 태스크(T) 성능(P) 개선.
- **ML 유형**: 지도 / 비지도 / 준지도(pseudo-label) / **강화(보상 기반)**.
- **딥러닝**: hidden layer 2개 이상. 층 깊을수록 복잡·추상 피처 학습.
- **뉴런**: `y = F(Σwᵢxᵢ + b)`, F=활성함수(ReLU=max(0,x)).
- **XOR 문제** ⭐: 퍼셉트론(선형)은 XOR 못 풂(비선형 분리 불가). **다층 + 비선형 활성**으로 해결 → 비선형 활성함수 필요성의 근거.
- **Softmax**: 최종층, logit → 확률분포(합=1). **Backpropagation**: 오차를 역전파해 각 가중치 기여도 계산·조정.

## Lec 2. NLP 기초 & RNN ⭐

- **NLP** = NLU(이해) + NLG(생성). 파이프라인: **전처리(토큰화) → 임베딩 → 모델링**.
- **Tokenization**: 텍스트→토큰. vocab 크기 트레이드오프(작으면 OOV, 크면 계산·희소성). 해결: character-level, **BPE(Byte Pair Encoding)**.
- **Word Embedding** ⭐: 단어→실수벡터.
  - BoW/One-hot(희소). **Word2Vec**(Mikolov 2013): 유사 문맥 단어를 가깝게. **CBOW**(문맥→중심어) / **Skip-gram**(중심어→문맥). hidden weight 행렬 행 = 단어벡터.
- **Language Model**: 단어열 확률 `P(w₁..wT)=∏P(wₜ|w₁..wₜ₋₁)`. 어순·어휘 선택 반영.
- **RNN** ⭐: 순차 데이터 자연 모델. 매 스텝 **같은 가중치 반복**, hidden state=메모리. 임의 길이 입력·모델크기 불변.
  - 학습: **Teacher Forcing**, **BPTT**(시간 역전파, 반복 가중치 gradient 합산). 생성=반복 샘플링.
  - **단점** ⚠️: **Vanishing gradient**(초기 입력 영향 소실), 느림(순차), 장기의존 어려움.
- **LSTM** 🔥: 3 gate(input/forget/output)로 장기 정보 보존 → vanishing 완화.

## Lec 3. Transformer & Attention ⭐⭐ (딥러닝 교양의 핵심)

- **진화**: Seq2Seq → +Attention → **Transformer**.
- **Seq2Seq**(encoder-decoder RNN): 입력을 단일 context vector로 압축 → **긴 문장에서 정보 병목**.
- **Attention** ⭐: 디코더가 매 스텝 **입력의 어느 부분에 집중**할지 결정. context vector = annotation 가중합, 가중치 = alignment score의 softmax.
- **Self-Attention** 🔥: 한 문장 내 각 단어가 다른 모든 단어에 attend → **문맥 민감 표현**. (예: "it"이 "animal"을 강하게 참조).
- **RNN vs Transformer** ⭐:
  | | RNN | Transformer |
  |---|---|---|
  | 병렬화 | 어려움(순차) | **쉬움→빠름** |
  | max path length | O(n) | **O(1)** |
  | 장기의존 | 약함 | 완전 연결 |
- **Transformer 구조**:
  - **Positional Embedding**(순서 정보 주입) + Token Embedding.
  - **Encoder** (N=6): Multi-Head Self-Attention + Position-wise FFN, 각 sublayer는 `LayerNorm(x+Sublayer(x))` (**residual connection**).
  - **Scaled Dot-Product Attention**: Q·K → softmax → V 가중합. **Multi-Head**: 여러 선형투영으로 병렬 attention.
  - **Decoder**: self-attention + **enc-dec attention**(Q=디코더, K·V=인코더) + FFN.
- **생성(NLG)**: 학습=cross-entropy(다음단어 분류). 추론=**autoregressive**(예측단어를 다시 입력). 탐색: **Greedy**(최고확률) vs **Beam Search**(상위 num_beams 경로 유지).

## Lec 4. LLM 사전학습과 진화 📎

- **사전학습→파인튜닝 패러다임** ⭐: 입력 일부를 가리고 복원 학습 → 강한 표현·초기화. 대량 unlabeled로 사전학습 후 소량 labeled로 downstream 적응.
- **아키텍처별 사전학습**: **BERT**(인코더, 양방향 문맥, masked+next-sentence) / **GPT**(디코더, 미래 조건 불가, 생성 강함) / **BART·T5**(enc-dec).
- **GPT-3** 📎: 175B 파라미터, 300B 토큰. **In-Context Learning** 🔥(zero/few-shot, gradient 업데이트 없이 프롬프트만으로 새 태스크). 한 모델로 요약·번역·감성 등 범용.
- **Misalignment 문제** ⭐: 학습목표(next-token) ≠ 원하는 것(지시 따르기). → **3H desiderata: Helpful, Honest, Harmless**.
- **정렬(Alignment)** 🔥:
  1. **SFT** (Supervised Fine-Tuning): 사람 시연으로 지도학습.
  2. **RLHF**: 사람이 출력 순위 매김 → **reward model** 학습 → **PPO**로 정책 최적화. (SFT에서 너무 벗어나지 않게 KL 페널티).
  - **RLHF > SFT 이유**: 보상이 미묘한 신호(품질 유사→보상 유사), 모델 자체 생성물 비평, preference 직접 포착, 데이터 효율↑.
  - **InstructGPT** = GPT-3 + SFT + RLHF → helpful·덜 hallucinate. **ChatGPT** = 대화형으로 확장.
- **RAG** ⭐: 사전학습 지식은 static → 검색으로 최신·외부 정보 주입 후 생성.
- **Base vs Instruct 모델**: Base=텍스트 완성, Instruct=지시 따르기(파인튜닝). 다운로드 시 구분.

## Lec 5. LLM Agent 📎 (Claude Code 같은 도구의 원리)

- **정의**: LLM은 곱셈·최신정보 등에 약함 → 외부시스템으로 강화 = **"Augmented LLM"** = 두뇌(effector) + Tools(actuator) + Memory + Planning.
- **Memory**:
  - **Short-term**: context window에 대화 이력 담기(또는 다른 LLM으로 요약).
  - **Long-term** ⭐: **벡터DB에 임베딩 저장 → RAG로 검색**. (semantic/working memory 등 유형 구분).
- **Tools & Function Calling** ⭐: 데이터 fetch / action 수행. LLM이 **JSON 구조로 어떤 도구 언제 쓸지 자율 결정**. **Toolformer**(API 호출 학습).
- **MCP (Model Context Protocol)** ⭐: Anthropic 표준. 많은 API의 수동 관리 문제 해결. **Host(LLM앱) + Client(1:1 연결) + Server(도구·컨텍스트 제공)**.
- **Planning + Reasoning**: 태스크를 실행 단계로 분해. Chain-of-Thought=추론만.
- **ReAct (Reason + Act)** 🔥: **Thought(추론) → Action(도구 실행) → Observation(결과 분석)** 순환.
- **Reflexion**: 실패에서 학습 (Actor + Evaluator + Self-reflection).
- **Multi-Agent**: 전문 에이전트들 + Supervisor 조율. (Agent Initialization + Orchestration).

## Lec 6. AI 하드웨어 & GPU 🔥 (실전: 모델 돌릴 때 필수)

- **CPU vs GPU** ⭐:
  | | CPU | GPU |
  |---|---|---|
  | 코어 | 소수·강력 | **수천·단순** |
  | 최적화 | latency(순차) | **throughput(병렬)** |
  | 용도 | OS·순차로직 | **딥러닝 행렬연산** |
- **Heterogeneous Programming**: CPU(host)가 메인, compute-집약 부분만 GPU로 offload. CPU↔GPU는 PCIe로 통신(느림).
- **정밀도(Precision)** 🔥: 비트폭↓ → 에너지·VRAM↓. FP32(4B) / **FP16·BF16(2B, 표준)** / INT8(1B) / INT4(0.5B, QLoRA).
- **VRAM 사이징** 🔥🔥 (Colab에서 중요):
  - **가중치**: 파라미터 × 바이트. 예: **Llama3-8B FP16 = 8B×2 ≈ 16GB**. INT4 ≈ 4GB.
  - **추론 VRAM 소비**: 가중치 + **KV Cache**(context length·batch에 선형 증가 → 긴 context OOM 위험).
  - **Full Fine-Tuning VRAM** ⚠️: 가중치 + **gradient(=가중치 크기)** + **optimizer states(AdamW=2배)** + activation. → **8B ≈ 64GB+**.
  - **16GB GPU(Colab T4/L4)**: FP16 추론 OK(빠듯), **full fine-tune 불가** → **PEFT/QLoRA(4bit)** 로 파인튜닝.

---

## 해커톤 적용 포인트 🔥

- **스포츠 tabular 문제라면 이 강의 대부분은 배경지식** — 우선순위 낮음. Tabular/시계열 강의에 집중.
- **단, 실전에서 유용**:
  - Tabular 강의의 **LLM 피처엔지니어링(CAAFE)·TabPFN·Qwen 실습**을 Colab에서 돌릴 때 → **VRAM 사이징 지식 필수**. 모델 크기×정밀도로 미리 계산, OOM 나면 정밀도↓(FP16/INT8) 또는 작은 모델.
  - 텍스트 피처(선수 코멘트·기사 등) 있으면 → Word2Vec/임베딩, 사전학습 임베딩(all-MiniLM 등) 활용.
  - 문제 해결에 도구 오케스트레이션 필요 시 ReAct/MCP 개념 참고.
- **Transformer·attention**은 시계열(TFT)·Tabular(FT-Transformer)와 연결되는 교양이므로 개념은 잡아둘 것.

---

> 💬 **Q&A 메모** — (질문 생기면 여기에 요약 기록)
