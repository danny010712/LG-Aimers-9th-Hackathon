# LG Aimers 강의정리

LG Aimers Phase I 온라인 교육 강의별 정리 모음. 궁극 목표 = **Phase II/III 스포츠 해커톤 우수 성적**.
스포츠 데이터는 tabular + time-series 성격이 강하므로 해당 강의를 최우선으로 심화.

## 중요도 표기 범례

| 배지 | 의미 |
|---|---|
| 🔥 | **해커톤 핵심** — 대회에서 바로 쓰는 실전 지식. 반드시 체화 |
| ⭐ | **중요** — 개념 이해 필수. 알아두면 성능·판단에 직결 |
| 📎 | **참고** — 배경/최신 연구/교양. 시간 남을 때 |

## 강의 목록

| # | 강의 | 정리 | 해커톤 관련도 |
|---|---|---|---|
| 1 | Tabular ML: Classical → Foundation Models | [01_Tabular_ML.md](01_Tabular_ML.md) | 🔥🔥🔥 최상 |
| 2 | 지도학습 | [02_지도학습.md](02_지도학습.md) | 🔥🔥 상 |
| 3 | Mathematics for ML | [03_Mathematics_for_ML.md](03_Mathematics_for_ML.md) | ⭐ 중 |
| 4 | Optimization/DFL + Time-Series | [04_Optimization_DFL_TimeSeries.md](04_Optimization_DFL_TimeSeries.md) | 🔥🔥 상 (시계열) |
| 5 | 딥러닝 NLP 기초 & LLM Agent | [05_딥러닝_NLP_LLM_Agent.md](05_딥러닝_NLP_LLM_Agent.md) | 📎 하 (문제 성격 따라) |
| 6 | LLM Application & Evaluation | [06_LLM_Application_Evaluation.md](06_LLM_Application_Evaluation.md) | 📎 하 |
| 7 | **LG 스포츠 해커톤 문제 소개** | [07_해커톤_문제소개.md](07_해커톤_문제소개.md) | 🔥🔥🔥 **대회 본체** |

## 🎯 대회 요약

**투수의 다음 투구 "제구 성공 확률" 예측** (tabular 이진분류, 확률 출력).
- 학습: 2019~2024 시즌 / 평가: 2025 시즌(Target 비공개, hidden test).
- 데이터 4축: 경기상황 · 경기중요도(WE·LI) · 과거이력(asof 누적) · Trackman 로그(구속·회전·무브·릴리스).
- 메인 전략: **LightGBM/CatBoost + 시간순 검증 + 피처엔지니어링** (강의1·4 직결).
- 📄 문제 개요: [07 문제소개](07_해커톤_문제소개.md) / 📄 **Phase 2 데이터·규칙·실행전략(실전 작업문서)**: [08 Phase2](08_Phase2_데이터_및_전략.md) 🔥
- 📄 **피처 아이디어·전제 검증 노트**: [09 피처분석 노트](09_피처분석_노트.md) 🔥 (모델 돌리기 전 데이터로 확인한 사실)
- 📄 **KBO 시즌 환경 분석 (2019~2025)**: [10 시즌환경 분석](10_KBO_시즌환경_분석.md) 🔥 (웹 사실 ↔ train.csv 대조 — `game_type` F 라벨 2023 교체, 2025 base rate 추정)

### Phase 2 핵심 수치 (외울 것)
- 평가지표 **Brier Skill Score** (확률 calibration 게임). 수료 기준 **≥549.51**(baseline RF).
- Target `control_success` 균형(train 0.524). **시즌별 하락: 2019=0.565 → 2024=0.486** → 2025 더 낮을 것. **2024로 검증·보정.**
- train 147만행(2019~24) / test 24.6만행(2025) / **코드제출(submit.zip, 추론≤10분, 오프라인)**.

## Q&A 메모 규칙

내가 질문 → 상세 답변 → 요약해서 해당 강의 md의 알맞은 섹션 하단 `> 💬 **Q&A 메모**` 블록에 기록.
