# 공통 지침

추가 정보가 있을 때 더 상세한 답변이 가능해진다면, 사용자에게 추가 설명을 요청한다. 이떄는 답변 없이 사용자에게 질문만 한다.
모든 한국어 문장은 간결하게, 명사형 어미로 끝나도록 한다. 다른 언어로 답변할 때도 문장을 간결하게 한다.
결론을 내리기에 충분한 정보가 없다면 알 수 없거나 모른다고 답변한다.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# CLAUDE.md — 10_LGAimers

LG Aimers 9기 + **LG스포츠 해커톤 Phase 2** 실전 디렉토리. 상위 `MyProjects/CLAUDE.md`(일반 지침)를 따르고 아래는 이 프로젝트 전용.
상세 근거는 `강의정리/08_Phase2_데이터_및_전략.md`(실험·제출기록) / `09_피처분석_노트.md`(라벨 복원) / `10_KBO_시즌환경_분석.md`.

## 1. 목표·디렉토리

Phase 2 과제 = **투구 단위 "제구 성공 확률" 예측** (tabular 이진분류, 확률 출력). 목표 = 순위.

```
강의자료/   원본 PDF/ipynb (수정 금지)
강의정리/   01~10 md + README(인덱스)
참고자료/   개요·규칙·평가 txt, DACON 가이드
open/data/  train.csv 1,475,092×49 (2019~24) | test.csv 2025 실제 245,789행(배포본은 5행 샘플)
            trackman_history.csv 1,793,078×30 (2019~24) | sample_submission.csv | first_game.csv(EDA용)
open/       baseline_submit/(주최측 RF), data_description.md
test/       train_local.py · train_offset.py · build_shift.py · recover_labels.py · recovered_labels.csv.gz
            probe_offset_forms.py(연도별 창 12모델) + _preds.csv.gz(4연도×3라벨 예측 캐시)
              + _score.py(전이 매트릭스 채점) ← offset 형태 재검정은 재학습 없이 이걸로
            common/{script.py, features.py, cond.py, requirements.txt}  ← zip에 그대로 들어감
            data/ → open/data (junction) · runs/<NNN_name>/{model/, result.json, submitNNN.zip}
            artifacts/{auxpred/*.npy, sub010.csv.gz}  ← offset 계수·logit_shift 적합 입력
```

**강의정리 규칙**: 강의당 md 1개, 배지 🔥해커톤핵심/⭐중요/📎참고, 상단 압축요약 → 본문 → 하단 적용포인트 + `💬 Q&A 메모`(사용자 질문·답변 요약 누적). 새 자료 추가 시 README 갱신.

## 2. 대회 사실

- **지표 = Brier Skill Score** `max(0, 100000×(1 − Brier/[r(1−r)]))`. 순위(AUC) 아님 = **확률 정확도 게임**. 상수예측 = 0점.
  - ⚠️ "확률이 중요" ≠ "사후 calibration을 걸어라". **사후 보정은 실측 −23** (§3).
- **수료선 549.51**(주최측 RF) — 통과 완료. 이제 순위 싸움.
- **Target `control_success`** 1=성공. 실패 = ①한복판 ②크게 벗어남 ③포수 요구 반대. 볼/스트라이크 판정과 다름. train 평균 0.524.
- ⚠️ **시즌 base rate 하락**: 2019 .565 → 2024 .486, 2025 더 낮음. → **랜덤 KFold 금지**(점수 3.5배 부풀림), 시간순만.
- **`asof_*` 19개** = 공식·leak-safe·test에도 제공. cold-start NaN.
  ⚠️ **시즌 리셋 없는 통산 누적**(`asof_pitcher_n` 2019 3,141 → 2024 15,449) → 편향(asof−실제) 2019 −.005 → 2024 +.025, 2025 +.03~.04 추정. rate는 **소수 6자리 반올림**(등식 검사에 `1e-9` 금지), **배타적 분할 아님**(겹침 5만건 → "합=1" 가정 금지).
- 🔥 **`game_type` 중요도 1위(24.24%)**, `season` 2위(22.05%) — 둘이 46%. 실력 피처 최상위는 6.13.
  - F(퓨처스) = 11%. 성공률 2022 .709 → 2023 .473 **체제 붕괴**. **test에도 F 있다**(245,789행 ≈ 시즌 전체 245,849; R만이면 219,015).
  - **F를 빼면 −85.4** → 빼지 말 것. 모델 분리도 사장.
- ✅ **숨은 투구 라벨 8종 복원됨** (`test/recover_labels.py` → `recovered_labels.csv.gz`). 답을 아는 `success`로 검증해 147만행 전부 일치, 결측 792행뿐. **실패 ⟺ (한복판∪반대) ⊎ 크게벗어남**, 예외 0건.
- **trackman**: 투수 단위 매핑은 복원 성공(등판 지문)했으나 `asof_*`와 **정보 중복**이라 이득 0 → 사장. 사유는 "불가능"이 아니라 "중복". **재시도 금지.**
- **금지**: 현재 투구 이후 정보 · 2025 trackman · **test 내부 행간 통계**(각 행 독립 예측) · 외부 데이터 · 외부 API.

## 3. 모델링 하드룰 (반드시 지킬 것)

- **`pitcher_id`/`batter_id` 범주형 금지.** 개인 암기 → 과적합 (LGBM 612.9→199.5, CatBoost 780.0→647.3). 수치형 유지 or 삭제(동일).
- **`season`은 수치형으로만.** 범주형이면 미관측 시즌에 값이 없어 붕괴(780→605). 수치형은 +450.
- **CatBoost `cat_features` 반드시 지정**(저cardinality만). 미지정 −114 = 최대 단일 이득.
- **옛 시즌을 버리거나 가중치를 낮추지 말 것.** 삭제도 감가도 손해. 데이터 양이 이긴다.
- **calibration 금지.** 로컬 천장 +89인데 팀 LB 실측 **−23**. offset의 `a`·`d` 적합도 같은 함정(§4).
- **depth 6 고정. 용량 늘리지 말 것.** depth 8 = LB **−34.2**. 미관측 시즌 외삽에는 얕은 트리가 유리 — 이 데이터에서 가장 확실한 단일 사실.
- **시드 평균 = fold 무관하게 작동하는 유일한 개선 수단**이나 **3시드에서 포화**(1→4 각 fold +10.5, 3→4는 +1.2). 3→7 = 로컬 +3.8 / LB +5.6이 사실상 천장.
- **앙상블 이득 = 분산 감소가 전부.** 구조 다양성(깊이 조합·grow_policy·신경망)은 모델 개수를 맞춰 재면 전부 0 이하. **새 앙상블 축 찾지 말 것.**

## 4. 채택 기준 🔴 (2026-08-11 3차 교정)

1. **fold 2024가 주 기준.** 2023은 제외(`best_iteration=0`인 죽은 fold), 2021·2022는 참고용 — 그 시절은 `game_type` F−R 격차 +0.19~0.21이라 점수의 60~75%가 공짜였다(2023부터 격차 소멸 → 2025도 동일). R만 재채점: 2022 `2354→595`, 2024 `788→785`.
2. **합계로 판정 금지. fold별 부호를 볼 것.** 규모가 2022:2354 / 2024:788이라 합계는 사실상 2022 판정이 된다.
3. **시드 4개 이상 평균 + 비교군 모델 개수 일치.** 4시드 표준오차 2021 ~10 / 2022 ~4 / 2024 ~2. 개수 안 맞추면 분산감소를 "다양성 이득"으로 오인(grow_policy 2 vs 4모델 +24.8 → 맞추니 +0.8).
4. 🔥 **적합 요소가 있으면 델타를 out-of-year로 잴 것.** 계수·상수를 **연도 T−1에서 맞춰 T에 적용**해 측정. fold 내부 델타는 LB에서 뒤집혔고(cond +12.5→−5.4, d8 +7.6→−34.2), out-of-year 델타는 맞았다(offset +26.6 → LB +63.7).

5. 🔥 **여러 후보를 한 측정으로 줄 세웠으면 그 순위는 증거가 아니다.** 최고값은 노이즈만으로도 평균보다 ~2.9σ 위다. **다른 전이 조합으로 재확인**해야 채택.
   실증: offset 항 246조합을 `2022→2024` 하나로 줄 세우니 상위 15개가 전부 `both`를 포함해
   "구조적 신호"처럼 보였다. 전이를 6개로 늘리니 **`both`는 그 측정 하나에서만 이겼다**
   (2022→2021 V0 +30.8 vs +8.0, 2024→2021 +31.8 vs +6.1, 2021 출처면 −1669). 선택 인공물.
   ⚠️ **"상위권 구성이 일관되다"는 독립 증거가 못 된다** — 측정이 하나면 그게 당연하다.

6. 🔥 **평균 낼 전이는 배포 조건과 같은 것만.** 전이를 늘리는 것만으로 5항이 지켜지지 않는다.
   **출처 연도(계수를 적합하는 연도)가 배포와 다르면** 평균이 결론을 뒤집는다.
   실증: O2a(`c=0`) 전이 12개 **평균 +32.9 / 중앙 +1.8**, 차이 전부가 **우리가 절대 적합하지 않는 2021** 출처.
   배포는 항상 **2024 출처 → 2025 목표** → 쓸 전이는 출처 2022~2024 · 목표 2023~2024, 최근접은 `2023→2024`.
   ⚠️ **방향을 구분할 것**: 이상한 연도가 **목표**면 멀쩡하고 **출처**일 때만 깨진다
   (2024→2021 **+36.8** vs 2021→2024 **−39.4**). 섞으면 "불안정"으로 오진한다.

⚠️ 이 기준을 한 세션에 세 번 고쳤다(합계→fold별→2024 주기준). 매번 "이제 맞다"고 판단했다. 새 기준으로 뭘 채택하기 전에 **그 기준이 뭘 측정하는지부터** 확인할 것.

## 5. 🔴 로컬 절대값은 LB 순위를 못 매긴다

| 로컬 2024 | LB 2025 | run |
|---|---|---|
| 199.5 | 274.50 | 001 LGBM(선수ID 범주형) |
| 415.6 | 549.51 | 주최측 baseline RF |
| 783.3 | 881.73 | **003** CatBoost+cat_features+FE10, d6, 3시드 |
| 795.8 | 876.37 | 004 = 003 +cond |
| 803.4 | 842.14 | 005 = 004 +depth8 |
| 810.2 | 945.40 | 009 = 003 +실패모드 offset |
| 807.6 | 950.96 | 010 = 009 +성공모델 시드 7 |
| — | 985.09 | 011 = 010 +**전역 로짓 이동 절반** (−0.0208) |
| — | 998.00 | 012 = 이동 전량 (−0.0416, 예측평균 0.4873→0.4770) |
| **858.6** | — | **013 = 003 +시즌내 성적 분해 4열** (003 대비 단일변수 **+75.3**) |
| — | **1051.73** | **015 = 013 +offset +이동**(−0.0438), 시드 3 ← **최고·현 기준선** |

🔥 **시즌 base rate 이동 보정 = +47.04.** 트리가 미관측 시즌을 외삽 못 해 2025 예측이
2024 수준에 갇혀 있었다. 10 문서의 KBO 환경분석 추정(2025 ≈ 0.477)을 상수로 박았다.
**절반 → 전량 2단계로 냈고 두 번 다 예측 적중**(+31.8→실측 +34.13, +13.0→+12.91).
2점으로 풀면 `δ(010 편향)=0.01086`, `B=r(1−r)=0.24998`, **최적 이동량 0.01086**(012는 0.01030)
→ **남은 여지 +0.12. 축 종료.**
- 🗑️ §7의 "사후 로짓 이동 −469.5"는 **폐기**. 그건 하락폭을 fold에서 **적합**한 것이고
  여기는 **외부 도메인 추정치를 상수로 박은 것**이다. 성격이 다르다.
- ⚠️ LB 반응으로 이동량을 역산해 더 미는 건 **δ 프로브 = 규칙 회색지대.** 안 했고 할 필요도 없다.
- ⚠️ 이동량은 **학습 때 계산해 meta에 저장**(`logit_shift`). test 평균을 보고 정하면 규정 위반.

🔥🔥🔥 **시즌내 성적 분해 = 로컬 +75.3 / LB +53.7 (08 §5-10).** 단일 피처 축 역대 최대.
`asof_*`는 시즌 리셋 없는 통산이라 리그 하락만큼 늘 위로 치우친다(2024 편차 +.0253).
**통산은 누적이므로 직전 시즌 말 기준점을 빼면 그 시즌 성적만 남는다**: `(n₁r₁−s₀)/(n₁−n₀)` → 편차 **+.0033**.
- 🔴 **모델이 스스로 못 만든다** — `n₀,s₀`가 어느 컬럼에도 없고 선수 ID 조회는 금지(§3). **새 피처가 아니라 새 정보.**
  §7의 FE 8종이 전멸한 이유가 이것이다(전부 기존 컬럼 재조합 = 새 정보 0). **새 축을 찾을 땐 "정보가 느는가"를 먼저 물을 것.**
- 🔴 **shift·offset과 겹치지 않는다.** 최적 평균보정 후에도 +74.9(레벨이 아니라 판별력), offset `b` −0.0928→−0.0990.
- 규정: 쓰는 것은 **그 행 자신의 `asof` + 학습데이터**뿐. test 다른 행 0개 = 규칙 4번이 요구하는 정확히 그 둘.
  기준점 표(투수 792·타자 830)는 `global_mean`·`mu`와 같이 zip에 싣는다. 실제 test 5행으로 전제 확인(dn=+380/+399).
- 결측 0: 신인은 `n₀=0` → 통산이 곧 그 시즌 성적(자동 정답). `dn<0` 0건.

**전이율**: offset +26.6→**+63.7**(2.40) · 시드 +3.8→+5.6(1.47) · **시즌내 +75.3→+59.2(0.79)** · cond +12.5→**−5.4** · depth8 +7.6→**−34.2**
→ **out-of-year 델타는 부호가 유지되고, fold 내부 델타는 부호가 뒤집힌다.**
→ 🔴 **"증폭된다"는 틀렸다** (2026-08-24 교정). 관측 2개로 세운 규칙을 세 번째(0.79)가 깼다.
  **배율은 예측 불가.** 밴드를 적을 땐 로컬 델타의 **0.8~2.4배** 구간으로 잡을 것.
→ 010은 로컬이 009보다 **낮은데 LB는 높다.** 로컬 절대값 최적화 중단(783 위에서 반비례 3점 확인).
→ **제출은 반드시 단일 변수.** #5가 cond+depth8을 동시에 바꿔 하루 날렸다.
→ **LB 공개 모델(주최측 baseline 등)의 로컬 점수를 재면 제출권 0으로 대응점이 는다.** 제출은 팀 공유 **5회/일**.

## 6. 현재 구성 — run 015 (LB 1051.73)

```
성공모델: CatBoost d6, lr .05, cat_features 지정, FE 10개 + 시즌내 분해 4열, 3시드
          ins_{pitcher,batter}_{success_rate,n}  ← build_anchor()가 만든 기준점으로 분해
offset  : logit(p) = logit(p_success) + b·(logit(p_mr)−mu_mr) + c·(logit(p_wayoff)−mu_wayoff)
          보조모델 3시드(009 복사, 003 피처 57열). b=−0.0990, c=+0.0074
shift   : logit_shift = −0.043768.  2025 추정 0.4762 = (선형외삽 .4747 + 검증편향역산 .4778)/2
          ← 둘 다 train만 사용. 012의 0.477(KBO 공개자료)은 규정 2-3 회색지대라 제거했다
```
⚠️ **보조모델은 아직 003 피처(57열)** 다. 주모델(61열)과 Pool을 공유하면 CatBoost가 죽는다
→ `offset.aux_feature_cols`에 목록 저장, `prepare()`로 **열 부분집합만** 다르게 뽑는다
(`engineer()`는 한 번만 — 61 ⊃ 57이므로).
⚠️ **피처를 바꾸면 `make_valpred.py`를 반드시 다시 돌린다.** offset 계수는 성공모델의
out-of-sample 2024 예측에서 적합하는데, 캐시가 옛 피처 구성이면 계수가 어긋난다.
- 🔴 **`a`(스케일)·`d`(절편)를 적합하면 안 된다** — 그게 calibration이고 시즌 전이가 깨진다(무제약형: 자기연도 +53.8 → 한 해 건너 **−210~−638**). **`a=1·d=0` 고정.**
- 🔴 **`mu`는 학습 때 계산해 meta에 저장.** test에서 평균 내면 규정 위반(test 내부 행간 통계).
- 이 구조의 장점 = **메인 재학습 불필요 → 용량 증가 0 + 재보정 없음** → §3의 두 함정을 모두 회피.
- 순위 (2026-08-24): 우리 **1051.73**(팀 제출 57회). 08-13의 998.003=212위에서 갱신.
  **1위 1421.99**(제출 49회) / 2위 1224.42 / 3위 1218.66(**제출 4회**) / 10위 1171.22 / **100위 1124.70**.
  → **100위까지 −73.0** (08-13 −112.0 → 39점 좁혔다). 컷은 08-13~24에 **+5.96/일**로 올랐다.
  Brier 환산: 우리 .24682 / 100위 .24664 / 1위 .24590.
  ⚠️ **상위권은 제출이 적다** — 3위 4회 / 31위 5회 / 41위 4회. **탐색으로 얻는 점수가 아니다.**
  10~100위가 1124~1171 폭 46에 밀집. 우리와의 거리(73)가 그 밀집 폭보다 크다.

🔴 **offset 축(라벨·형태)은 2026-08-13자로 닫혔다.** 단, **성분의 값을 더 정확히 재는 것**은 별개다(아래 2번).
- **라벨 축**: `y=0 ⟺ (M∪R) ⊎ W` 예외 0건 → 타깃 분해 성분은 **정확히 2개, 둘 다 사용 중** (08 §5-8).
- **형태 축**: 더하기(V1·V3·V6·`both`) 전멸, 빼기(O2a `c=0`)도 배포 최근접 전이 +0.1로 기각 (08 §5-4).

**다음 방향** — 이 순서로만:
1. **델타는 out-of-year, 제출은 단일 변수.** 지금까지 유일하게 LB에서 부호가 맞은 측정법이다.
2. 🔥 **시즌내 분해를 더 밀 것** (08 §5-10 미착수 항목). 지금 쓰는 건 `success` 하나뿐이다.
   - 보조모델(mr/wayoff)은 아직 003 피처 복사본 — 시즌내 분해를 안 받았다
   - `middle`/`reverse`/`ball`/`strike`도 같은 편차를 갖는다(통산 .1557→시즌내 .1715 등). 주모델 피처로 추가 가능
   - 시드 3 → 7 (**+5.6 기지**)
3. **팀원 트랙 병합**(league-rate baseline + p_matchup, 별도 +14.7 / 08 §5-11).
   ⚠️ 시즌내 분해와 **중복 가능** — baseline은 시즌 레벨의 *외삽*, 시즌내는 *관측*이다.
4. **로컬 관측점 확보**(LB 공개 모델의 로컬 점수 측정, 제출권 0).

🔥 **새 축을 찾을 때의 유일한 판별식: "정보가 느는가, 표현이 느는가."**
소진 목록 40항목은 **전부 기존 컬럼 재조합 = 정보 증가 0**이라 전멸했다. 시즌내 분해는
모델이 도달할 수 없는 값(`n₀,s₀`)을 넣어서 이겼다. **표현력을 늘리는 방향은 재시도 금지.**

## 7. 소진 목록 (재시도 금지)

하이퍼파라미터 9종(±15 노이즈) · FE 추가 8종 · 범주형 확장 · 시즌 가중치 · row_id · 깊이조합 앙상블 · grow_policy 혼합(개수 맞추니 +0.8) · 신경망(단독 311, 혼합 손해) · trackman 물리피처 · calibration · **asof 재중심화**(−14.9) · **BrierScore 조기중단**(결과 완전 동일, 무효) · **구종 피처**(+0.6) · **5클래스 다중분류**(용량 5배) · `1−s−m−r` 뺄셈 파생 ·
**최근성 개인기록 기각**: 1시즌 창 −28.5 / 편차형 −12.6 / 3시즌 창 fold별 +19.5·−21.6·−7.0으로 부호 갈림 → **개인기록은 신선도보다 표본 수가 지배.**
**로컬만 오르고 LB에서 진 것**: cond · depth 8 (교정 기준으로 재판정하니 둘 다 원래 탈락. LB −39.6과 일치).
**offset 축 전체**: 항 추가(`middle`계열 −33~−63 · `ball`/`strike`/`inplay` +0.2~1.4 · `both` 선택 인공물) · 항 제거(O2a `c=0`, 배포 최근접 +0.1) · 보조모델 시드 증가(0) · pre-2023 F 제외(−9.5) · **새 보조 라벨 탐색 자체**(성분 2개뿐 = 구조적 종료).

## 8. 제출 규칙·워크플로우

- `submit.zip` = **최상위에 `model/` + `script.py` + `requirements.txt`**. 추가 최상위 폴더 있으면 설치오류.
- script.py는 `./data/test.csv`·`sample_submission.csv` 읽고 → `./output/submission.csv`. `./data`는 읽기전용, 서버가 실제 test로 교체.
- 제한: 추론 ≤10분(24.6만행) · 설치 ≤10분 · zip ≤10GB · **1일 5회**. 오류 2종: 설치오류(차감 X) / **제출오류=script 실행실패(차감 O)**.
- 서버: Ubuntu22, Python **3.11**, L4 22.4GB, 6vCPU/28GB, **오프라인**. 기본설치에 **lightgbm/xgboost/catboost 없음** → requirements.txt에 명시(서버 기본 패키지는 재명시 말 것).

1. `train_local.py` 상단 `RUN` 이름 변경 후 실행 — **기존 RUN 있으면 실행 거부**(덮어쓰기 방지)
2. **`scratchpad/verify_submit.py <zip>` 전체규모 검증 필수** — 245,789행 가짜 test로 구조·시간·결측·범위 확인
3. DACON 업로드 → `result.json`의 `lb_2025`에 손으로 기록(대응표 축적)

- zip은 **Python `zipfile`**로 생성. `Compress-Archive`는 백슬래시 경로라 Linux에서 깨진다.
- 모델은 **네이티브 포맷**(`.cbm`/`.txt`), **pickle 금지**(로컬 3.14 vs 서버 3.11).
- **FE는 `common/features.py` 한 곳에만** — 학습·추론이 같은 함수를 써야 불일치가 원천 차단. zip에서 빠지면 `ModuleNotFoundError` = 제출 1회 차감.
- `global_mean`·`mu`·조건부 표는 **학습 때 만들어 zip에 실는다.** test에서 재계산 = 규정 위반 + 학습과 불일치.

## 9. 이 머신 함정

로컬 Python **3.14**: pandas3.0.5, numpy2.5.1, sklearn1.9.0, lightgbm4.7.0, **catboost1.2.10**, torch2.13.0+cpu, pypdf.

| 증상 | 원인 | 해결 |
|---|---|---|
| 즉시 종료, 로그 빔 | Windows stdout cp949에 유니코드 | `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')` — **파일당 1회만**(재래핑 시 `ValueError: I/O operation on closed file`) |
| 로그가 끝까지 안 보임 | UTF-8 래퍼 블록 버퍼링 | `print(..., flush=True)` |
| 종료코드 `-1073741676` | CatBoost `thread_count=0` | **CatBoost는 `-1`** (LightGBM만 0=전체) |
| 학습 4배 느려짐 | 6코어에 동시 실행 | 무거운 학습은 순차로 (431s→1819s 실측) |
| PDF 안 읽힘 | poppler 미설치 | `pypdf`로 텍스트 추출 → scratchpad txt → Read (원본 무수정) |
| `git add` 가 `open(...): No such file or directory` (파일은 실제로 있음) | 경로에 Windows 예약 장치명 — `aux` `con` `prn` `nul` `com1~9` `lpt1~9`. 디렉토리 이름이어도 걸린다 | 이름 변경(`aux` → `auxpred`). Python·MSYS `cp`·`ls`는 전부 통과하고 git만 막혀서 늦게 드러난다 |

- Bash 워드분할: 한글/공백 경로 loop 깨짐 → python glob.
- ⚠️ **스크립트로 문자열 치환했으면 반드시 검증**하고 출력할 것. 치환은 조용히 실패한다(공백·따옴표 한 글자). 실제로 4개 중 1개가 실패해 변수 미정의로 30분짜리 학습이 돌았다.
  ```python
  ast.parse(src)                     # 문법
  for pat in [...]: pat in src       # 각 치환이 실제 적용됐는지
  ```
