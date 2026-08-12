# LG스포츠 해커톤 Phase 2 — 제구 성공 확률 예측

투구 단위 이진분류. 지표 = **Brier Skill Score** `max(0, 100000×(1 − Brier/[r(1−r)]))`.
순위(AUC)가 아니라 **확률 정확도** 게임. 상수 예측 = 0점.

## 현황 (2026-08-12)

| 항목 | 값 |
|---|---|
| 우리 LB | **998.00** (run `012_shift_full`) |
| 100위 컷 | 1046.29 → **+95.3 필요** |
| 1위 | 1156.21 |
| 수료선 | 549.51 (통과 완료) |

Brier 환산으로 우리 .24743 vs 100위 .24719 — **차이 .00024**. 상위 10명이 1127~1156에 몰려 있음.

## 먼저 읽을 것 (순서대로)

1. **[CLAUDE.md](CLAUDE.md)** — 하드룰 + **§7 재시도 금지 목록**. 제출권 태우기 전에 필독
2. [강의정리/08_Phase2_데이터_및_전략.md](강의정리/08_Phase2_데이터_및_전략.md) — 실험·제출 전기록
3. [강의정리/09_피처분석_노트.md](강의정리/09_피처분석_노트.md) — 숨은 라벨 8종 복원
4. [강의정리/10_KBO_시즌환경_분석.md](강의정리/10_KBO_시즌환경_분석.md) — 2025 base rate 0.477 추정 (+47.04의 근거)

## LB에서 실제로 먹힌 것

| 변경 | LB 델타 |
|---|---|
| CatBoost + `cat_features` 지정 + 선수ID 범주형 제거 + `season` 수치형 + depth 6 + FE 10 | 274.50 → **881.73** |
| 실패모드 offset (`mr`/`wayoff` 보조모델, `a=1·d=0` 고정) | **+63.7** |
| 시즌 base rate 로짓 이동 (−0.0416, 2025 추정 0.477) | **+47.04** |
| 시드 3 → 7 | +5.6 |

## 로컬은 올랐는데 LB에서 진 것

| 변경 | 로컬 | LB |
|---|---|---|
| `cond` 조건부 개인기록 | +12.5 | **−5.4** |
| depth 8 | +7.6 | **−34.2** |
| 사후 calibration | 천장 +89 | **−23** |

## 방법론 — 이게 진짜 자산

1. **out-of-year 델타만 전이됨.** 계수를 연도 T−1에서 맞춰 T에 적용해 측정. fold 내부 델타는 LB에서 **부호가 뒤집힘**(위 표가 증거). out-of-year는 2.4배 증폭되어 전이됨.
2. **랜덤 KFold 금지.** base rate가 2019 .565 → 2024 .486로 하락 → 점수 3.5배 부풀림. 시간순 분할만.
3. **로컬 절대값은 LB 순위를 못 매김.** run 010은 009보다 로컬이 낮은데 LB는 높음.
4. **제출은 반드시 단일 변수.** 제출 #5가 `cond`+`depth8`을 동시에 바꿔 하루 날림.
5. **여러 후보를 한 측정으로 줄 세운 순위는 증거가 아님.** 최고값은 노이즈만으로 ~2.9σ 위. 다른 전이 조합으로 재확인해야 채택.
6. **fold 2024가 주 기준.** 2021·2022는 `game_type` F−R 격차가 커서 점수의 60~75%가 공짜였음.

## 현재 구성 — run 012

```
성공모델: CatBoost depth 6, lr .05, cat_features 지정, FE 10개, 7시드 평균
offset  : logit(p) = logit(p_success) + b·(logit(p_mr) − mu_mr) + c·(logit(p_wayoff) − mu_wayoff)
          b ≈ −0.105, c ≈ +0.011, a=1·d=0 고정, mu는 학습 때 저장
shift   : 위 결과에 전역 로짓 −0.041639 (예측 평균 0.4873 → 0.4770)
```

`a`·`d`를 적합하면 그게 calibration이고 시즌 전이가 깨짐. `mu`와 `logit_shift`는 **학습 때 계산해 meta에 저장** — test에서 평균 내면 test 행간 통계로 규정 위반.

## 셋업

데이터는 이 repo에 없음. DACON에서 받아 `open/data/`에 둘 것.

```
open/data/  train.csv  test.csv  trackman_history.csv  sample_submission.csv  first_game.csv
```

학습 스크립트는 `test/data/`를 읽음. `open/data`로 가는 **정션**을 만들어야 함 (Windows):

```cmd
mklink /J test\data open\data
```

Linux/macOS면 심볼릭 링크:

```bash
ln -s ../open/data test/data
```

## 재현 순서

```bash
cd test
python train_local.py     # 성공모델 학습. 상단 RUN 이름 변경 필수 (기존 RUN 있으면 실행 거부)
python train_offset.py    # 보조모델 + offset 계수
python build_shift.py     # 전역 로짓 이동 적용 → submit zip
```

제출 전 **전체규모 검증 필수** — 245,789행 가짜 test로 구조·시간·결측·범위 확인:

```bash
python scratchpad/verify_submit.py <zip>
```

## 제출 규칙

- `submit.zip` = **최상위에 `model/` + `script.py` + `requirements.txt`**. 추가 최상위 폴더 있으면 설치오류
- script.py는 `./data/test.csv`·`sample_submission.csv` 읽고 → `./output/submission.csv`
- 서버: Ubuntu 22, Python **3.11**, L4 22.4GB, 6vCPU/28GB, **오프라인**. catboost/lightgbm/xgboost 기본 미설치 → requirements.txt에 명시
- 추론 ≤10분, 설치 ≤10분, zip ≤10GB
- zip은 Python `zipfile`로 생성. `Compress-Archive`는 백슬래시 경로라 Linux에서 깨짐
- 모델은 네이티브 포맷(`.cbm`/`.txt`). **pickle 금지** (로컬 3.14 vs 서버 3.11)

## ⚠️ 팀 조율

**제출 1일 5회는 팀 전체 공유.** 제출 전 반드시 팀에 알릴 것.
제출오류(script 실행 실패)는 횟수 차감됨. 설치오류는 차감 안 됨.

결과는 `test/runs/<RUN>/result.json`의 `lb_2025`에 손으로 기록 — 로컬↔LB 대응표를 쌓는 게 유일한 나침반.
