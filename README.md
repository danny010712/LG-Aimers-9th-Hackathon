# LG Aimers Phase 2 — 제구 성공 확률 예측

투구 단위 이진분류.
지표 = **Brier Skill Score** `max(0, 100000×(1 − Brier/[r(1−r)]))`.

```
읽기 전에 :
- 코드의 많은 부분이 Claude Code로 작성되었습니다. 얘가 주석을 엄청 복잡하게 달아뒀는데.. 무시하셔도 됩니다.
- 조금이라도 이상하거나 필요없어 보이는 부분이 있다면 말씀해주세요. 어이없는 실수가 있을 수도 있습니다.
- 궁금한 거 있으시면 언제든 물어봐주세요.
```

## 현황 (2026-08-13)

| 항목 | 값 |
|---|---|
| 우리 점수 | **998.00** |
| 100위 컷 | 1059.12 → **+61.1 필요** |
| 1위 | 1295.47 |

## 현재 구성 — [submit012](test/runs/012_shift_full)

```
성공모델: CatBoost depth 6, lr .05, cat_features 지정, FE 10개, 7시드 평균
offset  : logit(p) = logit(p_success) + b·(logit(p_mr) − mu_mr) + c·(logit(p_wayoff) − mu_wayoff)
          b ≈ −0.105, c ≈ +0.011, a=1·d=0 고정, mu는 학습 때 저장
shift   : 위 결과에 전역 로짓 −0.041639 (예측 평균 0.4873 → 0.4770)
```

`a`·`d`를 적합하면 그게 calibration이고 시즌 전이가 깨짐. `mu`와 `logit_shift`는 **학습 때 계산해 meta에 저장** — test에서 평균 내면 test 행간 통계로 규정 위반.

- 자세한 설명은 아래 주요 변화 목록을 참고해주세요.

## 주요 코드
1. [features.py](test/common/features.py) : 피처 정의
2. [script.py](test/common/script.py) : 평가 서버 추론 스크립트
3. [train_local.py](test/train_local.py) : 로컬 학습 스크립트
4. [train_offset.py](test/train_offset.py) : 제구 실패 유형별 offset 빌드
5. [build_shift.py](test/build_shift.py) : 시즌 base rate 보정
6. [recover_labels.py](test/recover_labels.py) : 숨은 투구 라벨 복원 

## 주요 변화

| 변경 | 점수 변화 | 모델명 |
|---|---|---|
| CatBoost + `cat_features` 지정 + 선수ID 범주형 제거 + `season` 수치형 + depth 6 + FE 10 | **881.73** | [submit003](test/runs/003_catboost_fe) |
| 제구 실패 유형별 offset (`mr`/`wayoff` 보조모델, `a=1·d=0` 고정) | **+63.7** | [submit009](test/runs/009_offset) |
| 시드 3 → 7 | +5.6 | [submit010](test/runs/010_offset_seeds7) |
| 시즌 base rate 로짓 이동 (−0.0416, 2025 추정 0.477) | **+47.04** | [submit012](test/runs/012_shift_full) |

- 아래는 작성중입니다. 최대한 빨리 완성할게요..

### [submit003](test/runs/003_catboost_fe)

### [submit009](test/runs/009_offset)

### [submit010](test/runs/010_offset_seeds7)

### [submit012](test/runs/012_shift_full)

## 로컬은 올랐는데 실제 제출에서 효과 못 본 것

| 변경 | 로컬 | 제출 |
|---|---|---|
| `cond` 조건부 개인기록 | +12.5 | **−5.4** |
| depth 8 | +7.6 | **−34.2** |
| 사후 calibration | 천장 +89 | **−23** |

## 재현 순서

- 잘 돌아갈지 모르겠습니다. 해보고 안 되면 물어봐주세요.

```bash
cd test
python train_local.py     # 성공모델 학습. 상단 RUN 이름 변경 필수 (기존 RUN 있으면 실행 거부)
python train_offset.py    # 보조모델 + offset 계수
python build_shift.py     # 전역 로짓 이동 적용 → submit zip
```

세 스크립트는 모두 `test/`를 작업 디렉토리로 가정함. 경로가 전부 상대경로임.

`train_offset.py`와 `build_shift.py`는 **이전 run의 산출물을 그대로 복사**해서 단일 변수를 보장하는 구조라, 재학습 없이도 계보를 이어가려면 다음이 repo에 함께 들어 있음:

| 경로 | 용도 |
|---|---|
| `test/artifacts/auxpred/*.npy` | 2019~23 학습 → 2024 검증 예측 13개 (`success` 7시드, `mr`/`wayoff` 각 3시드). offset 계수 `b`·`c`와 `mu` 적합에 씀. 재생성하려면 CatBoost 13개 재학습 = 수 시간 |
| `test/artifacts/sub010.csv.gz` | run 010이 245,789행 가짜 test에 낸 예측. `logit_shift` 산출의 기준 (평균 0.487295) |
| `test/runs/{003,007,009,010,012}/model/` | 현 계보 모델. `BASE_RUN`·`AUX_FROM`이 여기서 파일째 복사됨 |

죽은 축(001·002·004 `cond`·005 `depth8`·006 `grow_policy`·011)의 모델은 제외. `result.json`에 로컬↔LB 기록은 남아 있음.

제출 전 **전체규모 검증 필수** — 245,789행 가짜 test로 구조·시간·결측·범위 확인:

```bash
python scratchpad/verify_submit.py <zip>
```

## CLAUDE가 작성한 학습 기록 및 분석 파일

1. [CLAUDE.md](CLAUDE.md) — 하드룰 + §7 재시도 금지 목록. 제출권 태우기 전에 필독
2. [강의정리/08_Phase2_데이터_및_전략.md](강의정리/08_Phase2_데이터_및_전략.md) — 실험·제출 전기록
3. [강의정리/09_피처분석_노트.md](강의정리/09_피처분석_노트.md) — 숨은 라벨 8종 복원
4. [강의정리/10_KBO_시즌환경_분석.md](강의정리/10_KBO_시즌환경_분석.md) — 2025 base rate 0.477 추정

- 위 문서들에 적힌 금지 사항을 반드시 지켜야할 필요는 없습니다. 그쪽에서 뭔가 성과가 나올지도 모르는 일이니까요.

## 제출 규칙

- `submit.zip` = **최상위에 `model/` + `script.py` + `requirements.txt`**. 추가 최상위 폴더 있으면 설치오류
- script.py는 `./data/test.csv`·`sample_submission.csv` 읽고 → `./output/submission.csv`
- 서버: Ubuntu 22, Python **3.11**, L4 22.4GB, 6vCPU/28GB, **오프라인**. catboost/lightgbm/xgboost 기본 미설치 → requirements.txt에 명시
- 추론 ≤10분, 설치 ≤10분, zip ≤10GB
- zip은 Python `zipfile`로 생성. `Compress-Archive`는 백슬래시 경로라 Linux에서 깨짐
- 모델은 네이티브 포맷(`.cbm`/`.txt`). **pickle 금지** (로컬 3.14 vs 서버 3.11)