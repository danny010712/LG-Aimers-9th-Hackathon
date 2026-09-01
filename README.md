# LG Aimers Phase 2 — 제구 성공 확률 예측

투구 단위 이진분류.
지표 = **Brier Skill Score** `max(0, 100000×(1 − Brier/[r(1−r)]))`.

```
읽기 전에 :
- 코드의 많은 부분이 Claude Code로 작성되었습니다. 얘가 주석을 엄청 복잡하게 달아뒀는데.. 무시하셔도 됩니다.
- 조금이라도 이상하거나 필요없어 보이는 부분이 있다면 말씀해주세요. 어이없는 실수가 있을 수도 있습니다.
- 궁금한 거 있으시면 언제든 물어봐주세요.
```

## 현황 (2026-09-02, 대회 마감)

| 항목 | 값 |
|---|---|
| LB 확정 최고 | **1080.349** ([submit220](test/runs/220_shift_multiclass_joint_bh)) |
| 마지막 배포(LB 미확인) | 222 = 220 +`team13_transition`. 마감 임박으로 **검증 스킵 긴급배포**([상세](CLAUDE.md#6)) |
| 100위 컷 | ~1130 |

⚠️ 대회가 2026-09-02 09:59에 마감됐고 222는 제출하지 못했습니다. 아래 "현재 구성"은
220/222가 쓰는 **조인트 MultiClass 구조**입니다 — 021까지 썼던 "성공모델+보조모델+offset
3단 파이프라인"(바로 아래 옛 설명)은 219에서 대체됐습니다. 자세한 배경은 [CLAUDE.md §6](CLAUDE.md).

## 현재 구성 (219~222, 조인트 MultiClass)

```
모델    CatBoost MultiClass(loss=eval=MultiClass), depth 6, lr .05, 3시드(42/7/2024)
        클래스: 0=mr(middle|reverse) 1=wayoff 2=success — 라벨 복원된 행만 학습
        하나의 모델이 success/mr/wayoff를 동시에 예측 → 옛 구조의 보조모델(mr/wayoff)이 불필요해짐
피처    FE + 시즌내 성적 분해 4열 + cond_ph(+dev) + cond_bh(+dev) [+team13_transition, 222부터]
offset  logit(p) = logit(P_success) + b·(logit(P_mr)−mu_mr) + c·(logit(P_wayoff)−mu_wayoff)
        같은 모델의 출력을 재사용(별도 보조모델 없음). b,c는 훨씬 작음(조인트학습이 흡수)
shift   전역 로짓 이동 (2025 base rate를 train만으로 추정해 상수만큼 미는 것, build_shift.py)
```

- `a`·`d`를 적합하면 그게 calibration이고 시즌 전이가 깨집니다(219 이후도 동일하게 고정).
- `mu`·`logit_shift`·기준점 표·cond 표는 **학습 때 계산해 zip에 싣습니다.**
  test에서 계산하면 test 행간 통계라 규정 위반입니다.
- 학습: `train_multiclass.py` 하나로 검증(3시드)→전체재학습→zip까지 끝납니다(옛 구조의
  `train_local.py`+`train_offset.py` 두 스크립트가 이걸로 대체됨).

<details>
<summary>옛 구성 (003~021, 2026-08-25 시점 — 참고용)</summary>

```
성공모델  CatBoost depth 6, lr .05, cat_features 지정, 3시드
          FE 10개 + 시즌내 성적 분해 4열 (61열)
offset    logit(p) += b·(logit(p_mr) − mu_mr) + c·(logit(p_wayoff) − mu_wayoff)
          b = −0.0990, c = +0.0074,  a=1·d=0 고정
shift     전역 로짓 −0.043768  (2025 base rate 추정 0.4762)
platoon   logit(p) += 2.12 · split[pitcher_id, 좌우일치]
```

- 보조모델(`mr`/`wayoff`)은 003 피처(57열)입니다. 주모델(61열)과 Pool을 공유하면 CatBoost가
  죽어서 `offset.aux_feature_cols`로 열 부분집합만 따로 뽑습니다.
</details>

## 주요 코드

| 파일 | 역할 |
|---|---|
| [features.py](test/common/features.py) | 피처 정의. 학습·추론이 **이 파일 하나를 공유**합니다 |
| [cond.py](test/common/cond.py) | 개체×상황 EB스무딩 조건부표(cond_ph/cond_bh 등) |
| [script.py](test/common/script.py) | 평가 서버 추론 스크립트 (옛/조인트 구조 둘 다 지원) |
| [train_multiclass.py](test/train_multiclass.py) | **현재 쓰는 학습 스크립트.** 조인트 MultiClass, 검증→재학습→zip을 한 번에 |
| [build_shift.py](test/build_shift.py) | 시즌 base rate 보정 (옛/조인트 구조 둘 다 지원) |
| [verify_submit.py](test/verify_submit.py) | **제출 전 필수.** 전체규모 실행 검증 |
| [recover_labels.py](test/recover_labels.py) | 숨은 투구 라벨 복원 |
| [audit.py](test/audit.py) | `why-not <키워드>`로 이미 닫힌 축 검색, `record`로 로컬Δ↔LBΔ 전수표 |
| [closed.tsv](test/closed.tsv) | 재시도 금지 목록(기각 실험 90여 건, 데이터) |
| [train_local.py](test/train_local.py)/[train_offset.py](test/train_offset.py)/[build_platoon.py](test/build_platoon.py) | 옛 구조(003~021)용 — 조인트 구조에선 안 씀, 하위호환용으로 남겨둠 |

## 주요 변화

| 변경 | 점수 | 모델 |
|---|---|---|
| CatBoost + `cat_features` 지정 + 선수ID 범주형 제거 + `season` 수치형 + depth 6 + FE 10 | **881.73** | [003](test/runs/003_catboost_fe) |
| 제구 실패 유형별 offset | **+63.7** | [009](test/runs/009_offset) |
| 시드 3 → 7 | +5.6 | [010](test/runs/010_offset_seeds7) |
| 시즌 base rate 로짓 이동 | **+47.04** | [012](test/runs/012_shift_full) |
| **시즌내 성적 분해** | **+53.7** | [015](test/runs/015_shift_inseason) |
| 투수 좌우편차 offset | +5.27 | [021](test/runs/021_platoon) |
| `cond_ph`(투수×타자손) 트리 피처 단독 | +5.55 | [044](test/runs/044_platoon_condph) |
| success/mr/wayoff를 CatBoost MultiClass 하나로 조인트학습(구조 전환) | +0.632 | [219](test/runs/219_shift_multiclass_joint) |
| 219 +`cond_bh`(타자×투수손) | **+12.6** (이 프로젝트 단일축 최대 실이득) | [220](test/runs/220_shift_multiclass_joint_bh) ← LB 확정 최고 |
| 220 +`team13_transition`(체제전환 지시자) | ? (검증 스킵 긴급배포, LB 미확인) | [222](test/runs/222_shift_multiclass_joint_team13) |

**003** — `cat_features`를 지정 안 하고 `OrdinalEncoder`로 넣으면 CatBoost의 ordered target
statistics가 아예 안 돕니다(+114). FE 10개(+98)는 주자·압박 플래그를 전부 뺀 것입니다.

**009** — 실패에는 세 종류(한복판/크게벗어남/포수요구반대)가 있는데 정답은 1비트뿐이라 합쳐지며
사라집니다. `asof_*_rate`가 `누적성공/누적투구` 구조라 **연속 두 투구의 차분으로 투구별 사건을
복원**했습니다([recover_labels.py](test/recover_labels.py), 147만 행 100% 일치). 그 라벨로 보조모델
둘을 학습해 로짓을 밉니다. `실패 ⟺ (한복판∪반대) ⊎ 크게벗어남`이 예외 0건이라 성분은 정확히 2개입니다.
본 모델은 **재학습하지 않고 파일째 복사**합니다 — `b=c=0`이면 003과 동일한 출력이라 진짜 단일 변수입니다.

**012** — 리그 성공률이 6년 연속 하락(.565 → .486)하는데 트리는 미관측 시즌을 외삽 못 해서 2025를
2024와 똑같이 취급합니다. Brier는 평균 편향 δ에 제곱으로 반응하니 최적 이동량은 `s = δ`입니다.
절반(011) → 전량(012) 두 단계로 냈고 **두 번 다 예측이 맞았습니다.**

**015** — `asof_*`는 **시즌 리셋이 없는 통산**이라 리그가 하락하는 만큼 늘 위로 치우칩니다
(2024 편차 **+.0253**). 통산은 누적이므로 **직전 시즌 말 기준점을 빼면 그 시즌만 남습니다**:

```
시즌내 = (n₁·r₁ − s₀) / (n₁ − n₀)      n₁,r₁ = 그 행의 asof / n₀,s₀ = train으로 만든 기준점
편차 +.0253 → +.0033
```

`n₀`,`s₀`는 **어느 컬럼에도 없고** 투수 ID 조회는 과적합 때문에 금지한 것입니다.
→ 기존 피처를 어떻게 조합해도 도달할 수 없는 값입니다.

> 지금까지 제가 시도한 FE 8종이 전부 실패한 이유도 이걸로 설명됩니다 — **전부 기존 컬럼의 재조합이라
> 정보가 하나도 늘지 않았습니다.** 새 아이디어를 볼 때 "표현력이 느는가"가 아니라
> **"정보가 느는가"** 를 먼저 물어보면 좋을 것 같습니다.

**021** — 투수마다 좌/우 타자 상대 제구가 다른데 그 **차이의 크기**가 데이터에 없습니다.
학습으로 표를 만들어 최종 로짓에 더합니다. 같은 정보를 트리 피처로 준 `cond`는 −5.4로 졌는데,
잔차 offset으로 주니 +5.27입니다(용량 증가 0, 재학습 없음).

## 🔴 반드시 지킬 것

| 하지 말 것 | 결과 |
|---|---|
| `pitcher_id`/`batter_id` 범주형 | 개인 암기 → 과적합 (612.9 → 199.5) |
| `season` 범주형 | 미관측 시즌에 값이 없어 붕괴 (780 → 605) |
| `cat_features` 미지정 | −114 |
| offset의 `a`·`d` 적합 | 자기연도 +53.8 → 한 해 건너 **−210~−638** |
| 사후 calibration | 로컬 천장 +89 / 실제 **−23** |
| depth 8 | 로컬 +7.6 / 실제 **−34.2** |

## 🔴 로컬 델타를 믿는 법

| 측정 방식 | 로컬 → 제출 |
|---|---|
| **out-of-year** (계수를 T−1에서 맞춰 T에 적용) | offset +26.6 → **+63.7** · 시즌내 +75.3 → **+59.2** |
| fold 내부 | cond +12.5 → **−5.4** · depth8 +7.6 → **−34.2** |

out-of-year로 잰 델타는 **부호가 유지되고** fold 내부 델타는 뒤집힙니다. 제출은 반드시 단일 변수로.

## 재현 순서

- 잘 돌아갈지 모르겠습니다. 해보고 안 되면 물어봐주세요.

```bash
cd test
python train_multiclass.py   # 조인트 모델. 상단 RUN 이름 변경 필수(기존 RUN 있으면 실행 거부)
                              # 검증(3시드)→전체재학습→zip까지 이 한 스크립트 안에서 끝남
python build_shift.py        # 전역 로짓 이동 → 최종 submit zip
python verify_submit.py runs/<RUN>/submit<NNN>.zip    # 제출 전 필수
```

옛 구조(003~021, `train_local.py`→`train_offset.py`→`build_shift.py`→`build_platoon.py`)도
하위호환으로 여전히 돌아가지만 지금 안 씁니다.

전부 `test/`를 작업 디렉토리로 가정합니다(경로가 상대경로입니다).

`train_offset.py`·`build_shift.py`·`build_platoon.py`는 **이전 run 산출물을 파일째 복사**해서
단일 변수를 보장하는 구조라, 다음이 repo에 함께 있어야 계보가 이어집니다.

| 경로 | 용도 |
|---|---|
| `test/artifacts/auxpred/*.npy` | 003 피처(57열)로 만든 2024 검증 예측. 009~012 계보 |
| `test/artifacts/auxpred_ins*/` | 시즌내 분해 피처로 만든 것. 013 이후 계보 |
| `test/runs/*/model/` | `BASE_RUN`·`AUX_FROM`이 여기서 복사됩니다 |

⚠️ **피처를 바꾸면 검증 예측 캐시가 낡습니다.** offset 계수를 옛 기준으로 잡는데 에러가 안 나서
알아채기 어렵습니다. (지금은 `train_local.py`가 캐시를 직접 저장합니다.)

⚠️ **`ANCHOR_SPECS`([features.py](test/common/features.py))를 바꾸면 `anchor.csv` 스키마가 같이 바뀝니다.**
옛 run 위에 뭔가를 얹을 때 어긋나면 추론이 `KeyError`로 죽습니다.

## 기록 문서

1. [CLAUDE.md](CLAUDE.md) — 하드룰 + §7 재시도 금지 목록.
2. [08_Phase2_데이터_및_전략.md](강의정리/08_Phase2_데이터_및_전략.md) — 실험·제출 전기록
3. [09_피처분석_노트.md](강의정리/09_피처분석_노트.md) — 숨은 라벨 복원, trackman 감사
4. [10_KBO_시즌환경_분석.md](강의정리/10_KBO_시즌환경_분석.md) — 2025 base rate 추정
   - ⚠️ 이 문서의 0.477은 KBO 공개자료 기반이라 **규정 2-3 회색지대**입니다.
     015부터는 train만으로 만든 0.4762로 대체했습니다.

- 위 문서들에 적힌 금지 사항을 반드시 지켜야할 필요는 없습니다. 그쪽에서 뭔가 성과가 나올지도 모르는 일이니까요.

## 제출 규칙

- `submit.zip` = **최상위에 `model/` + `script.py` + `requirements.txt`**. 추가 최상위 폴더 있으면 설치오류
- script.py는 `./data/test.csv`·`sample_submission.csv` 읽고 → `./output/submission.csv`
- 서버: Ubuntu 22, Python **3.11**, L4 22.4GB, 6vCPU/28GB, **오프라인**.
  catboost/lightgbm/xgboost 기본 미설치 → requirements.txt에 명시
- 추론 ≤10분, 설치 ≤10분, zip ≤10GB, **1일 5회**
- 설치오류는 횟수 차감 안 되지만 **script 실행 실패는 차감됩니다**
- zip은 Python `zipfile`로 생성. `Compress-Archive`는 백슬래시 경로라 Linux에서 깨짐
- 모델은 네이티브 포맷(`.cbm`). **pickle 금지** (로컬 3.14 vs 서버 3.11)
- `script.py`가 import하는 모듈은 **전부 zip에** 들어가야 합니다
  ([verify_submit.py](test/verify_submit.py)가 자동 대조합니다)
