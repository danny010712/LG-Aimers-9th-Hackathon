# league-rate baseline 도입 — 변경 요약

## 파일 배치
```
league_baseline/
├── train_local.py       (수정) — 주모델 학습, baseline 주입
├── train_offset.py      (수정) — offset 계수 적합, 캐시 경로 자동 판단
├── build_shift.py       (수정) — 잔여 shift만 계산 (자체 추론 포함)
├── README.md             (이 문서)
└── common/
    ├── league_rate.py   (신규) — baseline 계산·직렬화 유틸
    └── script.py         (수정) — 추론 시 baseline 재계산
```
`common/features.py`, `common/cond.py`, `common/requirements.txt`는 **변경 없음**
(기존 그대로 옆에 두면 됨).

## 검증 상태

실제 CatBoost 1.2.10으로 다음을 직접 확인했습니다.

1. **CatBoost `baseline`은 raw margin(로짓) 스케일이고, `.cbm` 파일에 저장되지
   않는다.** 추론 때 baseline을 다시 안 넣으면 예측 평균이 크게 틀어짐(검증:
   0.66→0.52로 왜곡) — 이게 이번 변경에서 제일 위험한 지점이라 `script.py`,
   `build_shift.py` 양쪽에 baseline 재계산 로직을 넣고 실제로 실행해 확인함.
2. **`train_local.py` → `train_offset.py` → `build_shift.py` → (zip 생성) →
   unzip → `script.py` 실행**까지 합성 데이터(6시즌×R/F, 1,800행)로 **전 과정을
   실제로 돌려 통과 확인.** (진짜 대회 데이터로 성능을 확인한 건 아님 — 코드
   경로가 안 끊기고 논리대로 동작하는지만 확인한 것.)
3. 최종 산출 확률 평균이 목표(0.477) 근처로 정확히 이동하는 것까지 확인.

## 각 파일이 하는 일 (변경점만)

### `common/league_rate.py` (신규)
- `build_table`: (season, game_type)별 실측 성공률 표
- `extrapolate` / `held_out_estimate`: out-of-year 외삽(내부 선형회귀) 또는
  외부 override(10문서 §6-C 추정치) 중 선택해서 미관측 시즌 값을 만듦
- `assign_baseline_logit`: df 각 행에 baseline(로짓) 배열을 붙임
- `table_to_json` / `table_from_json`: meta.json 저장/복원

### `train_local.py`
- 상단에 `USE_LEAGUE_BASELINE`, `LEAGUE_GROUP_COLS`,
  `LEAGUE_EST_2025_OVERRIDE` 추가
- 검증(2019~23→2024) 단계: 2024 baseline은 **2019~23만으로 외삽**(out-of-year,
  override 안 씀 — "진짜로 안 본 해"를 정직하게 시험)
- 검증 단계에서 `success_2024_{seed}.npy`를 `artifacts/auxpred_league/`에 저장
  (offset 계수 적합용 — 기존 `artifacts/auxpred`는 baseline 없는 구모델
  캐시라 재사용 불가)
- 최종(2019~24 전체) 재학습: baseline은 실측 표 그대로 사용
- `meta.json`에 `league_baseline` 블록(표, group_cols, test_override 등) 저장
- zip에 `league_rate.py` 추가 포함

### `train_offset.py`
- `BASE_RUN`을 새 league-baseline 런(`015_league_baseline`)으로 변경
- `fit_offset`이 읽는 success 예측 캐시 경로를 `base_meta["league_baseline"]`
  유무로 **자동 판단**(`artifacts/auxpred_league` ↔ `artifacts/auxpred`)
- mr/wayoff 보조모델은 **이번 변경 범위 밖** — baseline 없이 기존 방식 그대로
  (`AUX_FROM="009_offset"`에서 그대로 복사). 나중에 확장하고 싶으면 이 스크립트의
  주석에 자리를 남겨뒀음.
- `meta = dict(base_meta)`가 `league_baseline` 블록을 자동으로 물려받으므로
  추가 코드 없이 하위 run에 전파됨

### `build_shift.py`
- 기존엔 캐시(`artifacts/sub010.csv.gz`)를 그냥 읽었지만, 이제 **자체적으로
  가짜 test(2024 행 + season→2025)를 만들어 직접 추론**(baseline+offset까지
  반영)한 뒤 평균을 잼 — baseline 유무에 따라 평균이 달라지므로 옛 캐시를
  못 믿음
- 잔여폭이 0.005 미만이면 "baseline이 잘 작동했다"는 진단 메시지 출력,
  크면 경고 출력

### `common/script.py`
- `meta["league_baseline"]["enabled"]`가 True면, raw test.csv의 season/
  game_type으로 baseline을 다시 계산해 성공모델 Pool에 넣음
- mr/wayoff Pool은 baseline 없이(기존과 동일)

## 실행 전 확인할 것

1. `LEAGUE_GROUP_COLS`를 `["season"]`으로 바꾸면 game_type 구분 없이 시즌만
   기준으로 삼는 이전 방식으로 즉시 되돌릴 수 있습니다(요청하신 토글).
2. `train_offset.py`의 `BASE_RUN`, `train_offset.py`/`build_shift.py`의
   `AUX_FROM`, 파일 상단 `RUN` 이름은 실제 실행 시 상황에 맞게 확인해 주세요
   (지금은 015→016→017 체인으로 맞춰뒀습니다).
3. `artifacts/auxpred/mr_2024_*.npy`, `wayoff_2024_*.npy`(기존 캐시)는 그대로
   재사용됩니다 — 새로 만들 필요 없습니다.
4. §3-4 원칙대로, 이 축의 채택 여부는 **로컬 절대값이 아니라 out-of-year
   델타(2022→2024 등)와 실제 LB 제출**로 판단해야 합니다. `train_local.py`가
   출력하는 검증 점수 옆에 경고 메시지를 넣어뒀습니다.
