"""🔴 새 축을 제안하기 **전에** 돌린다. 문서를 읽는 대신 실행한다.

만든 이유 (2026-08-26): 하루에 기록된 함정을 네 번 밟았다.
  ① §3-E를 지지 근거로 인용했다 — 실제로는 그 축을 경고하는 문서였다
  ② "LB 2/2 적중"이라고 했다 — 최근 두 건만 봤다. 전수는 3/5다 (§4-4 위반)
  ③ 043 기대값을 이중으로 깎았다
원인은 지식 부족이 아니라 **검색 방식**이다. grep은 내가 이미 믿는 것을 돌려준다.
그래서 반박을 강제로 보여주는 도구가 필요하다.

  python audit.py record            주모델 델타 ↔ LB 델타 **전수** (표본 고르기 금지)
  python audit.py why-not <키워드>  문서에서 그 축의 🔴/⚠️/❌ 줄만 뽑는다
"""
import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DOCS = ["../CLAUDE.md", "../강의정리/08_Phase2_데이터_및_전략.md",
        "../강의정리/09_피처분석_노트.md"]

# 주모델이 바뀐 제출만. (run_주모델, run_LB, 라벨, 비고)
# 🔴 새 제출을 하면 **여기에 반드시 추가**한다. 빠뜨리면 표본 고르기가 된다.
PAIRS = [
    ("003_catboost_fe", "004_cond", "003_catboost_fe", "004_cond",
     "cond 4표 번들", "죽은 표 3개가 희석"),
    ("003_catboost_fe", "013_inseason", "012_shift_full", "015_shift_inseason",
     "시즌내 분해 4열", "⚠️ 시드 7→3 동반 = 순수 단일변수 아님"),
    ("013_inseason", "018_inseason_all", "015_shift_inseason",
     "020_shift_inseason_all", "ins 5열 확장", "5열이 서로 겹침(올해 환경)"),
    ("013_inseason", "041_condph", "021_platoon", "044_platoon_condph",
     "cond_ph 단독", "지속성 통과 표 1개"),
    ("041_condph", "049_condphbh", "044_platoon_condph", "052_platoon_phbh",
     "cond_bh 추가", "지속성 ph급인데 실패"),
]


def _val(run):
    j = json.load(open(f"runs/{run}/result.json", encoding="utf-8"))
    x = j.get("val_2024")
    if isinstance(x, dict):
        for k in ("score", "avg3", "after_oof"):
            if k in x:
                return x[k]
    return x


def _lb(run):
    return json.load(open(f"runs/{run}/result.json", encoding="utf-8")).get("lb_2025")


def record():
    print("주모델 델타(≤2023→2024, out-of-year)  ↔  LB 델타  — **전수**\n")
    print(f"{'변경':<18} {'주모델 Δ':>10} {'LB Δ':>9}  {'부호':<5} 비고")
    print("-" * 78)
    hit = pos_hit = pos_n = 0
    for a, b, la, lb_, name, note in PAIRS:
        try:
            dm, dl = _val(b) - _val(a), _lb(lb_) - _lb(la)
        except Exception as e:
            print(f"{name:<18}  (계산 불가: {e})")
            continue
        ok = (dm > 0) == (dl > 0)
        hit += ok
        if dm > 0:
            pos_n += 1
            pos_hit += ok
        print(f"{name:<18} {dm:>+10.2f} {dl:>+9.2f}  {'✅' if ok else '❌':<5} {note}")
    n = len(PAIRS)
    print("-" * 78)
    print(f"전체 부호 적중 {hit}/{n}    **주모델 Δ가 양수일 때 {pos_hit}/{pos_n}**")
    print("\n🔴 양수는 동전던지기다. 실패 둘의 공통점은 크기가 아니라 **중복**이었다:")
    print("   004 = 번들(죽은 표 3개) · 020 = 5열이 서로 겹침")
    print("   → 조건은 'δ>0'이 아니라 'δ>0 **그리고** 추가 열이 서로·기존과 안 겹친다'")


def _closed(kw):
    """closed.tsv 를 먼저 본다 — 판정은 산문이 아니라 여기 있다."""
    import csv
    if not os.path.exists("closed.tsv"):
        return 0
    pat = re.compile(kw, re.I)
    n = 0
    src = (l for l in open("closed.tsv", encoding="utf-8")
           if not l.startswith("#"))
    for r in csv.reader(src, delimiter="\t"):
        if len(r) < 4 or r[0] == "축":
            continue
        if pat.search(r[0]) or pat.search(r[1]):
            n += 1
            print("")
            print(f"🔴 [소진 목록] {r[0]}")
            print(f"   결과: {r[1]}")
            print(f"   근거: {r[2]}" + (f"   재현: {r[3]}" if r[3] else ""))
    return n


def why_not(kw):
    nc = _closed(kw)
    if nc:
        print("")
        print("   ^^ 이 축은 이미 닫혀 있다. 다시 열려면 "
              "**기존 기각 사유가 왜 틀렸는지**부터 답할 것.")
    pat = re.compile(kw, re.I)
    mark = re.compile(r"[🔴⚠️❌]|기각|반증|전멸|재시도 금지|상한")
    n = 0
    for d in DOCS:
        if not os.path.exists(d):
            continue
        lines = open(d, encoding="utf-8").read().splitlines()
        for i, ln in enumerate(lines):
            if not pat.search(ln):
                continue
            lo, hi = max(0, i - 4), min(len(lines), i + 8)
            hits = [lines[j] for j in range(lo, hi) if mark.search(lines[j])]
            if hits:
                n += 1
                print(f"\n── {os.path.basename(d)}:{i+1}  {ln.strip()[:90]}")
                for h in hits:
                    print(f"   {h.strip()[:110]}")
    print(f"\n{'='*70}\n경고 문맥 {n}건. 🔴 **지지 근거로 읽지 말 것.** "
          "이 축이 왜 닫혔는지부터 답하고 시작한다.")
    if not n:
        print("경고 없음 — 다만 '없다'가 '안전하다'는 뜻은 아니다. 소진 목록 표도 직접 볼 것.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "record"
    if cmd == "record":
        record()
    elif cmd == "why-not":
        why_not(sys.argv[2])
    else:
        print(__doc__)
