set -e
echo "=== [2/5] train_offset (042_offset_condph) ==="
python train_offset.py
echo "=== [3/5] build_shift (043_shift_condph) ==="
python build_shift.py
echo "=== [4/5] build_platoon (044_platoon_condph) ==="
python build_platoon.py
echo "=== [5/5] verify_submit ==="
python verify_submit.py runs/044_platoon_condph/submit044.zip
