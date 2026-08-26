set -e
python train_local.py
python train_offset.py
python build_shift.py
python build_platoon.py
python verify_submit.py runs/048_platoon_phpc/submit048.zip
