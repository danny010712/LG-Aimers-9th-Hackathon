set -e
python train_local.py
python train_offset.py
python build_shift.py
python build_platoon.py
python verify_submit.py runs/056_platoon_phb/submit056.zip
