> This learning exercise is now collected in [ML Foundations](https://github.com/zoga228/ml-foundations/tree/main/exercises/PRODIGY_ML_04). This repository is archived to preserve its original history.

# PRODIGY_ML_04 - Hand Gesture Recognition

Task: develop a hand gesture recognition model that can classify different hand gestures from image data.

Dataset: <https://www.kaggle.com/gti-upm/leapgestrecog>

## Run

Download and extract the Kaggle dataset into `data/`. The script expects images under class folders such as:

```text
data/leapGestRecog/00/01_palm/frame_00_01_0001.png
data/leapGestRecog/00/02_l/frame_00_02_0001.png
```

Train a CNN:

```bash
python src/hand_gesture_recognition.py --data-root data/leapGestRecog --epochs 8 --limit-per-class 800
```

Smoke-test without Kaggle data:

```bash
python src/hand_gesture_recognition.py --demo --epochs 2
```

Predict one image:

```bash
python src/hand_gesture_recognition.py --predict-image path/to/gesture.png --model outputs/hand_gesture_cnn.pt
```

Outputs are saved to `outputs/`.
