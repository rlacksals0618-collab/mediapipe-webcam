"""제스처 분류기 훈련 - gesture_data.csv를 읽어 gesture_model.pkl로 저장

사용법:  python gesture_train.py
"""
import csv
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gesture_common import CLASSIFIER_PATH, DATA_PATH


def load_data():
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH.name}이 없습니다. 먼저 gesture_collect.py로 데이터를 모으세요.")
    labels, features = [], []
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 헤더
        for row in reader:
            if row:
                labels.append(row[0])
                features.append([float(v) for v in row[1:]])
    return np.array(features, dtype=np.float32), np.array(labels)


def main():
    X, y = load_data()
    counts = Counter(y)
    print("수집된 데이터:")
    for label, n in sorted(counts.items()):
        print(f"  {label:<20} {n}개")

    if len(counts) < 2:
        raise SystemExit("제스처가 2개 이상 있어야 훈련할 수 있습니다.")
    if min(counts.values()) < 20:
        print("[주의] 샘플이 20개 미만인 제스처가 있습니다. 제스처당 200개 이상을 권장합니다.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000,
                      early_stopping=True, random_state=42),
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"\n테스트 정확도: {acc * 100:.1f}%  (훈련 {len(X_train)}개 / 테스트 {len(X_test)}개)\n")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("혼동 행렬 (행=정답, 열=예측):", [str(c) for c in model.classes_])
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))

    # 최종 모델은 전체 데이터로 다시 훈련
    model.fit(X, y)
    joblib.dump(model, CLASSIFIER_PATH)
    print(f"\n모델 저장: {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
