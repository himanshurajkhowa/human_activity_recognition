"""Run timestamp alignment, F48/D60 extraction and an SVM on generated signals."""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluation import metrics
from src.features import features48, features60
from src.preprocessing import Stream, prepare_pair


def example_recording(person, label):
    """Make one pair of toy streams with a gap in one held-person recording."""
    rng = np.random.default_rng(100 * person + label)
    time_ms = np.arange(0, 4600, 20)
    activity_rate = 0.8 + label * 0.4
    phase = 2 * np.pi * activity_rate * time_ms / 1000 + person * 0.15
    acc = np.column_stack((np.sin(phase), np.cos(phase), 9.81 + np.sin(2 * phase)))
    gyro = np.column_stack((np.cos(phase), np.sin(phase), np.cos(2 * phase)))
    acc += rng.normal(0, 0.04, acc.shape)
    gyro += rng.normal(0, 0.02, gyro.shape)

    # Offset the gyroscope clock so alignment needs interpolation.
    keep = np.ones(len(time_ms), dtype=bool)
    if person == 3 and label == 0:
        keep &= (time_ms < 1000) | (time_ms >= 1460)
    ids = np.arange(len(time_ms))
    return prepare_pair(
        Stream(time_ms, acc, ids, {}),
        Stream(time_ms[keep] + 7, gyro[keep], ids[keep], {}),
    )


def run_demo():
    training_windows, training_labels = [], []
    for person in (1, 2):
        for activity in range(7):
            windows = example_recording(person, activity)["tensors"]
            training_windows.extend(windows)
            training_labels.extend([activity] * len(windows))

    training_windows = np.asarray(training_windows)
    f48 = features48(training_windows)
    d60 = features60(training_windows)
    np.testing.assert_array_equal(f48, d60[:, :48])
    # Fit both the scaler and the SVM using the two training groups only.
    svm = make_pipeline(StandardScaler(), SVC(C=1, gamma=1 / 48))
    svm.fit(d60, training_labels)

    truth, predictions = [], []
    for activity in range(7):
        recording = example_recording(3, activity)
        eligible = recording["eligible"]
        predicted = np.full(len(eligible), 7, dtype=int)
        if eligible.any():
            predicted[eligible] = svm.predict(features60(recording["tensors"]))
        truth.extend([activity] * len(eligible))
        predictions.extend(predicted)

    result = metrics(truth, predictions)
    return d60.shape, len(truth), int(np.count_nonzero(np.array(predictions) == 7)), result


if __name__ == "__main__":
    shape, scheduled, missing, result = run_demo()
    print(f"Synthetic example: {shape[0]} training windows with {shape[1]} features")
    print(f"Held synthetic group: {scheduled} scheduled, {missing} unavailable")
    print(f"Toy all-scheduled macro-F1: {result['macro_f1']:.3f}")
