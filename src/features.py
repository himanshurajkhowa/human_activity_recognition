"""F48 sensor statistics and the twelve time comparisons added by D60.

The feature definitions follow the saved RealWorld experiments.
Fit feature scaling on the training participants.
"""

import numpy as np


def _base_names(modalities=2):
    names = [
        f"{axis}_{stat}"
        for stat in ["mean", "std", "min", "max"]
        for axis in ["x", "y", "z"]
    ]
    names += [
        "norm_mean",
        "norm_std",
        "norm_min",
        "norm_max",
        "norm_iqr",
        "xyz_dynamic_energy",
        "norm_mean_abs_diff",
        "norm_lag1",
        "norm_dominant_hz",
        "norm_power_0p4_3",
        "norm_power_3_8",
        "norm_power_8_25",
    ]
    return [prefix + "_" + n for prefix in ["acc", "gyr"][:modalities] for n in names]


def features48(windows):
    """Return the historical 24 features per sensor in the original order.

    Each sensor contributes XYZ mean/std/min/max, then twelve norm and
    motion descriptors. Population standard deviation uses ddof=0.
    FFT bands use demeaned magnitude and a nominal 50 Hz grid.
    """
    x = np.asarray(windows)
    if x.ndim != 3 or x.shape[1] != 125 or x.shape[2] not in (3, 6):
        raise ValueError("Expected N × 125 × 3 or 6 windows")
    if not np.isfinite(x).all():
        raise ValueError("Windows must contain finite, supported observations")

    parts = []
    for first in range(0, x.shape[2], 3):
        xyz = x[:, :, first : first + 3].astype(float)
        magnitude = np.linalg.norm(xyz, axis=2)
        centered = magnitude - magnitude.mean(1, keepdims=True)

        # Remove the magnitude mean, then measure power at each grid frequency.
        # DC is the zero-frequency value; leave it out of the power denominator.
        power = abs(np.fft.rfft(centered, axis=1)) ** 2
        frequency = np.fft.rfftfreq(125, 0.02)
        spectral_energy = power[:, 1:].sum(1)
        bands = []
        for low, high in [(0.4, 3), (3, 8), (8, 25.0001)]:
            selected = (frequency >= low) & (frequency < high)
            fraction = np.divide(
                power[:, selected].sum(1),
                spectral_energy,
                out=np.zeros(len(xyz)),
                where=spectral_energy > 1e-12,
            )
            bands.append(fraction)

        # Lag one compares adjacent magnitudes after subtracting their window mean.
        left, right = centered[:, :-1], centered[:, 1:]
        denominator = np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1))
        lag1 = np.divide(
            np.sum(left * right, axis=1),
            denominator,
            out=np.zeros(len(xyz)),
            where=denominator > 1e-12,
        )
        dominant = (
            frequency[1:][np.argmax(power[:, 1:], axis=1)] if len(xyz) else np.empty(0)
        )
        dominant = np.where(spectral_energy > 1e-12, dominant, 0)

        # Average the squared deviation from each axis mean.
        dynamic_energy = np.mean(
            np.sum((xyz - xyz.mean(1, keepdims=True)) ** 2, axis=2), axis=1
        )
        # Preserve the model input order: 12 axis statistics, then 12 summaries.
        block = np.column_stack(
            [
                xyz.mean(1),
                xyz.std(1),
                xyz.min(1),
                xyz.max(1),
                magnitude.mean(1),
                magnitude.std(1),
                magnitude.min(1),
                magnitude.max(1),
                np.quantile(magnitude, 0.75, axis=1)
                - np.quantile(magnitude, 0.25, axis=1),
                dynamic_energy,
                np.mean(abs(np.diff(magnitude, axis=1)), axis=1),
                lag1,
                dominant,
                *bands,
            ]
        )
        parts.append(block)

    # Round to float32 before fitting the SVM scaler.
    result = np.column_stack(parts).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite features")
    return result


LAGS = (5, 10, 25, 50)
EPSILON = 1e-12


def raw_windows(raw):
    x = np.asarray(raw)
    if x.ndim != 3 or x.shape[1:] != (125, 6) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite Nx125x6 raw SI windows")
    with np.errstate(over="ignore", invalid="ignore"):
        x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Input must be representable in Float32")
    return x.astype(np.float64)


def _cosine(left, right):
    # Sum across time and XYZ, then divide by the lengths of the two arrays.
    dot = np.sum(left * right, axis=(1, 2))
    den = np.sqrt(np.sum(left * left, axis=(1, 2)) * np.sum(right * right, axis=(1, 2)))
    out = np.zeros(len(left), dtype=np.float64)
    np.divide(dot, den, out=out, where=den > EPSILON)
    return np.clip(out, -1.0, 1.0)


def lagged_gram(raw):
    """Compare centered sensor patterns at four time shifts.

    Each comparison uses only the current window. Applying the same fixed
    rotation to both sensors leaves these 12 values unchanged. This does not
    cover time-varying rotations or make the directional F48 block invariant.
    The small-denominator rule avoids unstable division; it is not a sensor
    noise calibration.
    """
    x = raw_windows(raw)
    a = x[:, :, :3]
    g = x[:, :, 3:]
    a = a - a.mean(axis=1, keepdims=True)
    g = g - g.mean(axis=1, keepdims=True)
    out = []
    # Four acceleration comparisons, then four gyroscope comparisons.
    for block in [a, g]:
        out.extend(_cosine(block[:, :-lag], block[:, lag:]) for lag in LAGS)
    # The final four compare earlier acceleration with later gyroscope values.
    out.extend(_cosine(a[:, :-lag], g[:, lag:]) for lag in LAGS)
    return np.stack(out, axis=1)


def features60(raw):
    """Append lag12 to the unchanged Float32 F48 block, yielding Float64."""
    x = raw_windows(raw)
    return np.concatenate(
        [features48(x.astype(np.float32)).astype(np.float64), lagged_gram(x)], axis=1
    )


def feature_names(count=48):
    names = _base_names(2)
    if count == 60:
        names += [
            f"{sensor}_centered_cos_lag_{lag}"
            for sensor in ["acc", "gyr"]
            for lag in LAGS
        ]
        names += [f"acc_early_gyr_late_cos_lag_{lag}" for lag in LAGS]
    elif count != 48:
        raise ValueError("Choose 48 or 60 features")
    return names
