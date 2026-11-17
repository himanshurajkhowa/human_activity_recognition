# Methods

## Data and windowing

UCI HAPT was used for the first pipeline experiments. The main benchmark uses RealWorld HAR with walking, running, sitting, standing, lying, upstairs and downstairs. Jumping is outside this seven-class task. The benchmark uses thigh and waist recordings. The dataset itself has seven body locations.

Acceleration XYZ (m/s², including gravity) and angular velocity XYZ (rad/s) are joined by recorded time. The first acceleration timestamp is rounded up to a 20 ms grid. Exact readings are kept. Linear interpolation uses the closest readings on either side if their timestamps are at most 200 ms apart and they belong to the same uninterrupted segment. The implementation leaves unsupported points missing.

Each input has 125 readings per channel. Its first and last readings are 2.48 seconds apart. The first prediction is scheduled 2.5 seconds after the grid starts, with a new request every 500 ms. A usable window needs support for all six channels through the prediction time. The latest recorded endpoint needed for interpolation must arrive within 100 ms of that time. This is an input availability rule on the recorded clock. Phone runtime and physical sensor synchronization need separate measurements.

## Features and models

F48 uses 24 features per sensor. The first 12 are XYZ means, population standard deviations, minima and maxima. The rest describe the magnitude distribution, centered motion energy, changes between adjacent magnitudes, lag-one similarity and frequency content. Frequency features use the mean-centered magnitude, with FFT power summed over 0.4 to under 3 Hz, 3 to under 8 Hz, and 8 through 25 Hz. Each band is divided by total nonzero-frequency power. See the implementation for feature order, thresholds and data types. [Feature code](../src/features.py)

D60 adds twelve features. After subtracting each axis mean, it compares earlier and later XYZ values at lags of 0.1, 0.2, 0.5 and 1 second. The three comparisons at each lag are acceleration with itself, gyroscope with itself, and earlier acceleration with later gyroscope. For aligned slices `L` and `R`, the feature is their normalized dot product.

```text
similarity = sum(L * R) / sqrt(sum(L²) * sum(R²))
```

Results are clipped to [−1, 1], and denominators at or below 1e-12 give zero. A common fixed rotation of both triads preserves the lag block. F48 retains axis statistics, so the full D60 vector still depends on orientation.

The B11Q and D60 models used weighted training-only scaling and a radial basis function SVM. They share training rows and weights. Their SVM penalty `C` was 1, 10 or 1 across three predefined groups of fits, with `gamma = 1/48` for both feature sets. CNN-GRU C11 used all selected observations from eleven training people, more than the two quantity-matched SVM conditions. Its convolutions use left padding and feed a 24-unit GRU. The network returns seven class scores from each window. [Network code](../src/models.py)

## Evaluation

For each of nine development participants, the corresponding model used fitting rows from other people. Training retained every fifth scheduled window within each recording in the source plan. The SVMs used the matched subset described above. A participant's accuracy and macro-F1 are computed first, then averaged equally across participants. Macro-F1 averages the seven class F1 values, including zeros for undefined classes.

Usable-window scores cover 133,982 windows. All-scheduled scores also account for unavailable input among 155,845 scheduled prediction times. Each unavailable prediction counts as a missed true activity. The confusion matrix in the public example pools windows, so its diagonal shares can differ from recall averaged across participants. Evaluation windows overlap and the nine participants were inspected repeatedly during research.
