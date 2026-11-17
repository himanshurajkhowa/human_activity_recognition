"""Align sensor streams and check input availability on the recorded clock."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Stream:
    time_ms: np.ndarray
    xyz: np.ndarray
    ids: np.ndarray
    metadata: dict

    def __post_init__(self):
        t, x, ids = self.time_ms, self.xyz, self.ids
        if t.ndim != 1 or x.shape != (len(t), 3) or ids.shape != t.shape or len(t) < 2:
            raise ValueError("Expected at least two timestamps, IDs and XYZ rows")
        if (
            not np.isfinite(t).all()
            or not np.isfinite(x).all()
            or not np.isfinite(ids).all()
        ):
            raise ValueError("Nonfinite input: stop instead of silently repairing it")
        if not np.equal(t, np.floor(t)).all() or not np.equal(ids, np.floor(ids)).all():
            raise ValueError("Release timestamps and IDs must be integers")
        bounds = np.iinfo(np.int64)
        for values in (t, ids):
            if int(values.min()) < bounds.min or int(values.max()) > bounds.max:
                raise ValueError("Timestamp or ID cannot be represented as int64")
        if (t[1:] <= t[:-1]).any() or (ids[1:] <= ids[:-1]).any():
            raise ValueError("Duplicate or unordered input: do not sort or average it")
        self.time_ms = t.astype(np.int64)
        self.ids = ids.astype(np.int64)
        self.xyz = x.astype(np.float64)


@dataclass(frozen=True)
class Settings:
    interval_ms: int = 20
    bracket_ms: int = 100
    gap_ms: int = 200
    window_points: int = 125
    stride_points: int = 125

    def __post_init__(self):
        if any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
            for v in (
                self.interval_ms,
                self.bracket_ms,
                self.gap_ms,
                self.window_points,
                self.stride_points,
            )
        ):
            raise ValueError("Timing, window and stride settings must be integers")
        if (
            min(
                self.interval_ms,
                self.bracket_ms,
                self.gap_ms,
                self.window_points,
                self.stride_points,
            )
            <= 0
        ):
            raise ValueError("Timing, window and stride settings must be positive")


def acc_grid(acc: Stream, settings: Settings) -> np.ndarray:
    """Start the grid from acceleration timestamps, without using activity labels."""
    step = settings.interval_ms
    start = ((int(acc.time_ms[0]) + step - 1) // step) * step
    return np.arange(start, int(acc.time_ms[-1]) + 1, step, dtype=np.int64)


def interpolate(stream: Stream, grid: np.ndarray, settings: Settings) -> dict:
    """Exact observations need no bracket. Other points require two valid endpoints.

    left/right are zero-based original data-row indices (CSV line = index + 2).
    A nonexact point is available only after its right endpoint arrives.
    Invalid points remain NaN. No extrapolation, gap repair or anti-alias claim.
    """
    if grid.ndim != 1 or grid.dtype.kind not in "iu" or (grid[1:] <= grid[:-1]).any():
        raise ValueError("Grid must be a strictly increasing integer time vector")
    t = stream.time_ms
    pos = np.searchsorted(t, grid, side="left")
    right = np.clip(pos, 0, len(t) - 1)
    exact = (pos < len(t)) & (t[right] == grid)
    left = np.where(exact, right, np.clip(pos - 1, 0, len(t) - 1))
    inside = (grid >= t[0]) & (grid <= t[-1])
    segment = np.r_[0, np.cumsum(np.diff(t) > settings.gap_ms)]
    bracket = t[right] - t[left]
    same_segment = segment[left] == segment[right]
    valid = inside & same_segment & (exact | (bracket <= settings.bracket_ms))
    alpha = np.divide(
        grid - t[left],
        bracket,
        out=np.zeros(len(grid), dtype=float),
        where=bracket != 0,
    )
    values = stream.xyz[left] + alpha[:, None] * (stream.xyz[right] - stream.xyz[left])
    values[~valid] = np.nan
    return dict(
        time_ms=grid,
        xyz=values,
        valid=valid,
        exact=exact,
        left=left,
        right=right,
        source_left_id=stream.ids[left],
        source_right_id=stream.ids[right],
        alpha=alpha,
        bracket_ms=bracket,
        segment=segment[left],
        nearest_ms=np.minimum(np.abs(grid - t[left]), np.abs(t[right] - grid)),
        available_after_ms=t[right],
        outside=~inside,
        gap_crossing=inside & ~same_segment,
        bracket_failure=inside
        & same_segment
        & ~exact
        & (bracket > settings.bracket_ms),
    )


def requested_clock(acc: Stream) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schedule each window so its last point is at the prediction time."""
    grid = acc_grid(acc, Settings())
    if not len(grid):
        return grid, np.empty(0, np.int64), np.empty(0, np.int64)
    queries = np.arange(grid[0] + 2500, acc.time_ms[-1] + 1, 500, dtype=np.int64)
    starts = 1 + 25 * np.arange(len(queries), dtype=np.int64)
    if not np.array_equal(grid[starts + 124], queries):
        raise ValueError("Query/grid endpoint mismatch")
    return grid, queries, starts


def release_eligibility(accepted, ready, queries, budget):
    accepted = np.asarray(accepted, bool)
    ready, queries = np.asarray(ready, np.int64), np.asarray(queries, np.int64)
    if (
        accepted.shape != ready.shape
        or ready.shape != queries.shape
        or isinstance(budget, (bool, np.bool_))
        or not isinstance(budget, (int, np.integer))
        or budget < 0
    ):
        raise ValueError("Invalid eligibility inputs")
    if np.any(accepted & (ready < queries)):
        raise ValueError("Complete query-inclusive support cannot precede the query")
    release = np.maximum(ready, queries)
    eligible = accepted & (ready <= queries + budget)
    release[~accepted] = -1
    return eligible, release


def prepare_pair(acc, gyr, cap_ms=200, release_ms=100):
    """Return prediction times, input-availability flags and usable windows."""
    identity = ("subject", "site", "activity", "trial", "key")
    if any(field in stream.metadata for stream in (acc, gyr) for field in identity):
        if any(
            field not in stream.metadata for stream in (acc, gyr) for field in identity
        ):
            raise ValueError("Incomplete sensor recording identity")
        if any(acc.metadata[field] != gyr.metadata[field] for field in identity):
            raise ValueError("Modalities have different recording identities")
    grid, queries, starts = requested_clock(acc)
    a = interpolate(acc, grid, Settings(bracket_ms=cap_ms))
    g = interpolate(gyr, grid, Settings(bracket_ms=cap_ms))
    ix = starts[:, None] + np.arange(125)
    reasons = {}
    for name, z in [("acc", a), ("gyr", g)]:
        for field in ("outside", "gap_crossing", "bracket_failure"):
            reasons[name + "_" + field] = z[field][ix].any(axis=1)
        reasons[name + "_segment_change"] = (
            z["segment"][ix] != z["segment"][ix[:, :1]]
        ).any(axis=1)
    supported = ~np.logical_or.reduce(list(reasons.values()))
    ready = np.full(len(queries), -1, dtype=np.int64)
    if supported.any():
        ready[supported] = np.max(
            np.stack([z["available_after_ms"][ix[supported]] for z in (a, g)]),
            axis=(0, 2),
        )
    eligible, release = release_eligibility(supported, ready, queries, release_ms)
    values = np.concatenate([a["xyz"], g["xyz"]], axis=1)
    tensors = values[ix[eligible]].astype(np.float32)
    if not np.isfinite(tensors).all():
        raise ValueError("Nonfinite accepted window")
    return dict(
        grid_ms=grid,
        acc=a,
        gyr=g,
        query_ms=queries,
        starts=starts,
        supported=supported,
        eligible=eligible,
        ready_ms=ready,
        release_ms=release,
        reasons=reasons,
        tensors=tensors,
        accepted_query_index=np.flatnonzero(eligible),
    )
