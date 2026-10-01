"""전처리 캐시(data/processed)를 불러와 셀 단위 피처 테이블을 만든다.

피처는 Severson et al. (2019) 의 후보 피처를 기반으로 하며, 모두 초기 100 사이클 이내 정보만 사용한다.

사용 예:
    from src.features import load_processed, build_features
    cells, summary, curves = load_processed()
    feat = build_features(cells, summary, curves)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

EARLY_CYCLES = (2, 100)     # cycle 1 은 Batch 1 에서 측정 누락 → cycle 2 부터 사용
DQ_CYCLES = (10, 100)       # ΔQ(V) = Q_100(V) - Q_10(V)

# 물리적으로 불가능한 측정값 (센서 오류) 범위 밖은 NaN 처리 후 셀 내부 보간
VALID_RANGE = {
    "QD": (0.8, 1.2),
    "QC": (0.8, 1.2),
    "IR": (1e-4, 0.05),
    "Tavg": (20, 50),
    "Tmax": (20, 60),
    "Tmin": (20, 45),
    "chargetime": (5, 60),
}


def load_processed(include_excluded=False):
    cells = pd.read_parquet(PROCESSED_DIR / "cells.parquet")
    summary = pd.read_parquet(PROCESSED_DIR / "summary.parquet")
    npz = np.load(PROCESSED_DIR / "curves.npz", allow_pickle=True)
    curves = {
        "cell_id": list(npz["cell_id"]),
        "Qdlin": npz["Qdlin"],
        "Tdlin": npz["Tdlin"],
        "Vdlin": npz["Vdlin"],
    }
    if not include_excluded:
        cells = cells[cells["exclude"] == ""].reset_index(drop=True)
        summary = summary[summary["cell_id"].isin(cells["cell_id"])].reset_index(drop=True)
    return cells, summary, curves


def clean_summary(summary):
    """센서 오류값을 NaN 으로 바꾸고 셀 내부에서 선형 보간한다."""
    out = summary.copy()
    for col, (lo, hi) in VALID_RANGE.items():
        out.loc[~out[col].between(lo, hi), col] = np.nan
    cols = list(VALID_RANGE)
    out[cols] = out.groupby("cell_id")[cols].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    return out


def delta_q(curves, cell_id, cycles=DQ_CYCLES):
    """ΔQ(V) = Q_late(V) - Q_early(V). cycle 번호는 1부터 시작."""
    i = curves["cell_id"].index(cell_id)
    early, late = cycles
    return curves["Qdlin"][i, late - 1] - curves["Qdlin"][i, early - 1]


def charge_time_to_80(c1, q1, c2):
    """2단계 충전 정책의 0→80% 이론 충전 시간 (분)."""
    q1 = np.minimum(q1, 80) / 100
    return 60 * (q1 / c1 + (0.8 - q1) / c2)


def _linfit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def derived_dq_features(curves, cell_id):
    """EDA 결과를 바탕으로 만든 ΔQ(V) 파생변수."""
    V = curves["Vdlin"]
    i = curves["cell_id"].index(cell_id)
    q10, q20, q50, q100 = (curves["Qdlin"][i, c - 1] for c in (10, 20, 50, 100))
    dq = q100 - q10
    mid = (V >= 2.9) & (V <= 3.1)   # 가장 크게 패이는 구간
    low = (V >= 2.0) & (V < 2.7)    # 저전압 구간
    return {
        "dq_area_29_31": np.log10(abs(dq[mid].sum())),
        "dq_lowv_mean": np.log10(abs(dq[low].mean())),
        "dq_var_norm": np.log10(np.var(dq / q10[-1])),   # cycle 10 방전 용량으로 정규화
        "dq_var_100_20": np.log10(np.var(q100 - q20)),
        "dq_var_50_10": np.log10(np.var(q50 - q10)),
        "dq_v_at_min": V[np.argmin(dq)],
    }


def build_features(cells, summary, curves):
    """셀 단위 피처 테이블 (index = cell_id)."""
    s = clean_summary(summary)
    lo, hi = EARLY_CYCLES
    early = s[s["cycle"].between(lo, hi)]

    rows = []
    for _, cell in cells.iterrows():
        cid = cell["cell_id"]
        e = early[early["cell_id"] == cid].set_index("cycle")
        dq = delta_q(curves, cid)

        fade_slope, fade_int = _linfit(e.index, e["QD"])
        tail = e.loc[hi - 9:hi]
        tail_slope, tail_int = _linfit(tail.index, tail["QD"])

        rows.append({
            "cell_id": cid,
            "batch": cell["batch"],
            "policy": cell["policy"],
            "cycle_life": cell["cycle_life"],
            # ΔQ(V) 곡선 통계량
            "dq_min": np.log10(abs(dq.min())),
            "dq_mean": np.log10(abs(dq.mean())),
            "dq_var": np.log10(dq.var()),
            "dq_skew": np.log10(abs(stats.skew(dq))),
            "dq_kurt": np.log10(abs(stats.kurtosis(dq, fisher=False))),
            # 방전 용량 열화
            "qd_2": e.loc[lo, "QD"],
            "qd_max_minus_2": e["QD"].max() - e.loc[lo, "QD"],
            "qd_100_minus_2": e.loc[hi, "QD"] - e.loc[lo, "QD"],
            "fade_slope_2_100": fade_slope,
            "fade_int_2_100": fade_int,
            "fade_slope_91_100": tail_slope,
            "fade_int_91_100": tail_int,
            # 충전 조건
            "C1": cell["C1"],
            "Q1": cell["Q1"],
            "C2": cell["C2"],
            "t80_policy": charge_time_to_80(cell["C1"], cell["Q1"], cell["C2"]),
            "chargetime_2_6": e.loc[lo:lo + 4, "chargetime"].mean(),
            # 온도
            "tavg_mean": e["Tavg"].mean(),
            "tmax_max": e["Tmax"].max(),
            "tmin_min": e["Tmin"].min(),
            # 내부 저항
            "ir_2": e.loc[lo, "IR"],
            "ir_min": e["IR"].min(),
            "ir_100_minus_2": e.loc[hi, "IR"] - e.loc[lo, "IR"],
            # EDA 기반 파생변수
            **derived_dq_features(curves, cid),
        })
    return pd.DataFrame(rows).set_index("cell_id")


FEATURE_COLS = [
    "dq_min", "dq_mean", "dq_var", "dq_skew", "dq_kurt",
    "qd_2", "qd_max_minus_2", "qd_100_minus_2",
    "fade_slope_2_100", "fade_int_2_100", "fade_slope_91_100", "fade_int_91_100",
    "C1", "Q1", "C2", "t80_policy", "chargetime_2_6",
    "tavg_mean", "tmax_max", "tmin_min",
    "ir_2", "ir_min", "ir_100_minus_2",
]

DERIVED_COLS = [
    "dq_area_29_31", "dq_lowv_mean", "dq_var_norm",
    "dq_var_100_20", "dq_var_50_10", "dq_v_at_min",
]
