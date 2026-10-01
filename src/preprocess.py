"""MIT-Stanford 배터리 .mat(v7.3/HDF5) 파일을 분석용 캐시로 변환한다.

출력 (data/processed/):
    cells.parquet    셀 단위 메타데이터 (batch, policy, cycle_life, 제외 여부 등)
    summary.parquet  사이클 단위 요약값 (long format)
    curves.npz       초기 1~N_CYCLES 사이클의 Qdlin / Tdlin 곡선 + Vdlin 전압축

실행:
    python -m src.preprocess
"""
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data"
OUT_DIR = ROOT / "data" / "processed"

BATCHES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "b2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
N_CYCLES = 100          # 곡선 데이터를 저장할 초기 사이클 수
EOL_CAPACITY = 0.88     # 80% SOH (공칭 1.1Ah 기준)

SUMMARY_FIELDS = {
    "QDischarge": "QD", "QCharge": "QC", "IR": "IR",
    "Tavg": "Tavg", "Tmin": "Tmin", "Tmax": "Tmax", "chargetime": "chargetime",
}


def _str(f, ref):
    return "".join(chr(c) for c in f[ref][()].ravel())


def parse_policy(policy):
    """'5.4C(40%)-3.6C-newstructure' -> (C1=5.4, Q1=40, C2=3.6, newstructure=True)"""
    m = re.match(r"([\d.]+)C\((\d+)%\)-([\d.]+)C", policy)
    if not m:
        return np.nan, np.nan, np.nan, "newstructure" in policy
    c1, q1, c2 = float(m[1]), float(m[2]), float(m[3])
    return c1, q1, c2, "newstructure" in policy


def exclude_reason(cycle_life, qd):
    if np.isnan(cycle_life):
        return "no_cycle_life"
    # EOL 이전에 실험이 종료된 셀 (중도절단): 마지막 구간 용량이 EOL보다 높음
    if np.median(qd[-5:]) > EOL_CAPACITY + 0.01:
        return "censored"
    return ""


def load_batch(batch, fname):
    cells, summaries, qdlins, tdlins = [], [], [], []
    with h5py.File(RAW_DIR / fname, "r") as f:
        b = f["batch"]
        vdlin = f[b["Vdlin"][0, 0]][()].ravel()
        for i in range(b["summary"].shape[0]):
            cell_id = f"{batch}c{i}"
            policy = _str(f, b["policy_readable"][i, 0])
            cycle_life = float(f[b["cycle_life"][i, 0]][()].ravel()[0])

            sm = f[b["summary"][i, 0]]
            s = pd.DataFrame({new: sm[old][()].ravel() for old, new in SUMMARY_FIELDS.items()})
            s.insert(0, "cycle", sm["cycle"][()].ravel().astype(int))
            s.insert(0, "cell_id", cell_id)
            summaries.append(s)

            # cycles 구조체의 index k 가 cycle k+1 에 대응
            cyc = f[b["cycles"][i, 0]]
            n = min(N_CYCLES, cyc["Qdlin"].shape[0])
            q = np.full((N_CYCLES, len(vdlin)), np.nan, dtype=np.float32)
            t = np.full_like(q, np.nan)
            for k in range(n):
                qk = f[cyc["Qdlin"][k, 0]][()].ravel()
                tk = f[cyc["Tdlin"][k, 0]][()].ravel()
                if qk.size == len(vdlin):   # 측정 누락 사이클은 빈 배열로 저장되어 있음 -> NaN 유지
                    q[k], t[k] = qk, tk
            qdlins.append(q)
            tdlins.append(t)

            c1, q1, c2, newstruct = parse_policy(policy)
            cells.append({
                "cell_id": cell_id, "batch": batch, "idx": i, "policy": policy,
                "C1": c1, "Q1": q1, "C2": c2, "newstructure": newstruct,
                "cycle_life": cycle_life, "n_cycles": len(s),
                "exclude": exclude_reason(cycle_life, s["QD"].to_numpy()),
            })
            print(f"\r{batch}: {i + 1}/{b['summary'].shape[0]}", end="", flush=True)
    print()
    return cells, summaries, qdlins, tdlins, vdlin


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_cells, all_summary, all_q, all_t = [], [], [], []
    for batch, fname in BATCHES.items():
        cells, summaries, q, t, vdlin = load_batch(batch, fname)
        all_cells += cells
        all_summary += summaries
        all_q += q
        all_t += t

    cells = pd.DataFrame(all_cells)
    cells.to_parquet(OUT_DIR / "cells.parquet", index=False)
    pd.concat(all_summary, ignore_index=True).to_parquet(OUT_DIR / "summary.parquet", index=False)
    np.savez_compressed(
        OUT_DIR / "curves.npz",
        cell_id=cells["cell_id"].to_numpy(),
        Qdlin=np.stack(all_q), Tdlin=np.stack(all_t), Vdlin=vdlin,
    )

    print(cells.groupby("batch")["exclude"].value_counts().unstack(fill_value=0))
    print(f"saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
