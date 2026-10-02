"""Batch 1 학습 → Batch 2 평가 파이프라인.

평가 구조 (Notion Performance Reporting 기준)
    Train (Batch 1 CV)       : Batch 1 학습 구간에서 GroupKFold(group = 충전 정책) CV 평균 MAPE
    Valid (Batch 1 Hold-out) : 학습 구간과 정책이 겹치지 않는 Batch 1 Hold-out 셀의 MAPE
    Test  (Batch 2)          : Batch 1 전체로 다시 학습한 모델의 Batch 2 MAPE

Target 은 log10(cycle_life) 로 학습하고, 예측값을 10^x 로 되돌려 MAPE 를 계산한다.

사용 예:
    python -m src.train          # 전체 모델 × 피처 세트 비교 결과를 results/ 에 저장
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.features import FEATURE_COLS, build_features, load_processed

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

TARGET_MAPE = 9.1       # 원논문 Regression Target
N_VALID_POLICIES = 4    # Batch 1 20개 정책 중 Hold-out 으로 뗄 정책 수
N_CV_SPLITS = 5
SEED = 42

# IR 계열은 Batch 2 에 측정 결측 셀이 있어 비교용 전체 세트에서도 제외
ALL_NO_IR = [c for c in FEATURE_COLS if not c.startswith("ir_")]

FEATURE_SETS = {
    "S1 dq_var": ["dq_var"],
    "S2 + dq_lowv_mean": ["dq_var", "dq_lowv_mean"],
    "S3 + qd_2, fade_slope_91_100": ["dq_var", "dq_lowv_mean", "qd_2", "fade_slope_91_100"],
    "ALL 후보 20개 (비교용)": ALL_NO_IR,
}

# 1차 : EDA 기반 설계 전략의 3단계 피처 세트 × 전체 모델
PLAN_SETS = ["S1 dq_var", "S2 + dq_lowv_mean", "S3 + qd_2, fade_slope_91_100"]
# 2차 : 배치 간 관계가 일관된 피처 세트 × 학습 범위 밖 예측이 가능한 선형 계열 모델
ROBUST_SETS = ["S1 dq_var", "S2 + dq_lowv_mean"]
LINEAR_MODELS = ["Linear", "Ridge", "ElasticNet"]


def _scaled(model):
    return make_pipeline(StandardScaler(), model)


# 모델 이름 → (생성 함수, 하이퍼파라미터 후보)
MODELS = {
    "Linear": (lambda: _scaled(LinearRegression()), {}),
    "Ridge": (lambda alpha: _scaled(Ridge(alpha=alpha)), {"alpha": [0.01, 0.1, 1, 10, 100]}),
    "ElasticNet": (
        lambda alpha, l1_ratio: _scaled(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50_000)),
        {"alpha": [1e-4, 1e-3, 1e-2, 1e-1], "l1_ratio": [0.1, 0.5, 0.9]},
    ),
    "RandomForest": (
        lambda max_depth, min_samples_leaf: RandomForestRegressor(
            n_estimators=300, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEED),
        {"max_depth": [2, 3, None], "min_samples_leaf": [1, 3]},
    ),
    "XGBoost": (
        lambda max_depth, learning_rate: XGBRegressor(
            n_estimators=300, max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.8, random_state=SEED, verbosity=0),
        {"max_depth": [2, 3], "learning_rate": [0.03, 0.1]},
    ),
}


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_pred - y_true) / y_true) * 100)


def load_dataset(include_b3=False):
    """Batch 1 / Batch 2 (/ Batch 3) 피처 테이블."""
    cells, summary, curves = load_processed()
    feat = build_features(cells, summary, curves).join(cells.set_index("cell_id")[["newstructure"]])
    feat["log_cl"] = np.log10(feat["cycle_life"])
    batches = ["b1", "b2", "b3"] if include_b3 else ["b1", "b2"]
    return tuple(feat[feat["batch"] == b].copy() for b in batches)


def holdout_policies(b1, n=N_VALID_POLICIES):
    """평균 수명 순으로 정렬한 정책 중 n 개를 균등 간격으로 뽑는다 → Hold-out 이 수명 범위를 고르게 포함."""
    order = b1.groupby("policy")["cycle_life"].mean().sort_values().index
    idx = np.linspace(0, len(order) - 1, n + 2)[1:-1].round().astype(int)
    return list(order[idx])


def split_batch1(b1, valid_policies):
    is_valid = b1["policy"].isin(valid_policies)
    return b1[~is_valid], b1[is_valid]


def _predict(model, X):
    return 10 ** model.predict(X)


def cv_mape(make_model, params, df, cols, n_splits=N_CV_SPLITS):
    """GroupKFold(group = 정책) CV 평균 MAPE 와 셀별 out-of-fold 예측."""
    gkf = GroupKFold(n_splits=n_splits)
    oof = pd.Series(np.nan, index=df.index)
    scores = []
    for tr, va in gkf.split(df, groups=df["policy"]):
        m = make_model(**params).fit(df.iloc[tr][cols], df.iloc[tr]["log_cl"])
        p = _predict(m, df.iloc[va][cols])
        oof.iloc[va] = p
        scores.append(mape(df.iloc[va]["cycle_life"], p))
    return float(np.mean(scores)), oof


def tune(model_name, df, cols):
    """하이퍼파라미터 후보 중 CV MAPE 가 가장 낮은 조합."""
    make_model, grid = MODELS[model_name]
    keys = list(grid)
    best = None
    for values in product(*grid.values()) if keys else [()]:
        params = dict(zip(keys, values))
        score, oof = cv_mape(make_model, params, df, cols)
        if best is None or score < best[1]:
            best = (params, score, oof)
    return best


def evaluate(model_name, cols, b1, b2, valid_policies):
    """Train CV / Valid / Test MAPE 와 예측값."""
    make_model, _ = MODELS[model_name]
    train, valid = split_batch1(b1, valid_policies)

    params, train_cv, oof = tune(model_name, train, cols)
    m_train = make_model(**params).fit(train[cols], train["log_cl"])
    valid_pred = pd.Series(_predict(m_train, valid[cols]), index=valid.index)

    m_final = make_model(**params).fit(b1[cols], b1["log_cl"])   # Batch 1 전체로 재학습
    test_pred = pd.Series(_predict(m_final, b2[cols]), index=b2.index)

    return {
        "params": params,
        "train_cv": train_cv,
        "valid": mape(valid["cycle_life"], valid_pred),
        "test": mape(b2["cycle_life"], test_pred),
        "oof_pred": oof,
        "valid_pred": valid_pred,
        "test_pred": test_pred,
        "model": m_final,
    }


def compare_all(b1, b2, valid_policies, models=MODELS, feature_sets=FEATURE_SETS):
    rows, details = [], {}
    for (fs_name, cols), model_name in product(feature_sets.items(), models):
        r = evaluate(model_name, cols, b1, b2, valid_policies)
        details[(model_name, fs_name)] = r
        rows.append({
            "model": model_name, "feature_set": fs_name, "n_features": len(cols),
            "train_cv": r["train_cv"], "valid": r["valid"], "test": r["test"],
            "params": r["params"],
        })
    return pd.DataFrame(rows), details


def select_by_valid(comparison, models, feature_sets):
    """주어진 후보 안에서 Valid MAPE 가 가장 낮은 조합 (Test 는 사용하지 않음)."""
    cand = comparison[comparison["model"].isin(models) & comparison["feature_set"].isin(feature_sets)]
    return cand.sort_values("valid").iloc[0]


def covariate_shift(b1, b2, cols):
    """Batch 2 피처 분포가 Batch 1 에서 얼마나 벗어났는지 (Batch 2 정답 사용 안 함)."""
    rows = []
    for c in cols:
        lo, hi = b1[c].min(), b1[c].max()
        rows.append({
            "feature": c,
            "Batch 1 범위 밖 비율 (%)": ((b2[c] < lo) | (b2[c] > hi)).mean() * 100,
            "평균 차이 (Batch 1 표준편차 단위)": (b2[c].mean() - b1[c].mean()) / b1[c].std(),
        })
    return pd.DataFrame(rows).set_index("feature")


def predict(r, df, cols):
    """evaluate() 결과의 최종 모델(Batch 1 전체 학습)로 새 데이터를 예측."""
    return pd.Series(_predict(r["model"], df[cols]), index=df.index)


def report_table(r):
    """Notion Reporting format (Regression).

    MAPE 는 낮을수록 좋은 지표이므로, Notion 의 "(+) : 과적합 / 일반화 저하 의심" 의미에 맞도록
    모든 Gap 을 '뒤 단계 − 앞 단계' 로 계산한다 → (+) 면 뒤 단계에서 성능이 나빠졌다는 뜻.
    계산식은 '계산' 열에 함께 표기한다.
    """
    rows = [
        ("Train (Batch 1 CV)", r["train_cv"], "", ""),
        ("Valid (Batch 1 Hold-out)", r["valid"], "", ""),
        ("Test (Batch 2)", r["test"], "", ""),
        ("Gap (Train-Valid)", r["valid"] - r["train_cv"], "Valid − Train", "(+) : 과적합 의심"),
        ("Gap (Valid-Test)", r["test"] - r["valid"], "Test − Valid", "(+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)", r["test"] - TARGET_MAPE, f"Test − Target({TARGET_MAPE})", f"Target : 원논문 {TARGET_MAPE}%"),
    ]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "계산", "비고"]).set_index("구분").round(2)


def report_table_b3(r, test_b3):
    """Notion Reporting format (Batch 3 추가 검증, Regression)."""
    base = report_table(r)
    extra = pd.DataFrame([
        ("Test (Batch 3)", test_b3, "", ""),
        ("Gap (Batch2-Batch3)", test_b3 - r["test"], "Batch 3 − Batch 2", "Test 성능 간 비교"),
        ("Gap (Target-Test) · Batch 3", test_b3 - TARGET_MAPE, f"Batch 3 − Target({TARGET_MAPE})", "Batch 3 기준, 원논문 성능 비교"),
    ], columns=["구분", "MAPE (%)", "계산", "비고"]).set_index("구분").round(2)
    return pd.concat([base, extra])


def main():
    b1, b2, b3 = load_dataset(include_b3=True)
    valid_policies = holdout_policies(b1)
    print("Hold-out 정책 :", valid_policies)
    table, details = compare_all(b1, b2, valid_policies)
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(table.drop(columns="params").round(2).to_string(index=False))

    first = select_by_valid(table, list(MODELS), PLAN_SETS)
    final = select_by_valid(table, LINEAR_MODELS, ROBUST_SETS)
    print(f"\n1차 선택 (피처 세트 S1~S3, 전체 모델) : {first['model']} · {first['feature_set']}")
    print(f"최종 선택 (일관된 피처 세트, 선형 모델) : {final['model']} · {final['feature_set']}")
    final_r = details[(final["model"], final["feature_set"])]
    perf = report_table(final_r)
    perf.to_csv(RESULTS_DIR / "model_performance.csv")
    print(perf.to_string())

    # Batch 3 추가 검증 : 최종 모델을 바꾸지 않고 그대로 적용
    b3_pred = predict(final_r, b3, FEATURE_SETS[final["feature_set"]])
    perf_b3 = report_table_b3(final_r, mape(b3["cycle_life"], b3_pred))
    perf_b3.to_csv(RESULTS_DIR / "model_performance_batch3.csv")
    print("\n[Batch 3 추가 검증]")
    print(perf_b3.to_string())


if __name__ == "__main__":
    main()
