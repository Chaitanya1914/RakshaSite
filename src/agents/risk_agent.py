import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from src.agents.state import SafetyState

logger = logging.getLogger(__name__)

VIOLATION_WEIGHT_MAP = {
    "no_helmet": 40.0,
    "no_vest": 30.0,
    "no_harness": 50.0,
    "no_boots": 20.0,
    "no_gloves": 15.0,
}

ZONE_BONUS_MAP = {
    "general": 0.0,
    "ground_level": 0.0,
    "excavation_trench": 25.0,
    "restricted": 40.0,
    "active_machinery": 50.0,
    "heavy_machinery_zone": 45.0,
    "scaffolding_elevated": 45.0,
    "elevated": 45.0,
    "confined_space": 50.0,
}

VIOLATION_INDEX = {
    "no_helmet": 0,
    "no_vest": 1,
    "no_harness": 2,
    "no_boots": 3,
    "no_gloves": 4,
}

ZONE_INDEX = {
    "general": 0,
    "ground_level": 0,
    "excavation_trench": 1,
    "restricted": 2,
    "active_machinery": 3,
    "heavy_machinery_zone": 3,
    "scaffolding_elevated": 4,
    "elevated": 4,
    "confined_space": 5,
}


def _compute_nonlinear_hazard_interaction(v_type: str, zone: str, proximity: float, frames: float) -> float:
    """
    Domain-specific non-linear construction hazard interaction term:
    - Working on elevated scaffolding without a harness has catastrophic fall risk.
    - Working near active machinery without a helmet or high-visibility vest scales with proximity.
    - Persistent violations across many consecutive frames compound risk non-linearly.
    """
    interaction = 0.0
    if v_type == "no_harness" and zone in ("scaffolding_elevated", "elevated"):
        interaction += 12.0
    elif v_type in ("no_helmet", "no_vest") and zone in ("active_machinery", "heavy_machinery_zone"):
        interaction += 8.0 * (0.5 + proximity)
    elif v_type == "no_helmet" and zone == "restricted":
        interaction += 6.0

    # Non-linear temporal persistence boost (saturating logarithmically after 5 frames)
    if frames > 5.0:
        interaction += min(6.0, float(np.log1p(frames - 5.0) * 2.2))

    return interaction


def _extract_feature_vector(violation: Dict[str, Any]) -> List[float]:
    """
    Convert a violation dictionary into a 7-dimensional numerical feature vector:
    [v_idx, z_idx, base_weight, zone_bonus, confidence, consecutive_frames, proximity]
    """
    v_type = str(violation.get("violation_type", "no_helmet")).lower()
    zone = str(violation.get("zone", "general")).lower()
    conf = float(violation.get("confidence", 0.85))
    frames = float(violation.get("consecutive_frames", 5))
    proximity = float(violation.get("proximity", 0.2))

    v_idx = float(VIOLATION_INDEX.get(v_type, 0))
    z_idx = float(ZONE_INDEX.get(zone, 0))
    base_w = float(VIOLATION_WEIGHT_MAP.get(v_type, 30.0))
    z_bonus = float(ZONE_BONUS_MAP.get(zone, 0.0))

    return [v_idx, z_idx, base_w, z_bonus, conf, min(frames, 30.0), proximity]


def train_and_evaluate_risk_model(
    output_model_path: str = "models/risk_xgboost.json",
    n_samples: int = 4000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate the XGBoost Risk Assessment Model following ML best practices:
    1. Generate 4,000 construction incident records with realistic non-linear hazard interactions.
    2. Check for missing/NULL values.
    3. Split into Train (70%), Validation (15%), and Test (15%) sets BEFORE fitting preprocessors.
    4. Compare a Linear Ridge baseline against tuned XGBoost (`XGBRegressor`).
    5. Evaluate MAE, RMSE, R2, and Risk Tier Classification F1-Score & Confusion Matrix.
    6. Persist trained XGBoost model to `models/risk_xgboost.json`.
    """
    import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    v_types = list(VIOLATION_WEIGHT_MAP.keys())
    zones = ["general", "excavation_trench", "restricted", "active_machinery", "scaffolding_elevated", "confined_space"]

    X_rows: List[List[float]] = []
    y_rows: List[float] = []

    for _ in range(n_samples):
        v_t = str(rng.choice(v_types, p=[0.32, 0.28, 0.16, 0.12, 0.12]))
        z_t = str(rng.choice(zones, p=[0.32, 0.16, 0.16, 0.14, 0.12, 0.10]))
        conf = float(rng.uniform(0.70, 0.99))
        frames = float(rng.integers(5, 31))
        prox = float(rng.uniform(0.1, 0.95)) if z_t in ("active_machinery", "restricted") else float(rng.uniform(0.05, 0.4))

        sample = {
            "violation_type": v_t,
            "zone": z_t,
            "confidence": conf,
            "consecutive_frames": frames,
            "proximity": prox,
        }
        feats = _extract_feature_vector(sample)

        # Target composite severity score (0 - 100) with non-linear domain interactions
        base_score = VIOLATION_WEIGHT_MAP[v_t] + ZONE_BONUS_MAP[z_t]
        interaction = _compute_nonlinear_hazard_interaction(v_t, z_t, prox, frames)
        conf_adj = (conf - 0.80) * 8.0
        noise = float(rng.normal(0.0, 0.25))
        target = float(np.clip(base_score + interaction + conf_adj + noise, 0.0, 100.0))

        X_rows.append(feats)
        y_rows.append(target)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    # Check for missing / NaN values
    nan_count = int(np.isnan(X).sum() + np.isnan(y).sum())
    if nan_count > 0:
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

    # Strict train / val / test split BEFORE fitting any estimators
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed)

    def _to_tier(scores: np.ndarray) -> List[str]:
        tiers = []
        for s in scores:
            if s >= 80:
                tiers.append("CRITICAL")
            elif s >= 60:
                tiers.append("HIGH")
            elif s >= 40:
                tiers.append("MEDIUM")
            else:
                tiers.append("LOW")
        return tiers

    y_true_tiers = _to_tier(y_test)

    # 1. Baseline Model (Ridge Regression)
    baseline = Ridge(alpha=1.0)
    baseline.fit(X_train, y_train)
    base_preds = np.clip(baseline.predict(X_test), 0.0, 100.0)
    base_mae = float(mean_absolute_error(y_test, base_preds))
    base_rmse = float(np.sqrt(mean_squared_error(y_test, base_preds)))
    base_r2 = float(r2_score(y_test, base_preds))
    base_f1 = float(f1_score(y_true_tiers, _to_tier(base_preds), average="macro"))

    # 2. Tuned XGBoost Model
    xgb_model = xgb.XGBRegressor(
        n_estimators=260,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.92,
        colsample_bytree=0.95,
        reg_alpha=0.05,
        reg_lambda=0.8,
        random_state=seed,
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    test_preds = np.clip(xgb_model.predict(X_test), 0.0, 100.0)
    xgb_mae = float(mean_absolute_error(y_test, test_preds))
    xgb_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    xgb_r2 = float(r2_score(y_test, test_preds))

    y_pred_tiers = _to_tier(test_preds)
    macro_f1 = float(f1_score(y_true_tiers, y_pred_tiers, average="macro"))
    tier_acc = float(accuracy_score(y_true_tiers, y_pred_tiers))
    cm = confusion_matrix(
        y_true_tiers,
        y_pred_tiers,
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    ).tolist()

    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xgb_model.get_booster().save_model(str(out_path))
    RiskScorer._cached_booster = xgb_model.get_booster()

    metrics_summary = {
        "n_samples": n_samples,
        "nan_values_found": nan_count,
        "baseline_ridge": {
            "mae": round(base_mae, 4),
            "rmse": round(base_rmse, 4),
            "r2": round(base_r2, 4),
            "tier_macro_f1": round(base_f1, 4),
        },
        "xgboost": {
            "mae": round(xgb_mae, 4),
            "rmse": round(xgb_rmse, 4),
            "r2": round(xgb_r2, 4),
            "tier_accuracy": round(tier_acc, 4),
            "tier_macro_f1": round(macro_f1, 4),
        },
        "confusion_matrix_labels": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "confusion_matrix": cm,
        "model_path": str(out_path),
    }

    metrics_file = out_path.with_suffix(".metrics.json")
    metrics_file.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    logger.info(f"Trained XGBoost risk model saved to {out_path} (F1={macro_f1:.4f}, R2={xgb_r2:.4f})")
    return metrics_summary


class RiskScorer:
    _cached_booster = None

    def __init__(self, model_path: str = "models/risk_xgboost.json"):
        self.model_path = Path(model_path)
        self.model = None
        try:
            import xgboost as xgb

            if RiskScorer._cached_booster is not None:
                self.model = RiskScorer._cached_booster
            elif self.model_path.exists():
                booster = xgb.Booster()
                booster.load_model(str(self.model_path))
                RiskScorer._cached_booster = booster
                self.model = booster
        except ImportError:
            logger.warning("xgboost not found, falling back to rule-based scoring")
        except Exception as e:
            logger.warning(f"Could not load XGBoost model ({e}), using rule-based scoring")

    def _rule_based_score(self, violation: Dict[str, Any]) -> float:
        v_type = str(violation.get("violation_type", "")).lower()
        zone = str(violation.get("zone", "general")).lower()

        score = VIOLATION_WEIGHT_MAP.get(v_type, 30.0) + ZONE_BONUS_MAP.get(zone, 0.0)
        return min(score, 100.0)

    def score(self, violation: Dict[str, Any]) -> Dict[str, Any]:
        rule_score = self._rule_based_score(violation)

        if self.model is not None:
            try:
                import xgboost as xgb

                feats = np.array([_extract_feature_vector(violation)], dtype=np.float32)
                dmat = xgb.DMatrix(feats)
                xgb_pred = float(self.model.predict(dmat)[0])
                score_val = round(float(np.clip(0.5 * xgb_pred + 0.5 * rule_score, rule_score, 100.0)), 2)
            except Exception as e:
                logger.warning(f"XGBoost inference fallback ({e})")
                score_val = rule_score
        else:
            score_val = rule_score

        label = "LOW"
        if score_val >= 80:
            label = "CRITICAL"
        elif score_val >= 60:
            label = "HIGH"
        elif score_val >= 40:
            label = "MEDIUM"

        return {
            "score": score_val,
            "label": label,
        }


def risk_agent(state: SafetyState) -> SafetyState:
    logger.info(f"Running risk agent on {len(state.get('validated_violations', []))} violations")

    scorer = RiskScorer()
    risk_assessments = []

    for violation in state.get("validated_violations", []):
        risk = scorer.score(violation)
        assessment = {**violation, "risk_score": risk["score"], "risk_label": risk["label"]}
        risk_assessments.append(assessment)

    return {**state, "risk_assessments": risk_assessments}
