import numpy as np
import pandas as pd

from src.features.build_features import apply_feature_engineering
from src.utils.guardrails import validate_prediction_results
from src.utils.store import AssignmentStore
# *** NEW IMPORT for loading configuration ***
from src.utils.config import load_config 


@validate_prediction_results
def main():
    store = AssignmentStore()
    config = load_config() # <-- NEW: Load the configuration

    df_test = store.get_raw("test_data.csv")
    
    # --- FIX START: Pass the required 'target_col' argument ---
    df_test = apply_feature_engineering(df_test, target_col=config["target"])
    # --- FIX END ---

    model = store.get_model("saved_model.pkl")
    df_test["score"] = model.predict(df_test)

    selected_drivers = choose_best_driver(df_test)
    store.put_predictions("results.csv", selected_drivers)


def choose_best_driver(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("order_id").agg({"driver_id": list, "score": list}).reset_index()
    df["best_driver"] = df.apply(
        lambda r: r["driver_id"][np.argmax(r["score"])], axis=1
    )
    df = df.drop(["driver_id", "score"], axis=1)
    df = df.rename(columns={"best_driver": "driver_id"})
    return df


if __name__ == "__main__":
    main()