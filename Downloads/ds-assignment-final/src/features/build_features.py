import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import load_config # Ensure this import is present!

from src.features.transformations import (
    driver_distance_to_pickup,
    driver_historical_completed_bookings,
    hour_of_day,
    driver_historical_acceptance_rate,
)
from src.utils.store import AssignmentStore


def main():
    store = AssignmentStore()
    config = load_config() # Load the config here

    dataset = store.get_processed("dataset.csv")
    # Pass the target column name from config to the feature engineering pipeline
    dataset = apply_feature_engineering(dataset, target_col=config["target"])

    store.put_processed("transformed_dataset.csv", dataset)


def apply_feature_engineering(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Accepts the target column name and passes it to the necessary transformation."""
    return (
        df.pipe(driver_distance_to_pickup)
        .pipe(hour_of_day)
        .pipe(driver_historical_completed_bookings)
        # Pass the actual target column name to the new function
        .pipe(driver_historical_acceptance_rate, target_col=target_col) 
    )


if __name__ == "__main__":
    main()