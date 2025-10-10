import pandas as pd
import math
import numpy as np
from src.utils.time import robust_hour_of_iso_date

# --- (haversine, driver_distance_to_pickup, hour_of_day functions remain unchanged) ---
def haversine(coord1, coord2):
    """Calculate the great-circle distance between two points (lat, lon) in kilometers."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # convert decimal degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    # Earth radius in kilometers
    R = 6371.0
    return R * c


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized haversine distance between driver location and pickup location in kilometers."""
    df = df.copy()
    # handle missing values by filling with zeros
    lat1 = df['driver_latitude'].fillna(0).astype(float).to_numpy()
    lon1 = df['driver_longitude'].fillna(0).astype(float).to_numpy()
    lat2 = df['pickup_latitude'].fillna(0).astype(float).to_numpy()
    lon2 = df['pickup_longitude'].fillna(0).astype(float).to_numpy()
    # convert to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371.0
    df['driver_distance'] = R * c
    return df

def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df


def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Create a feature counting how many bookings each driver completed prior to the current event."""
    df = df.copy()
    
    # --- Task 1 Fix: Robustness check for 'is_completed' ---
    if 'is_completed' not in df.columns:
        if 'is_accepted' in df.columns:
            df['is_completed'] = df['is_accepted'] 
        else:
            df['is_completed'] = 0 
    # --------------------------------------------------------

    if not pd.api.types.is_datetime64_any_dtype(df.get('event_timestamp')):
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], utc=True, errors='coerce')
        
    df = df.sort_values(['driver_id', 'event_timestamp'])
    
    df['historical_completed'] = df.groupby('driver_id')['is_completed'].cumsum() - df['is_completed']
    df['historical_completed'] = df['historical_completed'].clip(lower=0)
    
    return df


def driver_historical_acceptance_rate(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Creates a non-leaking feature: driver's historical acceptance rate (ratio of accepted offers to total offers).
    Accepts the target column name dynamically (Task 2 Improvement).
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df.get('event_timestamp')):
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], utc=True, errors='coerce')
        
    df = df.sort_values(['driver_id', 'event_timestamp'])
    
    # --- FIX START: Use dynamic target_col ---
    df['is_accepted_flag'] = df[target_col] 
    # --- FIX END ---
    
    df['is_offer_flag'] = 1 

    df['driver_total_accepted_past'] = (
        df.groupby('driver_id')['is_accepted_flag'].cumsum().shift(1).fillna(0)
    )

    df['driver_total_offers_past'] = (
        df.groupby('driver_id')['is_offer_flag'].cumsum().shift(1).fillna(0)
    )

    df['historical_acceptance_rate'] = (
        df['driver_total_accepted_past'] / (df['driver_total_offers_past'] + 1e-6)
    )
    
    df = df.drop(columns=['is_accepted_flag', 'is_offer_flag', 'driver_total_accepted_past', 'driver_total_offers_past'])

    return df