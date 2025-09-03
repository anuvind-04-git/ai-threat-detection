import ipaddress
from typing import List
import pandas as pd

# Safe helpers that work even when some columns are missing
def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Ensure DataFrame has specified columns, filling missing ones with 0.
    
    Args:
        df: Input DataFrame
        cols: List of column names to ensure exist
    Returns:
        DataFrame with all specified columns
    """
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df

def to_hour(series: pd.Series, tz_aware: bool = False) -> pd.Series:
    """
    Convert timestamp series to hour of day (0-23).
    
    Args:
        series: Input timestamp series
        tz_aware: Whether to parse timestamps as UTC
    Returns:
        Series containing hour of day as int
    """
    ts = pd.to_datetime(series, errors="coerce", utc=tz_aware)
    return ts.dt.hour.fillna(0).astype(int)

def is_private_ip(ip_str: str) -> int:
    """
    Check if IP address is private.
    
    Args:
        ip_str: IP address string
    Returns:
        1 if private IP, 0 if public or invalid
    """
    try:
        return int(ipaddress.ip_address(ip_str).is_private)
    except Exception:
        return 0

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the DataFrame.
    
    Args:
        df: Input DataFrame
    Returns:
        DataFrame with added features
    """
    # Hour of day
    if "ts" in df.columns and "hour" not in df.columns:
        df["hour"] = to_hour(df["ts"])  # 0-23
    else:
        df["hour"] = df.get("hour", 0)
        
    # Private IP indicator
    if "ip" in df.columns and "is_private_ip" not in df.columns:
        df["is_private_ip"] = df["ip"].astype(str).map(is_private_ip)
    else:
        df["is_private_ip"] = df.get("is_private_ip", 0)

    return df