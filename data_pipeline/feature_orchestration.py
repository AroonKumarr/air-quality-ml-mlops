"""
Feature Pipeline - Scheduled script for hourly feature extraction.
"""
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

# Load .env for local development (GitHub Actions uses secrets)
from dotenv import load_dotenv
load_dotenv()

# Debug: Print if env vars are set
print(f"DEBUG: OPENWEATHERMAP_API_KEY set: {bool(os.getenv('OPENWEATHERMAP_API_KEY'))}")
print(f"DEBUG: HOPSWORKS_API_KEY set: {bool(os.getenv('HOPSWORKS_API_KEY'))}")

from app_core.feature_engineering.data_loader import AQIDataFetcher
from app_core.feature_engineering.feature_computation import compute_all_features
from app_core.feature_engineering.feature_registry import get_feature_store
from app_core.utils.logger import get_logger

import pandas as pd

logger = get_logger("feature_pipeline")


def run_feature_pipeline(cities: list = None):
    """
    Run the feature extraction pipeline.
    
    Args:
        cities: List of cities to fetch data for
    """
    if cities is None:
        cities = ["Karachi", "Lahore", "Islamabad"]
    
    logger.info("="*50)
    logger.info("Starting Feature Pipeline")
    logger.info(f"Cities: {cities}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*50)
    
    fetcher = AQIDataFetcher()
    feature_store = get_feature_store()
    
    all_data = []
    
    for city in cities:
        try:
            logger.info(f"Fetching data for {city}...")
            
            # Fetch current AQI data
            raw_data = fetcher.fetch_current_aqi(city)
            
            if raw_data:
                all_data.append(raw_data)
                logger.info(f"  ✓ Fetched data for {city}")
            else:
                logger.warning(f"  ✗ No data returned for {city}")
                
        except Exception as e:
            logger.error(f"  ✗ Error fetching data for {city}: {e}")
    
    if all_data:
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Compute feature_engineering
        logger.info("Computing feature_engineering...")
        df_feature_engineering, feature_cols, target_cols = compute_all_features(df)
        logger.info(f"  ✓ Computed {len(feature_cols)} feature_engineering")
        
        # Save to feature store
        logger.info("Saving to feature store...")
        feature_store.save_feature_engineering(df_feature_engineering, 'aqi_feature_engineering')
        logger.info(f"  ✓ Saved {len(df_feature_engineering)} records")
        
    else:
        logger.warning("No data to process")
    
    logger.info("Feature pipeline completed")
    logger.info("="*50)


if __name__ == "__main__":
    run_feature_pipeline()
