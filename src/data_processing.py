"""
Data loading and cleaning.
"""

import pandas as pd
import numpy as np


def load_and_clean_data(filepath):
    """Load Excel file and return cleaned dataframe."""

    # Load data
    df = pd.read_excel(filepath, sheet_name='Sheet1')

    # Rename columns
    df.columns = [
        'Condition', 'Sample_ID', 'Before_UV', 'UV_0.3mW', 'Duration_0.3mW',
        'UV_3mW', 'Duration_3mW', 'Measurement_Day', 'Refrigeration', 'Notes'
    ]

    # Remove rows with 'Average' in Sample_ID
    df = df[~df['Sample_ID'].astype(str).str.contains('Average', na=False)]

    # Forward-fill condition names
    df['Condition'] = df['Condition'].ffill()

    # Standardize condition names (remove extra spaces)
    df['Condition'] = df['Condition'].str.strip()
    df['Condition'] = df['Condition'].replace({
        'Dura+ PBS': 'Dura+PBS',
        '2mM + RF': '2mM+RF',
        '4mM + RF': '4mM+RF',
        '8mM + RF': '8mM+RF',
    })

    # Keep only rows with baseline measurements
    df = df.dropna(subset=['Before_UV'])

    # Convert stiffness columns to numeric
    for col in ['Before_UV', 'UV_0.3mW', 'UV_3mW']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # UNIT CONVERSION: RF conditions have post-UV values in MPa, PBS in kPa
    # Convert RF conditions to kPa (multiply by 1000) for consistency
    rf_mask = df['Condition'].str.contains('RF', na=False)
    df.loc[rf_mask, 'UV_0.3mW'] = df.loc[rf_mask, 'UV_0.3mW'] * 1000  # MPa to kPa
    df.loc[rf_mask, 'UV_3mW'] = df.loc[rf_mask, 'UV_3mW'] * 1000      # MPa to kPa

    # Add RF concentration as numeric
    rf_map = {'Dura+PBS': 0, '2mM+RF': 2, '4mM+RF': 4, '8mM+RF': 8}
    df['RF_Concentration'] = df['Condition'].map(rf_map)

    # Calculate fold changes
    df['Fold_Change_0.3mW'] = df['UV_0.3mW'] / df['Before_UV']
    df['Fold_Change_3mW'] = df['UV_3mW'] / df['Before_UV']

    # Reset index
    df = df.reset_index(drop=True)

    return df
