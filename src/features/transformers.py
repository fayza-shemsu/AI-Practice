import numpy as np
import pandas as pd

def feature_engineering(df):

    # interaction features
    df['area_per_room'] = df['area'] / df['bedrooms'].replace(0, np.nan)
    df['area_per_room'] = df['area_per_room'].fillna(df['area_per_room'].median())

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    df['area_squared'] = df['area'] ** 2

    df['log_area'] = np.log1p(df['area'])

    df['bed_bath_interaction'] = df['bedrooms'] * df['bathrooms']

    df['is_new_or_renov_like'] = (df['stories'] >= 2).astype(int)

    return df