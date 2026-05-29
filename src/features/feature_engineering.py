def create_features(df):

    df = df.copy()

    # =========================
    # 1. Engagement features
    # =========================
    df["support_per_tenure"] = df["Support Calls"] / (df["Tenure"] + 1)

    df["usage_per_tenure"] = df["Usage Frequency"] / (df["Tenure"] + 1)

    # =========================
    # 2. Spending behavior
    # =========================
    df["spend_per_month"] = df["Total Spend"] / (df["Tenure"] + 1)

    # =========================
    # 3. Recency signal
    # =========================
    df["inactive_score"] = df["Last Interaction"] / (df["Tenure"] + 1)

    # =========================
    # 4. Risk flags
    # =========================
    df["high_support_flag"] = (df["Support Calls"] > df["Support Calls"].mean()).astype(int)

    df["low_usage_flag"] = (df["Usage Frequency"] < df["Usage Frequency"].mean()).astype(int)

    return df