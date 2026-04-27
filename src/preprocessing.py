def clean_data(df):
    df = df.copy()

    ##Filling numeric columns
    num_col = df.select_dtypes(include = ['number']).columns
    df[num_col] = df[num_col].fillna(df[num_col].mean())

    ##Filling categorical columns
    cat_col = df.select_dtypes(include=['object']).columns
    for col in cat_col:
        if df[col].mode().empty:
            df[col] = df[col].fillna("Unknown")  # fallback
        else:
            df[col] = df[col].fillna(df[col].mode()[0])


    return df