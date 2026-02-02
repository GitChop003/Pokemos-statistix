import pandas as pd
import os

def load_and_preprocess(filepath):
    print(f"Loading {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None

    df = pd.read_csv(filepath)
    print(f"Initial shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    # 1. Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicates found: {duplicates}")
    if duplicates > 0:
        print("Dropping duplicates...")
        df = df.drop_duplicates()

    # 2. Check for missing values
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Fill missing values if necessary for critical columns
    # Type_2 usually has NaNs, which means no second type. We can fill with 'None' or leave as is.
    # We need to ensure 'Total' (target) is not null.
    if df['Total'].isnull().sum() > 0:
        print("Warning: Target 'Total' has missing values. Dropping those rows.")
        df = df.dropna(subset=['Total'])

    # 3. Optimize Data Types
    # Convert 'False'/'True' strings to booleans if they aren't already
    bool_cols = ['isLegendary', 'hasGender', 'hasMegaEvolution']
    for col in bool_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                 # sometimes loaded as strings
                 df[col] = df[col].astype(bool)
            print(f"Column {col} type: {df[col].dtype}")

    # 4. Data Integrity Check
    # Ensure Total is numeric
    if not pd.api.types.is_numeric_dtype(df['Total']):
         print("Error: 'Total' column is not numeric!")
    
    
    # Check for corruption (e.g. negative stats)
    stats = ['HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'Total']
    for stat in stats:
        if (df[stat] < 0).any():
            print(f"Warning: corruption detected. Negative values in {stat}")

    # Check consistency between hasGender and Pr_Male
    gender_mismatch = df[~df['hasGender'] & df['Pr_Male'].notnull()]
    if not gender_mismatch.empty:
        print(f"Warning: {len(gender_mismatch)} rows have hasGender=False but a Pr_Male value. Converting Pr_Male to NaN.")
        df.loc[~df['hasGender'], 'Pr_Male'] = None

    print("Data validation complete.")
    return df

if __name__ == "__main__":
    file_path = 'Pokemon Data.csv'
    df = load_and_preprocess(file_path)
    if df is not None:
        print("First 5 rows:")
        print(df.head())
        # Save cleaned version if needed, or we just verify it works for the app
        df.to_csv('Pokemon Data_cleaned.csv', index=False)
        print("Saved cleaned data to Pokemon Data_cleaned.csv")
