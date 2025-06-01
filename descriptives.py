import os
import pandas as pd

# 1. Load the CSV dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_name = 'combinedbrfss_18_23v9.csv' # Updated CSV file name
data_file_path = os.path.join(script_dir, csv_file_name)

print(f"Attempting to read CSV file: {data_file_path}...")
try:
    df = pd.read_csv(data_file_path)
    print(f"Successfully read {csv_file_name}.")
except FileNotFoundError:
    print(f"Error: CSV file not found at {data_file_path}")
    print(f"Please ensure you have run 'convert_to_csv.py' to generate '{csv_file_name}',")
    print(f"or that '{csv_file_name}' is in the same directory as this script.")
    import sys
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    import sys
    sys.exit(1)

# Ensure key numeric columns are indeed numeric after CSV load
numeric_cols_to_check = ['_LLCPWT', 'currentsmoker', 'URRU', '_AGE_G', 'SEXVAR', '_RACEGR3', '_EDUCAG', 'year_centered']
for col in numeric_cols_to_check:
    if col in df.columns:
        # Attempt to convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Optional: Check for NaNs introduced if conversion failed for some rows
        if df[col].isnull().any():
            print(f"Warning: Column '{col}' contains non-numeric values that were converted to NaN.")
    else:
        print(f"Warning: Expected numeric column '{col}' not found in CSV. This might lead to errors.")

# 2. Compute total weighted sample size
# Ensure _LLCPWT is not all NaN before summing, otherwise total_weight could be NaN
if df['_LLCPWT'].isnull().all():
    print("Error: '_LLCPWT' column is entirely NaN or missing. Cannot compute total weight.")
    import sys
    sys.exit(1)
total_weight = df['_LLCPWT'].sum()
if total_weight == 0:
    print("Warning: Total weighted sample size (_LLCPWT sum) is zero. Percentages will be NaN or Inf.")


# 3. Create categorical labels for each analytic variable
df['URRU_cat'] = df['URRU'].map({0: 'Urban', 1: 'Rural'}).fillna('Missing')

age_map = {
    1: '18–24',
    2: '25–34',
    3: '35–44',
    4: '45–54',
    5: '55–64',
    6: '65 or older'
}
df['Age_cat'] = df['_AGE_G'].map(age_map).fillna('Missing')

df['Sex_cat'] = df['SEXVAR'].map({1: 'Male', 2: 'Female'}).fillna('Missing')

race_map = {
    1: 'non-Hispanic White',
    2: 'non-Hispanic Black',
    3: 'non-Hispanic Other',
    4: 'non-Hispanic Multiracial',
    5: 'Hispanic'
}
df['Race_cat'] = df['_RACEGR3'].map(race_map).fillna('Missing')

edu_map = {
    1: 'Did not graduate high school',
    2: 'Graduated high school',
    3: 'Attended college or technical school',
    4: 'Graduated from college or technical school'
}
df['Edu_cat'] = df['_EDUCAG'].map(edu_map).fillna('Missing')

df['Year_cat'] = df['year_centered'].astype(int).astype(str)

# 4. Function to compute summary for a categorical column
def summarize(col_name):
    grouped = df.groupby(col_name).apply(
        lambda g: pd.Series({
            'Weighted sample size': g['_LLCPWT'].sum(),
            'Percentage': g['_LLCPWT'].sum() / total_weight * 100,
            'Smoking prevalence': (g['_LLCPWT'] * g['currentsmoker']).sum() / g['_LLCPWT'].sum() * 100
        })
    ).reset_index()
    grouped.columns = [col_name, 'Weighted sample size', 'Percentage', 'Smoking prevalence']
    return grouped

# 5. Generate and print summaries for each characteristic
urban_rural_summary = summarize('URRU_cat')
age_summary        = summarize('Age_cat')
sex_summary        = summarize('Sex_cat')
race_summary       = summarize('Race_cat')
edu_summary        = summarize('Edu_cat')
year_summary       = summarize('Year_cat')

print("\n=== Urban/Rural ===")
print(urban_rural_summary.to_string(index=False))

print("\n=== Age ===")
print(age_summary.to_string(index=False))

print("\n=== Sex ===")
print(sex_summary.to_string(index=False))

print("\n=== Race/Ethnicity ===")
print(race_summary.to_string(index=False))

print("\n=== Education ===")
print(edu_summary.to_string(index=False))

print("\n=== Year ===")
print(year_summary.to_string(index=False))

# … after all print() calls …
summaries = [
    ('Urban/Rural', urban_rural_summary),
    ('Age', age_summary),
    ('Sex', sex_summary),
    ('Race/Ethnicity', race_summary),
    ('Education', edu_summary),
    ('Year', year_summary),
]
combined_dfs = []
for name, df_sum in summaries:
    df2 = df_sum.copy()
    first_col = df2.columns[0]
    df2 = df2.rename(columns={first_col: 'Category'})
    df2.insert(0, 'Characteristic', name)
    combined_dfs.append(df2[['Characteristic','Category','Weighted sample size','Percentage','Smoking prevalence']])
combined_df = pd.concat(combined_dfs, ignore_index=True)
csv_file = os.path.join(script_dir, 'descriptives_summary.csv')
combined_df.to_csv(csv_file, index=False)
print(f"\nSaved combined summary table to {csv_file}")
