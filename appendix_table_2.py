"""
Generates Appendix Table 2: State-level Rural vs. Urban Smoking Prevalence
for 2018 & 2023.

Input:
- combinedbrfss_18_23v9.sas7bdat: SAS dataset expected in the same directory.
  Required variables: _STATE, year_centered, URRU, CURRENTSMOKER, _LLCPWT.

Output:
- appendix_table_2.csv: CSV file saved in the same directory, containing
  the formatted table with columns:
    - STATEFIPS
    - Rural_Prevalence_2018_CI
    - Urban_Prevalence_2018_CI
    - Ratio_2018
    - Rural_Prevalence_2023_CI
    - Urban_Prevalence_2023_CI
    - Ratio_2023
    - Change_In_Ratio
"""
import pandas as pd
import numpy as np
import os

def calculate_prevalence_ci(group):
    '''
    Calculates weighted smoking prevalence, effective sample size (n_eff),
    and the 95% confidence interval (CI) for a given data group.

    Args:
        group (pd.DataFrame): DataFrame subgroup with _LLCPWT (weights) and
                              CURRENTSMOKER (0 or 1 indicator).

    Returns:
        pd.Series: Contains 'prevalence', 'n_eff' (effective sample size),
                   'ci_lower' (CI lower bound), 'ci_upper' (CI upper bound),
                   and 'prevalence_ci_str' (formatted string for prevalence and CI).
    '''
    weighted_smokers = (group['_LLCPWT'] * group['CURRENTSMOKER']).sum()
    total_weight = group['_LLCPWT'].sum()

    # Handle cases where the group is empty or weights sum to zero
    if total_weight == 0:
        return pd.Series({
            'prevalence': np.nan,
            'n_eff': 0,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'prevalence_ci_str': 'N/A'
        })

    prevalence = weighted_smokers / total_weight

    # Calculate effective sample size (Kish's formula for weighted data)
    sum_weights_sq = (group['_LLCPWT']**2).sum()
    if sum_weights_sq == 0:
        n_eff = 0
    else:
        n_eff = (total_weight**2) / sum_weights_sq

    # Calculate 95% CI using Wilson score interval for proportions, adjusted for effective sample size
    if n_eff == 0 or not (0 <= prevalence <= 1): # Prevalence must be between 0 and 1
        ci_lower = np.nan
        ci_upper = np.nan
        # Provide prevalence percentage even if CI is N/A, unless prevalence itself is NaN
        prevalence_ci_str = f"{prevalence*100:.1f}% (N/A)" if not np.isnan(prevalence) else "N/A"
    else:
        # Ensure p*(1-p) is not negative due to potential floating point inaccuracies if p is slightly outside [0,1]
        p_for_ci = max(0, min(1, prevalence))
        margin_of_error = 1.96 * np.sqrt((p_for_ci * (1 - p_for_ci)) / n_eff)
        ci_lower = max(0, prevalence - margin_of_error) # CI bounds should not go below 0 or above 1
        ci_upper = min(1, prevalence + margin_of_error)
        prevalence_ci_str = f"{prevalence*100:.1f}% ({ci_lower*100:.1f}% - {ci_upper*100:.1f}%)"

    return pd.Series({
        'prevalence': prevalence,
        'n_eff': n_eff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'prevalence_ci_str': prevalence_ci_str
    })

def main():
    """
    Main function to load data, perform calculations, structure the table,
    and save it to a CSV file.
    """
    # --- 1. Load the SAS dataset ---
    # Determine the directory of the current script to make file paths relative
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'combinedbrfss_18_23v9.sas7bdat')

    try:
        # Attempt to load the SAS dataset using pandas
        # Specify encoding 'iso-8859-1' (latin1) as it's common for SAS files
        df = pd.read_sas(data_file, format='sas7bdat', encoding='iso-8859-1')
        print(f"Successfully loaded data from {data_file}")
    except FileNotFoundError:
        print(f"Error: The data file was not found at {data_file}")
        print("Please ensure 'combinedbrfss_18_23v9.sas7bdat' is in the same directory as the script.")
        return # Exit script if data not found
    except Exception as e:
        print(f"Error loading SAS dataset: {e}")
        return # Exit script on other loading errors

    # --- 2. Filter Data for 2018 and 2023 ---
    # The 'year_centered' column is expected: 2018 is -2, 2023 is 3 (assuming 2020 is the center year 0)
    if 'year_centered' not in df.columns:
        print("Error: 'year_centered' column not found in the dataset.")
        # The script could attempt to derive 'year_centered' from 'IYEAR' if available,
        # but current requirements specify 'year_centered' must be present.
        if 'IYEAR' in df.columns:
            print("Info: 'IYEAR' column found, but 'year_centered' is required for specific filtering logic.")
            # Example derivation: df['year_centered_calculated'] = df['IYEAR'] - 2020
            # However, sticking to the explicit requirement for 'year_centered'.
        print("Please ensure 'year_centered' column is present as specified.")
        return

    try:
        # Convert 'year_centered' to numeric, coercing errors. Non-numeric values become NaN.
        df['year_centered'] = pd.to_numeric(df['year_centered'], errors='coerce')
        # Drop rows where 'year_centered' could not be converted (became NaN)
        df.dropna(subset=['year_centered'], inplace=True)
    except Exception as e:
        print(f"Error converting 'year_centered' to numeric: {e}")
        return

    # Filter data for the specific years: 2018 (year_centered == -2) and 2023 (year_centered == 3)
    # .copy() is used to avoid SettingWithCopyWarning later on
    df_2018 = df[df['year_centered'] == -2].copy()
    df_2023 = df[df['year_centered'] == 3].copy()

    if df_2018.empty:
        print("Warning: No data found for the year 2018 (year_centered == -2).")
    else:
        print(f"Filtered {len(df_2018)} records for 2018.")

    if df_2023.empty:
        print("Warning: No data found for the year 2023 (year_centered == 3).")
    else:
        print(f"Filtered {len(df_2023)} records for 2023.")

    # Verify that other essential columns for calculation are present in the original DataFrame
    required_cols = ['_STATE', 'URRU', 'CURRENTSMOKER', '_LLCPWT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in the dataset: {', '.join(missing_cols)}. Cannot proceed.")
        return

    # Data type conversion and cleaning for key columns in the filtered DataFrames
    # CURRENTSMOKER: Expected to be 0 (non-smoker) or 1 (smoker).
    # URRU: Expected to be 0 (urban) or 1 (rural).
    for year_df in [df_2018, df_2023]:
        if not year_df.empty:
            year_df['CURRENTSMOKER'] = pd.to_numeric(year_df['CURRENTSMOKER'], errors='coerce')
            year_df['URRU'] = pd.to_numeric(year_df['URRU'], errors='coerce')
            year_df['_LLCPWT'] = pd.to_numeric(year_df['_LLCPWT'], errors='coerce')


    # Drop rows where key variables for calculation are missing (NaN) AFTER attempting conversion
    key_calc_cols = ['_LLCPWT', 'CURRENTSMOKER', 'URRU', '_STATE']
    df_2018.dropna(subset=key_calc_cols, inplace=True)
    df_2023.dropna(subset=key_calc_cols, inplace=True)

    # Report if data for a year was entirely dropped due to missing values in key columns
    if df_2018.empty and not df[df['year_centered'] == -2].empty: # Check original df to confirm data existed before dropna
        print("Warning: All data for 2018 was dropped due to missing values in key calculation columns (_LLCPWT, CURRENTSMOKER, URRU, _STATE).")
    if df_2023.empty and not df[df['year_centered'] == 3].empty:
        print("Warning: All data for 2023 was dropped due to missing values in key calculation columns (_LLCPWT, CURRENTSMOKER, URRU, _STATE).")

    # --- 3. Implement Calculation Logic for Each Year ---
    processed_years = {} # Dictionary to store summary DataFrames for each year

    for year_label, df_year_orig in [('2018', df_2018), ('2023', df_2023)]:
        if df_year_orig.empty:
            print(f"Skipping calculations for {year_label} as no data was available after filtering and cleaning.")
            processed_years[year_label] = pd.DataFrame() # Store empty DataFrame to simplify later steps
            continue

        print(f"Processing data for {year_label}...")
        df_year = df_year_orig.copy() # Work on a copy

        # Map numeric URRU (0/1) to categorical strings ('Urban'/'Rural') for pivoting
        # Ensure URRU is integer type before mapping (already converted by to_numeric and NaNs dropped)
        df_year['URRU'] = df_year['URRU'].astype(int)
        df_year['URRU_cat'] = df_year['URRU'].map({0: 'Urban', 1: 'Rural'})

        # Verify mapping - if URRU contained values other than 0 or 1, URRU_cat would have NaNs
        if df_year['URRU_cat'].isnull().any():
            print(f"Warning: Some URRU values in {year_label} data were not 0 or 1. Rows with unmapped URRU_cat will be excluded from URRU-specific grouping.")
            # Optionally, drop rows with NaN URRU_cat if they should not be included in any group
            # df_year.dropna(subset=['URRU_cat'], inplace=True)

        # Group data by state and urban/rural category, then apply the CI calculation function
        # _STATE column is used as is; its type (numeric FIPS or string) depends on SAS import
        summary_df = df_year.groupby(['_STATE', 'URRU_cat']).apply(calculate_prevalence_ci).reset_index()

        if summary_df.empty:
            print(f"No summary data could be calculated for {year_label}.")
        else:
            print(f"Calculated summary statistics for {year_label}.")

        processed_years[year_label] = summary_df

    # --- 4. Pivot and Structure Data for Each Year ---
    final_yearly_data = {} # Dictionary to store pivoted DataFrames for each year
    for year_label, summary_df in processed_years.items():
        if summary_df.empty:
            print(f"No summary data for {year_label} to pivot.")
            # Create an empty DataFrame with expected columns and _STATE index
            # This standardizes structure for the merge step, even if one year has no data.
            final_yearly_data[year_label] = pd.DataFrame(
                columns=['_STATE', f'Rural_Prevalence_{year_label}_CI',
                         f'Urban_Prevalence_{year_label}_CI', f'Ratio_{year_label}']
            ).set_index('_STATE')
            continue

        # Pivot the table: index by _STATE, columns by URRU_cat, values are prevalence and CI string
        pivot_df = summary_df.pivot_table(
            index='_STATE',
            columns='URRU_cat',
            values=['prevalence', 'prevalence_ci_str'] # Pivot both raw prevalence (for ratio) and CI string (for display)
        ).reset_index()

        # Flatten MultiIndex columns: ('prevalence', 'Rural') becomes 'prevalence_Rural'
        pivot_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in pivot_df.columns.values]

        # Ensure columns for both 'Urban' and 'Rural' categories exist after pivot.
        # A state might only have data for one category, so pivot might not create all columns.
        for ur_cat in ['Urban', 'Rural']:
            if f'prevalence_{ur_cat}' not in pivot_df.columns:
                pivot_df[f'prevalence_{ur_cat}'] = np.nan # Add missing prevalence column with NaN
            if f'prevalence_ci_str_{ur_cat}' not in pivot_df.columns:
                pivot_df[f'prevalence_ci_str_{ur_cat}'] = "N/A" # Add missing CI string column with "N/A"

        # Rename CI string columns to their final, year-specific format
        pivot_df.rename(columns={
            f'prevalence_ci_str_Rural': f'Rural_Prevalence_{year_label}_CI',
            f'prevalence_ci_str_Urban': f'Urban_Prevalence_{year_label}_CI',
        }, inplace=True)

        # Calculate Ratio: Rural prevalence / Urban prevalence
        # Uses the raw 'prevalence_Rural' and 'prevalence_Urban' columns.
        pivot_df[f'Ratio_{year_label}'] = pivot_df[f'prevalence_Rural'] / pivot_df[f'prevalence_Urban']
        # Replace infinite values (if Urban prevalence was 0) with NaN
        pivot_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Select and order columns for this year's final DataFrame structure
        year_final_df = pivot_df[['_STATE',
                                  f'Rural_Prevalence_{year_label}_CI',
                                  f'Urban_Prevalence_{year_label}_CI',
                                  f'Ratio_{year_label}']]

        year_final_df.set_index('_STATE', inplace=True) # Set _STATE as index for merging
        final_yearly_data[year_label] = year_final_df
        print(f"Pivoted and structured data for {year_label}.")

    # --- 5. Combine Yearly Data and Calculate Change ---
    df_2018_final = final_yearly_data.get('2018')
    df_2023_final = final_yearly_data.get('2023')

    # Ensure DataFrames exist for merging, even if empty from previous steps
    # This handles cases where one year might have had no data at all.
    if df_2018_final is None: # Should not happen if previous step correctly initializes empty DFs
        print("Critical Warning: df_2018_final was None before merge. Creating empty placeholder.")
        df_2018_final = pd.DataFrame(columns=['_STATE', f'Rural_Prevalence_2018_CI', f'Urban_Prevalence_2018_CI', f'Ratio_2018']).set_index('_STATE')
    if df_2023_final is None:
        print("Critical Warning: df_2023_final was None before merge. Creating empty placeholder.")
        df_2023_final = pd.DataFrame(columns=['_STATE', f'Rural_Prevalence_2023_CI', f'Urban_Prevalence_2023_CI', f'Ratio_2023']).set_index('_STATE')

    if df_2018_final.empty:
        print("Warning: No processed data available for 2018 to contribute to the final table.")
    if df_2023_final.empty:
        print("Warning: No processed data available for 2023 to contribute to the final table.")

    # Merge the 2018 and 2023 dataframes.
    # Using an outer merge to ensure all states from both years are included.
    # States not present in one year's data will have NaN for that year's columns.
    # Merging is done on '_STATE' (the index of df_2018_final and df_2023_final).
    final_table = pd.merge(df_2018_final, df_2023_final, on='_STATE', how='outer')

    # Calculate Change in Ratio: Ratio_2023 - Ratio_2018
    # This requires both ratio columns to be present (even if with NaNs).
    if f'Ratio_2023' in final_table.columns and f'Ratio_2018' in final_table.columns:
        final_table['Change_In_Ratio'] = final_table[f'Ratio_2023'] - final_table[f'Ratio_2018']
    else:
        # This might occur if one year had absolutely no data, and its ratio column wasn't created.
        # The earlier placeholder DataFrame creation should prevent this, but as a safeguard:
        print("Warning: Ratio columns for both 2023 and 2018 are not present. Cannot calculate Change_In_Ratio.")
        final_table['Change_In_Ratio'] = np.nan # Add column with NaNs

    # Reset index so _STATE becomes a regular column for the CSV output.
    final_table.reset_index(inplace=True)

    # Standardize state column name (e.g., to STATEFIPS, or keep as _STATE per preference)
    final_table.rename(columns={'_STATE': 'STATEFIPS'}, inplace=True)

    # --- 6. Format Final Table ---
    # Define the exact desired column order for the output CSV
    column_order = [
        'STATEFIPS',
        'Rural_Prevalence_2018_CI',
        'Urban_Prevalence_2018_CI',
        'Ratio_2018',
        'Rural_Prevalence_2023_CI',
        'Urban_Prevalence_2023_CI',
        'Ratio_2023',
        'Change_In_Ratio'
    ]

    # Ensure all specified columns exist in final_table, adding any missing ones with NaN.
    # This is a robust way to prevent errors if some columns weren't created due to missing data.
    for col in column_order:
        if col not in final_table.columns:
            final_table[col] = np.nan
            print(f"Warning: Column '{col}' was missing from final_table and has been added with NaN values.")

    # Reorder columns to match the specified 'column_order'
    final_table = final_table[column_order]

    print("Final table formatted with specified column order.")

    # Display the first few rows of the final table for review
    print("\n--- Combined Table (First 5 rows) ---")
    print(final_table.head())

    # --- 7. Save Output to CSV ---
    # Construct the full path for the output CSV file in the script's directory
    output_csv_file = os.path.join(script_dir, 'appendix_table_2.csv')
    try:
        # Save the final table to CSV, without writing the DataFrame index
        final_table.to_csv(output_csv_file, index=False)
        print(f"\nSuccessfully saved the final table to: {output_csv_file}")
    except Exception as e:
        print(f"\nError saving the table to CSV: {e}")

    # End of main function

if __name__ == '__main__':
    main()
