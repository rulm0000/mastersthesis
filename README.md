# Master Thesis Data Analysis

This repository contains scripts and datasets used to analyze BRFSS data and visualize odds ratios by state. Below is a brief overview of the important components and how to run the code.

## Repository Structure

- **CS_QR_finalresults17.xlsx**, **CS_QR_finalresults18.xlsx**: Excel files with model results.
- **combinedbrfss_18_23v9.csv**: Main BRFSS dataset in CSV format (large, tracked via Git LFS).
- **combinedbrfss_18_23v9.sas7bdat**: SAS version of the dataset (also in LFS).
- **descriptives.py**: Calculates weighted descriptive statistics and prints them.
- **descriptives2.py**: Similar to `descriptives.py`, exports summaries to `descriptives_summary.csv`.
- **generate_model_choropleth_maps.py**: Creates three-panel choropleth maps using Plotly and Matplotlib.
- **generate_square_choropleth.py**: Generates square-tile maps for odds ratios across three models. Running it produces `square_model_OR_maps.png`, which is not tracked in version control.
- **generate_geo_choropleth.py**: Creates three geographic maps across three models using state shapes.
- **us-states.json**: GeoJSON file used for state centroids in mapping.

## Dependencies

The scripts require Python with packages such as `pandas`, `numpy`, `matplotlib`, `plotly`, `geopandas`, and `openpyxl`. Install them with:

```bash
pip install pandas numpy matplotlib plotly geopandas openpyxl
```

## Running the Scripts

Run any of the Python files from the project root. For example:

```bash
python descriptives.py
python descriptives2.py
python generate_model_choropleth_maps.py
python generate_geo_choropleth.py
python generate_square_choropleth.py
```

The mapping scripts will produce PNG outputs in the repository directory.

## Next Steps

- Consider adding unit tests or validation scripts to ensure correct data processing.
- Create a `requirements.txt` or environment file for easier setup.
- Improve documentation with examples and expected outputs.
