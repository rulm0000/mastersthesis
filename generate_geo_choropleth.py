#!/usr/bin/env python3
"""Generate geographic choropleth maps for odds ratios from FinalResults18."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import geopandas as gpd

STATE_ABBR = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO',
    'Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY',
    'Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN',
    'Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
    'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
    'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
    'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}


def parse_p(val):
    try:
        if isinstance(val, str) and val.startswith('<'):
            return float(val.strip('<'))
        return float(val)
    except Exception:
        return np.nan


def load_states():
    gdf = gpd.read_file('us-states.json')
    gdf = gdf[gdf['name'] != 'Puerto Rico']
    gdf['abbr'] = gdf['name'].map(STATE_ABBR)
    return gdf[['abbr', 'geometry']]


def build_maps():
    states = load_states()
    results = pd.read_excel('CS_QR_finalresults18.xlsx', engine='openpyxl')
    brf = pd.read_csv('combinedbrfss_18_23v9.csv', usecols=['_STATE','URRU'])
    rural = brf[brf['URRU']==1].groupby('_STATE').size()
    results['abbr'] = results['State'].map(STATE_ABBR)
    results['_STATE'] = results['State Code'].astype(int)
    results['rural_n'] = results['_STATE'].map(rural).fillna(0).astype(int)

    models = [
        ('Model 1', 'Model 1_CS_OR', 'Model 1_CS_p'),
        ('Model 2', 'Model 2_CS_OR', 'Model 2_CS_p'),
        ('Model 3', 'Model 3_CS_OR', 'Model 3_CS_p')
    ]

    colors = ['lightgrey', '#a6bddb', '#3690c0', '#034e7b']
    labels = [
        'Insufficient sample or p≥0.05',
        'OR < 1.2',
        '1.2 ≤ OR ≤ 1.5',
        'OR > 1.5'
    ]
    cmap = {labels[i]: colors[i] for i in range(len(labels))}

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, (title, or_col, p_col) in zip(axes, models):
        df = results.copy()
        df['OR'] = pd.to_numeric(df[or_col], errors='coerce')
        df['p'] = df[p_col].apply(parse_p)

        def cat(row):
            if row['rural_n'] < 50 or row['p'] >= 0.05 or np.isnan(row['OR']):
                return labels[0]
            if row['OR'] < 1.2:
                return labels[1]
            elif row['OR'] <= 1.5:
                return labels[2]
            else:
                return labels[3]

        df['cat'] = df.apply(cat, axis=1)
        merged = states.merge(df[['abbr','cat']], on='abbr', how='left')
        merged['color'] = merged['cat'].map(cmap)
        merged.plot(color=merged['color'], linewidth=0.5, edgecolor='black', ax=ax)
        ax.axis('off')
        ax.set_title(title)

    legend_elems = [Patch(facecolor=colors[i], edgecolor='black', label=lab) for i, lab in enumerate(labels)]
    fig.legend(handles=legend_elems, loc='lower center', ncol=4, frameon=False)
    fig.tight_layout(rect=[0,0.05,1,1])
    out = 'geo_model_OR_maps.png'
    fig.savefig(out, dpi=300)
    print(f'Saved {out}')

if __name__ == '__main__':
    build_maps()
