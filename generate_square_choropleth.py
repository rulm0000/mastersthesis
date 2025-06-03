#!/usr/bin/env python3
"""Generate square tile choropleth maps for odds ratios from FinalResults18."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import geopandas as gpd
import os

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


def load_state_grid():
    """Create simple grid coordinates by sorting states by latitude and longitude."""
    gdf = gpd.read_file('us-states.json')
    gdf = gdf[gdf['name'] != 'Puerto Rico']
    gdf['abbr'] = gdf['name'].map(STATE_ABBR)
    cent = gdf.to_crs('EPSG:4326').centroid
    coords = pd.DataFrame({'abbr': gdf['abbr'], 'lon': cent.x, 'lat': cent.y})
    coords = coords.sort_values('lat', ascending=False).reset_index(drop=True)
    rows = 6
    cols = 10
    coords['row'] = coords.index // cols
    coords['col'] = coords.index % cols
    ordered = []
    for r in range(rows):
        sub = coords[coords['row'] == r].sort_values('lon').reset_index(drop=True)
        sub['col'] = range(len(sub))
        ordered.append(sub)
    coords = pd.concat(ordered).reset_index(drop=True)
    return coords[['abbr', 'row', 'col']]


def parse_p(val):
    try:
        if isinstance(val, str) and val.startswith('<'):
            return float(val.strip('<'))
        return float(val)
    except Exception:
        return np.nan


def build_maps():
    grid = load_state_grid()
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, (title, or_col, p_col) in zip(axes, models):
        df = results.copy()
        df['OR'] = pd.to_numeric(df[or_col], errors='coerce')
        df['p'] = df[p_col].apply(parse_p)
        mask = (df['p'] < 0.05) & (df['rural_n'] >= 50) & df['OR'].notna()
        sig_or = df.loc[mask, 'OR']
        if not sig_or.empty:
            q1, q2 = np.percentile(sig_or, [33.33, 66.67])
        else:
            q1 = q2 = np.nan
        labels = [
            'Insufficient sample or p≥0.05',
            f'OR ≤ {q1:.2f}',
            f'{q1:.2f} < OR ≤ {q2:.2f}',
            f'OR > {q2:.2f}'
        ]

        def cat(row):
            if row['rural_n'] < 50 or row['p'] >= 0.05 or np.isnan(row['OR']):
                return labels[0]
            if row['OR'] <= q1:
                return labels[1]
            elif row['OR'] <= q2:
                return labels[2]
            else:
                return labels[3]
        df['cat'] = df.apply(cat, axis=1)
        df = df.merge(grid, on='abbr', how='left')

        for _, r in df.iterrows():
            x, y = r['col'], 5 - r['row']  # flip y for display
            rect = Rectangle((x, y), 1, 1, facecolor=colors[labels.index(r['cat'])], edgecolor='black')
            ax.add_patch(rect)
            ax.text(x+0.5, y+0.5, r['abbr'], ha='center', va='center', fontsize=8)
        ax.set_xlim(0, grid['col'].max()+1)
        ax.set_ylim(-1, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title)

    legend_elems = [Patch(facecolor=colors[i], edgecolor='black', label=lab) for i, lab in enumerate(labels)]
    fig.legend(handles=legend_elems, loc='lower center', ncol=4, frameon=False)
    fig.tight_layout(rect=[0,0.05,1,1])
    out = 'square_model_OR_maps.png'
    fig.savefig(out, dpi=300)
    print(f'Saved {out}')

if __name__ == '__main__':
    build_maps()
