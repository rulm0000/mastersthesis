#!/usr/bin/env python3
"""
Generate three US choropleth maps (Model 1,2,3) of URRU odds ratios
using Plotly and combine into one PNG with a shared legend.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_p(val):
    try:
        if isinstance(val, str) and val.startswith('<'):
            return float(val.strip('<'))
        return float(val)
    except:
        return np.nan

def main():
    # Read model results
    df = pd.read_excel('CS_QR_finalresults17.xlsx', sheet_name=0, engine='openpyxl')
    # State name to USPS code
    state_abbrev = {
        'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
        'Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL',
        'Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
        'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD',
        'Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO',
        'Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ',
        'New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH',
        'Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
        'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT',
        'Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
    }
    df['abbr'] = df['State'].map(state_abbrev)
    df['_STATE'] = df['State Code'].astype(int)

    # Model specifications
    models = [
        ('Model 1', 'Model 1_CS_OR', 'Model 1_CS_p'),
        ('Model 2', 'Model 2_CS_OR', 'Model 2_CS_p'),
        ('Model 3', 'Model 3_CS_OR', 'Model 3_CS_p')
    ]
    # Color definitions
    colors = ['lightgrey', '#a6bddb', '#3690c0', '#034e7b']

    # Temporary images
    imgs = []
    for i, (title, or_col, p_col) in enumerate(models, start=1):
        # Parse OR and p
        df['OR'] = pd.to_numeric(df[or_col], errors='coerce')
        df['p'] = df[p_col].apply(parse_p)
        # Identify significant and sufficient
        mask_sig = (df['p'] < 0.05) & df['OR'].notna()
        or_sig = df.loc[mask_sig, 'OR']
        # Compute tertiles
        if not or_sig.empty:
            q1, q2 = np.percentile(or_sig, [33.33, 66.67])
        else:
            q1 = q2 = np.nan
        # Define labels
        lbl0 = 'Insignificant (p≥0.05 or OR is null)'
        lbl1 = f'OR ≤ {q1:.2f}'
        lbl2 = f'{q1:.2f} < OR ≤ {q2:.2f}'
        lbl3 = f'OR > {q2:.2f}'
        lbl_map = {0:lbl0, 1:lbl1, 2:lbl2, 3:lbl3}
        # Category assignment
        def cat(row):
            if row['p'] >= 0.05 or np.isnan(row['OR']):
                return lbl0
            if row['OR'] <= q1:
                return lbl1
            elif row['OR'] <= q2:
                return lbl2
            else:
                return lbl3
        df['cat'] = df.apply(cat, axis=1)

        # Plotly choropleth
        try:
            import plotly.express as px
        except ImportError:
            raise ImportError('plotly is required: pip install plotly kaleido')
        fig = px.choropleth(
            df,
            locations='abbr',
            color='cat',
            color_discrete_map={lbl0:colors[0], lbl1:colors[1], lbl2:colors[2], lbl3:colors[3]},
            scope='usa',
            labels={'cat':'Category'},
            title=title
        )
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        # Use Albers USA projection for improved state shape rendering (includes AK & HI)
        fig.update_geos(scope='usa', projection_type='albers usa', visible=False)
        img_file = f'model_map_{i}.png'
        fig.write_image(img_file, width=400, height=300, scale=2)
        imgs.append(img_file)

    # Combine images in matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img_file, (title, _, _) in zip(axes, imgs, models):
        img = plt.imread(img_file)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    # Shared legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=colors[i], edgecolor='black', label=lbl) for i, lbl in enumerate([lbl0, lbl1, lbl2, lbl3])]
    fig.legend(handles=legend_elems, loc='lower center', ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = 'model_OR_maps.png'
    fig.savefig(out, dpi=300)
    print(f'Map figure saved to {out}')

if __name__ == '__main__':
    main()