import pandas as pd
import plotly.express as px
import sys
import os
sys.path.append('../lib')
from prepro import clean_df

print("loading data...")
df_raw = pd.read_csv('../data/data.txt', sep='\t')

df = df_raw.copy()
df_grouped = df.groupby(['大分類', '中分類', '小分類']).size().reset_index(name='小分類出現数')
fig = px.sunburst(
    df_grouped,
    path=['大分類', '中分類', '小分類'],
    values='小分類出現数',
    title='大分類・中分類・小分類の階層構造（サンバーストチャート）'
)
fig.show()