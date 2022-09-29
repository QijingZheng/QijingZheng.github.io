#!/usr/bin/env python

import pandas as pd
import plotly.graph_objects as go

df         = pd.read_csv('site_visits_counts.csv')
df['Date'] = df['Date'].apply(pd.to_datetime)

NoDays     = (df.iloc[-1, 0] - df.iloc[0, 0]).days
print(NoDays)

############################################################
fig = go.Figure()
############################################################
fig.add_trace(
    go.Scatter(
	x=df['Date'], y=df['Hits_Staff'],
        name='staff',
	mode='lines+markers',
        line=dict(color='red', width=1.0,),
	marker=dict(color='blue', size=6, symbol='circle-open'),
    )
)
fig.add_trace(
    go.Scatter(
	x=df['Date'], y=df['Hits_Home'],
        name='home',
	mode='lines+markers',
        line=dict(color='blue', width=1.0,),
	marker=dict(color='green', size=6, symbol='circle-open'),
    )
)

fig.update_layout(
    width=720, height=400,
    title=dict(
        text="My homepage visit count in <span style='color: red'>{}</span> Days".format(NoDays),
        x=0.5,
        font=dict(color='black'),
    ),
    xaxis=dict(
	title=dict(
            text="DATE",
            font=dict(),
        )
    ),
    yaxis=dict(
	title=dict(
            text="HITs",
            font=dict(),
        )
    ),
)

fig.show()
