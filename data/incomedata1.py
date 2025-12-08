import pandas as pd
path = "HDPulse_data_export.csv"
income = pd.read_csv(path, skiprows=4)
income = income[income["FIPS"] >= 13001]
income["FIPS"] = income["FIPS"].astype(int).astype(str).str.zfill(5)
income["MedianIncome"] = (
    income["Value (Dollars)"]
    .str.replace(",", "", regex=False)
    .astype(int))

# optional: check
print(income.head())

import json
import requests
url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
counties_geojson = requests.get(url).json()

import plotly.express as px

fig = px.choropleth(
    income,
    geojson=counties_geojson,
    locations="FIPS",              # must match the 'id' field in the geojson
    color="MedianIncome",
    color_continuous_scale="Viridis",
    scope="usa",
    hover_data={"County": True, "MedianIncome": ":,$"},  # show county + $ formatting
    labels={"MedianIncome": "Median family income ($)"}
)

# zoom in to just Georgia
fig.update_geos(fitbounds="locations", visible=False)

# nicer hover text (optional)
fig.update_traces(
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"      # County name
        "Median income: $%{customdata[1]:,}<extra></extra>"
    ),
    customdata=income[["County", "MedianIncome"]].to_numpy()
)

fig.update_layout(
    title="Median Family Income by County, Georgia",
    margin={"r":0,"t":40,"l":0,"b":0}
)

fig.show()

from dash import Dash, dcc, html
import plotly.express as px

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1("Georgia County Median Income Map"),
    dcc.Graph(figure=fig)
])

fig.write_html("georgia_income_map.html")