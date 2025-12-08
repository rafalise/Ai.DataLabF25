import pandas as pd
import requests
import plotly.express as px

# ---------- INCOME MAP ----------
path = "HDPulse_data_export.csv"
income = pd.read_csv(path, skiprows=4)

# keep only GA counties
income = income[income["FIPS"] >= 13001]

income["FIPS"] = income["FIPS"].astype(int).astype(str).str.zfill(5)
income["MedianIncome"] = (
    income["Value (Dollars)"]
    .str.replace(",", "", regex=False)
    .astype(int)
)

# Get counties geojson once
url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
counties_geojson = requests.get(url).json()

fig_income = px.choropleth(
    income,
    geojson=counties_geojson,
    locations="FIPS",
    color="MedianIncome",
    color_continuous_scale="Viridis",
    scope="usa",
    hover_data={"County": True, "MedianIncome": ":,$"},
    labels={"MedianIncome": "Median family income ($)"}
)

fig_income.update_geos(fitbounds="locations", visible=False)
fig_income.update_traces(
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "Median income: $%{customdata[1]:,}<extra></extra>"
    ),
    customdata=income[["County", "MedianIncome"]].to_numpy()
)
fig_income.update_layout(
    title="Median Family Income by County, Georgia",
    margin={"r":0,"t":40,"l":0,"b":0}
)

# ---------- RACE/ETHNICITY MAP ----------
# here you reuse your existing code that builds `censuscountiesclean`
# and gives each county a FIPS and % race columns (WA_TOTAL, BA_TOTAL, etc.)

# ... your data wrangling here ...

# Example: map of % White (you can later add other race % to the dropdown)

path = "censuscountiesclean.csv"
censuscountiesclean = pd.read_csv(path)

fig_race = px.choropleth(
    censuscountiesclean,
    geojson=counties_geojson,
    locations="FIPS",
    color="WA_TOTAL",   # % White
    color_continuous_scale="Blues",
    scope="usa",
    hover_name='CTYNAME',
    hover_data={'FIPS': False},
    labels={"WA_TOTAL": "% White"}
)

fig_race.update_geos(fitbounds="locations", visible=False)
fig_race.update_traces(
    hovertemplate=censuscountiesclean['hover_text']  # from your script
)
fig_race.update_layout(
    title="% White by County, Georgia",
    margin={"r":0,"t":40,"l":0,"b":0}
)
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Georgia Demographics & Income Dashboard"),

    # Toggle to switch between maps
    dcc.RadioItems(
        id='map-toggle',
        options=[
            {'label': 'Median Income', 'value': 'income'},
            {'label': 'Race/Ethnicity (% White)', 'value': 'race'},
        ],
        value='income',   # default map shown
        inline=True,
        style={"marginBottom": "10px"}
    ),

    dcc.Graph(id='map-graph')
])

# callback chooses which figure to show
@app.callback(
    Output('map-graph', 'figure'),
    Input('map-toggle', 'value')
)
def update_map(selected_map):
    if selected_map == 'income':
        return fig_income
    else:
        return fig_race

if __name__ == "__main__":
    app.run_server(debug=True)
