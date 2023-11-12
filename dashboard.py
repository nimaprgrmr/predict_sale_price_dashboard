import dash
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State
# import plotly.express as px
import plotly.graph_objs as go
from utils import make_period_time
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# import dash_bootstrap_components as dbc

# Loading model and scaler
FILE_NAME_MODEL = 'models/bamland_predictor.pickle'
FILE_NAME_SCALER = 'models/bamland_scaler.pickle'
MODEL = pickle.load(open(FILE_NAME_MODEL, 'rb'))
SCALER = pickle.load(open(FILE_NAME_SCALER, 'rb'))


# Define the available options for year, month, and day
YEARS = [str(year) for year in range(1405, 1396, -1)]
MONTHS = [str(month).zfill(2) for month in range(1, 13)]
DAYS = [str(day).zfill(2) for day in range(1, 32)]


def make_predictions(start_date:str, end_date:str, model, scaler):

    start_date = list(start_date.split("-"))
    end_date = list(end_date.split("-"))

    start_date = [int(x) for x in start_date]
    end_date = [int(x) for x in end_date]

    input_features = make_period_time(start_date, end_date)
    scaler = scaler
    scaled_input = scaler.transform(input_features)
    model = model
    predictions = model.predict(scaled_input)

    return predictions


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Creating my dashboard application
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the font type you want to use for buttons
font_style = {
    'font-family': 'Helvetica, Arial, sans-serif',
    'font-size': '13px',  # Adjust the font size as needed
    'color': 'Black'  # Change 'Arial' to the desired font family
}

app.layout = html.Div([
    html.Div([
        html.Label('Select Start Year'),
        dcc.Dropdown(id='start-year-dropdown', options=[{'label': year, 'value': year} for year in YEARS],
                     value='1402', style={'background-color': 'rgba(221,235,241,0.7)'}),
    ], style={'display': 'inline-block', 'margin-right': '15px', 'margin-left': '20px'}),

    html.Div([
        html.Label('Start Month'),
        dcc.Dropdown(id='start-month-dropdown', options=[{'label': month, 'value': month} for month in MONTHS],
                     value='01', style={'background-color': 'rgba(221,235,241,0.7)'}),
    ], style={'display': 'inline-block', 'margin-right': '15px'}),

    html.Div([
        html.Label('Start Day'),
        dcc.Dropdown(id='start-day-dropdown', options=[{'label': day, 'value': day} for day in DAYS], value='01',
                     style={'background-color': 'rgba(221,235,241,0.7)'}),
    ], style={'display': 'inline-block', 'margin-right': '30px'}),

    html.Div([
        html.Label('Select End Year'),
        dcc.Dropdown(id='end-year-dropdown', options=[{'label': year, 'value': year} for year in YEARS], value='1402',
                     style={'background-color': 'rgba(248,194,127,0.6)'}),
    ], style={'display': 'inline-block', 'margin-right': '15px'}),

    html.Div([
        html.Label('End Month'),
        dcc.Dropdown(id='end-month-dropdown', options=[{'label': month, 'value': month} for month in MONTHS],
                     value='02', style={'background-color': 'rgba(248,194,127,0.6)'}),
    ], style={'display': 'inline-block', 'margin-right': '15px'}),

    html.Div([
        html.Label('End Day'),
        dcc.Dropdown(id='end-day-dropdown', options=[{'label': day, 'value': day} for day in DAYS], value='01',
                     style={'background-color': 'rgba(248,194,127,0.6)'}),
    ], style={'display': 'inline-block', 'margin-right': '30px'}),

    html.Div([
        html.Button('Predict Sales', id='predict-button', n_clicks=0, style={'background-color': 'green', 'color': 'white'}),
    ], style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-top': '55px'}),

    # # Spacer to push the Help button to the right
    # html.Div(style={'flex': 0.8}),

    html.Div([
        html.Button('Help', id='help-button', n_clicks=0, style={'background-color': 'rgb(13,152,186)', 'color': 'white'}),
    ], style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-top': '55px', 'margin-right': '20px',
              'margin-left': '550px'}),

    html.Div([
        dcc.Graph(id='prediction-plot'), ], style={'margin-top': '20px'}),

    html.Div(id='help-modal', style={'display': 'none'}, children=[
        html.Div([
            html.H4("Dashboard Guide"),
            dcc.Markdown("""
                    This is a guide for using your dashboard.

                    1. Use the dropdowns to select start and end dates.
                    2. Click on the "Predict Sales" button to generate predictions.
                    3. Click on the "Help" button to close this guide.

                    Enjoy using the dashboard!
                """, style={'color': 'black', 'background-color': 'rgba(13,152,186, 0.3)', 'padding': '10px', 'border-radius': '10px'}),

        ], style={'position': 'fixed', 'top': '15%', 'left': '65%', 'transform': 'translate(-25%, -50%)'}),
        html.Div(id='modal-background',
                 style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'height': '80%',
                        'background-color': 'rgba(0, 0, 0, 0.5)', 'display': 'none'}),
    ]),
])


@app.callback(
    Output('prediction-plot', 'figure'),
    Input('predict-button', 'n_clicks'),
    [
        State('start-year-dropdown', 'value'),
        State('start-month-dropdown', 'value'),
        State('start-day-dropdown', 'value'),
        State('end-year-dropdown', 'value'),
        State('end-month-dropdown', 'value'),
        State('end-day-dropdown', 'value')
    ]
)
def update_prediction_plot(n_clicks, start_year, start_month, start_day, end_year, end_month, end_day):
    if n_clicks > 0:
        # Combine the selected start date and end date into strings
        start_date = f"{start_year}-{start_month}-{start_day}"
        end_date = f"{end_year}-{end_month}-{end_day}"
        # Convert the input strings to the appropriate format if needed
        # Call your sales prediction function with start_time and end_time
        predictions = make_predictions(start_date, end_date, model=MODEL, scaler=SCALER)
        # Convert values by dividing by 10
        predictions = [value / 10 for value in predictions]
        total_predictions = sum(predictions)
        # Create a Plotly bar plot
        fig = go.Figure()
        # Add a line trace
        fig.add_trace(go.Scatter(x=list(range(1, len(predictions) + 1)), y=predictions, mode='lines+markers', name='Line Plot'))
        # Add an annotation
        fig.add_annotation(
            text=f"<b>Total sale is {total_predictions:,.0f} Tomans</b>",
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            font=dict(size=12),
        )
        fig.update_layout(
            height=600,
            title="Sales Predictions of `Bamland` branch",
            title_font=dict(size=20),
            xaxis_title="days",
            xaxis_title_font=dict(size=16),
            yaxis_title="sales",
            yaxis_title_font=dict(size=16),
            # font=dict(family="Arial", size=18, color="black"),
            paper_bgcolor="white",
            plot_bgcolor="rgba(221,235,241,0.7)",
            xaxis=dict(tickfont=dict(size=15), gridcolor='white'),
            # yaxis=dict(gridcolor='white'),  # Color of y-axis grid lines
        )
        # Change the theme to "plotly_dark"
        # fig.update_layout(template="plotly_dark") # for changing the background of plot
        return fig

    return dash.no_update


@app.callback(
    Output('help-modal', 'style'),
    Output('modal-background', 'style'),
    Input('help-button', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_modal(help_clicks):
    if help_clicks is None:
        help_clicks = 0
    if help_clicks % 2 == 1:  # Odd number of clicks on Help or Close button
        return {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}


if __name__ == '__main__':
    app.run_server(debug=True)
