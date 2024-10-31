import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px


def register_callbacks(dash_app):
    
    df = pd.read_csv('../data/merged_data.csv')
    country_fraud_counts = df[df['class'] == 1].groupby('country').size().reset_index(name='fraud_cases')
    df.purchase_time = pd.to_datetime(df.purchase_time)
    df_sorted = df.sort_values(by="purchase_time")
    # Update summary statistics
    @dash_app.callback(
        [
            Output('total-transaction', 'children'),
            Output('total-fraud', 'children'),
            Output('fraud-percentage', 'children'),
        ],
        [Input('interval-component', 'n_intervals')] 
    )

    def update_summary(n_intervals):
        # Load and preprocess data
        total_transaction = df['user_id'].count()
        total_fraud = df[df['class']==1].shape[0]
        fraud_percent = (total_fraud / total_transaction)*100
        return (
            f"{total_transaction:}",
            f"{total_fraud:}",
            f"{fraud_percent:.2f}"
        )
    
    # Update geo statistics
    @dash_app.callback(
        [
            Output('geo-fig', 'figure')
        ],
        [Input('geo-fig', 'id')] 
    )

    def geofig(id):
        # Aggregate data by country and count fraud cases
        # Create a choropleth map
        figure = px.choropleth(
            country_fraud_counts,
            locations='country',
            locationmode='country names',
            color='fraud_cases',
            hover_name='country',
            color_continuous_scale=px.colors.sequential.Plasma,
            title='Fraud Cases by Country')
        # fig.update_layout(dragmode=False, paper_bgcolor="#e5ecf6", height=600, margin={"l":0, "r":0})
        return figure
        
     # Update geo statistics
    @dash_app.callback(
        Output("line-chart", "figure"),
        Input("line-chart", "id")  # Trigger the callback on app load (or add specific inputs if needed)
     )
    def update_line_chart(_):
        fig = px.line(df_sorted, x="purchase_time", y="class", title="Transactions Over Time")
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        return fig
        