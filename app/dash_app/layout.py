from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout():

    # first_card = dbc.Card(
    #                 [
    #                 dbc.CardBody([
    #                     html.H5("Total Transaction", className="card-title"),
    #                     html.H2(id='total-transaction', className="card-text")
    #                             ])
    #                 ]
    #                     )

    layout =  dbc.Container([

        # Interval component for periodic updates
        dcc.Interval(
            id='interval-component',
            interval=60*1000,  # Update every 60 seconds
            n_intervals=0
        ),

        dbc.Row([
            dbc.Col(html.H1("Fraud Data Dashboard"), className="mb-2")
        ]),


        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.H3("Total Transactions"),
                                    html.P(id='total-transaction')
                                ], className="summary-box"),
                                
                                html.Div([
                                    html.H3("Fraud Cases"),
                                    html.P(id='total-fraud')
                                ], className="summary-box"),
                                
                                dbc.Card([
                                dbc.CardBody([
                                    html.H5("Fraud percentage %", className="card-title"),
                                    html.H2(id='fraud-percentage', className="card-text")
                                ])
                            ], className='summary-box') 
                            ], style={'display': 'flex', 'justify-content': 'space-around'})
                        ])
                    ]
                )
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                         html.Div([
                            dcc.Graph(id='geo-fig', className=" rounded-lg")
                        ], className="p-4 rounded-lg shadow-lg mb-8"),
                    ]
                )
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                         html.Div([
                            dcc.Graph(
                                id='line-chart',  className=" rounded-lg")
                            ],className="p-4 rounded-lg shadow-lg mb-8"),
                    ]
                )
            ]
        )
        # dbc.Row(
        # [
        #     dbc.Col(first_card, width=4),


        #     dbc.Col(
        #         dbc.Card(
        #             dbc.CardBody(
        #                 [
        #                     html.H5("Total fraud", className="card-title"),
        #                     html.H2(id='total-fraud', className="card-text")
        #                 ]
        #             )
        #         , className="mb-2")
        #     ,  width=4),


        #     dbc.Col(
        #         dbc.Card([
        #             dbc.CardBody([
        #                 html.H5("Fraud percentage %", className="card-title"),
        #                 html.H2(id='fraud-percentage', className="card-text")
        #             ])
        #         ], className='text-center m-4') , xs=12, sm=6, md=4,  width=4),
        # ])



        
    ], fluid=True)

    return layout