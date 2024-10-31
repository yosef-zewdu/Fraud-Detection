from dash import Dash
from .layout import create_layout
from .callbacks import register_callbacks

def create_dash_app(server):
    # Initialize the Dash app with Flask server
    dash_app = Dash(__name__, server=server, url_base_pathname='/dashboard/')

    # Set up the layout and callbacks
    dash_app.layout = create_layout()
    register_callbacks(dash_app)

    # Run the app
    if __name__ == '__main__':
        dash_app.run_server(debug=True)


    return dash_app
