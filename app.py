import dash
from components.layout import create_layout
from components.callbacks import register_callbacks

# Init App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Set Layout
app.layout = create_layout()

# Register Callbacks
register_callbacks(app)

# Run
if __name__ == '__main__':
    app.run(debug=True)