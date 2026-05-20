import os
import re

from dash import html, dcc
import dash_cytoscape as cyto

from config_rl import NUM_NODES, INITIAL_ENERGY_JOULE, MAX_COMM_DISTANCE, SINK_POSITION
from components.logic import initial_positions_raw, get_available_models

# ── Dynamic sizing ───────────────────────────────────────────────────────
if NUM_NODES <= 20:
    DYN_SIZE = '20px'; DYN_FONT = '10px'
elif NUM_NODES <= 50:
    DYN_SIZE = '12px'; DYN_FONT = '8px'
else:
    DYN_SIZE = '6px';  DYN_FONT = '4px'

# ── Cytoscape stylesheet ─────────────────────────────────────────────────
CYTO_STYLESHEET = [
    {'selector': 'node', 'style': {
        'content': 'data(label)', 'font-size': DYN_FONT,
        'text-valign': 'bottom', 'text-margin-y': '5px',
        'color': '#f5f6fa', 'text-outline-width': 2,
        'text-outline-color': '#2d3436',
        'width': DYN_SIZE, 'height': DYN_SIZE,
        'background-color': '#7f8c8d', 'z-index': 10
    }},
    {'selector': '.sink-node', 'style': {
        'shape': 'rectangle', 'width': '30px', 'height': '30px',
        'background-color': '#0000FF',
        'label': 'SINK', 'font-weight': 'bold', 'font-size': '10px'
    }},
    {'selector': '.active-node', 'style': {
        'background-color': '#2ecc71',
        'border-width': 2, 'border-color': '#27ae60'
    }},
    # Node yang dipilih user via klik (inspector selection)
    {'selector': '.selected-node', 'style': {
        'border-width': 3,
        'border-color': '#f1c40f',       # kuning terang — mudah dilihat
        'border-style': 'solid',
        'width':  '20px', 'height': '20px',   # diperbesar saat dipilih
        'z-index': 999,
        'font-size': '11px',
        'font-weight': 'bold',
        'color': '#f1c40f',
        'text-outline-color': '#2d3436',
        'text-outline-width': 2,
    }},
    {'selector': '[type = "ghost"]', 'style': {
        'width': 1, 'line-color': '#bdc3c7', 'line-style': 'solid',
        'opacity': 0.3, 'events': 'no', 'z-index': 1
    }},
    {'selector': '[type = "active"]', 'style': {
        'width': 1, 'line-color': 'data(color)',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': 'data(color)',
        'curve-style': 'bezier', 'opacity': 1, 'z-index': 999
    }},
    {'selector': '[energy<=0]', 'style': {
        'background-color': '#95a5a6', 'label': '',
        'width': '8px', 'height': '8px'
    }},
]


# ── Helper: build dropdown options at server startup ─────────────────────
def _build_model_options():
    """
    Dipanggil SEKALI saat server start. Mengembalikan (options, default_value)
    sehingga dropdown sudah terisi tanpa bergantung pada callback apapun.
    """
    available = get_available_models()   # pakai default dir dari logic.py
    if not available:
        return [], None

    options = []
    for m in available:
        fname = m['filename']
        name  = fname.replace('.pth', '')
        match = re.search(r'E(\d+)_T(\d+)_(\d{8})_(\d{6})', name)
        if match:
            e, t = match.group(1), match.group(2)
            ds   = match.group(3)
            label = (f"[E{e} T{t}]  "
                     f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}  "
                     f"({m['size_kb']} KB)")
        else:
            label = f"{fname}  ({m['size_kb']} KB)"

        if m['is_active']:
            label = "✅ " + label

        options.append({'label': label, 'value': fname})

    # Pilih model aktif sebagai default value; jika tidak ada, pilih index 0
    default = next(
        (o['value'] for o in options if o['label'].startswith('✅')),
        options[0]['value']
    )
    return options, default


# ── Layout ───────────────────────────────────────────────────────────────
def create_layout():
    # Build model options SEKALI di sini — tidak ada callback yang menimpa value
    _model_opts, _model_val = _build_model_options()

    return html.Div(className='main-container', children=[

        # ── Stores ──────────────────────────────────────────────────────
        dcc.Store(id='st-nodes',
                  data={str(i): {'energy': INITIAL_ENERGY_JOULE, 'buffer': 0}
                        for i in range(NUM_NODES)}),
        dcc.Store(id='st-pos',
                  data={str(k): [v[0], v[1]]
                        for k, v in initial_positions_raw.items() if k != 'sink'}),
        dcc.Store(id='st-sink-pos', data=list(SINK_POSITION)),
        dcc.Store(id='st-hist',
                  data={'step': [], 'sig': [], 'eng': [],
                        'sel_sig': [], 'sel_eng': [], 'last_sel': None}),
        dcc.Store(id='st-route', data={}),
        dcc.Store(id='st-logs',  data=[]),
        dcc.Store(id='st-active-model',
                  data={'name': 'Initializing...', 'loaded': False}),
        dcc.Store(id='st-sel-node', data=None),   # ← simpan sel_id yang dipilih user

        # ── Header ──────────────────────────────────────────────────────
        html.Div(className='header', children=[
            html.Div([
                html.H2("WSN Digital Twin",
                        style={'margin': 0, 'fontSize': '20px', 'color': '#ffffff'}),
                html.Span("Context-Aware Adaptive QoS Research",
                          style={'fontSize': '12px', 'color': '#98edf3'})
            ]),
            html.Div([
                html.Span("⏱️ Step: 0", id='live-timer',
                          style={'fontSize': '14px', 'fontWeight': 'bold',
                                 'color': '#f1c40f', 'marginRight': '15px'}),
                html.Span("🔄 Loading...", id='model-header-badge',
                          style={'fontSize': '12px', 'fontWeight': 'bold',
                                 'color': '#ffffff', 'padding': '5px 10px',
                                 'backgroundColor': '#636e72', 'borderRadius': '15px'})
            ])
        ]),

        # ── Content ──────────────────────────────────────────────────────
        html.Div(className='content-wrapper', children=[

            # Map
            html.Div(className='map-container', children=[
                cyto.Cytoscape(
                    id='wsn-map', layout={'name': 'preset'},
                    style={'width': '100%', 'height': '100%',
                           'backgroundColor': '#1e272e'},
                    stylesheet=CYTO_STYLESHEET, elements=[],
                    minZoom=0.5, maxZoom=2.0, responsive=True
                ),
                html.Div(className='map-legend', children=[
                    html.Div([html.Span("———", style={'color': '#2ecc71', 'fontWeight': 'bold'}),
                              " Strong Route"]),
                    html.Div([html.Span("———", style={'color': '#e74c3c', 'fontWeight': 'bold'}),
                              " Weak Route"]),
                    html.Div([html.Span("------", style={'color': '#bdc3c7', 'fontWeight': 'bold'}),
                              " Physical Range (Ghost)"]),
                ])
            ]),

            # Sidebar
            html.Div(className='control-panel', children=[

                # ════════ CARD 1 — MODEL SELECTION ════════
                html.Div(className='ui-card', children=[
                    html.H4("🤖 Model Selection",
                            style={'marginTop': 0, 'fontSize': '13px', 'color': '#00d2d3'}),

                    html.Label("Available Trained Models",
                               style={'fontSize': '11px', 'fontWeight': 'bold',
                                      'color': '#f5f6fa', 'marginBottom': '6px',
                                      'display': 'block'}),

                    # ── Dropdown — options & value diisi dari server ──
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=_model_opts,   # ← sudah terisi saat startup
                        value=_model_val,      # ← default = model aktif
                        clearable=False,
                        searchable=False,
                        style={
                            'fontSize': '11px',
                            'backgroundColor': '#2d3436',
                            'color': '#2d3436',
                            'borderColor': '#485460',
                            'marginBottom': '8px',
                        }
                    ),

                    # ── Metadata panel ───────────────────────────────
                    html.Div(id='model-meta-panel', style={'marginBottom': '8px'},
                             children=[
                                 html.P("Pilih model untuk melihat detail.",
                                        style={'color': '#636e72', 'fontSize': '10px',
                                               'fontStyle': 'italic', 'margin': 0})
                             ]),

                    # ── Buttons ──────────────────────────────────────
                    html.Div(style={'display': 'flex', 'gap': '8px', 'alignItems': 'center'},
                             children=[
                                 html.Button(
                                     "⚡ Load Model", id='btn-load-model',
                                     n_clicks=0,
                                     style={
                                         'flex': '1', 'backgroundColor': '#6c5ce7',
                                         'color': 'white', 'border': 'none',
                                         'borderRadius': '4px', 'padding': '7px 10px',
                                         'fontWeight': 'bold', 'fontSize': '11px',
                                         'cursor': 'pointer', 'margin': 0
                                     }
                                 ),
                                 html.Button(
                                     "🔄", id='btn-refresh-models',
                                     n_clicks=0,
                                     title='Refresh daftar model',
                                     style={
                                         'backgroundColor': '#485460', 'color': 'white',
                                         'border': 'none', 'borderRadius': '4px',
                                         'padding': '7px 12px', 'cursor': 'pointer',
                                         'fontSize': '13px', 'margin': 0
                                     }
                                 ),
                             ]),

                    # ── Status badge setelah load ─────────────────────
                    html.Div(id='model-load-status', style={'marginTop': '8px'})
                ]),

                # ════════ CARD 2 — ENVIRONMENT SETTINGS ════════
                html.Div(className='ui-card', children=[
                    html.H4("Environment Settings",
                            style={'marginTop': 0, 'fontSize': '13px', 'color': '#95a5a6'}),

                    html.Label("Visual Radio Range",
                               style={'fontSize': '11px', 'fontWeight': 'bold'}),
                    dcc.Slider(id='range-slider', min=10, max=80, step=5,
                               value=MAX_COMM_DISTANCE,
                               marks={10: '10m', 40: '40m', 80: '80m'}),
                    html.Div("Mengatur jangkauan transmisi fisik node",
                             style={'fontSize': '10px', 'color': '#ccc',
                                    'fontStyle': 'italic', 'marginBottom': '15px'}),

                    html.Label("Environmental Noise (EDF)",
                               style={'fontSize': '11px', 'fontWeight': 'bold',
                                      'marginTop': '10px'}),
                    dcc.Slider(id='edf-slider', min=0, max=0.9, step=0.1, value=0,
                               marks={
                                   0:   {'label': 'Clear', 'style': {'color': '#27ae60'}},
                                   0.5: {'label': 'Rain',  'style': {'color': '#f39c12'}},
                                   0.9: {'label': 'Storm', 'style': {'color': '#e74c3c'}},
                               }),
                    html.Div("Simulasi hambatan eksternal (cuaca/interferensi)",
                             style={'fontSize': '10px', 'color': '#ccc',
                                    'fontStyle': 'italic', 'marginBottom': '20px'}),

                    html.Label("Routing Algorithm",
                               style={'fontSize': '11px', 'fontWeight': 'bold',
                                      'marginTop': '10px'}),
                    dcc.RadioItems(
                        id='mode-selector',
                        options=[
                            {'label': 'Adaptive Switching QoS',    'value': 'AI'},
                            {'label': 'Greedy Algorithm (Baseline)', 'value': 'GREEDY'},
                        ],
                        value='AI', inline=True,
                        labelStyle={'fontSize': '11px', 'marginRight': '10px',
                                    'color': '#ffffff'},
                        style={'marginBottom': '15px'}
                    ),

                    html.Button("RESET SIMULATION", id='btn-reset',
                                className='ui-btn-reset'),

                    dcc.Checklist(
                        id='auto-drain-toggle',
                        options=[{'label': ' Enable Real-time Auto Drain (Dry Run)',
                                  'value': 'enable'}],
                        value=[],
                        style={'marginTop': '15px', 'color': '#f1c40f',
                               'fontSize': '12px', 'fontWeight': 'bold'}
                    ),
                ]),

                # ════════ CARD 3 — NODE INSPECTOR ════════
                html.Div(className='ui-card', children=[
                    html.H4("Node Inspector",
                            style={'marginTop': 0, 'fontSize': '13px', 'color': '#95a5a6'}),
                    html.Div(id='inspector-content', children=[
                        html.P("Klik node untuk detail...",
                               style={'color': '#bdc3c7', 'fontStyle': 'italic',
                                      'textAlign': 'center'})
                    ]),
                    html.Div(id='inspector-controls', style={'display': 'none'}, children=[
                        html.Hr(style={'borderColor': '#f0f0f0'}),
                        html.Label("Battery Level:",
                                   style={'fontSize': '11px', 'fontWeight': 'bold'}),
                        dcc.Slider(0, INITIAL_ENERGY_JOULE, 100,
                                   id='energy-slider', value=INITIAL_ENERGY_JOULE),
                        html.Button("🔥 DRAIN BATTERY", id='btn-drain',
                                    className='ui-btn-drain')
                    ]),
                ]),

                # ════════ CARD 4 — SWITCHING LOG ════════
                html.Div(className='ui-card',
                         style={'height': '150px', 'display': 'flex',
                                'flexDirection': 'column'},
                         children=[
                    html.H4("Switching Log",
                            style={'marginTop': 0, 'fontSize': '13px', 'color': '#95a5a6'}),
                    html.Div(id='log-container', style={
                        'flex': '1', 'overflowY': 'auto', 'fontSize': '11px',
                        'fontFamily': 'monospace', 'border': '1px solid #f0f0f0',
                        'padding': '5px', 'backgroundColor': '#fafafa', 'height': '100%'
                    }, children=[html.Div("Waiting for events...",
                                          style={'color': '#ccc'})])
                ]),

                # ════════ CARD 5 — NETWORK HEALTH ════════
                html.Div(className='ui-card',
                         style={'flex': '1', 'minHeight': '300px',
                                'display': 'flex', 'flexDirection': 'column'},
                         children=[
                    html.H4("Network Health (Real-time)",
                            style={'marginTop': 0, 'fontSize': '13px', 'color': '#95a5a6'}),
                    dcc.Graph(id='live-graph',
                              config={'displayModeBar': False, 'responsive': True},
                              style={'height': '100%', 'width': '100%',
                                     'minHeight': '300px'})
                ]),
            ])
        ])
    ])