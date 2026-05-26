import datetime
import math
import csv
import os
import re

from dash import Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

from components.logic import (
    get_logic_state, SCALE,
    load_model_from_path, get_active_model_info, get_available_models
)
from config_rl import NUM_NODES, INITIAL_ENERGY_JOULE, MAX_COMM_DISTANCE, SINK_POSITION

# ── Directories ──────────────────────────────────────────────────────────
if not os.path.exists('experiment_data'):
    os.makedirs('experiment_data')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Global simulation state ──────────────────────────────────────────────
LOG_CSV_PATH   = None
LOG_TXT_PATH   = None
LAST_EDF_VAL   = None
LAST_RANGE_VAL = None
GLOBAL_STEP    = 0
MODEL_JUST_SWITCHED = False 

def write_to_log_file(txt_path, message):
    if txt_path and os.path.exists(txt_path):
        try:
            with open(txt_path, 'a', encoding='utf-8') as f:
                f.write(message + "\n")
        except Exception:
            pass

# ── Node drag tracking (untuk deteksi pergeseran node oleh user) ─────────
# key: node_id (str), value: [x_meter, y_meter] posisi terakhir yang DICATAT
_LAST_LOGGED_POSITIONS = {}
# key: node_id, value: step terakhir posisi berubah (untuk delay debounce)
_DRAG_CHANGED_AT_STEP  = {}
DRAG_LOG_DELAY_STEPS   = 3   # catat ke TXT setelah posisi stabil N step
DRAG_MIN_DISTANCE_M    = 2.0 # minimum pergeseran (meter) agar dianggap drag

# ── Helper ───────────────────────────────────────────────────────────────
def parse_model_filename(filename: str) -> dict:
    name  = filename.replace('.pth', '')
    match = re.search(r'E(\d+)_T(\d+)_(\d{8})_(\d{6})', name)
    if match:
        ds = match.group(3)
        ts = match.group(4)
        return {
            'episodes':  int(match.group(1)),
            'timesteps': int(match.group(2)),
            'date': f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}",
            'time': f"{ts[:2]}:{ts[2:4]}:{ts[4:6]}",
            'valid': True,
        }
    return {'valid': False}


def register_callbacks(app):

    # ════════════════════════════════════════════════════════════════════
    # CB-1  Refresh dropdown options (TIDAK menyentuh value)
    #       Dipicu HANYA oleh tombol 🔄 atau initial load
    # ════════════════════════════════════════════════════════════════════
    @app.callback(
        Output('model-dropdown', 'options'),
        Input('btn-refresh-models', 'n_clicks'),
        prevent_initial_call=False,   # ← terpicu sekali saat startup untuk sinkronisasi
    )
    def refresh_model_options(n_clicks):
        available = get_available_models(MODELS_DIR)
        if not available:
            return []

        options = []
        for m in available:
            meta = parse_model_filename(m['filename'])
            if meta['valid']:
                label = (f"[E{meta['episodes']} T{meta['timesteps']}]  "
                         f"{meta['date']}  ({m['size_kb']} KB)")
            else:
                label = f"{m['filename']}  ({m['size_kb']} KB)"
            if m['is_active']:
                label = "✅ " + label
            options.append({'label': label, 'value': m['filename']})

        return options
    # ── TIDAK ada Output untuk 'model-dropdown.value' di sini ────────────


    # ════════════════════════════════════════════════════════════════════
    # CB-2  Tampilkan metadata model yang dipilih
    # ════════════════════════════════════════════════════════════════════
    @app.callback(
        Output('model-meta-panel', 'children'),
        Input('model-dropdown', 'value'),
        prevent_initial_call=False,
    )
    def show_model_meta(selected_filename):
        if not selected_filename:
            return html.P("Pilih model dari dropdown.",
                          style={'color': '#636e72', 'fontSize': '10px',
                                 'fontStyle': 'italic', 'margin': 0})

        path    = os.path.join(MODELS_DIR, selected_filename)
        size_kb = round(os.path.getsize(path) / 1024, 1) if os.path.isfile(path) else '?'
        meta    = parse_model_filename(selected_filename)
        is_active = (get_active_model_info().get('path') == path)

        rows = [
            ("💾 File",    selected_filename, '#f5f6fa'),
            ("📦 Size",    f"{size_kb} KB",  '#f5f6fa'),
        ]
        if meta['valid']:
            rows += [
                ("📊 Episodes",  str(meta['episodes']),  '#00d2d3'),
                ("⏱ Timesteps", str(meta['timesteps']), '#00d2d3'),
                ("📅 Trained",   f"{meta['date']} {meta['time']}", '#95a5a6'),
            ]

        else:
    # ← Tambahkan ini agar model lama tetap informatif
            rows += [
                ("📊 Episodes",  "Unknown (nama tidak terformat)", '#f39c12'),
                ("⏱ Timesteps", "Unknown",                        '#f39c12'),
                ("📅 Trained",   "Unknown",                        '#f39c12'),
                ("ℹ️ Info",
                "Rename ke format: E{ep}_T{ts}_YYYYMMDD_HHMMSS",  '#636e72'),
            ]

        grid = [
            html.Div(style={
                'display': 'flex', 'justifyContent': 'space-between',
                'borderBottom': '1px solid #485460', 'padding': '3px 0',
            }, children=[
                html.Span(lbl, style={'fontSize': '10px', 'color': '#95a5a6'}),
                html.Span(val, style={'fontSize': '10px', 'color': col,
                                      'fontWeight': 'bold',
                                      'overflow': 'hidden',
                                      'textOverflow': 'ellipsis',
                                      'whiteSpace': 'nowrap',
                                      'maxWidth': '170px'}),
            ])
            for lbl, val, col in rows
        ]

        badge = html.Span("● ACTIVE MODEL", style={
            'fontSize': '10px', 'color': '#27ae60', 'fontWeight': 'bold',
            'backgroundColor': '#eafaf1', 'borderRadius': '8px',
            'padding': '2px 8px', 'marginTop': '6px', 'display': 'inline-block',
        }) if is_active else None

        return html.Div([html.Div(grid)] + ([badge] if badge else []))


    # ════════════════════════════════════════════════════════════════════
    # CB-3  Load model yang dipilih ke memori PyTorch
    # ════════════════════════════════════════════════════════════════════
    @app.callback(
        [Output('model-load-status', 'children'),
         Output('st-active-model',   'data'),
         Output('model-dropdown',    'options', allow_duplicate=True),
         Output('model-meta-panel',  'children', allow_duplicate=True)],
        Input('btn-load-model', 'n_clicks'),
        [State('model-dropdown', 'value'),
         State('model-dropdown', 'options')],
        prevent_initial_call=True,
    )
    def load_selected_model(n_clicks, selected_filename, current_options):
        print(f"[CB-3] load_selected_model | n_clicks={n_clicks} | file={selected_filename}")

        if not n_clicks or not selected_filename:
            return no_update, no_update, no_update, no_update

        path    = os.path.join(MODELS_DIR, selected_filename)
        success, msg = load_model_from_path(path)

        if success:
            status_ui = html.Div([
                html.Span("✅ "),
                html.Span(f"Loaded: {msg}",
                          style={'fontSize': '11px', 'color': '#27ae60',
                                 'fontWeight': 'bold'}),
            ], style={'backgroundColor': '#eafaf1', 'borderRadius': '6px',
                      'padding': '6px 10px', 'border': '1px solid #27ae60'})

            store_data = {'name': selected_filename, 'loaded': True, 'path': path}

            # Perbarui label ✅ di options
            updated_opts = []
            for opt in (current_options or []):
                lbl = opt['label'].replace('✅ ', '')
                if opt['value'] == selected_filename:
                    lbl = '✅ ' + lbl
                updated_opts.append({'label': lbl, 'value': opt['value']})

            # Perbarui panel meta (tambahkan badge ACTIVE)
            size_kb   = round(os.path.getsize(path) / 1024, 1)
            meta      = parse_model_filename(selected_filename)
            rows = [("💾 File",  selected_filename, '#f5f6fa'),
                    ("📦 Size",  f"{size_kb} KB",   '#f5f6fa')]
            if meta['valid']:
                rows += [("📊 Episodes",  str(meta['episodes']),  '#00d2d3'),
                         ("⏱ Timesteps", str(meta['timesteps']), '#00d2d3'),
                         ("📅 Trained",  f"{meta['date']} {meta['time']}", '#95a5a6')]

            grid = [
                html.Div(style={
                    'display': 'flex', 'justifyContent': 'space-between',
                    'borderBottom': '1px solid #485460', 'padding': '3px 0',
                }, children=[
                    html.Span(lbl, style={'fontSize': '10px', 'color': '#95a5a6'}),
                    html.Span(val, style={'fontSize': '10px', 'color': col,
                                         'fontWeight': 'bold'}),
                ])
                for lbl, val, col in rows
            ]
            active_badge = html.Span("● ACTIVE MODEL", style={
                'fontSize': '10px', 'color': '#27ae60', 'fontWeight': 'bold',
                'backgroundColor': '#eafaf1', 'borderRadius': '8px',
                'padding': '2px 8px', 'marginTop': '6px', 'display': 'inline-block',
            })
            updated_meta = html.Div([html.Div(grid), active_badge])

            global MODEL_JUST_SWITCHED
            MODEL_JUST_SWITCHED = True

        else:
            status_ui = html.Div([
                html.Span("❌ "),
                html.Span(f"Error: {msg}",
                          style={'fontSize': '11px', 'color': '#e74c3c',
                                 'fontWeight': 'bold'}),
            ], style={'backgroundColor': '#ffeef0', 'borderRadius': '6px',
                      'padding': '6px 10px', 'border': '1px solid #e74c3c'})
            store_data   = {'name': 'Load Failed', 'loaded': False, 'path': None}
            updated_opts = current_options or []
            updated_meta = no_update

        return status_ui, store_data, updated_opts, updated_meta


    # ════════════════════════════════════════════════════════════════════
    # CB-4  Perbarui badge model di header
    # ════════════════════════════════════════════════════════════════════
    @app.callback(
        [Output('model-header-badge', 'children'),
         Output('model-header-badge', 'style')],
        Input('st-active-model', 'data'),
        prevent_initial_call=False,
    )
    def update_header_badge(model_data):
        BASE_STYLE = {
            'fontSize': '12px', 'fontWeight': 'bold', 'color': '#ffffff',
            'padding': '5px 10px', 'borderRadius': '15px',
        }
        info = get_active_model_info()
        if info['loaded']:
            meta = parse_model_filename(info['name'])
            if meta['valid']:
                display = f"E{meta['episodes']} T{meta['timesteps']} · {meta['date']}"
            else:
                n = info['name']
                display = n[:28] + ('…' if len(n) > 28 else '')
            return f"🟢 {display}", {**BASE_STYLE, 'backgroundColor': '#27ae60'}
        return "🔴 No AI Model", {**BASE_STYLE, 'backgroundColor': '#c0392b'}


    # ════════════════════════════════════════════════════════════════════
    # CB-5  Main simulation callback (tidak berubah)
    # ════════════════════════════════════════════════════════════════════
    @app.callback(
        [Output('st-nodes',           'data'),
         Output('st-pos',             'data'),
         Output('st-hist',            'data'),
         Output('st-route',           'data'),
         Output('st-logs',            'data'),
         Output('wsn-map',            'elements'),
         Output('live-graph',         'figure'),
         Output('inspector-content',  'children'),
         Output('inspector-controls', 'style'),
         Output('energy-slider',      'value'),
         Output('log-container',      'children'),
         Output('auto-drain-toggle',  'value'),
         Output('live-timer',         'children'),
         Output('st-sel-node',        'data')],      # ← simpan sel_id terpilih
        [Input('wsn-map',          'tapNodeData'),
         Input('wsn-map',          'elements'),
         Input('energy-slider',    'value'),
         Input('btn-drain',        'n_clicks'),
         Input('btn-reset',        'n_clicks'),
         Input('range-slider',     'value'),
         Input('edf-slider',       'value'),
         Input('mode-selector',    'value'),
         Input('auto-drain-toggle','value'),
         Input('live-graph',       'clickData')],
        [State('st-nodes',    'data'),
         State('st-pos',      'data'),
         State('st-hist',     'data'),
         State('st-route',    'data'),
         State('st-logs',     'data'),
         State('st-sink-pos', 'data'),
         State('st-sel-node', 'data')],            # ← baca sel_id sebelumnya
    )
    def update_system(tap_node, cyto_els, slider_val, n_drain, n_reset,
                      range_val, edf_val, mode_val, auto_drain_val, graph_click,
                      node_data, pos_data, history, old_route, log_data,
                      sink_pos_data, prev_sel_id):

        global LOG_CSV_PATH, LOG_TXT_PATH, LAST_EDF_VAL, LAST_RANGE_VAL, GLOBAL_STEP, MODEL_JUST_SWITCHED, _LAST_LOGGED_POSITIONS, _DRAG_CHANGED_AT_STEP

        ctx          = callback_context
        full_trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else 'init'
        trigger      = full_trigger.split('.')[0]
        timestamp_now = datetime.datetime.now().strftime("%H:%M:%S")

        if node_data is None and trigger not in ('init', 'btn-reset', 'mode-selector'):
            raise PreventUpdate
        
        # ── Tentukan sel_id berdasarkan SUMBER TRIGGER yang tepat ──────────
        # Hanya update sel_id saat user benar-benar memilih node baru
        # Jangan overwrite saat trigger dari wsn-map.elements (re-render biasa)
        if full_trigger == 'wsn-map.tapNodeData':
            # Klik node di Cytoscape map
            if tap_node and 'id' in tap_node and tap_node['id'] != 'sink':
                sel_id = str(tap_node['id'])
            else:
                sel_id = prev_sel_id   # tap kosong → pertahankan pilihan lama

        elif full_trigger == 'live-graph.clickData' and graph_click:
            # Klik dot di scatter plot
            try:
                pt = graph_click['points'][0]
                nid_from_graph = str(pt.get('customdata', ''))
                if not nid_from_graph.isdigit():
                    point_text = pt.get('text', '')
                    nid_from_graph = (point_text.split('<br>')[0]
                                      .replace('Node ', '').strip())
                sel_id = nid_from_graph if nid_from_graph.isdigit() else prev_sel_id
            except (KeyError, IndexError, AttributeError):
                sel_id = prev_sel_id

        elif trigger in ('btn-reset', 'mode-selector'):
            sel_id = None   # reset selection saat simulasi direset

        else:
            # Semua trigger lain (slider, elements re-render, dsb)
            # → pertahankan pilihan node sebelumnya
            sel_id = prev_sel_id
        # ───────────────────────────────────────────────────────────────────

        if node_data is None:
            node_data = {str(i): {'energy': INITIAL_ENERGY_JOULE, 'buffer': 0}
                         for i in range(NUM_NODES)}
        if old_route is None: old_route = {}
        if log_data  is None: log_data  = []

        # user_events  : hanya aksi user → masuk TXT + UI log
        # system_events: otomatis jaringan → hanya UI log (difilter)
        user_events   = []
        current_step_events = []   # gabungan sementara untuk UI log

        # --- Deteksi aksi user ---
        if trigger in ('btn-reset', 'mode-selector'):
            node_data = {str(i): {'energy': INITIAL_ENERGY_JOULE, 'buffer': 0}
                         for i in range(NUM_NODES)}
            history   = {'step': [], 'sig': [], 'eng': [],
                         'sel_sig': [], 'sel_eng': [], 'last_sel': None}
            log_data  = []
            LOG_CSV_PATH   = None
            LOG_TXT_PATH   = None
            LAST_EDF_VAL   = None
            LAST_RANGE_VAL = None
            GLOBAL_STEP    = 0
            auto_drain_val = []
            _LAST_LOGGED_POSITIONS = {}
            _DRAG_CHANGED_AT_STEP  = {}
            if trigger == 'mode-selector':
                ev = f"⚙️ MODE CHANGED TO {mode_val} (Auto-Reset)"
            else:
                ev = f"🔄 SIMULATION RESET (Mode: {mode_val})"
            user_events.append(ev)
            current_step_events.append(ev)
        else:
            if trigger not in ('init', 'auto-drain-toggle'):
                GLOBAL_STEP += 1

        # Deteksi auto-drain ON/OFF
        if trigger == 'auto-drain-toggle':
            state_str = "ON ✅" if (auto_drain_val and 'enable' in auto_drain_val) else "OFF ⛔"
            ev = f"🔋 Auto Drain: {state_str}"
            user_events.append(ev)
            current_step_events.append(ev)

        if trigger == 'btn-drain' and sel_id and sel_id in node_data:
            old_e = node_data[sel_id]['energy']
            if old_e <= 0:
                # Node sudah mati — abaikan, jangan catat event
                pass
            else:
                new_e = max(0, old_e - (INITIAL_ENERGY_JOULE * 0.2))
                node_data[sel_id]['energy'] = new_e
                ev = f"🔻 Node {sel_id}: Manual Drain ({old_e:.0f}J -> {new_e:.0f}J)"
                user_events.append(ev)
                current_step_events.append(ev)

        elif trigger == 'energy-slider' and sel_id and sel_id in node_data:
            if abs(node_data[sel_id]['energy'] - slider_val) > 1.0:
                node_data[sel_id]['energy'] = slider_val

        if edf_val is not None:
            if LAST_EDF_VAL is None: 
                LAST_EDF_VAL = edf_val
            elif LAST_EDF_VAL != edf_val:
                # HANYA rekam ke log jika pemicunya murni dari pergerakan slider
                if trigger == 'edf-slider':
                    ev = f"🌩️ EDF/Noise Changed: {LAST_EDF_VAL} -> {edf_val}"
                    user_events.append(ev)
                    current_step_events.append(ev)
                    LAST_EDF_VAL = edf_val
                # Jika bukan dari slider (stale requests), diamkan saja.

        # --- PERBAIKAN BUG RANGE ---
        if range_val is not None:
            if LAST_RANGE_VAL is None: 
                LAST_RANGE_VAL = range_val
            elif LAST_RANGE_VAL != range_val:
                if trigger == 'range-slider':
                    ev = f"📡 Radio Range Changed: {LAST_RANGE_VAL} -> {range_val}m"
                    user_events.append(ev)
                    current_step_events.append(ev)
                    LAST_RANGE_VAL = range_val

        # --- Sync posisi ---
        current_positions = pos_data.copy() if pos_data else {}
        current_positions['sink'] = sink_pos_data if sink_pos_data else list(SINK_POSITION)

        if cyto_els:
            for el in cyto_els:
                if 'position' in el and 'data' in el and 'id' in el['data']:
                    node_id = str(el['data']['id'])
                    current_positions[node_id] = [
                        el['position']['x'] / SCALE,
                        el['position']['y'] / SCALE,
                    ]
                    if node_id != 'sink':
                        if pos_data is None: pos_data = {}
                        pos_data[node_id] = current_positions[node_id]

        # --- Deteksi drag node oleh user (dengan delay debounce) ---
        if trigger == 'wsn-map' and cyto_els:
            import math as _math
            for el in cyto_els:
                if 'position' not in el or 'data' not in el: continue
                nid = str(el['data'].get('id', ''))
                if nid == 'sink' or not nid.isdigit(): continue

                cur_x = el['position']['x'] / SCALE
                cur_y = el['position']['y'] / SCALE

                prev = _LAST_LOGGED_POSITIONS.get(nid)
                if prev is not None:
                    dx = cur_x - prev[0]
                    dy = cur_y - prev[1]
                    dist_moved = _math.sqrt(dx*dx + dy*dy)

                    if dist_moved >= DRAG_MIN_DISTANCE_M:
                        # Tandai bahwa node ini bergerak di step ini
                        _DRAG_CHANGED_AT_STEP[nid] = GLOBAL_STEP
                else:
                    # Inisialisasi posisi awal
                    _LAST_LOGGED_POSITIONS[nid] = [cur_x, cur_y]

            # Cek node mana yang sudah stabil selama DRAG_LOG_DELAY_STEPS
            for nid, changed_at in list(_DRAG_CHANGED_AT_STEP.items()):
                if GLOBAL_STEP - changed_at >= DRAG_LOG_DELAY_STEPS:
                    prev   = _LAST_LOGGED_POSITIONS.get(nid, [0, 0])
                    cur_pos = current_positions.get(nid, [0, 0])
                    ev = (f"📍 Node {nid} dipindahkan: "
                          f"({prev[0]:.1f}, {prev[1]:.1f}) → "
                          f"({cur_pos[0]:.1f}, {cur_pos[1]:.1f}) m")
                    user_events.append(ev)
                    current_step_events.append(ev)
                    _LAST_LOGGED_POSITIONS[nid] = [cur_pos[0], cur_pos[1]]
                    del _DRAG_CHANGED_AT_STEP[nid]

        # --- Logika AI / Greedy ---
        elements, avg_e, avg_s, new_route, new_system_events, networks_metrics = get_logic_state(
            current_positions, node_data, old_route, range_val, edf_val, mode_val,
            sel_id=sel_id   # ← teruskan sel_id untuk highlight di map
        )

        # --- Auto drain ---
        if (auto_drain_val and 'enable' in auto_drain_val
                and trigger not in ('btn-reset', 'mode-selector', 'init')):
            safe_edf = edf_val if edf_val is not None else 0.0
            for nid, r_info in new_route.items():
                if nid in node_data and node_data[nid]['energy'] > 0:
                    drain = 0.5
                    if r_info.get('parent') is not None and r_info.get('alive'):
                        drain += r_info.get('dist', 0) * 0.05
                        drain += safe_edf * 3.0
                    node_data[nid]['energy'] = max(0, node_data[nid]['energy'] - drain)

        # system_events dari jaringan (route switch, died, dll) → UI log saja
        current_step_events.extend(new_system_events)
        # user_events TIDAK diperluas dengan new_system_events

        # --- Log UI — hanya event penting, hilangkan route switch & connected ──
        UI_SKIP_KEYWORDS = ['Route Switch', 'connected to']
        for ev in current_step_events:
            if any(kw in ev for kw in UI_SKIP_KEYWORDS):
                continue   # ← abaikan dari UI log
            icon = "📝"
            if "Auto Drain"    in ev: icon = "🔋"   # ← harus sebelum "Drain"
            elif "Drain"       in ev: icon = "🔻"
            elif "EDF"         in ev: icon = "🌩️"
            elif "Range"       in ev: icon = "📡"
            elif "DIED"        in ev: icon = "💀"
            elif "lost"        in ev: icon = "🔌"
            elif "RESET"       in ev or "CHANGED" in ev: icon = "🔄"
            elif "dipindahkan" in ev: icon = "📍"
            elif "switched"    in ev: icon = "🔁"
            txt = f"[{timestamp_now}] {ev}" if icon in ev else f"[{timestamp_now}] {icon} {ev}"
            
            write_to_log_file(LOG_TXT_PATH, txt)  # Simpan ke file TXT

            if any(kw in ev for kw in UI_SKIP_KEYWORDS):
                continue   # ← abaikan dari UI log

            log_data.append(txt)
        log_data = log_data[-100:]

        # --- History ---
        if not isinstance(history, dict) or 'step' not in history:
            history = {'step': [], 'sig': [], 'eng': [],
                       'sel_sig': [], 'sel_eng': [], 'last_sel': None}

        for k in ('sel_sig', 'sel_eng', 'last_sel'):
            if k not in history:
                history[k] = [] if k != 'last_sel' else None

        history['step'].append(GLOBAL_STEP)
        history['eng'].append(avg_e if avg_e is not None else 0)
        history['sig'].append(avg_s if avg_s is not None else 0)

        if sel_id != history['last_sel']:
            history['sel_sig']  = [None] * len(history['step'])
            history['sel_eng']  = [None] * len(history['step'])
            history['last_sel'] = sel_id

        cur_sel_eng = cur_sel_sig = None
        if sel_id and sel_id in node_data:
            cur_sel_eng = node_data[sel_id]['energy']
            if sel_id in new_route:
                dist_p   = new_route[sel_id].get('dist', 0)
                max_lim  = range_val * SCALE if range_val else MAX_COMM_DISTANCE * SCALE
                safe_edf = edf_val if edf_val is not None else 0.0
                cur_sel_sig = max(0, 1 - (dist_p / (max_lim * 1.2))) * (1.0 - safe_edf)
            else:
                cur_sel_sig = 0

        history['sel_eng'].append(cur_sel_eng)
        history['sel_sig'].append(cur_sel_sig)

        MAX_HIST = 50
        if len(history['step']) > MAX_HIST:
            for k in history:
                if isinstance(history[k], list):
                    history[k] = history[k][-MAX_HIST:]

        # --- CSV / TXT logging ---
        psr_value = networks_metrics['psr']
        avg_retries = networks_metrics['avg_retries']
        dominant_dr = networks_metrics['dominant_dr']
        model_name  = get_active_model_info().get('name', 'N/A')
        dead_nodes  = sum(1 for n in node_data.values() if n['energy'] <= 0)

        if MODEL_JUST_SWITCHED and LOG_TXT_PATH is not None:
            try:
                with open(LOG_TXT_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*50}\n")
                    f.write(f"=== MODEL SWITCHED TO: {model_name} ===\n")
                    f.write(f"=== Log dilanjutkan di file baru      ===\n")
                    f.write(f"{'='*50}\n")
            except Exception as e:
                print(f"[CB-5] Gagal tulis penutup: {e}")
            LOG_CSV_PATH = None
            LOG_TXT_PATH = None
            MODEL_JUST_SWITCHED = False
            # ← model_name sudah terdefinisi di atas, aman dipakai di sini
            ev = f"🔁 Model switched → {model_name} | Log baru dibuat"
            user_events.append(ev)
            current_step_events.append(ev)

        if LOG_CSV_PATH is None:
            ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            meta_log  = parse_model_filename(model_name)
            if meta_log['valid']:
                # Format: Experiment_AI_E1000_T200_20260427_100911
                base_name = (f"{mode_val}_"
                             f"E{meta_log['episodes']}_"
                             f"T{meta_log['timesteps']}_"
                             f"{ts}")
            else:
                # Model tanpa format standar (misal wsn_dqn_model.pth)
                # Format: Experiment_AI_Eunk_Trunk_20260427_100911
                base_name = f"{mode_val}_Eunk_Trunk_{ts}"
            LOG_CSV_PATH = f"experiment_data/Experiment_{base_name}.csv"
            LOG_TXT_PATH = f"experiment_data/LogEvents_{base_name}.txt"
            print(f"[CB-5] File log baru: {LOG_CSV_PATH}")
            with open(LOG_CSV_PATH, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'Step','Avg_Energy','Alive_Nodes','Dead_Nodes',
                    'EDF_Noise','Mode','PSR(%)','Avg_Retries',
                    'Dominant_DR','Active_Model',
                ])
            with open(LOG_TXT_PATH, 'w', encoding='utf-8') as f:
                f.write(f"=== SIMULATION LOG ===\nMode: {mode_val}\n"
                        f"Model: {model_name}\nStart: {ts}\n\n")

        try:
            if trigger != 'init':
                with open(LOG_CSV_PATH, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        GLOBAL_STEP, f"{avg_e:.2f}",
                        NUM_NODES - dead_nodes, dead_nodes,
                        edf_val, mode_val,
                        f"{psr_value:.2f}", f"{avg_retries:.2f}",
                        dominant_dr, model_name,
                    ])
        except Exception:
            pass

        # TXT hanya mencatat user_events (aksi eksplisit user)
        if user_events and LOG_TXT_PATH is not None and trigger != 'init':
            try:
                with open(LOG_TXT_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"\n--- Step {GLOBAL_STEP} | {timestamp_now} ---\n")
                    for ev in user_events:
                        f.write(f"  {ev}\n")
            except Exception as e:
                print(f"[CB-5] Gagal tulis TXT: {e}")

        # --- Scatter graph ---
        sink_x, sink_y = (sink_pos_data if sink_pos_data else list(SINK_POSITION))
        safe_edf = edf_val if edf_val is not None else 0.0
        x_dist = []; y_energy = []; colors = []; sizes = []; hovers = []
        node_ids_scatter = []   # ← urutan node ID sesuai dengan titik scatter

        for nid, data in node_data.items():
            if nid == 'sink': continue
            pos = current_positions.get(str(nid), [0, 0])
            try:
                d2s = math.sqrt((pos[0]-sink_x)**2 + (pos[1]-sink_y)**2)
            except Exception:
                d2s = 0
            x_dist.append(d2s)
            eng = data.get('energy', 0)
            y_energy.append(eng)
            node_ids_scatter.append(str(nid))   # ← simpan ID sesuai urutan

            nr      = new_route.get(str(nid), {})
            dp      = nr.get('dist', 0)
            max_lim = range_val * SCALE if range_val else MAX_COMM_DISTANCE * SCALE
            sig_q   = max(0, 1 - (dp / max_lim)) * (1.0 - safe_edf) if max_lim else 0

            if eng <= 0:      c = '#636e72'
            elif sig_q > 0.7: c = '#55efc4'
            elif sig_q > 0.4: c = '#ffeaa7'
            else:             c = '#ff7675'
            colors.append(c)
            sizes.append(18 if str(sel_id) == str(nid) else 10)
            hovers.append(f"Node {nid}<br>Dist: {d2s:.1f}m<br>Eng: {eng:.1f}J")

        BG, TXT = '#2d3436', '#dfe6e9'
        fig = {
            'data': [{'x': x_dist, 'y': y_energy, 'mode': 'markers', 'type': 'scatter',
                      'marker': {'color': colors, 'size': sizes,
                                 'line': {'width': 1, 'color': 'white'}, 'opacity': 0.9},
                      'text': hovers, 'hoverinfo': 'text',
                      'customdata': node_ids_scatter,   # ← ID node per titik, urutan terjamin
                      }],
            'layout': {
                'title': {'text': f"Energy Distribution (Mode: {mode_val})",
                          'font': {'size': 14, 'color': TXT}, 'x': 0.05, 'y': 0.98},
                'plot_bgcolor': BG, 'paper_bgcolor': BG,
                'font': {'color': TXT},
                'margin': {'t': 40, 'b': 40, 'l': 50, 'r': 20},
                'autosize': True, 'uirevision': 'constant_scatter',
                'xaxis': {'title': 'Distance to Sink (m)', 'range': [0, 120],
                          'showgrid': True, 'gridcolor': '#636e72',
                          'zeroline': False, 'color': TXT},
                'yaxis': {'title': 'Residual Energy (Joule)',
                          'range': [-50, INITIAL_ENERGY_JOULE * 1.1],
                          'showgrid': True, 'gridcolor': '#636e72',
                          'zeroline': False, 'color': TXT},
                'hovermode': 'closest',
            }
        }

        # --- Inspector ---
        insp_content = html.P("Pilih node...",
                               style={'color': '#ccc', 'textAlign': 'center',
                                      'fontStyle': 'italic'})
        insp_style = {'display': 'none'}
        slider_out = INITIAL_ENERGY_JOULE

        if sel_id and sel_id in node_data:
            insp_style = {'display': 'block'}
            slider_out = node_data[sel_id]['energy']
            r = new_route.get(sel_id, {})
            is_drop   = r.get('dropped', False)
            stat_txt  = "Failed/Drop" if is_drop else "Success"
            stat_col  = "#e74c3c"     if is_drop else "#2ecc71"
            insp_content = html.Div([
                html.Div([
                    html.Span(f"NODE {sel_id}",
                              style={'fontSize': '20px', 'fontWeight': 'bold',
                                     'color': '#f5f6fa'}),        # ← putih, kontras di background gelap
                    html.Span(f" {node_data[sel_id]['energy']:.0f}J",
                              style={'float': 'right', 'color': '#e67e22',
                                     'fontWeight': 'bold'}),
                ]),
                html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr',
                                'gap': '5px', 'marginTop': '10px'}, children=[
                    html.Div([html.Span("Routes via", style={'fontSize': '10px', 'color': '#95a5a6'}),
                              html.Div(f"→ Node {r.get('parent','?')}",
                                       style={'fontWeight': 'bold', 'color': '#2980b9'})]),
                    html.Div([html.Span("Data Rate", style={'fontSize': '10px', 'color': '#95a5a6'}),
                              html.Div(f"{r.get('dr','?')}",
                                       style={'fontWeight': 'bold', 'color': '#27ae60'})]),
                    html.Div([html.Span("Retries", style={'fontSize': '10px', 'color': '#95a5a6'}),
                              html.Div(f"{r.get('retries',0)}",
                                       style={'fontWeight': 'bold', 'color': '#d35400'})]),
                    html.Div([html.Span("Last Status", style={'fontSize': '10px', 'color': '#95a5a6'}),
                              html.Div(stat_txt,
                                       style={'fontWeight': 'bold', 'color': stat_col})]),
                ]),
            ])

        log_ui = [
            html.Div(line, style={
                'borderBottom': '1px solid #485460', 'padding': '4px 0',
                'color': '#f5f6fa', 'fontFamily': 'monospace', 'fontSize': '11px',
            })
            for line in reversed(log_data)
        ]

        return (node_data, pos_data, history, new_route, log_data,
                elements, fig, insp_content, insp_style, slider_out,
                log_ui, auto_drain_val, f"⏱ Step: {GLOBAL_STEP}",
                sel_id)   # ← simpan ke st-sel-node Store