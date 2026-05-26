# app.py

import io
import time
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from inference import MultiModelPredictor, MODEL_VARIANTS, ENSEMBLE_SEEDS


# Konfigurasi halaman Streamlit

st.set_page_config(
    page_title="BACE-1 · Predict",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# THEME STATE — light / dark toggle

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'


# Custom CSS — Lab Terminal aesthetic

DARK_VARS = """
  --bg:        #0d0f0e;
  --bg-2:      #14181a;
  --line:      #253025;
  --line-2:    #334033;
  --fg:        #ffffff;
  --fg-2:      #f8f8f4;
  --muted:     #dde8dd;
  --muted-2:   #c8d4c8;
  --accent:    #d4ff70;
  --accent-d:  #a0d855;
  --warn:      #ffce5a;
  --err:       #ff7163;
  --mol-bg-r:  0.078;
  --mol-bg-g:  0.094;
  --mol-bg-b:  0.086;
  --grid-rgba: rgba(212,255,112,0.025);
  --metric-bg: var(--bg-2);
"""

LIGHT_VARS = """
  --bg:        #f6f7f2;
  --bg-2:      #ffffff;
  --line:      #d8e0d2;
  --line-2:    #b9c6b2;
  --fg:        #0d120d;
  --fg-2:      #1a201a;
  --muted:     #3a463a;
  --muted-2:   #6a766a;
  --accent:    #4f8a2c;
  --accent-d:  #36631c;
  --warn:      #b97a0a;
  --err:       #c43d2e;
  --mol-bg-r:  1.0;
  --mol-bg-g:  1.0;
  --mol-bg-b:  1.0;
  --grid-rgba: rgba(79,138,44,0.05);
  --metric-bg: #f1f4ec;
"""


def build_css(theme: str) -> str:
    vars_block = DARK_VARS if theme == 'dark' else LIGHT_VARS
    btn_text_color = '#0d120d' if theme == 'dark' else '#ffffff'
    btn_glow = '0 0 24px rgba(212,255,112,0.32)' if theme == 'dark' else '0 4px 14px rgba(79,138,44,0.28)'
    btn_glow_hover = '0 0 32px rgba(212,255,112,0.50)' if theme == 'dark' else '0 6px 18px rgba(79,138,44,0.42)'
    btn_hover_bg = '#deff88' if theme == 'dark' else '#5fa237'
    return f"""
<style>
  /* ---- Font import ---- */
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

  :root {{
    {vars_block}
  }}

  /* ---- Global page ---- */
  html, body, [class*="css"], .stApp {{
      background: var(--bg) !important;
      color: var(--fg-2) !important;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
      font-size: 16px !important;
  }}

  .stApp {{
      background:
        radial-gradient(circle at 12% 10%, rgba(212,255,112,0.045) 0%, transparent 28%),
        radial-gradient(circle at 88% 88%, rgba(212,255,112,0.025) 0%, transparent 35%),
        linear-gradient(180deg, var(--bg) 0%, #11140f 100%);
      background-attachment: fixed;
  }}

  .block-container {{
      padding-top: 1.4rem !important;
      padding-bottom: 4rem !important;
      max-width: 1280px;
  }}

  /* ---- Top bar (status strip) ---- */
  .lab-topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 14px;
      border: 1px solid var(--line);
      background: var(--bg-2);
      font-family: 'JetBrains Mono', monospace;
      font-size: 12.5px;
      color: var(--muted);
      margin-bottom: 18px;
      letter-spacing: 0.05em;
  }}
  .lab-topbar .stamp {{
      color: var(--accent);
      font-weight: 600;
  }}
  .lab-topbar .led {{
      display: inline-block;
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent);
      margin-right: 8px;
      vertical-align: middle;
  }}

  /* ---- Headers ---- */
  .lab-h1 {{
      font-family: 'JetBrains Mono', monospace;
      color: var(--fg);
      font-size: 38px;
      font-weight: 700;
      letter-spacing: -0.01em;
      margin: 8px 0 6px 0;
      line-height: 1.15;
  }}
  .lab-h1 .slash {{
      color: var(--accent);
      font-weight: 500;
  }}
  .lab-lede {{
      color: var(--muted);
      font-size: 16.5px;
      line-height: 1.6;
      max-width: 780px;
      margin-bottom: 24px;
  }}
  .lab-lede b {{ color: var(--fg-2); }}

  /* ---- Sectioning ---- */
  .lab-section-h {{
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent-d);
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      border-top: 1px dashed var(--line);
      padding: 18px 0 10px 0;
      margin-top: 14px;
      margin-bottom: 8px;
  }}
  .lab-section-h::before {{
      content: '// ';
      color: var(--muted-2);
  }}

  /* ---- Sidebar ---- */
  [data-testid="stSidebar"] {{
      background: var(--bg-2) !important;
      border-right: 1px solid var(--line);
  }}
  [data-testid="stSidebar"] > div:first-child {{
      padding-top: 1rem;
  }}
  .lab-brand {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 17px;
      font-weight: 700;
      color: var(--fg);
      letter-spacing: 0.02em;
      margin-bottom: 4px;
  }}
  .lab-brand .led {{
      display: inline-block;
      width: 9px; height: 9px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 10px var(--accent);
      margin-right: 8px;
      vertical-align: middle;
  }}
  .lab-brand-id {{
      font-family: 'JetBrains Mono', monospace;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.04em;
      margin-bottom: 16px;
      border-bottom: 1px dashed var(--line);
      padding-bottom: 12px;
  }}
  .lab-brand-id .slash {{ color: var(--accent); }}

  .lab-sec-label {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11.5px;
      letter-spacing: 0.12em;
      color: var(--muted-2);
      margin: 16px 0 6px 0;
      font-weight: 600;
  }}

  .lab-cfg-row {{
      display: flex;
      justify-content: space-between;
      padding: 4px 0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12.5px;
      color: var(--muted);
      border-bottom: 1px dotted rgba(255,255,255,0.04);
  }}
  .lab-cfg-row b {{ color: var(--fg-2); }}

  /* ---- Buttons ---- */
  .stButton > button {{
      background: var(--accent) !important;
      color: {btn_text_color} !important;
      border: 1px solid var(--accent-d) !important;
      border-radius: 4px !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-weight: 600 !important;
      font-size: 14px !important;
      letter-spacing: 0.05em !important;
      padding: 10px 18px !important;
      transition: all 0.15s ease !important;
      box-shadow: {btn_glow} !important;
  }}
  .stButton > button:hover {{
      background: {btn_hover_bg} !important;
      transform: translateY(-1px);
      box-shadow: {btn_glow_hover} !important;
  }}
  .stButton > button[kind="secondary"] {{
      background: transparent !important;
      color: var(--muted) !important;
      border: 1px solid var(--line-2) !important;
      box-shadow: none !important;
  }}
  .stButton > button[kind="secondary"]:hover {{
      border-color: var(--accent) !important;
      color: var(--accent) !important;
  }}

  /* ---- Inputs ---- */
  .stTextInput input, .stTextArea textarea, .stSelectbox > div > div {{
      background: var(--bg-2) !important;
      color: var(--fg-2) !important;
      border: 1px solid var(--line) !important;
      border-radius: 4px !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 14.5px !important;
  }}
  .stTextInput input:focus, .stTextArea textarea:focus {{
      border-color: var(--accent) !important;
      box-shadow: 0 0 0 1px var(--accent) !important;
  }}
  .stTextInput label, .stTextArea label, .stSelectbox label, .stFileUploader label, .stRadio label {{
      color: var(--muted) !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 12.5px !important;
      letter-spacing: 0.05em !important;
      text-transform: uppercase !important;
  }}

  /* ---- Radio ---- */
  .stRadio > div {{ gap: 8px !important; }}
  .stRadio label[data-baseweb="radio"] {{
      background: var(--bg-2);
      border: 1px solid var(--line);
      padding: 6px 12px !important;
      border-radius: 4px !important;
      cursor: pointer;
      font-size: 13.5px !important;
      text-transform: none !important;
      letter-spacing: 0 !important;
  }}

  /* ---- File uploader ---- */
  [data-testid="stFileUploader"] section {{
      background: var(--bg-2) !important;
      border: 1px dashed var(--line-2) !important;
      border-radius: 4px !important;
  }}

  /* ---- Metric ---- */
  [data-testid="stMetric"] {{
      background: var(--metric-bg);
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 4px;
  }}
  [data-testid="stMetricLabel"] {{
      color: var(--muted-2) !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 11.5px !important;
      letter-spacing: 0.08em !important;
      text-transform: uppercase !important;
  }}
  [data-testid="stMetricValue"] {{
      color: var(--fg) !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 26px !important;
      font-weight: 700 !important;
  }}

  /* ---- Result window ---- */
  .lab-result {{
      background: var(--bg-2);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 0;
      margin: 14px 0 6px 0;
      overflow: hidden;
  }}
  .lab-result-bar {{
      background: var(--bg);
      padding: 8px 14px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11.5px;
      color: var(--muted-2);
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      letter-spacing: 0.05em;
  }}
  .lab-result-bar .c {{ color: var(--accent-d); }}
  .lab-verdict {{
      display: flex;
      align-items: center;
      gap: 26px;
      padding: 20px 22px;
      flex-wrap: wrap;
  }}
  .lab-verdict .tag {{
      font-family: 'JetBrains Mono', monospace;
      color: var(--muted-2);
      font-size: 11.5px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 4px;
  }}
  .lab-verdict .big {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      font-size: 34px;
      letter-spacing: -0.01em;
      margin: 0;
  }}
  .lab-verdict .big.aktif    {{ color: var(--accent); }}
  .lab-verdict .big.inaktif  {{ color: var(--err); }}
  .lab-verdict .big.lowconf  {{ color: var(--warn); }}
  .lab-verdict .big .pct {{
      color: var(--muted);
      font-weight: 500;
      font-size: 22px;
      margin-left: 8px;
  }}

  .lab-conf {{
      display: flex;
      flex-direction: column;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: var(--muted);
      letter-spacing: 0.05em;
  }}
  .lab-conf-bars {{
      display: flex;
      gap: 2px;
      margin-top: 6px;
      margin-bottom: 4px;
  }}
  .lab-conf-bars span {{
      display: inline-block;
      width: 12px; height: 14px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
  }}
  .lab-conf-bars span.on   {{ background: var(--accent);  border-color: var(--accent-d); }}
  .lab-conf-bars span.warn {{ background: var(--warn);    border-color: var(--warn); opacity: 0.8; }}
  .lab-conf-bars span.err  {{ background: var(--err);     border-color: var(--err); opacity: 0.85; }}

  /* ---- Warning box ---- */
  .lab-warning {{
      background: var(--bg-2);
      border: 1px solid var(--warn);
      border-left: 4px solid var(--warn);
      padding: 14px 18px;
      border-radius: 4px;
      color: var(--fg-2);
      font-size: 14.5px;
      margin: 14px 0;
  }}
  .lab-warning .h {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
      color: var(--warn);
      letter-spacing: 0.05em;
      margin-bottom: 6px;
      font-size: 13.5px;
      text-transform: uppercase;
  }}
  .lab-warning ul {{
      margin: 8px 0 0 0;
      padding-left: 20px;
      font-size: 13.5px;
      color: var(--muted);
  }}

  /* ---- Compare grid (mode 3) ---- */
  .lab-compare-card {{
      background: var(--bg-2);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 14px 16px;
      margin-bottom: 10px;
  }}
  .lab-compare-card.aktif   {{ border-left: 4px solid var(--accent); }}
  .lab-compare-card.inaktif {{ border-left: 4px solid var(--err); }}
  .lab-compare-card.lowconf {{ border-left: 4px solid var(--warn); }}
  .lab-compare-card .vname {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: var(--muted-2);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 4px;
  }}
  .lab-compare-card .vclass {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 22px;
      font-weight: 700;
      margin: 2px 0 4px 0;
  }}
  .lab-compare-card .vclass.aktif   {{ color: var(--accent); }}
  .lab-compare-card .vclass.inaktif {{ color: var(--err); }}
  .lab-compare-card .vclass.lowconf {{ color: var(--warn); }}
  .lab-compare-card .vdetail {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12.5px;
      color: var(--muted);
      letter-spacing: 0.03em;
      line-height: 1.5;
  }}
  .lab-compare-card .vdetail b {{ color: var(--fg-2); }}

  /* ---- Structure viewer ---- */
  .lab-struct-h {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11.5px;
      color: var(--muted-2);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 6px;
  }}

  /* ---- Code block ---- */
  pre, code, .stCode {{
      background: var(--bg) !important;
      border: 1px solid var(--line) !important;
      color: var(--accent) !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 13px !important;
      border-radius: 4px !important;
  }}

  /* ---- DataFrame ---- */
  .stDataFrame {{
      border: 1px solid var(--line) !important;
      border-radius: 4px !important;
  }}

  /* ---- Spinner override ---- */
  .stSpinner > div {{
      border-top-color: var(--accent) !important;
      border-right-color: var(--accent) !important;
  }}

  /* ---- Captions ---- */
  .stCaption, [data-testid="stCaptionContainer"] {{
      color: var(--muted-2) !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 12px !important;
      letter-spacing: 0.03em !important;
  }}

  /* ---- Expander ---- */
  .streamlit-expanderHeader {{
      background: var(--bg-2) !important;
      border: 1px solid var(--line) !important;
      border-radius: 4px !important;
      font-family: 'JetBrains Mono', monospace !important;
      color: var(--muted) !important;
  }}

  /* ---- Hide Streamlit chrome ---- */
  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
  header {{ visibility: hidden; }}

  /* ---- Selectbox dropdown ---- */
  [data-baseweb="select"] > div {{
      background: var(--bg-2) !important;
      border-color: var(--line) !important;
  }}
</style>
"""


st.markdown(build_css(st.session_state.theme), unsafe_allow_html=True)


# Konstanta dan contoh SMILES

EXAMPLE_SMILES = {
    "Verubecestat (BACE-1 inhibitor klinis)":
        "O=S1(=O)CC(C)(c2ccc(NC(=O)c3cc(F)cnc3C)cc2F)N=C(N)N1C",
    "Senyawa training (aktif)":
        "C#CCOc1cnc(/C(F)=C/c2cc(F)c(F)c([C@]3(C)CS(=O)(=O)C(C)(C)C(N)=N3)c2)cn1",
    "Senyawa training (tidak aktif)":
        "Brc1ccc(/C=N\\C2CCN(Cc3ccccc3)C2)cc1",
    "Asetaldehid (out-of-domain, untuk demo warning)":
        "CC=O",
}


# Cache predictor — load 4 varian sekali saja

@st.cache_resource
def load_predictor():
    return MultiModelPredictor(models_dir='models', device='cpu')


# Helpers — visualisasi molekul (theme-aware)

def smiles_to_image(smiles, size=(460, 360)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        if st.session_state.theme == 'dark':
            opts.setBackgroundColour((0.078, 0.094, 0.086))
            opts.setAtomPalette({
                -1:  (0.96, 0.96, 0.93),
                6:   (0.96, 0.96, 0.93),
                7:   (0.66, 0.88, 0.42),
                8:   (1.00, 0.42, 0.36),
                9:   (0.66, 0.88, 0.42),
                16:  (0.96, 0.73, 0.29),
                17:  (0.66, 0.88, 0.42),
                35:  (1.00, 0.42, 0.36),
            })
        else:
            opts.setBackgroundColour((1.0, 1.0, 1.0))
            opts.setAtomPalette({
                -1:  (0.05, 0.07, 0.05),
                6:   (0.05, 0.07, 0.05),
                7:   (0.18, 0.45, 0.10),
                8:   (0.77, 0.24, 0.18),
                9:   (0.18, 0.45, 0.10),
                16:  (0.73, 0.48, 0.04),
                17:  (0.18, 0.45, 0.10),
                35:  (0.77, 0.24, 0.18),
            })
        opts.bondLineWidth = 1.6
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        from PIL import Image
        return Image.open(io.BytesIO(drawer.GetDrawingText()))
    except Exception:
        return Draw.MolToImage(mol, size=size)


def scaffold_to_image(smiles, size=(460, 360)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
            return None
        AllChem.Compute2DCoords(scaffold_mol)
        return smiles_to_image(Chem.MolToSmiles(scaffold_mol), size)
    except Exception:
        return None


def verdict_class(classification, confidence):
    if confidence == 'Low':
        return 'lowconf'
    if classification == 'Aktif':
        return 'aktif'
    return 'inaktif'


def conf_bars_html(probability, confidence):
    if confidence == 'Low':
        cells_on = 4
        cls = 'warn'
    else:
        cells_on = int(round(abs(probability - 0.5) * 2 * 10))
        cells_on = max(1, min(10, cells_on))
        cls = 'on' if probability >= 0.5 else 'err'
    out = []
    for i in range(10):
        if i < cells_on:
            out.append(f'<span class="{cls}"></span>')
        else:
            out.append('<span></span>')
    return ''.join(out)


def session_id():
    if '_sid' not in st.session_state:
        import random, string
        st.session_state._sid = ''.join(random.choices(string.hexdigits.upper(), k=4))
    return st.session_state._sid


def fmt_metric(val, decimals=4):
    # Format float aman, return '-' jika None.
    if val is None:
        return '-'
    try:
        return f"{val:.{decimals}f}"
    except Exception:
        return str(val)


# Load predictor (sebelum sidebar, supaya bisa enumerate varian)

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.info("Pastikan folder `models/` ada dengan struktur yang benar. Lihat README.")
    st.stop()


LOADED_VARIANTS = predictor.get_loaded_variants()
if not LOADED_VARIANTS:
    st.error("Tidak ada model yang ter-load. Periksa folder models/")
    st.stop()


# Sidebar

with st.sidebar:
    st.markdown(
        f"""
        <div class="lab-brand"><span class="led"></span> BACE-1 · LIVE</div>
        <div class="lab-brand-id">bace<span class="slash">/</span>predict<span class="slash">·</span>v2</div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Theme toggle ----
    st.markdown('<div class="lab-sec-label">// TAMPILAN</div>', unsafe_allow_html=True)
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        if st.button(
            "☾ Gelap",
            use_container_width=True,
            type="primary" if st.session_state.theme == 'dark' else "secondary",
            key="btn_theme_dark",
        ):
            if st.session_state.theme != 'dark':
                st.session_state.theme = 'dark'
                st.rerun()
    with t_col2:
        if st.button(
            "☀ Terang",
            use_container_width=True,
            type="primary" if st.session_state.theme == 'light' else "secondary",
            key="btn_theme_light",
        ):
            if st.session_state.theme != 'light':
                st.session_state.theme = 'light'
                st.rerun()

    # ---- Mode selector ----
    st.markdown('<div class="lab-sec-label">// MODE</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Mode",
        ["Single SMILES", "Batch SMILES", "Compare Models", "About"],
        index=0,
        label_visibility="collapsed",
    )

    # ---- Model selector ----
    st.markdown('<div class="lab-sec-label">// MODEL.SELECT</div>', unsafe_allow_html=True)
    variant_labels = {
        vkey: predictor.get_variant_info(vkey)['label']
        for vkey in LOADED_VARIANTS
    }
    default_key = predictor.get_default_model_key()
    default_idx = LOADED_VARIANTS.index(default_key) if default_key in LOADED_VARIANTS else 0

    # Compare mode tidak butuh selector (pakai semua), tapi tetap render untuk konsistensi
    selected_variant = st.selectbox(
        "Pilih model",
        options=LOADED_VARIANTS,
        index=default_idx,
        format_func=lambda k: variant_labels.get(k, k),
        label_visibility="collapsed",
        disabled=(mode == "Compare Models"),
    )

    if mode == "Compare Models":
        st.caption(f"Mode compare aktif → pakai semua {len(LOADED_VARIANTS)} model")

    # ---- Model config card (dinamis sesuai selected_variant) ----
    info = predictor.get_variant_info(selected_variant)
    cfg = info['config']
    st.markdown('<div class="lab-sec-label">// MODEL.CFG</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="lab-cfg-row"><span>arch</span><b>{info['model_type']}</b></div>
        <div class="lab-cfg-row"><span>split</span><b>{info['split']}</b></div>
        <div class="lab-cfg-row"><span>layers</span><b>{cfg['num_layers']}</b></div>
        <div class="lab-cfg-row"><span>hidden</span><b>{cfg['hidden_dim']}</b></div>
        <div class="lab-cfg-row"><span>dropout</span><b>{cfg['dropout']}</b></div>
        <div class="lab-cfg-row"><span>seeds</span><b>×{info['n_seeds_loaded']}</b></div>
        <div class="lab-cfg-row"><span>edge_feat</span><b>{'yes' if info['model_type'] == 'GINE' else 'no'}</b></div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Metrics card (dari multi-seed mean) ----
    mean_metrics = info.get('multi_seed_mean', {})
    std_metrics = info.get('multi_seed_std', {})
    st.markdown('<div class="lab-sec-label">// METRICS.TEST</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="lab-cfg-row"><span>PR-AUC</span><b>{fmt_metric(mean_metrics.get('pr_auc'))}</b></div>
        <div class="lab-cfg-row"><span>ROC-AUC</span><b>{fmt_metric(mean_metrics.get('roc_auc'))}</b></div>
        <div class="lab-cfg-row"><span>F1</span><b>{fmt_metric(mean_metrics.get('f1'))}</b></div>
        <div class="lab-cfg-row"><span>Accuracy</span><b>{fmt_metric(mean_metrics.get('accuracy'))}</b></div>
        <div class="lab-cfg-row"><span>Precision</span><b>{fmt_metric(mean_metrics.get('precision'))}</b></div>
        <div class="lab-cfg-row"><span>Recall</span><b>{fmt_metric(mean_metrics.get('recall'))}</b></div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("multi-seed mean (±std lihat About)")

    st.markdown('<div class="lab-sec-label">// NOTICE</div>', unsafe_allow_html=True)
    st.caption(
        "Prediksi bersifat computational dan tidak menggantikan "
        "uji eksperimental laboratorium."
    )


# Top bar (shared chrome)

now = time.strftime('%H:%M:%S')
theme_label = 'DARK' if st.session_state.theme == 'dark' else 'LIGHT'
active_model_label = info['label'] if mode != "Compare Models" else f"COMPARE × {len(LOADED_VARIANTS)}"
st.markdown(
    f"""
    <div class="lab-topbar">
        <span>session · 0x{session_id()} · {now} UTC · theme:{theme_label} · model:{active_model_label}</span>
        <span class="stamp"><span class="led"></span> MODEL READY</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# MODE 1 — Single SMILES

if mode == "Single SMILES":
    st.markdown(
        '<div class="lab-h1">predict<span class="slash">/</span>single</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="lab-lede">Klasifikasi binary inhibitor BACE-1. Model: <b>{info["label"]}</b>. '
        f'Ensemble dari {info["n_seeds_loaded"]} graph neural networks, '
        f'agregasi <b>mean</b> probabilitas dengan reporting <b>std deviation</b> sebagai estimasi epistemic uncertainty.</p>',
        unsafe_allow_html=True,
    )

    col_example, _ = st.columns([3, 1])
    with col_example:
        example_choice = st.selectbox(
            "Contoh SMILES",
            options=["(masukkan SMILES sendiri)"] + list(EXAMPLE_SMILES.keys()),
            index=0,
        )

    default_smiles = ""
    if example_choice != "(masukkan SMILES sendiri)":
        default_smiles = EXAMPLE_SMILES[example_choice]

    smiles_input = st.text_input(
        "SMILES",
        value=default_smiles,
        placeholder="$ smiles >  contoh: CCO  atau  O=S1(=O)CC(C)(...)",
    )

    predict_btn = st.button("▶ Execute prediction", type="primary", use_container_width=True)

    if predict_btn:
        if not smiles_input.strip():
            st.warning("[input.error] Mohon masukkan SMILES terlebih dahulu.")
        else:
            with st.spinner(f"Running {info['n_seeds_loaded']} forward passes through ensemble..."):
                start = time.time()
                result = predictor.predict(smiles_input.strip(), model_key=selected_variant)
                elapsed = time.time() - start

            if not result['success']:
                st.error(f"[predict.failed] {result['error']}")
            else:
                classification = result['classification']
                probability    = result['probability']
                uncertainty    = result['uncertainty']
                confidence     = result['confidence_level']
                v_cls          = verdict_class(classification, confidence)

                st.markdown(
                    f"""
                    <div class="lab-result">
                      <div class="lab-result-bar">
                        <span class="c">┌ result.window</span>
                        <span>model:{info['label']} · t = {elapsed:.2f}s · {info['n_seeds_loaded']} forward passes</span>
                        <span class="c">┐</span>
                      </div>
                      <div class="lab-verdict">
                        <div>
                          <div class="tag">classification.label</div>
                          <h2 class="big {v_cls}">{classification.upper()}<span class="pct">·{probability*100:.1f}%</span></h2>
                        </div>
                        <div class="lab-conf">
                          <span>confidence: {confidence.lower()}</span>
                          <div class="lab-conf-bars">{conf_bars_html(probability, confidence)}</div>
                          <span style="margin-top:6px">σ ensemble · <b style="color:var(--fg)">{uncertainty:.4f}</b></span>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if not result['in_domain']:
                    warns_html = ''.join(f"<li>{w}</li>" for w in result['domain_warnings'])
                    st.markdown(
                        f"""
                        <div class="lab-warning">
                          <div class="h">⚠ out-of-domain warning</div>
                          Senyawa ini berada di luar rentang training. Prediksi mungkin kurang akurat.
                          <ul>{warns_html}</ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown('<div class="lab-section-h">key metrics</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("P(active) · μ", f"{probability:.4f}",
                            help=f"Mean probability dari {info['n_seeds_loaded']} model ensemble")
                col2.metric("σ ensemble",   f"{uncertainty:.4f}",
                            help=f"Std deviasi probabilitas antar {info['n_seeds_loaded']} model")
                col3.metric("Confidence",   confidence, help="High / Medium / Low")

                with st.expander("› ensemble · probabilitas per seed"):
                    probs_df = pd.DataFrame({
                        'seed':       result['seeds_used'],
                        'P(active)':  [f"{p:.4f}" for p in result['probs_per_model']],
                    })
                    st.dataframe(probs_df, use_container_width=True, hide_index=True)
                    st.caption(f"μ = {probability:.4f}  ·  σ = {uncertainty:.4f}")

                st.markdown('<div class="lab-section-h">molecule.properties</div>', unsafe_allow_html=True)
                props = result['properties']
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MW · Da",   f"{props['MW']:.2f}")
                c2.metric("AlogP",     f"{props['AlogP']:.2f}")
                c3.metric("Atoms",     f"{props['num_atoms']}")
                c4.metric("Bonds",     f"{props['num_bonds']}")
                c5, c6 = st.columns(2)
                c5.metric("Rings",     f"{props['num_rings']}")
                c6.metric("Inference", f"{elapsed:.2f}s")

                st.markdown('<div class="lab-section-h">structure.visual</div>', unsafe_allow_html=True)
                col_struct, col_scaffold = st.columns(2)

                with col_struct:
                    st.markdown('<div class="lab-struct-h">molecule.2d · canonical</div>', unsafe_allow_html=True)
                    img = smiles_to_image(result['canonical_smiles'])
                    if img:
                        st.image(img, use_container_width=True)
                    st.code(result['canonical_smiles'], language=None)

                with col_scaffold:
                    st.markdown('<div class="lab-struct-h">scaffold.murcko</div>', unsafe_allow_html=True)
                    if props['scaffold']:
                        scaffold_img = scaffold_to_image(props['scaffold'])
                        if scaffold_img:
                            st.image(scaffold_img, use_container_width=True)
                        else:
                            st.info("[scaffold.empty] Tidak dapat divisualisasi")
                        st.code(props['scaffold'], language=None)
                    else:
                        st.info("[scaffold.empty] Senyawa tidak memiliki scaffold")


# MODE 2 — Batch SMILES

elif mode == "Batch SMILES":
    st.markdown(
        '<div class="lab-h1">predict<span class="slash">/</span>batch</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="lab-lede">Prediksi banyak SMILES sekaligus dengan model <b>{info["label"]}</b>. '
        f'Input via upload CSV atau text area · maks <b>100</b> per batch.</p>',
        unsafe_allow_html=True,
    )

    input_method = st.radio(
        "Input method",
        ["Upload CSV", "Text Area (satu SMILES per baris)"],
        horizontal=True,
    )

    smiles_list = []

    if input_method == "Upload CSV":
        st.info("[csv.spec] Kolom bernama `smiles` (case-insensitive). Maksimal 100 baris per batch.")
        uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])

        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                smiles_col = None
                for col in df_input.columns:
                    if col.lower().strip() == 'smiles':
                        smiles_col = col
                        break

                if smiles_col is None:
                    st.error(f"[csv.error] kolom 'smiles' tidak ditemukan. Available: {list(df_input.columns)}")
                else:
                    smiles_list = df_input[smiles_col].dropna().astype(str).tolist()
                    st.success(f"[csv.loaded] {len(smiles_list)} SMILES ready.")
                    if len(smiles_list) > 100:
                        st.warning("[csv.truncated] >100 baris — hanya 100 pertama yang diproses.")
                        smiles_list = smiles_list[:100]
            except Exception as e:
                st.error(f"[csv.error] {e}")
    else:
        text_input = st.text_area(
            "SMILES list",
            height=200,
            placeholder="CCO\nCC(=O)O\nc1ccccc1\n...",
        )
        if text_input.strip():
            smiles_list = [s.strip() for s in text_input.strip().split('\n') if s.strip()]
            if len(smiles_list) > 100:
                st.warning("[input.truncated] >100 baris — hanya 100 pertama yang diproses.")
                smiles_list = smiles_list[:100]

    if smiles_list:
        st.markdown(
            f'<div class="lab-cfg-row" style="margin-top:8px"><span>queue.size</span>'
            f'<b style="color:var(--accent)">{len(smiles_list)}</b></div>',
            unsafe_allow_html=True,
        )

        if st.button("▶ Execute batch", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Initializing ensemble...")
            start = time.time()
            results = []
            for idx, smi in enumerate(smiles_list):
                res = predictor.predict(smi, model_key=selected_variant)
                results.append(res)
                progress = (idx + 1) / len(smiles_list)
                progress_bar.progress(progress, text=f"› forward pass · {idx + 1}/{len(smiles_list)}")
            progress_bar.empty()
            elapsed = time.time() - start

            rows = []
            for idx, res in enumerate(results):
                row = {
                    'no':           idx + 1,
                    'input':        res['input_smiles'],
                    'status':       'ok' if res['success'] else 'err',
                    'class':        res.get('classification', '-') if res['success'] else '-',
                    'P(active)':    f"{res['probability']:.4f}" if res.get('probability') is not None else '-',
                    'σ':            f"{res['uncertainty']:.4f}" if res.get('uncertainty') is not None else '-',
                    'conf':         res.get('confidence_level', '-') or '-',
                    'in_domain':    res.get('in_domain') if res.get('in_domain') is not None else '-',
                    'MW':           f"{res['properties']['MW']:.2f}" if res.get('properties') else '-',
                    'AlogP':        f"{res['properties']['AlogP']:.2f}" if res.get('properties') else '-',
                    'atoms':        res['properties']['num_atoms'] if res.get('properties') else '-',
                    'error':        res['error'] if not res['success'] else '',
                }
                rows.append(row)
            df_result = pd.DataFrame(rows)

            n_total    = len(results)
            n_success  = sum(1 for r in results if r['success'])
            n_active   = sum(1 for r in results if r['success'] and r['classification'] == 'Aktif')
            n_inactive = n_success - n_active
            n_error    = n_total - n_success
            n_low_conf = sum(1 for r in results if r['success'] and r['confidence_level'] == 'Low')

            st.markdown('<div class="lab-section-h">batch.summary</div>', unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total",          n_total)
            col2.metric("Aktif",          n_active)
            col3.metric("Tidak Aktif",    n_inactive)
            col4.metric("Low Conf",       n_low_conf)
            col5.metric("Error",          n_error)
            st.caption(f"Model: {info['label']}  ·  Total time {elapsed:.1f}s  ·  avg {elapsed/n_total:.2f}s / SMILES")

            st.markdown('<div class="lab-section-h">batch.results</div>', unsafe_allow_html=True)

            if st.session_state.theme == 'dark':
                bg_row     = '#14181a'
                aktif_c    = '#d4ff70'
                inaktif_c  = '#ff7163'
                err_c      = '#ff7163'
            else:
                bg_row     = '#ffffff'
                aktif_c    = '#36631c'
                inaktif_c  = '#c43d2e'
                err_c      = '#c43d2e'

            def _style_row(row):
                cls = row.get('class', '-')
                status = row.get('status', '-')
                if status == 'err':
                    return [f'color: {err_c}'] * len(row)
                if cls == 'Aktif':
                    return [f'color: {aktif_c}'] * len(row)
                if cls == 'Tidak Aktif':
                    return [f'color: {inaktif_c}'] * len(row)
                return [''] * len(row)

            styled = (df_result.style
                      .apply(_style_row, axis=1)
                      .set_properties(**{
                          'background-color': bg_row,
                          'font-family': 'JetBrains Mono, monospace',
                          'font-size': '14px',
                      }))
            st.dataframe(styled, use_container_width=True, hide_index=True)

            csv_buffer = io.StringIO()
            df_result.to_csv(csv_buffer, index=False)
            st.download_button(
                label="↓ Download results.csv",
                data=csv_buffer.getvalue(),
                file_name=f"bace1_predictions_{selected_variant}_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# MODE 3 — Compare Models

elif mode == "Compare Models":
    st.markdown(
        '<div class="lab-h1">predict<span class="slash">/</span>compare</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="lab-lede">Bandingkan prediksi <b>{len(LOADED_VARIANTS)} varian model</b> untuk satu SMILES. '
        f'Mode ini berguna untuk melihat konsistensi prediksi antar arsitektur (GIN vs GINE) '
        f'dan antar split strategy (scaffold vs stratified).</p>',
        unsafe_allow_html=True,
    )

    col_example, _ = st.columns([3, 1])
    with col_example:
        example_choice = st.selectbox(
            "Contoh SMILES",
            options=["(masukkan SMILES sendiri)"] + list(EXAMPLE_SMILES.keys()),
            index=0,
            key="cmp_example",
        )

    default_smiles = ""
    if example_choice != "(masukkan SMILES sendiri)":
        default_smiles = EXAMPLE_SMILES[example_choice]

    smiles_input = st.text_input(
        "SMILES",
        value=default_smiles,
        placeholder="$ smiles >  contoh: CCO  atau  O=S1(=O)CC(C)(...)",
        key="cmp_smi_input",
    )

    compare_btn = st.button("▶ Execute compare", type="primary", use_container_width=True)

    if compare_btn:
        if not smiles_input.strip():
            st.warning("[input.error] Mohon masukkan SMILES terlebih dahulu.")
        else:
            with st.spinner(f"Running {len(LOADED_VARIANTS)} models × seeds forward passes..."):
                start = time.time()
                all_results = predictor.predict_all(smiles_input.strip())
                elapsed = time.time() - start

            # Cek apakah ada error pada SMILES (sama untuk semua varian karena preprocessing sama)
            first_ok_result = next((r for r in all_results.values() if r['success']), None)
            any_failed = any(not r['success'] for r in all_results.values())

            if first_ok_result is None:
                # Semua gagal — pasti error preprocessing SMILES
                error_msg = next(iter(all_results.values()))['error']
                st.error(f"[predict.failed] {error_msg}")
            else:
                # Tampilkan info SMILES
                st.markdown(
                    f"""
                    <div class="lab-result">
                      <div class="lab-result-bar">
                        <span class="c">┌ compare.window</span>
                        <span>{len(all_results)} models · t = {elapsed:.2f}s total</span>
                        <span class="c">┐</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # OOD warning (sama untuk semua, ambil dari yang pertama)
                if not first_ok_result['in_domain']:
                    warns_html = ''.join(f"<li>{w}</li>" for w in first_ok_result['domain_warnings'])
                    st.markdown(
                        f"""
                        <div class="lab-warning">
                          <div class="h">⚠ out-of-domain warning</div>
                          Senyawa ini berada di luar rentang training. Prediksi mungkin kurang akurat.
                          <ul>{warns_html}</ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Grid 4 model
                st.markdown('<div class="lab-section-h">compare.results</div>', unsafe_allow_html=True)

                # Render 2 kolom × 2 baris (atau menyesuaikan jumlah varian)
                variant_list = list(all_results.keys())
                cols_per_row = 2
                for row_start in range(0, len(variant_list), cols_per_row):
                    row_keys = variant_list[row_start:row_start + cols_per_row]
                    cols = st.columns(len(row_keys))
                    for col, vkey in zip(cols, row_keys):
                        res = all_results[vkey]
                        vinfo = predictor.get_variant_info(vkey)
                        with col:
                            if not res['success']:
                                st.markdown(
                                    f"""
                                    <div class="lab-compare-card">
                                      <div class="vname">{vinfo['label']}</div>
                                      <div class="vclass inaktif">ERROR</div>
                                      <div class="vdetail">{res['error']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                cls = res['classification']
                                prob = res['probability']
                                std = res['uncertainty']
                                conf = res['confidence_level']
                                vcls = verdict_class(cls, conf)
                                st.markdown(
                                    f"""
                                    <div class="lab-compare-card {vcls}">
                                      <div class="vname">{vinfo['label']}</div>
                                      <div class="vclass {vcls}">{cls.upper()}</div>
                                      <div class="vdetail">
                                        P(active) = <b>{prob:.4f}</b><br>
                                        σ ensemble = <b>{std:.4f}</b><br>
                                        confidence = <b>{conf.lower()}</b><br>
                                        arch · L{vinfo['config']['num_layers']} H{vinfo['config']['hidden_dim']}
                                      </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                # Tabel ringkasan
                st.markdown('<div class="lab-section-h">compare.summary</div>', unsafe_allow_html=True)
                summary_rows = []
                for vkey, res in all_results.items():
                    vinfo = predictor.get_variant_info(vkey)
                    if res['success']:
                        summary_rows.append({
                            'model':       vinfo['label'],
                            'arch':        vinfo['model_type'],
                            'split':       vinfo['split'],
                            'class':       res['classification'],
                            'P(active)':   f"{res['probability']:.4f}",
                            'σ ensemble':  f"{res['uncertainty']:.4f}",
                            'confidence':  res['confidence_level'],
                            'val PR-AUC':  fmt_metric(vinfo['multi_seed_mean'].get('pr_auc')),
                        })
                    else:
                        summary_rows.append({
                            'model':       vinfo['label'],
                            'arch':        vinfo['model_type'],
                            'split':       vinfo['split'],
                            'class':       'ERROR',
                            'P(active)':   '-',
                            'σ ensemble':  '-',
                            'confidence':  '-',
                            'val PR-AUC':  '-',
                        })
                df_cmp = pd.DataFrame(summary_rows)

                if st.session_state.theme == 'dark':
                    bg_row, aktif_c, inaktif_c = '#14181a', '#d4ff70', '#ff7163'
                else:
                    bg_row, aktif_c, inaktif_c = '#ffffff', '#36631c', '#c43d2e'

                def _style_cmp(row):
                    if row['class'] == 'Aktif':
                        return [f'color: {aktif_c}'] * len(row)
                    if row['class'] == 'Tidak Aktif':
                        return [f'color: {inaktif_c}'] * len(row)
                    return [f'color: {inaktif_c}'] * len(row)

                styled = (df_cmp.style
                          .apply(_style_cmp, axis=1)
                          .set_properties(**{
                              'background-color': bg_row,
                              'font-family': 'JetBrains Mono, monospace',
                              'font-size': '14px',
                          }))
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # Konsistensi prediksi
                classes = [r['classification'] for r in all_results.values() if r['success']]
                n_aktif = classes.count('Aktif')
                n_inaktif = classes.count('Tidak Aktif')
                if len(classes) > 0:
                    if n_aktif == len(classes):
                        consensus_text = f"✓ Semua {len(classes)} model setuju: AKTIF"
                        consensus_color = "var(--accent)"
                    elif n_inaktif == len(classes):
                        consensus_text = f"✓ Semua {len(classes)} model setuju: TIDAK AKTIF"
                        consensus_color = "var(--err)"
                    else:
                        consensus_text = f"⚠ Disagreement: {n_aktif} aktif vs {n_inaktif} tidak aktif"
                        consensus_color = "var(--warn)"

                    st.markdown(
                        f"""
                        <div class="lab-cfg-row" style="margin-top:14px; font-size:14px;">
                          <span>consensus</span><b style="color:{consensus_color}">{consensus_text}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Struktur molekul
                st.markdown('<div class="lab-section-h">structure.visual</div>', unsafe_allow_html=True)
                col_struct, col_scaffold = st.columns(2)
                with col_struct:
                    st.markdown('<div class="lab-struct-h">molecule.2d · canonical</div>', unsafe_allow_html=True)
                    img = smiles_to_image(first_ok_result['canonical_smiles'])
                    if img:
                        st.image(img, use_container_width=True)
                    st.code(first_ok_result['canonical_smiles'], language=None)
                with col_scaffold:
                    st.markdown('<div class="lab-struct-h">scaffold.murcko</div>', unsafe_allow_html=True)
                    props = first_ok_result['properties']
                    if props and props['scaffold']:
                        scaffold_img = scaffold_to_image(props['scaffold'])
                        if scaffold_img:
                            st.image(scaffold_img, use_container_width=True)
                        st.code(props['scaffold'], language=None)
                    else:
                        st.info("[scaffold.empty] Senyawa tidak memiliki scaffold")


# MODE 4 — About

elif mode == "About":
    st.markdown(
        '<div class="lab-h1">readme<span class="slash">/</span>bace-1</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="lab-lede">Dokumentasi singkat tentang model, dataset, dan keterbatasan. Bacalah sebelum menafsirkan hasil prediksi.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="lab-section-h">background</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **BACE-1** (β-site Amyloid Precursor Protein Cleaving Enzyme 1) adalah
        target terapi penting dalam pengembangan obat untuk penyakit Alzheimer.
        Enzim ini berperan dalam pembentukan plak amiloid-β yang diduga menjadi
        penyebab utama degenerasi neuron pada Alzheimer.

        Aplikasi ini menggunakan model **Graph Neural Network (GNN)** untuk
        mengklasifikasikan apakah suatu senyawa kimia berpotensi sebagai
        **inhibitor BACE-1** (aktif) atau tidak, berdasarkan struktur molekulnya
        yang direpresentasikan dalam format **SMILES**.
        """
    )

    st.markdown('<div class="lab-section-h">available models</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Aplikasi ini menyediakan **4 varian model** untuk mengakomodasi
        berbagai kebutuhan analisis:

        - **GIN · scaffold**: Graph Isomorphism Network tanpa fitur edge,
          dilatih dengan scaffold split (uji generalisasi ke scaffold baru).
        - **GIN · stratified**: GIN dengan stratified split (uji performa i.i.d.).
        - **GINE · scaffold**: GIN dengan fitur edge (bond type, stereo,
          konjugasi, ring), scaffold split.
        - **GINE · stratified**: GINE dengan stratified split.

        **Scaffold split** lebih realistis untuk skenario drug discovery
        karena memastikan struktur inti molekul di test set berbeda dengan
        training set. **Stratified split** menjadi upper-bound performa
        pada distribusi i.i.d.

        Pilih model lewat sidebar, atau gunakan mode **Compare Models** untuk
        melihat prediksi keempat varian secara bersamaan.
        """
    )

    st.markdown('<div class="lab-section-h">performance summary</div>', unsafe_allow_html=True)
    perf_rows = []
    for vkey in LOADED_VARIANTS:
        vinfo = predictor.get_variant_info(vkey)
        mean = vinfo.get('multi_seed_mean', {})
        std = vinfo.get('multi_seed_std', {})
        perf_rows.append({
            'model':       vinfo['label'],
            'arch':        vinfo['model_type'],
            'split':       vinfo['split'],
            'layers':      vinfo['config']['num_layers'],
            'hidden':      vinfo['config']['hidden_dim'],
            'PR-AUC':      f"{mean.get('pr_auc', 0):.4f} ± {std.get('pr_auc', 0):.4f}",
            'ROC-AUC':     f"{mean.get('roc_auc', 0):.4f} ± {std.get('roc_auc', 0):.4f}",
            'F1':          f"{mean.get('f1', 0):.4f} ± {std.get('f1', 0):.4f}",
            'Accuracy':    f"{mean.get('accuracy', 0):.4f} ± {std.get('accuracy', 0):.4f}",
        })
    st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)
    st.caption("Metrik ditampilkan sebagai mean ± std dari 5 multi-seed runs di test set.")

    st.markdown('<div class="lab-section-h">arsitektur</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **Graph Isomorphism Network (GIN)** adalah salah satu arsitektur GNN
        terkuat untuk klasifikasi graf, dengan kekuatan ekspresif setara
        Weisfeiler-Lehman test. Komponen utama:

        1. **Node Encoder**: linear projection dari 43-dim atom features
        2. **Message Passing Layers**: GINConv (atau GINEConv untuk GINE)
           dengan MLP 2-layer dan learnable epsilon
        3. **Batch Normalization** setelah setiap layer
        4. **Jumping Knowledge Concat (JK-Concat)**: agregasi representasi
           dari semua layer
        5. **Classifier MLP** 2-layer dengan dropout

        **GINE** menambahkan **edge encoder per layer** untuk memproyeksikan
        12-dim bond features ke hidden dim, kemudian ditambahkan ke node
        features di dalam agregasi pesan.

        Setiap varian model adalah **ensemble dari 5 model** yang dilatih
        dengan seed berbeda (42, 123, 456, 789, 2024). Prediksi final adalah
        **mean probabilitas** dari kelima model, dengan **standard deviation**
        sebagai estimasi *epistemic uncertainty*.
        """
    )

    st.markdown('<div class="lab-section-h">dataset</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Sumber**: ChEMBL v36 — semua senyawa dengan aktivitas
          terhadap target BACE-1 (CHEMBL4822)
        - **Jumlah senyawa**: 7,829 setelah preprocessing
        - **Threshold aktivitas**: IC50 ≤ 1 μM (pIC50 ≥ 6.0) untuk kelas aktif
        - **Train / Val / Test**: 70% / 15% / 15%
        - **Class balance**: ~70% aktif, ~30% tidak aktif
          (di-handle dengan class weighting saat training)
        """
    )

    st.markdown('<div class="lab-section-h">how to interpret</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **P(active)**: probabilitas senyawa termasuk kelas aktif (0 - 1).
          Threshold default klasifikasi adalah 0.5.
        - **σ ensemble**: standar deviasi probabilitas antar 5 model.
          Nilai > 0.10 mengindikasikan disagreement antar model.
        - **Confidence**:
          - **High**: prediksi yakin (P jauh dari 0.5 dan σ rendah)
          - **Medium**: prediksi normal
          - **Low**: prediksi tidak yakin — verifikasi manual disarankan
        - **Out-of-domain warning**: muncul jika senyawa di luar rentang
          training (MW, jumlah atom, atau atom types). Prediksi mungkin
          kurang akurat untuk senyawa OOD.
        """
    )

    st.markdown('<div class="lab-section-h">limitations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. Model **hanya** memprediksi aktivitas terhadap BACE-1, bukan
           toksisitas, ADMET, atau target off-target.
        2. Prediksi bersifat **computational** dan tidak menggantikan uji
           eksperimental laboratorium.
        3. Senyawa di luar domain training (terutama dengan MW > 1000 Da
           atau atom langka) mungkin kurang akurat.
        4. **Scaffold split** memberikan estimasi performa yang lebih realistis
           untuk virtual screening senyawa baru, sementara **stratified split**
           cenderung over-optimistic. Untuk skenario drug discovery, percayalah
           pada metrik scaffold split.
        """
    )

    st.markdown('<div class="lab-section-h">credits</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **GIN**: Xu et al. (2019), "How Powerful are Graph Neural Networks?", ICLR
        - **GINE**: Hu et al. (2020), "Strategies for Pre-training Graph Neural Networks", ICLR
        - **Dataset**: ChEMBL Database, BACE-1 inhibitors (CHEMBL4822)
        - **Frameworks**: PyTorch, PyTorch Geometric, RDKit, Streamlit
        """
    )