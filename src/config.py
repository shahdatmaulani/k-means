import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


def configure_visuals():
    """Konfigurasi default untuk matplotlib & seaborn."""
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    sns.set_style("whitegrid")
    sns.set_palette("Set2")


# --- Konstanta Global ---
PREFERRED_COLUMNS = ["price", "width", "height"]
SESSION_KEYS = ["df_encoded", "timestamp", "current_file_key"]
DATA_DIR = "data"
DEFAULT_RANDOM_STATE = 42


# --- Konfigurasi Sidebar ---
SIDEBAR_OPTIONS = {
    "Z-Score": "cloud-upload",
    "Data Visual": "bar-chart"
}

SIDEBAR_STYLES = {
    "container": {"padding": "5px", "background-color": "#003366"},
    "icon": {"color": "white", "font-size": "25px"},
    "nav-link": {
        "font-size": "16px",
        "text-align": "left",
        "margin": "0px",
        "color": "white",
        "--hover-color": "#006699",
    },
    "nav-link-selected": {
        "background-color": "#005580",
    },
}
