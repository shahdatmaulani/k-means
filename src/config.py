import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

def configure_visuals():
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

# Konstanta global
PREFERRED_COLUMNS = ["price", "width", "height"]
SESSION_KEYS = ["df_encoded", "timestamp", "current_file_key"]