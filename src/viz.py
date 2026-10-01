"""노트북 공통 시각화 설정."""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BATCH_COLORS = {"b1": "#2a78d6", "b2": "#eb6834", "b3": "#1baf7a"}
BATCH_LABELS = {"b1": "Batch 1 (2017-05-12)", "b2": "Batch 2 (2018-02-20)", "b3": "Batch 3 (2018-04-12)"}

# 수명(크기) 표현용 단일 hue 순차 컬러맵: 밝음(단수명) → 어두움(장수명)
LIFE_CMAP = LinearSegmentedColormap.from_list("life", ["#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
# 상관계수 표현용 발산 컬러맵: 음(파랑) ↔ 0(회색) ↔ 양(빨강)
CORR_CMAP = LinearSegmentedColormap.from_list("corr", ["#1c5cab", "#f0efec", "#e34948"])

TEXT = "#3d3d3a"
MUTED = "#8a897f"
GRID = "#e6e5e0"


def set_style():
    plt.rcParams.update({
        "font.family": "Apple SD Gothic Neo",
        "axes.unicode_minus": False,
        "figure.figsize": (12, 5),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })
