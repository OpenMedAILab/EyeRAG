from math import pi
import os
from matplotlib import pyplot as plt
import json

from eye_rag.handle_results.rank_result_plot import LLM_ANSWER_TYPE_MAPPING, DISPLAY_ORDER, LARGE_FONTSIZE

PREDEFINED_COLORS = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#FF7F00",  # orange
    "#984EA3",  # purple
    "#FFD92F",  # yellow
    "#A65628",  # brown
    "#F781BF",  # pink
    "#999999",  # gray
]


def make_radar_plot(data, dimensions=None, title=None, savefig_path=None, figsize=(10, 10),
                    show_legend=True, legend_fontsize=LARGE_FONTSIZE, colors=None, line_width=2,
                    dimension_fontsize=LARGE_FONTSIZE):
    """
    Create a radar plot.

    Args:
        data: Dictionary format, key is method name, value is list of dimension scores
              e.g., {'Method1': [4.2, 3.8, 4.1, 3.9, 4.0], 'Method2': [...]}
        dimensions: List of dimension names. Supports '\n' for multi-line display.
        title: Chart title
        savefig_path: Save path
        figsize: Figure size
        show_legend: Whether to show legend
        legend_fontsize: Legend font size
        colors: Color list. If None, auto-generated.
        line_width: Line width
        dimension_fontsize: Dimension label font size
    """
    methods = DISPLAY_ORDER

    if dimensions is None:
        dimensions = [f"Dimension_{i + 1}" for i in range(len(data[methods[0]]))]
    else:
        if not data:
            print("Warning: data is empty, cannot draw radar plot")
            return

        representative_method = None
        for m in methods:
            if m in data:
                representative_method = m
                break
        if representative_method is None:
            representative_method = next(iter(data))
            print(f"Warning: Methods in DISPLAY_ORDER not found in data, using '{representative_method}' as baseline. Available methods: {list(data.keys())}")

        expected_len = len(data[representative_method])
        if len(dimensions) != expected_len:
            print(f"Warning: Dimension count ({len(dimensions)}) does not match data dimensions ({expected_len})")
            dimensions = dimensions[:expected_len]

    angles = [n / float(len(dimensions)) * 2 * pi for n in range(len(dimensions))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    if colors is None:
        colors = PREDEFINED_COLORS[:len(methods)]

    for i, (method, scores) in enumerate(data.items()):
        scores_closed = scores + scores[:1]
        ax.plot(angles, scores_closed, 'o-', linewidth=line_width,
                label=method, color=colors[i], alpha=0.8)

    ax.set_xticks(angles[:-1])

    for i, (angle, dimension) in enumerate(zip(angles[:-1], dimensions)):
        if angle == 0:
            ha, va = 'left', 'center'
        elif 0 < angle < pi / 2:
            ha, va = 'left', 'bottom'
        elif angle == pi / 2:
            ha, va = 'center', 'bottom'
        elif pi / 2 < angle < pi:
            ha, va = 'right', 'bottom'
        elif angle == pi:
            ha, va = 'right', 'center'
        elif pi < angle < 3 * pi / 2:
            ha, va = 'right', 'top'
        elif angle == 3 * pi / 2:
            ha, va = 'center', 'top'
        else:
            ha, va = 'left', 'top'

        ax.text(angle, ax.get_ylim()[1] * 0.99, dimension,
                horizontalalignment=ha, verticalalignment=va,
                fontsize=dimension_fontsize, fontweight='bold',
                multialignment='center')

    ax.set_xticklabels([])

    all_values = [val for scores in data.values() for val in scores]
    max_val = max(all_values)

    ax.set_ylim(0, max_val + 0.05)
    yticks = list(range(1, int(max_val) + 1))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(x) for x in yticks], fontsize=LARGE_FONTSIZE)

    if title:
        ax.set_title(title, size=LARGE_FONTSIZE, fontweight='bold', pad=40)
    ax.grid(True, alpha=0.3)

    if show_legend:
        plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.22),
            fontsize=legend_fontsize,
            frameon=True,
            ncol=2
        )

    plt.tight_layout()

    if savefig_path:
        plt.savefig(savefig_path, dpi=300, bbox_inches='tight')
        print(f"Radar plot saved to: {savefig_path}")
        plt.close()
    else:
        plt.show()


def get_filepath_prefix(filepath):
    basename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)
    prefix = basename.split('_')[0]
    return os.path.join(dirname, prefix)


class DimensionRadarPlot:
    """Dimension radar plot class."""

    def __init__(self, json_filepath, dimension_names=None, name_mapping=LLM_ANSWER_TYPE_MAPPING):
        self.json_filepath = json_filepath
        self.dimension_names = dimension_names or ["Clinical Accuracy", "Completeness", "Logical Consistency", "Relevance", "Clarity"]
        self.name_mapping = name_mapping or {}
        self.data = None

    def plot_radar(self, title='Average Rank (Lower is Better)',
                   savefig_path=None, dimension_fontsize=LARGE_FONTSIZE,
                   figsize=(12, 12), show_legend=True, legend_fontsize=10):
        """Draw dimension radar plot."""
        with open(self.json_filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        radar_data = {}
        for method_name, dimensions in json_data.items():
            clean_name = LLM_ANSWER_TYPE_MAPPING.get(method_name, method_name)
            ranking_values = list(dimensions.values())
            score_values = ranking_values
            radar_data[clean_name] = score_values

        display_dimension_name_dict = {
            "Clinical Accuracy and Safety": "Clinical\nAccuracy\nand Safety",
            "Patient-Centered Response": "Patient-Centered Response",
            "Professional Communication and Clarity": "Professional\nCommunication\nand Clarity",
            "Completeness and Practical Applicability": "Completeness\nand\nPractical\nApplicability",
            "Patient-Readiness": "Patient-Readiness",
        }
        display_dimension_names = [display_dimension_name_dict[dim] for dim in dimensions.keys()]

        max_score = max([max(scores) for scores in radar_data.values()])

        savefig_path = savefig_path if savefig_path else get_filepath_prefix(self.json_filepath) + '_radar_avg_ranking_plot.png'
        os.makedirs(os.path.dirname(savefig_path), exist_ok=True)

        make_radar_plot(
            data=radar_data,
            dimensions=display_dimension_names,
            title=title,
            dimension_fontsize=dimension_fontsize,
            savefig_path=savefig_path,
            figsize=(12, 12)
        )

        print(f"Data range: 1 to {max_score}")
        print(f"Radar plot generated: {os.path.abspath(savefig_path)}")
        return radar_data
