import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import json

from eye_rag.handle_results.cal_p import calculate_friedman_p_value
from eye_rag.llm import official_name_mapping

FONTSIZE = 16
LARGE_FONTSIZE = 20

LLM_ANSWER_TYPE_MAPPING = {
    'LLM_Response': 'Vanilla LLM',
    'LLM_NaiveRAG_Response': 'Naive RAG',
    'LLM_HypotheticalRAG_Response': 'Hypothetical RAG',
    'LLM_HierarchicalIndexRAG_Response': 'Hierarchical Index RAG',
    "LLM_LightRAG_Hybrid_Distillation_Response": "EyeRAG",
}

DISPLAY_ORDER = [
    "EyeRAG",
    'Naive RAG',
    'Hierarchical Index RAG',
    'Hypothetical RAG',
    'Vanilla LLM',
]


def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def plot_ranking_per_group_by_llm(sub_df, title='', savefig_path=None, legend_outside=True):
    llms = sub_df['llm'].unique()
    width = 0.15
    plt.figure(figsize=(12, 7))

    all_names = sorted(sub_df['name'].unique())
    palette = sns.color_palette('Set2', n_colors=len(all_names))
    color_map = {name: palette[i] for i, name in enumerate(all_names)}

    group_centers = []
    for idx, llm in enumerate(llms):
        group = sub_df[sub_df['llm'] == llm][['name', 'rank']]
        group_sorted = group.sort_values('rank')
        names = group_sorted['name'].tolist()
        ranks = group_sorted['rank'].tolist()
        x_positions = [idx + width * i for i in range(len(names))]
        bars = plt.bar(
            x_positions,
            ranks,
            width,
            label=None
        )
        for bar, name in zip(bars, names):
            bar.set_color(color_map[name])
        if x_positions:
            group_centers.append(np.mean(x_positions))
        else:
            group_centers.append(idx)

    plt.ylabel('Rank (Lower is Better)', fontsize=FONTSIZE)
    plt.title(f'{title}', fontsize=FONTSIZE)

    ranking_llms_to_display = [get_official_llm_name(ranking_llm) for ranking_llm in llms]
    plt.xticks(group_centers, ranking_llms_to_display, rotation=30, ha='right', fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[name]) for name in all_names]
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if legend_outside:
        plt.legend(
            handles, all_names, fontsize=FONTSIZE,
            loc='upper left', bbox_to_anchor=(1.01, 1),
            ncol=1, frameon=False
        )
    else:
        plt.legend(handles, all_names, fontsize=FONTSIZE)
    plt.tight_layout()
    if savefig_path:
        plt.savefig(savefig_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# python
def calc_avg_ranking(df):
    """
    计算每个方法的平均排名与标准差。
    - 将 `rank` 强制为数值并丢弃无法转换的行。
    - 计算 mean 和 std，若 std 为 NaN（例如单样本），则用 0.0 代替。
    - 按 avg_rank 升序返回。
    """
    df = df.copy()
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df = df.dropna(subset=['rank'])

    stats = df.groupby('name')['rank'].agg(['mean', 'std']).rename(columns={'mean': 'avg_rank', 'std': 'std_rank'})
    stats['std_rank'] = stats['std_rank'].fillna(0.0)
    stats = stats.sort_values('avg_rank')
    return stats


def plot_ranked_first_ratio(
    data,
    figsize=(12, 8),
    save_path=None,
    dpi=300,
    fontsize=None,
    ylabel='Ranked First Rate',
    title='',
):
    """Plot bar chart for ranked first ratio."""
    if fontsize is None:
        fontsize = FONTSIZE

    data = [
        [LLM_ANSWER_TYPE_MAPPING.get(row[0], row[0])] + row[1:]
        for row in data
    ]
    data = sorted(data, key=lambda x: x[1], reverse=True)

    labels = [row[0] for row in data]
    values = [row[1] for row in data]

    x_pos = np.arange(len(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    plt.figure(figsize=figsize)

    bars = plt.bar(
        x_pos, values, color=colors,
        alpha=0.8, edgecolor='black', linewidth=1.5
    )
    plt.ylabel(ylabel, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)
    plt.xticks(x_pos, labels, fontsize=fontsize, rotation=30, ha='right')

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for bar, mean_val in zip(bars, values):
        height = bar.get_height()
        y = height + max(values) * 0.01
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            y,
            f'{mean_val:.2f}%',
            ha='center', va='bottom', fontsize=fontsize - 2
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_bar_comparison(
    data,
    figsize=(12, 8),
    save_path=None,
    dpi=300,
    fontsize=None,
    ylabel='Average Rank',
    title='',
    with_error_bar=True
):
    """Plot bar chart with optional error bars."""
    if fontsize is None:
        fontsize = FONTSIZE

    labels = [row[0] for row in data]
    means = [row[1] for row in data]
    stds = [row[2] for row in data]

    x_pos = np.arange(len(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    plt.figure(figsize=figsize)
    if with_error_bar:
        bars = plt.bar(
            x_pos, means, yerr=stds, capsize=8, color=colors,
            alpha=0.8, edgecolor='black', linewidth=1.5
        )
    else:
        bars = plt.bar(
            x_pos, means, color=colors,
            alpha=0.8, edgecolor='black', linewidth=1.5
        )

    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.xticks(x_pos, labels, fontsize=fontsize, rotation=30, ha='right')

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for bar, mean_val, std_val in zip(bars, means, stds):
        height = bar.get_height()
        y = height + (std_val if with_error_bar else 0) + max(means) * 0.01
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            y,
            f'{mean_val:.2f}±{std_val:.2f}',
            ha='center', va='bottom', fontsize=fontsize - 2
        )

    plt.subplots_adjust(top=0.78)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def change_method_name(df):
    df['name'] = df['name'].map(lambda x: LLM_ANSWER_TYPE_MAPPING[x] if x in LLM_ANSWER_TYPE_MAPPING else x)
    return df


def cal_p_value(df):
    pivot_table = df.pivot_table(index='llm', columns='name', values='rank')
    ranking_data = pivot_table.values.tolist()
    statistic, p_value = calculate_friedman_p_value(ranking_data)
    return statistic, p_value


def cal_p_value_independent_rank(df):
    pivot_table = df.pivot_table(index=['llm', 'ranking_llm'], columns='name', values='rank')
    ranking_data = pivot_table.values.tolist()
    statistic, p_value = calculate_friedman_p_value(ranking_data)
    return statistic, p_value


def get_official_llm_name(llm_name):
    return official_name_mapping[llm_name] if llm_name in official_name_mapping else llm_name


class RankResultPlot:
    def __init__(self, filepath):
        assert os.path.isfile(filepath)
        self.df = load_json_data(filepath)
        self.df = change_method_name(self.df)
        self.filepath = filepath

        self.result_dir = os.path.dirname(filepath)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        assert os.path.isdir(self.result_dir)

    @property
    def avg_rank(self):
        return calc_avg_ranking(self.df)

    def plot_results(self):
        self.analyze_avg_ranking_by_llm()
        self.plot_ranking_by_llm()
        self.analyze_avg_ranking()

    def plot_ranking_by_llm(self):
        ranking_llms = self.df['ranking_llm'].unique()
        for ranking_llm in ranking_llms:
            savefig_path = self.get_filepath_prefix() + f"_individual_ranking_by_{ranking_llm}.png"
            sub_df = self.df[self.df['ranking_llm'] == ranking_llm]

            friedman_stat, p_val = cal_p_value(sub_df)
            ranking_llm_to_display = get_official_llm_name(ranking_llm)
            plot_ranking_per_group_by_llm(
                sub_df,
                title=f'Ranking LLM: {ranking_llm_to_display}\n(Friedman Test Statistic: {friedman_stat:.2f}, P-value={p_val:.3f})',
                savefig_path=savefig_path)

    def analyze_avg_ranking(self):
        ranking_llms = self.df['ranking_llm'].unique()
        ranking_llms_to_display = [get_official_llm_name(ranking_llm) for ranking_llm in ranking_llms]
        ranking_llm_str = '_'.join(ranking_llms)
        ranking_llm_str_for_title = ', '.join(ranking_llms_to_display)
        avg_rank = calc_avg_ranking(self.df)
        print(f"Ranking LLM: {ranking_llms}, Average Ranking by Method: ")
        print(avg_rank)
        friedman_stat, p_val = cal_p_value(self.df)
        print(f"Average Rank: Friedman Test Statistic: {friedman_stat:.2f}")
        print(f"P-value: {p_val:.3f}")
        friedman_stat, p_val = cal_p_value_independent_rank(self.df)

        print(f"Independent Rank: Friedman Test Statistic: {friedman_stat:.2f}")
        print(f"P-value: {p_val:.3f}")
        P = f'={p_val:.1e}'
        savefig_path = self.get_filepath_prefix() + f"_combined_avg_ranking_by_{ranking_llm_str}_error_bar_plot.png"
        data = avg_rank.reset_index().values.tolist()
        plot_error_bar_comparison(
            data=data,
            title=f'Ranking LLM: {ranking_llm_str_for_title}\n(Friedman Test Statistic: {friedman_stat:.2f}, P-value {P})',
            save_path=savefig_path,
            with_error_bar=True,
            fontsize=LARGE_FONTSIZE,
        )

    def get_filepath_prefix(self):
        basename = os.path.basename(self.filepath)
        dirname = os.path.dirname(self.filepath)
        prefix = basename.split('_')[0]
        return os.path.join(dirname, prefix)

    def analyze_avg_ranking_by_llm(self):
        ranking_llms = self.df['ranking_llm'].unique()
        for ranking_llm in ranking_llms:
            sub_df = self.df[self.df['ranking_llm'] == ranking_llm]
            avg_rank = calc_avg_ranking(sub_df)
            friedman_stat, p_val = cal_p_value(sub_df)
            print(f"Ranking LLM: {ranking_llm}, Friedman Test Statistic: {friedman_stat:.2f}, P-value: {p_val:.3f}")

            savefig_path = self.get_filepath_prefix() + f"_avg_ranking_by_{ranking_llm}_error_bar_plot.png"
            data = avg_rank.reset_index().values.tolist()
            ranking_llm_to_display = get_official_llm_name(ranking_llm)

            plot_error_bar_comparison(
                data=data,
                title=f'Ranking LLM: {ranking_llm_to_display}\n(Friedman Test Statistic: {friedman_stat:.2f}, P-value={p_val:.3f})',
                save_path=savefig_path,
                with_error_bar=True,
                fontsize=FONTSIZE,
            )

            savefig_path = self.get_filepath_prefix() + f"_avg_ranking_by_{ranking_llm}_bar_plot.png"
            plot_error_bar_comparison(
                data=data,
                title=f'Ranking LLM: {ranking_llm_to_display}',
                save_path=savefig_path,
                with_error_bar=False,
            )


def load_and_plot_ranking_vs_num_questions(
    filepath, save_filepath=None, fontsize=FONTSIZE, draw_right_top_border=True
):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if save_filepath is None:
        save_filepath = filepath.replace('.txt', '.png')

    method_names = []
    data_dict = {}

    for line in lines:
        parts = line.split(',')
        number_of_question = int(parts[0])
        i = 1
        while i < len(parts) - 1:
            method = LLM_ANSWER_TYPE_MAPPING.get(parts[i], parts[i])
            rank = float(parts[i + 1])
            if method not in method_names:
                method_names.append(method)
            if method not in data_dict:
                data_dict[method] = []
            data_dict[method].append((number_of_question, rank))
            i += 2

    for method in data_dict:
        data_dict[method].sort(key=lambda x: x[0])

    plt.figure(figsize=(10, 6))
    custom_colors = ['#1f77b4', '#2ca02c', '#e377c2', '#ff7f0e', '#9467bd']
    color_map = {name: custom_colors[idx % len(custom_colors)] for idx, name in enumerate(method_names)}
    lines = []
    for method in method_names:
        x = [item[0] for item in data_dict[method]]
        y = [item[1] for item in data_dict[method]]
        line, = plt.plot(x, y, label=method, color=color_map[method], linewidth=2)
        lines.append(line)
    plt.xlabel('Number of Questions', fontsize=fontsize)
    plt.ylabel('Average Rank', fontsize=fontsize)
    plt.ylim(0, 6)
    plt.xlim(0, 270)

    ax = plt.gca()
    ax.spines['right'].set_visible(draw_right_top_border)
    ax.spines['top'].set_visible(draw_right_top_border)

    legend_names = [name for name in DISPLAY_ORDER if name in method_names]
    legend_handles = [
        plt.Line2D([0], [0], color=color_map[name], marker='o', linestyle='-', linewidth=2)
        for name in legend_names
    ]

    if draw_right_top_border:
        plt.legend(
            legend_handles, legend_names, fontsize=fontsize,
            loc='upper center', bbox_to_anchor=(0.5, 0.98),
            ncol=3, frameon=True
        )
    else:
        plt.legend(
            legend_handles, legend_names, fontsize=fontsize,
            loc='upper center', bbox_to_anchor=(0.5, 1.05),
            ncol=3, frameon=True
        )

    plt.tight_layout()
    if save_filepath:
        plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def load_and_plot_p_vs_num_questions(
    filepath, save_filepath=None, fontsize=16, use_log_scale=True, show_grid=False, show_threshold_line=True
):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if save_filepath is None:
        save_filepath = filepath.replace('.txt', '.png')

    group_names = [
        'Llama-3.3-70B-Instruct',
        'gpt-4o',
        'gemini-2.0-flash',
        'Average Rank',
        'Independent Rank'
    ]
    custom_colors = ['#1f77b4', '#2ca02c', '#e377c2', '#ff7f0e', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    p_values = {name: [] for name in group_names}
    num_questions = []

    for idx, line in enumerate(lines):
        parts = line.split(',')
        num_questions.append(idx + 1)
        p_values[group_names[0]].append(float(parts[2]))
        p_values[group_names[1]].append(float(parts[5]))
        p_values[group_names[2]].append(float(parts[8]))
        p_values[group_names[3]].append(float(parts[11]))
        p_values[group_names[4]].append(float(parts[14]))

    plt.figure(figsize=(10, 6))
    for i, name in enumerate(group_names):
        display_name = get_official_llm_name(name)
        plt.plot(
            num_questions, p_values[name],
            label=display_name, color=custom_colors[i], linewidth=1.8,
            marker=markers[i], markersize=5, markevery=max(1, len(num_questions)//15)
        )

    plt.xlabel('Number of Questions', fontsize=fontsize)
    plt.xlim(1, 270)
    if use_log_scale:
        plt.ylabel('Friedman Test: P-Value (log scale)', fontsize=fontsize)
        plt.yscale('log')
        plt.ylim(5e-8, 10)
        plt.yticks([1e-7, 0.00005, 0.0005, 0.005, 0.05, 0.5], ['1e-7', '0.00005', '0.0005', '0.005', '0.05', '0.5'])
        if show_threshold_line:
            plt.axhline(0.05, color='red', linestyle='--', linewidth=1.5)
    else:
        plt.ylabel('Friedman Test: P-Value', fontsize=fontsize)
        plt.ylim(0, 10)
        if show_threshold_line:
            plt.axhline(0.05, color='red', linestyle='--', linewidth=1.5)

    plt.legend(
        fontsize=fontsize,
        loc='upper center', bbox_to_anchor=(0.5, 1.01),
        ncol=2, frameon=True
    )

    plt.tight_layout()
    if show_grid:
        plt.grid(axis='y', alpha=0.3, linestyle='--', which='both' if use_log_scale else 'major')
    if save_filepath:
        plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
