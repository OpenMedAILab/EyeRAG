import copy
import json
import random
from collections import Counter, defaultdict

from eye_rag import LLM_ANSWER_TYPE
from eye_rag.utils import save_dict_to_json
from eye_rag.handle_results.cal_p import calculate_friedman_p_value
from eye_rag.ranking.llm_as_a_judge import JudgeBase, SingleJudgerRanking, members_to_group_ranking_data, \
    split_participant_name_split
from eye_rag.ranking.rank import SCORE_ITEM_NAMES


# ============================================
# Multi Judger Ranking
# ============================================
class MultiJudgerRanking(JudgeBase):
    """Multiple LLM judgers for ranking responses."""

    def __init__(
        self,
        json_response_dir,
        ranking_llm_names,
        question_ids,
        responding_llm_list,
        answer_types=LLM_ANSWER_TYPE,
        experiment_name='',
    ):
        super().__init__(
            json_response_dir=json_response_dir,
            question_ids=question_ids,
            responding_llm_list=responding_llm_list,
            answer_types=answer_types,
            experiment_name=experiment_name
        )
        if isinstance(ranking_llm_names, str):
            ranking_llm_names = [ranking_llm_names]
        self.ranking_llm_names = ranking_llm_names

    def get_ranking_list(self, questions_ids=None):
        """Get list of single judger rankers."""
        return [
            SingleJudgerRanking(
                json_response_dir=self.json_response_dir,
                ranking_llm_name=x,
                question_ids=self.question_ids if questions_ids is None else questions_ids,
                responding_llm_list=self.responding_llm_list,
                answer_types=self.answer_types,
                experiment_name=self.experiment_name,
            ) for x in self.ranking_llm_names
        ]

    def eval_evey_llm_group(self):
        """Evaluate every LLM group using all rankers."""
        group_ranking_results_combined = []
        all_participants = []
        ranker_list = self.get_ranking_list()

        for k, ranker in enumerate(ranker_list):
            group_ranking_results = ranker.eval_evey_llm_group()
            group_ranking_results_combined = self.flatten_group_ranking_results(
                group_ranking_results=group_ranking_results,
                group_ranking_results_combined=group_ranking_results_combined,
            )
            participants = copy.deepcopy(ranker.all_participants)
            for participant in participants:
                participant.group = f"{participant.group}_{k}"
            all_participants += participants

        participants_to_ranking_data(all_participants, exp_name="Combined Results", display_sort_by_avg_rank=True)

        dimension_ranking_results = calculate_dimension_based_avg_ranking(all_participants)
        dimension_ranking_file = self.get_save_file_path('combined_dimension_analysis.json')
        save_dict_to_json(out_file=dimension_ranking_file, dict_to_save=dimension_ranking_results)
        print(f"Dimension-based average ranking saved to: {dimension_ranking_file}")

        from eye_rag.handle_results.plot_radar_data import DimensionRadarPlot
        plot_tool = DimensionRadarPlot(dimension_ranking_file)
        plot_tool.plot_radar()

        out_file = self.get_save_file_path('.json')
        save_dict_to_json(out_file=out_file, dict_to_save=group_ranking_results_combined)
        print(f"Saved to: {out_file}")

        from eye_rag.handle_results.rank_result_plot import RankResultPlot
        plot_tool = RankResultPlot(filepath=out_file)
        plot_tool.plot_results()

    def performance_vs_num_questions(self):
        """Analyze performance vs number of questions."""
        friedman_test = []
        group_ranking_results_combined = []
        question_ids = copy.deepcopy(self.question_ids)
        random.shuffle(question_ids)

        for l in range(1, len(question_ids) + 1):
            q_ids = question_ids[:l]
            all_participants = []
            ranker_list = self.get_ranking_list(questions_ids=q_ids)

            friedman_test_result = []
            ranking = []

            for k, ranker in enumerate(ranker_list):
                groups = ranker.init_group_by_llm_base_name()
                ranking_results_single_rank = ranker.collect_complete_matching_result(groups=groups)
                final_ranking_single_rank = merge_group_ranking_data(ranking_results_single_rank)
                final_ranking_by_avg_single_rank = sorted(final_ranking_single_rank, key=lambda x: x['answer_type'])
                ranking.append([participant['average_ranking'] for participant in final_ranking_by_avg_single_rank])

                participants = copy.deepcopy(ranker.all_participants)
                statistic, p_value = calculate_friedman_p_for_participants(participants)
                friedman_test_result += [ranker.ranking_llm_name, statistic, p_value]

                for participant in participants:
                    participant.group = f"{participant.group}_{k}"
                all_participants += participants

            final_ranking = participants_to_ranking_data(all_participants)
            final_ranking_by_avg = sorted(final_ranking, key=lambda x: x['answer_type'])
            result = [l + 1]
            for participant in final_ranking_by_avg:
                result += [participant['answer_type'], participant['average_ranking']]
            group_ranking_results_combined.append(result)

            statistic, p_value = calculate_friedman_p_value(ranking)
            friedman_test_result += ['friedman_test_for_final_rank_per_ranker', statistic, p_value]

            ranks = [x['participant_rankings'] for x in final_ranking]
            ranks_transposed = list(zip(*ranks))
            statistic, p_value = calculate_friedman_p_value(ranks_transposed)
            friedman_test_result += ['response_llm_ranking_llm_as_independent_group', statistic, p_value]

            friedman_test.append(friedman_test_result)

        resulting_file = self.get_save_file_path('performance_vs_num_questions.txt')
        save_2d_list_to_txt(data_2d=group_ranking_results_combined, filepath=resulting_file)

        resulting_file = self.get_save_file_path('friedman_test_vs_num_questions.txt')
        save_2d_list_to_txt(data_2d=friedman_test, filepath=resulting_file)

    def ranked_first_ratio(self):
        """Calculate ratio of ranked first for each answer type."""
        result_all = []

        for response_llm_name in self.responding_llm_list:
            for ranking_llm_name in self.ranking_llm_names:
                for q_id in self.question_ids:
                    ranker = SingleJudgerRanking(
                        json_response_dir=self.json_response_dir,
                        ranking_llm_name=ranking_llm_name,
                        question_ids=[q_id],
                        responding_llm_list=[response_llm_name],
                        answer_types=self.answer_types,
                        experiment_name=self.experiment_name,
                    )
                    groups = ranker.init_group_by_llm_base_name()
                    ranker.collect_complete_matching_result(groups=groups)
                    final_ranking = participants_to_ranking_data(ranker.all_participants)
                    final_ranking_by_score = sorted(final_ranking, key=lambda x: x['total_score'], reverse=True)

                    for res in final_ranking_by_score:
                        if res['average_ranking'] == 1 and res['total_score'] > 0:
                            ranks = [q_id, ranking_llm_name, response_llm_name, res['answer_type'],
                                     res['total_score'], res['average_ranking']]
                            result_all.append(ranks)

        answer_type_counter = Counter([row[3] for row in result_all])
        total_questions = len(self.question_ids) * len(self.ranking_llm_names) * len(self.responding_llm_list)

        ratio_lines = []
        ratio_data = []
        for answer_type in self.answer_types:
            count = answer_type_counter.get(answer_type, 0)
            ratio = count / total_questions * 100 if total_questions > 0 else 0
            line = f"{answer_type}, {count}/{total_questions} = {ratio:.1f}%"
            print(line)
            ratio_lines.append(line)
            ratio_data.append([answer_type, ratio])

        ratio_file = self.get_save_file_path('ranked_first_ratio_summary.txt')
        with open(ratio_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(ratio_lines))
        print(f"Ranked first ratio saved to: {ratio_file}")

        resulting_file = self.get_save_file_path('ranked_first_ratio_detail.txt')
        save_2d_list_to_txt(data_2d=result_all, filepath=resulting_file)

        from eye_rag.handle_results.rank_result_plot import plot_ranked_first_ratio, LARGE_FONTSIZE
        plot_filepath = self.get_save_file_path('ranked_first_ratio.png')
        plot_ranked_first_ratio(ratio_data, save_path=plot_filepath, fontsize=LARGE_FONTSIZE)

    def collect_llm_group_eval(self,):
        group_ranking_results_combined = []
        ranker_list = self.get_ranking_list()
        for k, ranker in enumerate(ranker_list):
            group_ranking_results = ranker.eval_evey_llm_group()
            group_ranking_results_combined = self.flatten_group_ranking_results(
                group_ranking_results=group_ranking_results,
                group_ranking_results_combined=group_ranking_results_combined,
            )

        # save the result to json file
        filepath = self.get_save_file_path('multi_ranker_matching_result.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(group_ranking_results_combined, f, ensure_ascii=False, indent=4)
        print(f"Saved to: {filepath}")


# ============================================
# Ranking Data Processing Functions
# ============================================
def calculate_friedman_p_for_participants(all_participants):
    """Calculate Friedman p-value for participants."""
    final_ranking = participants_to_ranking_data(all_participants)
    ranks = [x['participant_rankings'] for x in final_ranking]
    ranks_transposed = list(zip(*ranks))
    statistic, p_value = calculate_friedman_p_value(ranks_transposed)
    return statistic, p_value


def participants_to_ranking_data(all_participants, exp_name='', display_sort_by_avg_rank=False, display_sort_by_score=False):
    """Convert participants to ranking data."""
    group_ranking_results = participants_to_group_ranking_results(all_participants)
    final_ranking = merge_group_ranking_data(
        group_ranking_results,
        exp_name=exp_name,
        display_sort_by_avg_rank=display_sort_by_avg_rank,
        display_sort_by_score=display_sort_by_score
    )
    return final_ranking


def participants_to_group_ranking_results(all_participants):
    """Convert participants to group ranking results format."""
    group_ranking_results = defaultdict(list)
    groups = defaultdict(list)

    for participant in all_participants:
        group_name = participant.group
        groups[group_name].append(participant)

    for group_name, members in groups.items():
        group_ranking_results[group_name] = members_to_group_ranking_data(
            members=members, ranking_llm_name='Combined', verbose=False
        )

    return group_ranking_results


def merge_group_ranking_data(group_ranking_results, exp_name='', display_sort_by_avg_rank=False, display_sort_by_score=False):
    """Merge group ranking data from multiple groups."""
    participant_scores = defaultdict(float)
    participant_groups = defaultdict(list)
    participant_rankings = defaultdict(list)

    for group_name, group_members in group_ranking_results.items():
        for member_data in group_members:
            current_rank = member_data['rank']
            score = member_data['score']
            _, answer_type = split_participant_name_split(member_data['name'])

            participant_groups[answer_type].append(group_name)
            participant_scores[answer_type] += score
            participant_rankings[answer_type].append(current_rank)

    final_ranking = []
    for answer_type, total_score in participant_scores.items():
        avg_ranking = sum(participant_rankings[answer_type]) / len(participant_rankings[answer_type])
        final_ranking.append({
            'answer_type': answer_type,
            'total_score': total_score,
            'average_ranking': avg_ranking,
            'groups_participated': participant_groups[answer_type],
            'participant_rankings': participant_rankings[answer_type],
            'num_groups': len(participant_groups[answer_type])
        })

    if display_sort_by_avg_rank:
        sort_by_avg_ranking(final_ranking, exp_name=exp_name)
    if display_sort_by_score:
        sort_by_total_score(final_ranking, exp_name=exp_name)

    return final_ranking


def sort_by_avg_ranking(final_ranking, exp_name=''):
    """Sort and display ranking by average ranking."""
    final_ranking_by_avg = sorted(final_ranking, key=lambda x: x['average_ranking'])
    print(f"\n----------\n {exp_name} Sorted by Average Ranking \n----------")

    for rank, participant in enumerate(final_ranking_by_avg, 1):
        print(f"  {rank:2d}. {participant['answer_type']} "
              f"(avg rank: {participant['average_ranking']:.2f}, "
              f"total score: {participant['total_score']:4.1f})")


def sort_by_total_score(final_ranking, exp_name=''):
    """Sort and display ranking by total score."""
    final_ranking_by_score = sorted(final_ranking, key=lambda x: x['total_score'], reverse=True)
    print(f"\n----------\n {exp_name} Sorted by Total Score \n---------")

    for rank, participant in enumerate(final_ranking_by_score, 1):
        print(f"  {rank:2d}. {participant['answer_type']} "
              f"(total score: {participant['total_score']:4.1f}, "
              f"avg rank: {participant['average_ranking']:.2f})")


def calculate_dimension_based_avg_ranking(all_participants):
    """Calculate dimension-based average ranking for all participants."""
    participant_dimension_avg_ranks = defaultdict(dict)

    for dimension in SCORE_ITEM_NAMES:
        groups_data = defaultdict(list)

        for participant in all_participants:
            groups_data[participant.group].append({
                'name': participant.name.replace(f"{participant.group}_", "").split("_", 1)[1],
                'score': participant.dimension_scores[dimension]
            })

        participant_ranks = defaultdict(list)

        for group_name, group_participants in groups_data.items():
            sorted_participants = sorted(group_participants, key=lambda x: x['score'], reverse=True)

            for i, participant_data in enumerate(sorted_participants):
                rank = i + 1
                if i > 0 and participant_data['score'] == sorted_participants[i - 1]['score']:
                    rank = sorted_participants[i - 1].get('rank', rank)
                participant_data['rank'] = rank
                participant_ranks[participant_data['name']].append(rank)

        for participant_name, ranks in participant_ranks.items():
            avg_rank = sum(ranks) / len(ranks)
            participant_dimension_avg_ranks[participant_name][dimension] = avg_rank

    return participant_dimension_avg_ranks


def save_2d_list_to_txt(data_2d, filepath, delimiter=',', header=None):
    """Save 2D list to text file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        if header:
            f.write(delimiter.join(map(str, header)) + '\n')
        for row in data_2d:
            f.write(delimiter.join(map(str, row)) + '\n')
    print(f"Data saved to: {filepath}")
