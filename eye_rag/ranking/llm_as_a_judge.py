"""
LLM-as-a-Judge evaluation module for comparing RAG responses.

This module provides classes and functions for:
- Single and multi-judger ranking of LLM responses
- Group-based evaluation and scoring
- Dimension-based analysis
"""
import inspect
import json
import os
import random
import shutil
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

from tqdm.contrib.concurrent import process_map

from config import LLM_ANSWER_TYPE
from eye_rag.qa.patient_data import get_clinical_data_for_query, get_question, get_response
from eye_rag.ranking.rank import (
    LLMRank,
    LLMParticipant,
    update_match_points_cur_question,
    rank_result_to_score,
    points_to_score,
    SCORE_ITEM_NAMES,
)
from eye_rag.utils import save_dict_to_json

# ============================================
# Constants
# ============================================
CPU_COUNT = max(1, int(os.cpu_count() / 2))
DEBUG = False


# ============================================
# Helper Functions
# ============================================
def execute_ranking_task(tasks):
    """Execute ranking tasks in parallel using multiprocessing."""
    random.shuffle(tasks)

    llm_models = [arg[-1] for arg in tasks]
    model_counts = Counter(llm_models)

    print("Tasks grouped by LLM model:")
    for model, count in model_counts.items():
        print(f"  {model}: {count} tasks")

    max_workers = min(CPU_COUNT, len(tasks))
    print(f"Using {max_workers} parallel workers for ranking tasks")
    optimal_chunksize = max(1, len(tasks) // (max_workers * 4))

    try:
        results = process_map(
            _execute_rank_static_wrapper,
            tasks,
            max_workers=max_workers,
            desc="Ranking",
            chunksize=optimal_chunksize
        )
        print(f"Completed {len(results)} ranking tasks")
    except Exception as e:
        print(f"Multiprocessing error: {e}")
        raise


def _execute_rank_static_wrapper(args):
    """Wrapper function for unpacking arguments."""
    return _rank_static(*args)


def _rank_static(question_id, question_content, clinical_data, llm_a_response, llm_b_response,
                 llm_a_obj, llm_b_obj, match_result_file, ranking_llm_name):
    """Execute a single ranking comparison between two LLM responses."""
    llm_rank = LLMRank(llm_name=ranking_llm_name)
    rank_result = llm_rank.ranking(question_content, clinical_data, llm_a_response, llm_b_response)

    if rank_result is not None:
        a_match_points, b_match_points, llm_a_rank_wins, llm_b_rank_wins, question_winner_str, detailed_score = (
            rank_result_to_score(llm_a_obj, llm_b_obj, rank_result)
        )

        match_detail = {
            'question_id': question_id,
            'llm_a': llm_a_obj.name,
            'llm_b': llm_b_obj.name,
            'a_match_points_cur_question': a_match_points,
            'b_match_points_cur_question': b_match_points,
            'llm_a_rank_wins': llm_a_rank_wins,
            'llm_b_rank_wins': llm_b_rank_wins,
            'question_winner_str': question_winner_str,
            'rank_results': rank_result,
            'ranking_llm_name': ranking_llm_name,
            'dimension_scores': detailed_score
        }
        save_dict_to_json(dict_to_save=match_detail, out_file=match_result_file)
        return match_detail
    else:
        print(f"Warning: Ranking failed for question {question_id} between {llm_a_obj.name} and {llm_b_obj.name}")
        return None


def generate_result_dir(ranking_result_dir, experiment_name=''):
    """Generate experiment result directory."""
    result_dir = os.path.join(ranking_result_dir, experiment_name) if experiment_name else ranking_result_dir
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def members_to_group_ranking_data(members, ranking_llm_name='', verbose=True):
    """Convert group members to ranking data format."""
    sorted_members = sorted(members, key=lambda x: x.score, reverse=True)
    group_ranking_data = []
    current_rank = 1

    for i, member_data in enumerate(sorted_members):
        if i > 0 and sorted_members[i].score < sorted_members[i - 1].score:
            current_rank = i + 1

        group_ranking_data.append({
            "name": member_data.name,
            "score": member_data.score,
            'ranking_llm': ranking_llm_name,
            'rank': current_rank
        })
        if verbose:
            print(f"\t Rank {current_rank}: {member_data.name} (score: {member_data.score})")

    return group_ranking_data


def split_participant_name_split(participant_name):
    """Split participant name into base name and answer type."""
    result = participant_name.split('_', 1)
    assert len(result) == 2
    base_name, answer_type = result
    return base_name, answer_type


def to_participant_name(base_name, answer_type):
    """Create participant name from base name and answer type."""
    return f"{base_name}_{answer_type}"


def load_llm_query_and_response(input_filepath):
    """Load LLM query and response from JSON file."""
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            patient_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_filepath}. Skipping.")
        return None, None, None
    except Exception as e:
        print(f"Error processing {input_filepath}: {e}")
        return None, None, None

    if 'clinical_data' in patient_data:
        clinical_data = patient_data['clinical_data']
    else:
        clinical_data = get_clinical_data_for_query(patient_data)

    patient_question = get_question(patient_data)
    llm_answer_text = get_response(patient_data)
    return clinical_data, patient_question, llm_answer_text


# ============================================
# Base Judge Class
# ============================================
class JudgeBase:
    """Base class for LLM-as-a-Judge evaluation."""

    def __init__(
        self,
        json_response_dir,
        question_ids,
        responding_llm_list,
        answer_types=LLM_ANSWER_TYPE,
        experiment_name='',
    ):
        self.json_response_dir = json_response_dir
        self.responding_llm_list = responding_llm_list
        self.answer_types = answer_types
        self.question_ids = question_ids
        self.experiment_name = experiment_name

        self.participant_failed = []
        self.ranking_result_dir = json_response_dir + '_match_results'
        self.problem_response_dir = os.path.join(self.json_response_dir, 'PROBLEM_RESPONSE')
        # os.makedirs(self.problem_response_dir, exist_ok=True)

        self.result_dir = generate_result_dir(
            ranking_result_dir=self.ranking_result_dir,
            experiment_name=experiment_name
        )

    @property
    def NUM_QUESTIONS(self):
        return len(self.question_ids)

    def get_save_file_path(self, suffix):
        """Get save file path with suffix."""
        if hasattr(self, 'ranking_llm_name'):
            ranking_llm_name = self.ranking_llm_name
        elif hasattr(self, 'ranking_llm_names'):
            ranking_llm_name = '_'.join(self.ranking_llm_names)
        else:
            raise NotImplementedError

        result = os.path.join(self.result_dir, f'QA{self.NUM_QUESTIONS}_{ranking_llm_name}')
        return f'{result}_{suffix}' if suffix else result

    def flatten_group_ranking_results(self, group_ranking_results, group_ranking_results_combined=None):
        """Flatten group ranking results into a single list."""
        if group_ranking_results_combined is None:
            group_ranking_results_combined = []

        for key, value in group_ranking_results.items():
            for res in value:
                res.update({"llm": key, 'num_question': self.NUM_QUESTIONS})
                res['name'] = res['name'].replace(f'{key}_', '')
                group_ranking_results_combined.append(res)

        return group_ranking_results_combined


# ============================================
# Single Judger Ranking
# ============================================
class SingleJudgerRanking(JudgeBase):
    """Single LLM judger for ranking responses."""

    def __init__(
        self,
        json_response_dir,
        ranking_llm_name,
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
        self.ranking_llm_name = ranking_llm_name
        self.llm_rank = LLMRank(llm_name=ranking_llm_name)
        self.all_participants = self._init_all_participants()

    def _init_all_participants(self):
        """Initialize all participants."""
        all_participants = []
        for base in self.responding_llm_list:
            for r_type in self.answer_types:
                all_participants.append(LLMParticipant(name=to_participant_name(base, r_type)))
        return all_participants

    def init_group_by_llm_base_name(self):
        """Group participants by LLM base name."""
        groups = defaultdict(list)
        for participant in self.all_participants:
            base_name, _ = split_participant_name_split(participant.name)
            groups[base_name].append(participant)
            participant.group = base_name
        return groups

    def init_group_by_method(self, num_member_per_group=None):
        """Group participants by answer method/type."""
        temp_groups = defaultdict(list)
        for participant in self.all_participants:
            base_name = participant.name.replace(participant.name.split('_')[0], '').replace('_', '', 1)
            temp_groups[base_name].append(participant)

        if num_member_per_group is None:
            final_groups = defaultdict(list)
            for group_name, members in temp_groups.items():
                final_groups[group_name] = members
                for participant in members:
                    participant.group = group_name
            return final_groups

        # Split large groups into subgroups
        final_groups = defaultdict(list)
        for group_name, members in temp_groups.items():
            if len(members) <= num_member_per_group:
                final_groups[group_name] = members
                for participant in members:
                    participant.group = group_name
            else:
                num_subgroups = (len(members) + num_member_per_group - 1) // num_member_per_group
                for i in range(num_subgroups):
                    start_idx = i * num_member_per_group
                    end_idx = min((i + 1) * num_member_per_group, len(members))
                    subgroup_members = members[start_idx:end_idx]
                    subgroup_name = f"{group_name}_SubGroup{i + 1}"
                    final_groups[subgroup_name] = subgroup_members
                    for participant in subgroup_members:
                        participant.group = subgroup_name

        return final_groups

    def get_ranking_result_json_data(self, question_id, llm_obj: LLMParticipant):
        """Load ranking result JSON data for a question and LLM."""
        filename = f'{question_id}_{llm_obj.name}.json'
        response_file_path = os.path.join(self.json_response_dir, filename)

        if not os.path.isfile(response_file_path):
            print(f"No file {response_file_path}")
            return None, None, None

        clinical_data, patient_question, llm_answer_text = load_llm_query_and_response(response_file_path)

        if not (llm_answer_text and clinical_data and patient_question):
            if not llm_answer_text:
                print(f"No response in {response_file_path}")
            if not clinical_data:
                print(f"No clinical data in {response_file_path}")
            if not patient_question:
                print(f"No patient question in {response_file_path}")

            tar_file = os.path.join(self.problem_response_dir, filename)
            print(f"Move problem response file {response_file_path} to {tar_file}")
            shutil.move(response_file_path, tar_file)

        return clinical_data, patient_question, llm_answer_text

    def batch_load_ranking_result_json_data(self, question_ids, llm_objs):
        """Batch load ranking result JSON data."""
        results = {}
        for question_id in question_ids:
            for llm_obj in llm_objs:
                clinical_data, patient_question, llm_answer_text = self.get_ranking_result_json_data(
                    question_id, llm_obj
                )
                if llm_answer_text is not None and clinical_data is not None and patient_question is not None:
                    results[(question_id, llm_obj.name)] = {
                        'clinical_data': clinical_data,
                        'patient_question': patient_question,
                        'llm_answer_text': llm_answer_text
                    }
        return results

    def _collect_matching_task(self, llm_a_obj: LLMParticipant, llm_b_obj: LLMParticipant):
        """Collect matching tasks for two LLMs."""
        tasks = []
        all_data = self.batch_load_ranking_result_json_data(self.question_ids, [llm_a_obj, llm_b_obj])

        out_dir = os.path.join(self.ranking_result_dir, self.llm_rank.llm_name)
        os.makedirs(out_dir, exist_ok=True)

        for question_id in self.question_ids:
            sorted_llm_names = sorted([llm_a_obj.name, llm_b_obj.name])
            filename = f"{question_id}_{sorted_llm_names[0]}|{sorted_llm_names[1]}.json"
            match_result_file = os.path.join(out_dir, filename)

            if not os.path.isfile(match_result_file):
                llm_a_data = all_data.get((question_id, llm_a_obj.name))
                llm_b_data = all_data.get((question_id, llm_b_obj.name))

                if llm_a_data and llm_b_data:
                    clinical_data = llm_a_data['clinical_data']
                    question_content = llm_a_data['patient_question']
                    llm_a_response = llm_a_data['llm_answer_text']
                    llm_b_response = llm_b_data['llm_answer_text']

                    tasks.append((
                        question_id, question_content, clinical_data,
                        llm_a_response, llm_b_response,
                        llm_a_obj, llm_b_obj, match_result_file, self.ranking_llm_name
                    ))

        return tasks

    def collect_single_group_matching_task(self, members):
        """Collect matching tasks for a single group."""
        all_tasks = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                tasks = self._collect_matching_task(members[i], members[j])
                all_tasks += tasks
        return all_tasks

    def collect_complete_matching_task(self, groups):
        """Collect all matching tasks for all groups in parallel."""
        all_tasks = []

        def collect_group_tasks(group_data):
            _, members = group_data
            return self.collect_single_group_matching_task(members)

        with ThreadPoolExecutor(max_workers=min(4, len(groups))) as executor:
            results = executor.map(collect_group_tasks, groups.items())
            for tasks in results:
                all_tasks.extend(tasks)

        return all_tasks

    def _conduct_matching(self, all_tasks):
        """Execute matching tasks."""
        if len(all_tasks) > 0:
            if not DEBUG:
                execute_ranking_task(tasks=all_tasks)
            else:
                for k, task in enumerate(all_tasks):
                    (question_id, question_content, clinical_data,
                     llm_a_response, llm_b_response,
                     llm_a_obj, llm_b_obj, match_result_file, ranking_llm_name) = task
                    print(f"| {k + 1}/{len(all_tasks)}, processing match task {match_result_file}")
                    _rank_static(
                        question_id, question_content, clinical_data,
                        llm_a_response, llm_b_response,
                        llm_a_obj, llm_b_obj, match_result_file,
                        self.ranking_llm_name
                    )

    def conduct_single_group_matching(self, members):
        """Conduct matching for a single group."""
        all_tasks = self.collect_single_group_matching_task(members)
        self._conduct_matching(all_tasks)

    def conduct_multi_group_matching(self, groups):
        """Conduct matching for multiple groups."""
        all_tasks = self.collect_complete_matching_task(groups)
        self._conduct_matching(all_tasks)

    def advance_members_in_group(self, members, top_k, remove_failed=False):
        """Advance top members from a group."""
        members.sort(key=lambda x: x.score, reverse=True)
        advanced_members = members[:top_k]

        for rank, member in enumerate(members):
            print(f"    {rank + 1}. {member.name} (score: {member.score})")
        print(f"Advanced members: {[m.name for m in advanced_members]}")

        if remove_failed:
            failed_members = members[top_k:]
            for failed_member in failed_members:
                self.participant_failed.append(failed_member)
                if failed_member in self.all_participants:
                    self.all_participants.remove(failed_member)
            print(f"Failed members: {[m.name for m in failed_members]}")

        return advanced_members

    def get_advanced_group_members(self, groups_matched, top_k, remove_failed=True):
        """Get advanced members from matched groups."""
        groups_advanced = []
        for k, (group_name, members) in enumerate(groups_matched.items()):
            print(f"\nRanking LLM: {self.ranking_llm_name}\n Group {k}: {group_name} (members: {[m.name for m in members]})")
            top_k = len(members) if top_k is None else top_k
            advanced_members = self.advance_members_in_group(members=members, top_k=top_k, remove_failed=remove_failed)
            groups_advanced.append(advanced_members)
        return groups_advanced

    def save_group_ranking_results(self, group_ranking_results, save_suffix=''):
        """Save group ranking results to JSON file."""
        group_ranking_file = self.get_save_file_path(f"{save_suffix}.json")
        with open(group_ranking_file, 'w', encoding='utf-8') as f:
            json.dump(group_ranking_results, f, indent=4, ensure_ascii=False)

    def _analyze_matching_result(self, llm_a_obj: LLMParticipant, llm_b_obj: LLMParticipant):
        """Analyze matching results between two LLMs."""
        a_match_points = 0
        b_match_points = 0

        out_dir = os.path.join(self.ranking_result_dir, self.llm_rank.llm_name)

        a_total_dimension_scores = {dimension: 0 for dimension in SCORE_ITEM_NAMES}
        b_total_dimension_scores = {dimension: 0 for dimension in SCORE_ITEM_NAMES}

        def process_questions():
            for q_id in self.question_ids:
                sorted_llm_names = sorted([llm_a_obj.name, llm_b_obj.name])
                filename = f"{q_id}_{sorted_llm_names[0]}|{sorted_llm_names[1]}.json"
                result_file = os.path.join(out_dir, filename)
                if os.path.isfile(result_file):
                    yield q_id, result_file
                else:
                    print(f'Error: File {os.path.abspath(result_file)} not exist, ignore it ...')

        for question_id, match_result_file in process_questions():
            try:
                with open(match_result_file, 'r', encoding='utf-8') as f:
                    match_detail = json.load(f)
            except Exception as e:
                print(f"Error reading file: {os.path.abspath(match_result_file)}")
                print(f"Exception: {e}")
                if os.path.isfile(match_result_file):
                    os.remove(match_result_file)
                    print(f"Deleted problematic file: {os.path.abspath(match_result_file)}")
                continue

            rank_result = update_match_points_cur_question(match_detail)
            if rank_result is None:
                print(f"Error in ranking result in {match_result_file}, skip this match.")
                continue

            (a_match_points_cur_question, b_match_points_cur_question, _, _, _, dimension_scores) = (
                rank_result_to_score(llm_a_obj, llm_b_obj, rank_result)
            )

            if 'llm_a' not in match_detail or 'llm_b' not in match_detail:
                print(f"Warning: Missing 'llm_a' or 'llm_b' in {match_result_file}, skip this match.")

            assert {match_detail['llm_a'], match_detail['llm_b']} == {llm_a_obj.name, llm_b_obj.name}, \
                f"{match_result_file}\nLLM names in match detail do not match with participants!"

            if dimension_scores:
                if match_detail['llm_a'] == llm_a_obj.name:
                    a_dim_scores = dimension_scores['llm_a']
                    b_dim_scores = dimension_scores['llm_b']
                else:
                    a_dim_scores = dimension_scores['llm_b']
                    b_dim_scores = dimension_scores['llm_a']

                for dimension in SCORE_ITEM_NAMES:
                    a_total_dimension_scores[dimension] += a_dim_scores[dimension]
                    b_total_dimension_scores[dimension] += b_dim_scores[dimension]

                llm_a_obj.add_dimension_scores(a_dim_scores, question_id, llm_b_obj.name)
                llm_b_obj.add_dimension_scores(b_dim_scores, question_id, llm_a_obj.name)

            if match_detail['llm_a'] != llm_a_obj.name:
                match_detail['a_match_points_cur_question'] = b_match_points_cur_question
                match_detail['b_match_points_cur_question'] = a_match_points_cur_question
            else:
                match_detail['a_match_points_cur_question'] = a_match_points_cur_question
                match_detail['b_match_points_cur_question'] = b_match_points_cur_question

            score1, score2 = points_to_score(
                match_detail['a_match_points_cur_question'],
                match_detail['b_match_points_cur_question']
            )
            a_match_points += score1
            b_match_points += score2

        llm_a_obj.add_match_score(a_match_points)
        llm_b_obj.add_match_score(b_match_points)

    def collect_single_group_matching_result(self, members):
        """Collect matching results for a single group."""
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                self._analyze_matching_result(members[i], members[j])

        group_ranking_data = members_to_group_ranking_data(
            members=members, ranking_llm_name=self.ranking_llm_name, verbose=True
        )
        return group_ranking_data

    def group_stage_match(self, groups, skip_matching=False, save_suffix=''):
        """Conduct group stage matching."""
        print("\n--- Group Stage Match ---")
        group_ranking_results = {}

        for k, (group_name, members) in enumerate(groups.items()):
            print(f"Ranking LLM: {self.ranking_llm_name}")
            print(f"Group {k}: {group_name} (members: {[m.name for m in members]})")

            if not skip_matching:
                self.conduct_single_group_matching(members=members)

            group_ranking_data = self.collect_single_group_matching_result(members)
            group_ranking_results[group_name] = group_ranking_data

        if save_suffix:
            self.save_group_ranking_results(group_ranking_results, save_suffix=save_suffix)

        return group_ranking_results

    def collect_complete_matching_result(self, groups):
        """Collect complete matching results (skip matching, only collect)."""
        return self.group_stage_match(groups, skip_matching=True)

    def merge_group_results_and_save_final_ranking(self, group_ranking_results):
        """Merge group results and save final ranking."""
        # Lazy import to avoid circular dependency
        from eye_rag.ranking.multi_judge import merge_group_ranking_data
        exp_name = f"Ranking LLM: {self.ranking_llm_name}, Final Ranking (Number of Questions = {self.NUM_QUESTIONS})"
        merge_group_ranking_data(
            group_ranking_results,
            display_sort_by_avg_rank=True,
            display_sort_by_score=False,
            exp_name=exp_name
        )

    def eval_evey_llm_group(self, skip_matching=False):
        """Evaluate every LLM group."""
        func_name = inspect.currentframe().f_code.co_name
        groups = self.init_group_by_llm_base_name()
        group_ranking_results = self.group_stage_match(
            groups=groups, skip_matching=skip_matching, save_suffix=f'{func_name}'
        )
        self.merge_group_results_and_save_final_ranking(group_ranking_results)
        return group_ranking_results





