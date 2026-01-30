"""
Evaluation tools for EyeRAG responses using LLM-as-a-judge ranking.

This module provides utilities for evaluating and comparing LLM responses
using multiple judge LLMs for ranking.
"""
from typing import List

from eye_rag import init_patient_data
from eye_rag.ranking.multi_judge import MultiJudgerRanking
from eye_rag.qa.dialog import conduct_dialog_experiments
from eye_rag.qa.patient_data import get_question_ids_per_disease, get_question_ids_first_k


# Supported evaluation types
EVAL_TYPES = {
    'evey_llm_group': 'eval_evey_llm_group',
    'ranked_first_ratio': 'ranked_first_ratio',
    'collect_llm_group_eval': 'collect_llm_group_eval',
    'performance_vs_num_questions': 'performance_vs_num_questions',
}


class Analyze:
    """Analyze LLM responses with ranking evaluation."""

    # Can be overridden in subclasses
    ranker_class = MultiJudgerRanking
    eval_types = EVAL_TYPES

    def __init__(
            self,
            llm_answer_type: List[str],
            response_llm_names: List[str],
            json_response_dir: str,
            data_file: str = "Data/patient_data.xlsx",
    ):
        """
        Initialize the Analyze class.

        Args:
            llm_answer_type: List of answer types to evaluate
            response_llm_names: List of LLM names used for responses
            json_response_dir: Directory containing JSON response files
            data_file: Path to patient data file
        """
        self.loaded_patient_data = init_patient_data(data_file)
        self.json_response_dir = json_response_dir
        self.response_llm_names = response_llm_names
        self.llm_answer_type = llm_answer_type

    @staticmethod
    def get_question_ids(num_questions: int, start_id: int = None, fold: int = None) -> List[int]:
        """Get question IDs based on fold or start_id."""
        if fold is not None:
            return get_question_ids_per_disease(fold=fold, num_questions=num_questions)
        return get_question_ids_first_k(start_id=start_id or 1, num_questions=num_questions)

    def response_question(self, num_questions: int, start_id: int = None, fold: int = None):
        """Generate responses for questions."""
        question_ids = self.get_question_ids(num_questions, start_id, fold)
        self.answer_question(question_ids)

    def answer_question(self, question_ids: List[int]):
        """Conduct dialog experiments for given question IDs."""
        conduct_dialog_experiments(
            self.loaded_patient_data,
            result_dir=self.json_response_dir,
            llm_names=self.response_llm_names,
            question_ids=question_ids,
            llm_answer_types=self.llm_answer_type,
        )

    def eval_rank(
            self,
            question_ids: List[int],
            ranking_llm_names: List[str],
            experiment_name: str = '',
            eval_type: str = 'evey_llm_group',
    ):
        """
        Evaluate ranking using the configured ranker class.

        Args:
            question_ids: List of question IDs to evaluate
            ranking_llm_names: List of LLM names to use as judges
            experiment_name: Name suffix for experiment results
            eval_type: Type of evaluation to perform
        """
        if eval_type not in self.eval_types:
            supported = list(self.eval_types.keys())
            raise ValueError(f"Unsupported eval_type '{eval_type}'. Supported: {supported}")

        ranker = self.ranker_class(
            json_response_dir=self.json_response_dir,
            ranking_llm_names=ranking_llm_names,
            question_ids=question_ids,
            responding_llm_list=self.response_llm_names,
            answer_types=self.llm_answer_type,
            experiment_name=experiment_name,
        )

        method_name = self.eval_types[eval_type]
        getattr(ranker, method_name)()


def eval_rank(
        json_response_dir: str,
        compare_llm_answer_type: List[str],
        response_llm_names: List[str],
        question_ids: List[int],
        ranking_llm_names: List[str],
        exp_name: str = '',
        eval_type: str = 'evey_llm_group',
):
    """
    Convenience function for ranking evaluation.

    Args:
        json_response_dir: Directory containing JSON response files
        compare_llm_answer_type: Answer types to compare
        response_llm_names: LLM names used for responses
        question_ids: Question IDs to evaluate
        ranking_llm_names: LLM names to use as judges
        exp_name: Experiment name suffix
        eval_type: Type of evaluation
    """
    analyzer = Analyze(
        json_response_dir=json_response_dir,
        response_llm_names=response_llm_names,
        llm_answer_type=compare_llm_answer_type,
    )
    analyzer.eval_rank(
        question_ids=question_ids,
        ranking_llm_names=ranking_llm_names,
        experiment_name=exp_name or eval_type,
        eval_type=eval_type,
    )


def pairwise_eval_rank(
        out_dir: str,
        llm_answer_types: List[str],
        response_llm_names: List[str],
        question_ids: List[int],
        ranking_llm_names: List[str],
        exp_name: str = '',
):
    """Evaluate each answer type against LLM_Response baseline."""
    for answer_type in llm_answer_types:
        if answer_type == 'LLM_Response':
            continue
        eval_rank(
            json_response_dir=out_dir,
            compare_llm_answer_type=['LLM_Response', answer_type],
            response_llm_names=response_llm_names,
            question_ids=question_ids,
            ranking_llm_names=ranking_llm_names,
            exp_name=exp_name,
        )
