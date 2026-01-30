"""
LLM-as-a-Judge Pairwise Ranking Module.

This module implements pairwise comparison of LLM responses using another LLM as a judge.
It uses a two-pass evaluation (original and swapped order) to mitigate positional bias.

Scoring System:
- Win: 3 points
- Draw: 1 point each
- Loss: 0 points
"""

import json
from typing import Dict, List, Literal, Optional, Tuple, Any

from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

from eye_rag.llm import get_chat_llm


# Evaluation dimensions for medical response quality
SCORE_DIMENSIONS = [
    "Clinical Accuracy and Safety",
    "Patient-Centered Response",
    "Professional Communication and Clarity",
    "Completeness and Practical Applicability",
    "Patient-Readiness",
]

SCORING_CRITERIA = {
    "Clinical Accuracy and Safety": (
        "Assesses whether the information is medically accurate, factually correct, "
        "and aligned with current clinical guidelines, while ensuring patient safety "
        "and preventing harm."
    ),
    "Patient-Centered Response": (
        "Evaluates how well the response is tailored to the individual patient's "
        "specific concerns, clinical context, and needs."
    ),
    "Professional Communication and Clarity": (
        "Assesses the use of appropriate medical terminology, tone, and structure "
        "to ensure the answer is clear, professional, and easy to understand."
    ),
    "Completeness and Practical Applicability": (
        "Measures whether the response fully addresses all aspects of the patient's "
        "question and provides actionable, realistic clinical recommendations."
    ),
    "Patient-Readiness": (
        "Evaluates whether the response is appropriate for direct communication "
        "with a patient, considering tone, emotional sensitivity, and readability."
    ),
}

# Prompt template for pairwise comparison
COMPARISON_PROMPT = """Please compare the two anonymized responses (Response A and Response B) to the patient's question.
Your evaluation should focus solely on the "{scoring_dimension}" aspect.
Based on the scoring criteria for this aspect, determine which response is better (A or B) or if both are equivalent (E).

---
Scoring Criterion and Reference Standard for '{scoring_dimension}': {scoring_criteria}.
---
Patient Question: {question_content}
---
Response A: {llm_a_response}
---
Response B: {llm_b_response}
"""

_ranking_prompt = PromptTemplate(
    template=COMPARISON_PROMPT,
    input_variables=[
        "question_content", "clinical_data", "llm_a_response",
        "llm_b_response", "scoring_dimension", "scoring_criteria"
    ],
)


class RankingResult(BaseModel):
    """Schema for LLM ranking output."""
    choice: Literal['A', 'B', 'E'] = Field(
        description="'A' if Response A is better, 'B' if Response B is better, 'E' if equivalent."
    )


def aggregate_choices(choice1: str, choice2: str) -> str:
    """
    Aggregate two evaluation passes into a final choice.

    Uses a two-pass evaluation to mitigate positional bias:
    - If both passes agree, use that choice
    - If passes conflict (A vs B), treat as equivalent (E)
    - If one pass is E, use the non-E choice (non-conservative approach)

    Args:
        choice1: Normalized choice from pass 1 ('A', 'B', 'E', or 'Error')
        choice2: Normalized choice from pass 2 ('A', 'B', 'E', or 'Error')

    Returns:
        Final aggregated choice ('A', 'B', 'E', or 'Error')
    """
    if 'Error' in (choice1, choice2):
        return 'Error'

    if choice1 == choice2:
        return choice1

    # Conflicting choices (A vs B) indicate positional bias - treat as equivalent
    if {choice1, choice2} == {'A', 'B'}:
        return 'E'

    # One choice is E, the other is A or B - use the non-E choice
    return choice1 if choice2 == 'E' else choice2


def update_ranking_results(match_detail: Dict) -> Optional[Dict]:
    """
    Update ranking results by recalculating final choices.

    Args:
        match_detail: Dictionary containing 'rank_results' key

    Returns:
        Updated rank results dictionary, or None if error encountered
    """
    rank_result = match_detail["rank_results"]
    if isinstance(rank_result, str):
        import ast
        rank_result = ast.literal_eval(rank_result)

    for dimension in SCORE_DIMENSIONS:
        item = rank_result[dimension]
        choice1, choice2 = item['choice1'], item['choice2']
        final_choice = aggregate_choices(choice1, choice2)

        if final_choice != item['final_choice']:
            print(f"Updated {dimension}: {item['final_choice']} -> {final_choice}")

        item['final_choice'] = final_choice

        if final_choice == 'Error':
            print(f"Error in ranking for {dimension}, stopping processing.")
            return None

    return rank_result


def calculate_match_scores(
    llm_a: 'LLMParticipant',
    llm_b: 'LLMParticipant',
    rank_result: Dict[str, Any]
) -> Tuple[int, int, int, int, str, Dict]:
    """
    Calculate match scores from ranking results.

    Scoring: Win=3, Draw=1, Loss=0 per dimension.

    Args:
        llm_a: First LLM participant
        llm_b: Second LLM participant
        rank_result: Ranking results dictionary

    Returns:
        Tuple of (a_points, b_points, a_wins, b_wins, winner_name, detailed_scores)
    """
    if isinstance(rank_result, str):
        rank_result = json.loads(rank_result)

    a_points = b_points = 0
    a_wins = b_wins = 0
    detailed_scores: Dict[str, Dict[str, int]] = {'llm_a': {}, 'llm_b': {}}

    for dimension in SCORE_DIMENSIONS:
        eval_data = rank_result.get(dimension, {})
        choice = eval_data.get('final_choice', '').strip().upper()

        if choice == 'E':
            a_points += 1
            b_points += 1
            detailed_scores['llm_a'][dimension] = 1
            detailed_scores['llm_b'][dimension] = 1
        elif choice == 'A':
            a_points += 3
            a_wins += 1
            detailed_scores['llm_a'][dimension] = 3
            detailed_scores['llm_b'][dimension] = 0
        elif choice == 'B':
            b_points += 3
            b_wins += 1
            detailed_scores['llm_a'][dimension] = 0
            detailed_scores['llm_b'][dimension] = 3
        else:
            raise ValueError(f"Invalid choice '{choice}' for {dimension}")

    if a_points > b_points:
        winner = llm_a.name
    elif b_points > a_points:
        winner = llm_b.name
    else:
        winner = "Draw"

    return a_points, b_points, a_wins, b_wins, winner, detailed_scores


def convert_points_to_score(point_a: int, point_b: int) -> Tuple[int, int]:
    """
    Convert raw points to final scores.

    Returns raw points directly (preserving per-dimension scoring).

    Args:
        point_a: Raw points for participant A
        point_b: Raw points for participant B

    Returns:
        Tuple of final scores (score_a, score_b)
    """
    return point_a, point_b


class LLMRanker:
    """
    LLM-based pairwise ranker for comparing response quality.

    Uses structured output to get reliable A/B/E choices from the judge LLM.
    Performs two-pass evaluation (original and swapped order) to reduce positional bias.
    """

    SYSTEM_PROMPT = (
        "You are a professional medical AI scoring system. Your role is to objectively "
        "and accurately evaluate AI model responses to medical inquiries based on "
        "predefined criteria. Ensure all scores are fair and solely based on the "
        "provided information."
    )

    def __init__(self, llm_name: str):
        """
        Initialize the ranker with a specific judge LLM.

        Args:
            llm_name: Name of the LLM model to use as judge
        """
        self.llm_name = llm_name
        self.llm = get_chat_llm(model=llm_name, temperature=0)

    def _evaluate_single_dimension(
        self,
        question: str,
        clinical_data: str,
        response_a: str,
        response_b: str,
        dimension: str
    ) -> Dict[str, str]:
        """
        Evaluate a single dimension for one comparison.

        Args:
            question: Patient question
            clinical_data: Clinical context
            response_a: First response
            response_b: Second response
            dimension: Scoring dimension name

        Returns:
            Dictionary with 'choice' and 'reason' keys
        """
        criteria = SCORING_CRITERIA[dimension]
        chain = _ranking_prompt | self.llm.with_structured_output(RankingResult)

        input_data = {
            "question_content": question,
            "clinical_data": clinical_data,
            "llm_a_response": response_a,
            "llm_b_response": response_b,
            "scoring_dimension": dimension,
            "scoring_criteria": criteria,
        }

        try:
            output = chain.invoke(input_data)
            if output is None or output.choice is None:
                return {"choice": "Error", "reason": "LLM output is None"}
            return {"choice": output.choice, "reason": ""}
        except Exception as e:
            print(f"Error evaluating {dimension}: {e}")
            return {"choice": "Error", "reason": str(e)}

    def _normalize_pass2_choice(self, raw_choice: str) -> str:
        """
        Normalize pass 2 choice (where responses are swapped).

        In pass 2, A and B are swapped, so we need to flip the choice.
        """
        if raw_choice == 'A':
            return 'B'
        elif raw_choice == 'B':
            return 'A'
        return raw_choice  # 'E' or 'Error' stays the same

    def rank(
        self,
        patient_question: str,
        clinical_data: str,
        response_a: str,
        response_b: str
    ) -> Optional[str]:
        """
        Perform pairwise ranking of two responses.

        Uses two-pass evaluation to mitigate positional bias:
        - Pass 1: A vs B (original order)
        - Pass 2: B vs A (swapped order)

        Args:
            patient_question: The patient's question
            clinical_data: Clinical context information
            response_a: First LLM response
            response_b: Second LLM response

        Returns:
            JSON string with ranking results, or None if error
        """
        # Validate inputs
        missing = []
        if not response_a:
            missing.append("response_a")
        if not response_b:
            missing.append("response_b")
        if not patient_question:
            missing.append("patient_question")
        if not clinical_data:
            missing.append("clinical_data")

        if missing:
            print(f"Warning: Missing inputs: {', '.join(missing)}")
            return json.dumps({})

        results = {}

        for dimension in SCORE_DIMENSIONS:
            # Pass 1: Original order
            pass1 = self._evaluate_single_dimension(
                patient_question, clinical_data,
                response_a, response_b, dimension
            )
            choice1 = pass1.get('choice', 'Error').strip().upper()

            # Pass 2: Swapped order
            pass2 = self._evaluate_single_dimension(
                patient_question, clinical_data,
                response_b, response_a, dimension
            )
            raw_choice2 = pass2.get('choice', 'Error').strip().upper()
            choice2 = self._normalize_pass2_choice(raw_choice2)

            # Aggregate results
            final_choice = aggregate_choices(choice1, choice2)

            if final_choice == 'Error':
                return None

            results[dimension] = {
                'final_choice': final_choice,
                'choice1': choice1,
                'reason_pass1': pass1.get('reason', ''),
                'choice2': choice2,
                'reason_pass2': pass2.get('reason', '')
            }

        return json.dumps(results)


class LLMParticipant:
    """
    Represents an LLM participant in ranking evaluation.

    Tracks total score and per-dimension scores across multiple matches.
    """

    def __init__(self, name: str, score: int = 0):
        """
        Initialize a participant.

        Args:
            name: Participant identifier
            score: Initial score (default 0)
        """
        self.name = name
        self.score = score
        self.group: Optional[str] = None
        self.dimension_scores: Dict[str, int] = {d: 0 for d in SCORE_DIMENSIONS}
        self.dimension_history: List[Dict] = []

    def add_match_score(self, points: int) -> None:
        """Add points to total score."""
        self.score += points

    def add_dimension_scores(
        self,
        scores: Dict[str, int],
        question_id: str,
        opponent_name: str
    ) -> None:
        """
        Record dimension scores from a match.

        Args:
            scores: Dictionary mapping dimension names to scores
            question_id: Identifier for the question
            opponent_name: Name of the opponent in this match
        """
        for dimension, score in scores.items():
            if dimension in self.dimension_scores:
                self.dimension_scores[dimension] += score

        self.dimension_history.append({
            'question_id': question_id,
            'opponent': opponent_name,
            'scores': scores.copy()
        })

    def get_dimension_averages(self) -> Dict[str, float]:
        """
        Calculate average score per dimension across all matches.

        Returns:
            Dictionary mapping dimension names to average scores
        """
        if not self.dimension_history:
            return {d: 0.0 for d in SCORE_DIMENSIONS}

        n_matches = len(self.dimension_history)
        return {
            d: self.dimension_scores[d] / n_matches
            for d in SCORE_DIMENSIONS
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize participant to dictionary."""
        return {
            "name": self.name,
            "score": self.score,
            "dimension_scores": self.dimension_scores,
            "dimension_history": self.dimension_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMParticipant':
        """
        Deserialize participant from dictionary.

        Args:
            data: Dictionary with participant data

        Returns:
            LLMParticipant instance
        """
        participant = cls(data["name"], data["score"])
        participant.dimension_scores = data.get(
            "dimension_scores",
            {d: 0 for d in SCORE_DIMENSIONS}
        )
        participant.dimension_history = data.get("dimension_history", [])
        return participant


# Backward compatibility aliases
SCORE_ITEM_NAMES = SCORE_DIMENSIONS
PAIRWISE_SCORING_CRITERIA = SCORING_CRITERIA
get_final_choice = aggregate_choices
update_match_points_cur_question = update_ranking_results
rank_result_to_score = calculate_match_scores
points_to_score = convert_points_to_score


class LLMRank(LLMRanker):
    """Backward compatibility alias for LLMRanker."""

    def ranking(self, patient_question: str, clinical_data, llm_a_response, llm_b_response):
        """Alias for rank() method for backward compatibility."""
        return self.rank(patient_question, clinical_data, llm_a_response, llm_b_response)
