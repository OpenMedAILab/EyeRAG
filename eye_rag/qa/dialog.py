"""
Dialog experiment runner for EyeRAG.
Supports answer types defined in eye_rag.config.LLM_ANSWER_TYPE
"""
import json
import os
import random
from multiprocessing import Pool

from config import (
    LLM_ANSWER_TYPE,
    LLM_RESPONSE_TEMPERATURE,
    DEFAULT_LIGHTRAG_MODE,
)
from eye_rag.qa.patient_data import get_clinical_data_for_query
from eye_rag.utils import get_temperature_from_answer_type
from eye_rag.graph_node import execute_agent_graph

# Import graphs for each answer type
from eye_rag.graph.vanilla_llm_graph import graph as vanilla_llm_graph
from eye_rag.graph.naive_rag_graph import graph as naive_rag_graph
from eye_rag.graph.hypothetical_rag_graph import graph as hypothetical_rag_graph
from eye_rag.graph.hi_rag_graph import graph as hi_rag_graph
from eye_rag.graph.lightrag_graph import graph as lightrag_graph
from eye_rag.graph.lightrag_distill_context_graph import graph as lightrag_distill_graph

from eye_rag.graph.ablation.naive_rag_distill_context_graph import graph as naive_rag_distill_context_graph


CPU_COUNT = os.cpu_count()
DEBUG = False  # if DEBUG, then parallel processing, else, not parallels


def get_graph_for_answer_type(answer_type: str):
    """
    Get the appropriate graph based on answer type.

    Supported answer types (from LLM_ANSWER_TYPE):
    - LLM_Response: Direct LLM response without RAG
    - LLM_NaiveRAG_Response: Naive RAG with question rewriting
    - LLM_HypotheticalRAG_Response: Hypothetical RAG
    - LLM_HierarchicalIndexRAG_Response: Hierarchical index RAG
    - LLM_LightRAG_Hybrid_Distillation_Response: LightRAG with distillation
    """
    if answer_type == 'LLM_Response':
        return vanilla_llm_graph
    elif answer_type.find('LLM_NaiveRAG_Response') > -1:
        return naive_rag_graph
    elif answer_type == 'LLM_HypotheticalRAG_Response':
        return hypothetical_rag_graph
    elif answer_type == 'LLM_HierarchicalIndexRAG_Response':
        return hi_rag_graph
    elif answer_type.find('LLM_LightRAG_Hybrid_Distillation_Response') > -1:
        return lightrag_distill_graph

    elif answer_type.find('LLM_LightRAG_Hybrid_Response') > -1:
        return lightrag_graph
    elif answer_type.find('LLM_NaiveRAG_DistillContext_Response') > -1:
        return naive_rag_distill_context_graph

    else:
        raise ValueError(
            f"Unsupported answer_type: {answer_type}. "
            f"Supported types: {LLM_ANSWER_TYPE}"
        )


def generate_answer(
    question: str,
    clinical_data: dict,
    responding_llm: str,
    answer_type: str,
    question_id: str = '',
    initial_response_file_path: str = None,
) -> dict:
    """
    Generate an answer using the specified LLM and RAG strategy.

    Args:
        question: The clinical question to answer
        clinical_data: Patient clinical data
        responding_llm: Name of the LLM to use
        answer_type: Type of answer strategy (from LLM_ANSWER_TYPE)
        question_id: Optional question identifier
        initial_response_file_path: Path to save/load initial response

    Returns:
        dict with 'response' and 'context' keys
    """
    temperature = get_temperature_from_answer_type(answer_type)
    print(f"Using temperature: {temperature} for {answer_type}")

    input_data = {
        'question': question,
        'clinical_data': clinical_data,
        'responding_llm': responding_llm,
        'messages': [question],
        'initial_response_file_path': initial_response_file_path,
        'question_id': str(question_id),
        'temperature': temperature,
        'lightrag_mode': DEFAULT_LIGHTRAG_MODE,
    }

    print("--- STARTING GRAPH EXECUTION ---")
    graph = get_graph_for_answer_type(answer_type)

    try:
        result = execute_agent_graph(agent=graph, inputs=input_data)
        return result
    except Exception as e:
        # Handle any API or other errors that might not be serializable
        print(f"Error during graph execution: {str(e)}")
        # Return a safe result that can be serialized by multiprocessing
        return {
            'response': f"Error: Failed to generate response due to API or connectivity issue. Details: {str(e)[:500]}",
            'context': '',
            'error': str(e)
        }


def _get_question_ids(loaded_patient_data: dict, question_ids: list = None) -> list:
    """Filter and return valid question IDs from patient data."""
    if question_ids is None:
        return list(map(str, loaded_patient_data.keys()))

    return [
        str(qid)
        for qid in question_ids
        if str(qid) in loaded_patient_data
    ]


def qa(question_id: str, llm_name: str, answer_type: str, result_dir: str, patient_data: dict):
    """Run Q&A for a single question and save result."""
    out_file = os.path.join(result_dir, f"{question_id}_{llm_name}_{answer_type}.json")

    if os.path.exists(out_file):
        return

    question = patient_data['Question']
    clinical_data = get_clinical_data_for_query(patient_data)

    temperature = get_temperature_from_answer_type(answer_type)
    if temperature == LLM_RESPONSE_TEMPERATURE:
        initial_response_file_path = os.path.join(
            result_dir, f"{question_id}_{llm_name}_LLM_Response.json"
        )
    else:
        initial_response_file_path = os.path.join(
            result_dir, f"{question_id}_{llm_name}_LLM_Response_Temperature{temperature}.json"
        )

    try:
        result = generate_answer(
            clinical_data=clinical_data,
            question=question,
            responding_llm=llm_name,
            answer_type=answer_type,
            question_id=question_id,
            initial_response_file_path=initial_response_file_path,
        )
    except Exception as e:
        # Handle any errors that occur during generate_answer
        print(f"Error during generate_answer: {str(e)}")
        result = {
            'response': f"Error: Failed to generate response due to API or connectivity issue. Details: {str(e)[:500]}",
            'context': '',
            'error': str(e)
        }

    result.update({
        'QuestionID': question_id,
        'llm_name': llm_name,
        'Question': question,
        'clinical_data': clinical_data,
        'llm_answer_type': answer_type,
        'temperature': temperature,
    })

    print(f"\n\nDone: ---\n\n{json.dumps(result.get('response', '')[:500], indent=4, ensure_ascii=False)}...")

    response = result.get('response', '')
    if response and (not response.find('Error:') > -1):
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"Wrote {out_file}")
    else:
        print(f"Warning: No response from {llm_name} for question ID {question_id}")


def _qa_wrapper(args):
    """Wrapper for multiprocessing."""
    return qa(*args)


def conduct_dialog_experiments(
    loaded_patient_data: dict,
    result_dir: str,
    llm_names: list,
    llm_answer_types: list = None,
    question_ids: list = None,
):
    """
    Conduct dialog experiments with multiple LLMs and answer types.

    Args:
        loaded_patient_data: Dictionary of patient data keyed by question ID
        result_dir: Directory to save results
        llm_names: List of LLM names to use
        llm_answer_types: List of answer types (defaults to LLM_ANSWER_TYPE)
        question_ids: Optional list of specific question IDs to process
    """
    os.makedirs(result_dir, exist_ok=True)

    if llm_answer_types is None:
        llm_answer_types = LLM_ANSWER_TYPE

    # # Validate answer types
    # for answer_type in llm_answer_types:
    #     if answer_type not in LLM_ANSWER_TYPE:
    #         raise ValueError(
    #             f"Invalid answer_type: {answer_type}. "
    #             f"Must be one of: {LLM_ANSWER_TYPE}"
    #         )

    question_ids = _get_question_ids(loaded_patient_data, question_ids)
    print(f"Processing {len(question_ids)} questions with {len(llm_names)} LLMs and {len(llm_answer_types)} answer types")

    tasks = []
    for answer_type in llm_answer_types:
        for k, question_id in enumerate(question_ids):
            patient_data = loaded_patient_data[question_id]

            for llm_name in llm_names:
                signature = f"Q{question_id} ({k+1}/{len(question_ids)}), {llm_name}, {answer_type}"
                out_file = os.path.join(result_dir, f"{question_id}_{llm_name}_{answer_type}.json")

                if os.path.exists(out_file):
                    continue

                print(f"Queuing: {signature}")

                if DEBUG:
                    qa(question_id, llm_name, answer_type, result_dir, patient_data)
                else:
                    tasks.append((question_id, llm_name, answer_type, result_dir, patient_data))

    if tasks and not DEBUG:
        random.shuffle(tasks)
        print(f"Running {len(tasks)} tasks in parallel (CPU_COUNT={CPU_COUNT})")
        with Pool(processes=CPU_COUNT) as pool:
            pool.map(_qa_wrapper, tasks)


