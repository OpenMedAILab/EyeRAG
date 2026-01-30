# python
import os
from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

from config import EXP_CACHE_DIR
from eye_rag.eye_rag_utils import generate_key, load_json_data, save_dict_to_json

from eye_rag.llm import get_chat_llm


class RewriteQuestion(BaseModel):
    """
    Output schema for the rewritten question.
    """
    rewritten_question: List[str] = Field(
        description="The improved questions optimized for vectorstore retrieval."
    )

rewrite_prompt_template = """
You are a question re-writer that converts an input question to a better version optimized for information retrieval.
Your task is to rewrite a patient's question into clear, concise questions that are optimized for information retrieval from medical guide databases and web searches. 
Based on the patient's question and their clinical data, generate a list of 1 questions. 
Do not explicitly mention eye laterality (left/right); use neutral terms like 'the affected eye' or 'the eye with the condition.'
Do not explicitly mention age information for elderly patients; use terms like 'the elderly patient' or 'the older adult.'

Input you receive:
Patient Query: {question}
Patient Clinical Data: {clinical_data}
"""

def rewrite_question_for_retrieval(question: str, clinical_data) -> List[str]:
    """
    Independent function that only generates rewritten questions via the LLM.
    No caching or filesystem I/O happens here.
    """
    rewrite_llm = get_chat_llm()
    rewrite_prompt = PromptTemplate(
        template=rewrite_prompt_template,
        input_variables=["clinical_data", "question"],
    )
    question_rewriter = rewrite_prompt | rewrite_llm.with_structured_output(RewriteQuestion)
    result = question_rewriter.invoke({"clinical_data": clinical_data, "question": question})
    return result.rewritten_question


USE_CACHE_FILE = True


def rewrite_question(state: dict) -> dict:
    """
    Graph-node wrapper that handles caching and updates state with rewritten questions.
    """
    state["curr_state"] = "rewrite_question"
    question = state.get("question")
    clinical_data = state.get("clinical_data")

    cache_dir = os.path.join(EXP_CACHE_DIR, "rewritten_question")
    filename = generate_key(question + str(clinical_data) + rewrite_prompt_template) + ".json"
    cache_file_path = os.path.join(cache_dir, filename)

    if USE_CACHE_FILE and os.path.isfile(cache_file_path):
        data = load_json_data(cache_file_path)
        state['rewritten_questions'] = data.get('rewritten_questions', [])
        print(f"Loaded rewritten questions from cache: {cache_file_path}")
        for k, x in enumerate(state['rewritten_questions']):
            print(f'Question {k}: {x}')
        return state

    # No cache (or cache miss) -> call independent generator
    new_questions = rewrite_question_for_retrieval(question, clinical_data)
    state['rewritten_questions'] = new_questions
    for k, x in enumerate(new_questions):
        print(f'Question {k}: {x}')

    if USE_CACHE_FILE:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file_path)), exist_ok=True)
        save_dict_to_json(
            dict_to_save={"clinical_data": clinical_data, "question": question, "rewritten_questions": new_questions},
            out_file=cache_file_path
        )

    return state
