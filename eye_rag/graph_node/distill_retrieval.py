"""Context distillation module for extracting key points from retrieved context."""

import asyncio
import logging
from typing import Any, List, Dict

from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

from eye_rag.llm import get_chat_llm
from eye_rag.utils import wrap_context
from config import NUM_RECORD_FOR_WHOLE_CONTEXT

logger = logging.getLogger(__name__)

NUM_RECORD_PER_CONTEXT_ITEM = 5

USE_CACHE_FILE = True

prompt_summarize_context = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Goal---

Generate a response consisting of a list of {num_record} comprehensive analysis points that responds to the user's question, providing detailed summaries of all relevant information in the input data.

You should use the data (could be Knowledge Graph or Document Chunks) provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each analysis point in the response should have the following element:
- Description: A detailed, comprehensive analysis that includes:
  * Specific facts and details from the source data
  * Context and background information when relevant
  * Multiple sentences providing thorough explanation (minimum 2-3 sentences)
  * Quantitative data, statistics, or specific examples when available
  * Clear connections between different pieces of information
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

**Writing Guidelines:**
- Each description should be substantial and informative (aim for 50-150 words per point)
- Include specific details, numbers, procedures, or examples from the source data
- Provide sufficient context to make each point self-contained and meaningful
- Connect related information to create comprehensive explanations
- Use clear, professional language with proper medical/technical terminology when applicable

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Detailed comprehensive analysis of point 1 with specific facts, context, and thorough explanation [Source: Primary Angle-Closure Disease PPP (source ids)]", "score": score_value}},
        {{"description": "Detailed comprehensive analysis of point 2 with specific facts, context, and thorough explanation [Source: Cataract in the Adult Eye PPP (source ids)]", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Source: Primary Angle-Closure Disease PPP]"

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

**Data tables:**
{context_data}

**User Query:**
{query}
"""


class DistillTextPoints(BaseModel):
    points: List[Dict[str, Any]] = Field(
        description="A list of key points extracted from the text data, each with description and score. Score 0 if source-query mismatch or content lacks direct relevance."
    )


global_map_rag_points_prompt = PromptTemplate(
    template=prompt_summarize_context,
    input_variables=["query", "context_data", "source", "num_record"],
)

distill_text_llm = get_chat_llm()
distill_text_chain = (
    global_map_rag_points_prompt
    | distill_text_llm.with_structured_output(DistillTextPoints, method="function_calling")
)


def _distill_context_to_points(query: str, context_data: list[dict] | str) -> List[Dict]:
    """
    Distill text data into key points.

    Args:
        query: The query question
        context_data: Text data list, each element should contain 'source' and 'content' fields

    Returns:
        List of distilled key points, sorted by score in descending order
    """
    if not context_data:
        return []

    formatted_context = wrap_context(context_data)

    input_data = {
        "query": query,
        "context_data": formatted_context,
        'num_record': NUM_RECORD_FOR_WHOLE_CONTEXT
    }

    try:
        output = distill_text_chain.invoke(input_data)
        points = output.points

        valid_points = [p for p in points if p.get("score", 0) > 0]
        sorted_points = sorted(valid_points, key=lambda x: x.get("score", 0), reverse=True)

        return sorted_points
    except Exception as e:
        logger.error(f"Error distilling text points: {e}")
        return []


async def distill_text_dict_to_points(query: str, text_dict: dict) -> List[Dict]:
    """
    Distill a text group into key points (generic async function).

    Args:
        query: The query question
        text_dict: Text dictionary with keys:
            'id': int
            'content': str
            'file_path': str
            'source': str

    Returns:
        List of distilled key points (scored based on source and content relevance)
    """
    assert isinstance(text_dict, dict), "The input context for distillation must be a dictionary with data source"

    source = text_dict.get("source", "Unknown Source")
    text_context = text_dict["content"]

    input_data = {
        "query": query,
        "context_data": dict(text_context=text_context, source=source),
        "num_record": NUM_RECORD_PER_CONTEXT_ITEM
    }

    try:
        output = distill_text_chain.invoke(input_data)

        valid_points = [p for p in output.points if p.get("score", 0) > 0]

        if not valid_points:
            logger.info(f"Data source '{source}': {text_context[:300]} is not relevant to query '{query}'")

        return valid_points
    except Exception as e:
        logger.error(f"Error processing text group: {e}")
        return []


def distill_context(state):
    """
    Generic context distillation function, serving as the single entry point for distillation.
    Can handle various context formats.

    Args:
        state: Dictionary containing:
            - "rewritten_questions": Rewritten questions
            - "context": Context data (can be string, list, dict, etc.)

    Returns:
        Updated state with "filtered_context" containing distilled context list
    """
    state["curr_state"] = "distill_context"
    query = state["rewritten_questions"]
    context = state.get("context", [])

    distilled_points = _distill_context_to_points(query, context)

    valid_points = [p for p in distilled_points if p.get("score", 0) > 0]
    sorted_points = sorted(valid_points, key=lambda x: x.get("score", 0), reverse=True)

    state["filtered_context"] = points_to_context_list(sorted_points)
    return state


def _process_context_generic(query: str, context: Any) -> List[Dict]:
    """
    Generic context processing function supporting multiple input formats.

    Args:
        query: The query question
        context: Context data supporting the following formats:
            - str: Plain string
            - list: List of strings or dictionaries
            - dict: Dictionary format
            - Other formats will be converted to string

    Returns:
        List of distilled key points
    """
    if not context:
        return []

    if isinstance(context, str):
        return _distill_context_to_points(query, context)
    elif isinstance(context, list):
        return _process_context_list_async(query, context)
    else:
        return _distill_context_to_points(query, context)


def _process_context_list_async(query: str, context_list: list) -> List[Dict]:
    """
    Asynchronously process a context list.

    Args:
        query: The query question
        context_list: Context list

    Returns:
        List of distilled key points
    """

    async def _process_all_groups():
        responses = await asyncio.gather(
            *[distill_text_dict_to_points(query, group) for group in context_list],
            return_exceptions=True
        )

        all_points = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.warning(f"Text group {i} processing failed: {response}")
            else:
                all_points.extend(response)

        return all_points

    all_points = asyncio.run(_process_all_groups())
    return all_points


def points_to_context_list(final_points):
    """Convert points to context list."""
    return [x['description'] for x in final_points]
