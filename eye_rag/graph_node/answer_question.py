"""
Answer generation nodes for EyeRAG graph workflows.
"""
import re

from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from eye_rag.eye_rag_utils import wrap_context, load_cache_file, save_cache_file
from eye_rag.llm import get_chat_llm

# ============================================
# Constants
# ============================================
VALID_ANSWER_MIN_LEN = 500
REANSWER_MAX_TRY = 3
USE_CACHE_FILE = True

DOCTOR_SYSTEM_PROMPT = (
    "You are an experienced ophthalmology specialist providing evidence-based medical guidance."
)

# ============================================
# Prompts
# ============================================
QUERY_WITH_PATIENT_DATA_PROMPT = """
TASK: Response to the patient's question using their clinical data.

RESPONSE GUIDELINES:
1. **Evidence-Based**: Base your response STRICTLY on the provided clinical data of the patient
2. **Clarity**: Ensure your answer is clear, concise, and directly addresses the patient's question
3. **Communication**: Use professional yet compassionate tone with clear, simple language
4. **Boundaries**: Focus on patient education and guidance; do NOT provide differential diagnoses or speculate beyond available data
5. **Language**: The response language must match the user's question language
6. **Person**: Address the patient directly using second person ("you", "your") throughout the response
7. **Completeness**: Provide a thorough explanation that includes:
   - Direct answer to the patient's specific question
   - Relevant context from their clinical data
   - Educational information to help patient understanding
   - Any appropriate recommendations or next steps
8. **Structure**: Format your response with clear paragraphs and use markdown formatting where appropriate
9. **Length**: Aim for a comprehensive response (minimum 100-150 words) that fully addresses the patient's concerns

---Clinical Data---
{clinical_data}

---Question---
{question}

---Response---
"""

QUERY_WITH_PATIENT_DATA_AND_RAG_CONTEXT_PROMPT = """
TASK: Provide a comprehensive response to the patient's question using both their clinical data and retrieved medical literature.

RESPONSE GUIDELINES:
1. **Evidence-Based**: Base your response STRICTLY on the provided clinical data and retrieved medical literature
2. **Clarity**: Ensure your answer is clear, concise, and directly addresses the patient's question
3. **Communication**: Use professional yet compassionate tone with clear, simple language
4. **Boundaries**: Focus on patient education and guidance; do NOT provide differential diagnoses or speculate beyond available data
5. **Language**: The response language must match the user's question language
6. **Person**: Address the patient directly using second person ("you", "your") throughout the response
7. **CompletenListess**: Provide a thorough explanation that includes:
   - Direct answer to the patient's specific question
   - Relevant context from their clinical data
   - Educational information to help patient understanding
   - Any appropriate recommendations or next steps
8. **Structure**: Format your response with clear paragraphs and use markdown formatting where appropriate
9. **Length**: Aim for a comprehensive response (minimum 100-150 words) that fully addresses the patient's concerns
10. **Integration**: Strictly adhere to the provided medical literature. Do not invent or assume information not present in the source data
11. **Context Priority**: If the context does not directly address the question, answer based on general medical knowledge, but always prioritize the provided context when relevant

---Clinical Data---
{clinical_data}

---Question---
{question}

---Retrieved Medical Guide---
{context}

---Response---
"""


# ============================================
# Output Schema
# ============================================
class QuestionAnswer(BaseModel):
    response: str = Field(
        description="Generates an answer to a query based on a given patient data."
    )


# ============================================
# Helper Functions
# ============================================
def safe_structured_output(llm_chain, messages, output_schema):
    """Safely get structured output from LLM with fallback to JSON parsing."""
    try:
        output = llm_chain.with_structured_output(output_schema).invoke(messages)
        return output.response
    except Exception as e:
        # Check if it's a privacy/compliance error
        error_str = str(e)
        if "No endpoints found matching your data policy" in error_str or "404" in error_str:
            print(f"Privacy/compliance error detected: {e}")
            print("This error occurs due to OpenRouter privacy settings restricting model access.")
            print("Please visit https://openrouter.ai/settings/privacy to adjust your data policy.")
            print("Alternatively, consider using models that comply with your current privacy settings.")
            # Return a meaningful error message instead of failing completely
            return f"Error: Model access restricted due to privacy settings. Please check OpenRouter privacy configuration. Details: {str(e)}"

        # Check if it's a validation error (like the JSON validation error)
        if "validation error" in error_str or "Invalid JSON" in error_str or "json_invalid" in error_str:
            print(f"Validation error during structured output: {e}")
            print("Model returned plain text instead of expected JSON format. Attempting to extract response...")

            # Try to call the model again without structured output to get raw response
            try:
                response = llm_chain.invoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)

                # If the response looks like a valid answer, return it as is
                if isinstance(response_text, str) and len(response_text) > 20:
                    return response_text

                # If all else fails, return the raw response
                return response_text
            except Exception as plain_invoke_error:
                print(f"Plain text invocation also failed: {plain_invoke_error}")
                return f"Error: Failed to generate response. Details: {str(plain_invoke_error)}"

        # Check if it's a length limit error (common with long responses)
        if "length limit was reached" in error_str or "could not parse response content" in error_str.lower():
            print(f"Length limit error during structured output: {e}")
            print("Model response exceeded length limit for structured parsing. Attempting to extract response...")

            # Try to call the model again without structured output to get raw response
            try:
                response = llm_chain.invoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)

                # If the response looks like a valid answer, return it as is
                if isinstance(response_text, str) and len(response_text) > 20:
                    return response_text

                # If all else fails, return the raw response
                return response_text
            except Exception as plain_invoke_error:
                print(f"Plain text invocation also failed: {plain_invoke_error}")
                return f"Error: Failed to generate response. Details: {str(plain_invoke_error)}"

        print(f"Structured output failed, falling back to plain text invocation: {e}")

        # Try plain text invocation without JSON parsing to avoid encoding issues
        try:
            response = llm_chain.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # If we got a valid response, return it directly
            if isinstance(response_text, str) and len(response_text) > 20:
                # Try to extract JSON response if present
                try:
                    match = re.search(r'"response"\s*:\s*"([^"]*)"', response_text, re.DOTALL)
                    if match:
                        return match.group(1)
                except Exception:
                    pass
                return response_text

            return response_text
        except Exception as invoke_error:
            print(f"LLM invocation failed: {invoke_error}")
            return f"Error: Failed to generate response due to model access issues. Details: {str(invoke_error)}"


# ============================================
# Independent Generation Functions
# ============================================
def generate_response_with_clinical_data(
    question: str,
    clinical_data: str,
    model: str,
    temperature: float = 0.0,
) -> str:
    """
    Generate a response based on clinical data only.

    Args:
        question: The patient's question
        clinical_data: Patient clinical data as string
        model: Name of the LLM model to use
        temperature: LLM temperature setting

    Returns:
        Generated response string
    """
    try:
        llm = get_chat_llm(model=model, temperature=temperature)
    except Exception as e:
        if "No endpoints found matching your data policy" in str(e) or "404" in str(e):
            print(f"Privacy/compliance error during model initialization: {e}")
            print("This error occurs due to OpenRouter privacy settings restricting model access.")
            print("Please visit https://openrouter.ai/settings/privacy to adjust your data policy.")
            return f"Error: Model access restricted due to privacy settings. Please check OpenRouter privacy configuration. Details: {str(e)}"
        else:
            raise e

    human_prompt_template = PromptTemplate(
        template=QUERY_WITH_PATIENT_DATA_PROMPT,
        input_variables=["clinical_data", "question"],
    )

    human_prompt = human_prompt_template.invoke({
        "clinical_data": clinical_data,
        "question": question
    })

    messages = [
        SystemMessage(content=DOCTOR_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt.text)
    ]

    print("Answering the question with patient data...")

    response = ""
    for attempt in range(REANSWER_MAX_TRY):
        try:
            if attempt % 10 == 0 and attempt > 0:
                current_temperature = temperature + (attempt // 10) * 0.02
                print(f"{model} attempt {attempt + 1}: adjusting temperature to {current_temperature}")
                llm = get_chat_llm(model=model, temperature=current_temperature)

            response = safe_structured_output(llm, messages, QuestionAnswer)

            # Check if response indicates privacy error
            if isinstance(response, str) and "Model access restricted" in response:
                return response  # Return the error message directly

            if isinstance(response, str) and len(response) > VALID_ANSWER_MIN_LEN:
                return response
            else:
                print(f"{model} attempt {attempt + 1}: invalid response (length: {len(response) if isinstance(response, str) else 'N/A'})")
                if attempt == REANSWER_MAX_TRY - 1:
                    print(f"{model} max retries ({REANSWER_MAX_TRY}) reached, using last response")

        except Exception as e:
            # Check for privacy/compliance errors
            if "No endpoints found matching your data policy" in str(e) or "404" in str(e):
                print(f"Privacy/compliance error during generation: {e}")
                print("This error occurs due to OpenRouter privacy settings restricting model access.")
                print("Please visit https://openrouter.ai/settings/privacy to adjust your data policy.")
                return f"Error: Model access restricted due to privacy settings. Please check OpenRouter privacy configuration. Details: {str(e)}"

            print(f"{model} attempt {attempt + 1} failed: {e}")
            if attempt == REANSWER_MAX_TRY - 1:
                print(f"{model} all retries failed, setting empty response")
                return ""

    return response if isinstance(response, str) else str(response)


def generate_response_with_context(
    question: str,
    clinical_data: str,
    context: str,
    model: str,
    temperature: float = 0.0,
) -> str:
    """
    Generate a response using clinical data and RAG context.

    Args:
        question: The patient's question
        clinical_data: Patient clinical data as string
        context: Retrieved RAG context
        model: Name of the LLM model to use
        temperature: LLM temperature setting

    Returns:
        Generated response string
    """
    try:
        llm = get_chat_llm(model=model, temperature=temperature)
    except Exception as e:
        if "No endpoints found matching your data policy" in str(e) or "404" in str(e):
            print(f"Privacy/compliance error during model initialization: {e}")
            print("This error occurs due to OpenRouter privacy settings restricting model access.")
            print("Please visit https://openrouter.ai/settings/privacy to adjust your data policy.")
            return f"Error: Model access restricted due to privacy settings. Please check OpenRouter privacy configuration. Details: {str(e)}"
        else:
            raise e

    question_answer_from_context_prompt = PromptTemplate(
        template=QUERY_WITH_PATIENT_DATA_AND_RAG_CONTEXT_PROMPT,
        input_variables=["context", "question", "clinical_data"],
    )

    human_prompt = question_answer_from_context_prompt.invoke({
        "context": context,
        "question": question,
        "clinical_data": clinical_data
    })

    messages = [
        SystemMessage(content=DOCTOR_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt.text)
    ]

    print("Answering the question from the retrieved context...")

    response = ""
    for attempt in range(REANSWER_MAX_TRY):
        try:
            if attempt % 10 == 0 and attempt > 0:
                current_temperature = temperature + (attempt // 10) * 0.02
                print(f"{model} attempt {attempt + 1}: adjusting temperature to {current_temperature}")
                llm = get_chat_llm(model=model, temperature=current_temperature)

            response = safe_structured_output(llm, messages, QuestionAnswer)

            # Check if response indicates privacy error
            if isinstance(response, str) and "Model access restricted" in response:
                return response  # Return the error message directly

            if isinstance(response, str) and len(response) > VALID_ANSWER_MIN_LEN:
                return response
            else:
                print(f"{model} attempt {attempt + 1}: invalid response (length: {len(response) if isinstance(response, str) else 'N/A'})")
                if attempt == REANSWER_MAX_TRY - 1:
                    print(f"{model} max retries ({REANSWER_MAX_TRY}) reached, using last response")

        except Exception as e:
            # Check for privacy/compliance errors
            if "No endpoints found matching your data policy" in str(e) or "404" in str(e):
                print(f"Privacy/compliance error during generation: {e}")
                print("This error occurs due to OpenRouter privacy settings restricting model access.")
                print("Please visit https://openrouter.ai/settings/privacy to adjust your data policy.")
                return f"Error: Model access restricted due to privacy settings. Please check OpenRouter privacy configuration. Details: {str(e)}"

            print(f"{model} attempt {attempt + 1} failed: {e}")
            if attempt == REANSWER_MAX_TRY - 1:
                print(f"{model} all retries failed, setting empty response")
                return ""

    return response if isinstance(response, str) else str(response)


# ============================================
# Graph Node Functions (with State & Cache)
# ============================================
def answer_question_with_clinical_data(state):
    """
    Graph node: Answers a question using clinical data with caching support.

    Args:
        state (dict): Contains question, clinical_data, responding_llm, temperature, etc.

    Returns:
        dict: Updated state with 'response' field.
    """
    state["curr_state"] = "answer_question_with_clinical_data"
    cache_file_path = state.get("initial_response_file_path", '')

    # Try loading from cache
    response = load_cache_file(use_cache_file=USE_CACHE_FILE, cache_file_path=cache_file_path, key="response")
    if response and len(response) > VALID_ANSWER_MIN_LEN:
        state['response'] = response
        return state

    # Generate response using independent function
    response = generate_response_with_clinical_data(
        question=state["question"],
        clinical_data=state["clinical_data"],
        model=state['responding_llm'],
        temperature=state['temperature'],
    )
    state['response'] = response

    # Save to cache
    if USE_CACHE_FILE and cache_file_path:
        question_id = state['question_id']
        param_dict = dict(
            question=state["question"],
            model=state['responding_llm'],
            clinical_data=state["clinical_data"],
            system_prompt=DOCTOR_SYSTEM_PROMPT,
            user_prompt=QUERY_WITH_PATIENT_DATA_PROMPT,
        )
        save_cache_file(
            cache_file_path=cache_file_path,
            data_dict_to_save={'question_id': question_id, "response": response},
            param_dict=param_dict)

    return state


def answer_question_with_context(state):
    """
    Graph node: Answers a question using clinical data and RAG context.

    Args:
        state (dict): Contains question, clinical_data, context/filtered_context, responding_llm, temperature.

    Returns:
        dict: Updated state with 'response' field.
    """
    state["curr_state"] = "answer_question_with_context"
    print(f"Temperature for answering questions: {state['temperature']}")

    # Get context (prefer filtered_context if available)
    context = state["filtered_context"] if "filtered_context" in state else state["context"]
    context = wrap_context(context)

    # Generate response using independent function
    response = generate_response_with_context(
        question=state["question"],
        clinical_data=state["clinical_data"],
        context=context,
        model=state['responding_llm'],
        temperature=state['temperature'],
    )
    state['response'] = response

    return state
