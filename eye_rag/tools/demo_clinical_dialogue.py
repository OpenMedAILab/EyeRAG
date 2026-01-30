#!/usr/bin/env python3
"""
EyeRAG Clinical Dialogue Demonstration Script

This script demonstrates how to conduct clinical dialogue using EyeRAG's
LightRAG with context distillation pipeline. It showcases the complete
process from clinical data input to generating evidence-based responses.

Usage:
    python -m eye_rag.tools.demo_clinical_dialogue

The script will:
1. Display example clinical patient data
2. Show the clinical question
3. Run the LightRAG distillation pipeline
4. Display the retrieved context (medical knowledge)
5. Show the final clinical response
"""

import json
import sys
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, '.')

from config import DEFAULT_LIGHTRAG_MODE
from eye_rag.llm import LLMModelName
from eye_rag.graph_node import execute_agent_graph
from eye_rag.graph.lightrag_distill_context_graph import graph as lightrag_distill_graph


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

class DisplayFormatter:
    """Utility class for formatted console output."""

    WIDTH = 80

    # Box characters
    TOP_LEFT = "╔"
    TOP_RIGHT = "╗"
    BOTTOM_LEFT = "╚"
    BOTTOM_RIGHT = "╝"
    HORIZONTAL = "═"
    VERTICAL = "║"

    # Colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"

    @classmethod
    def print_header(cls, title: str):
        """Print a styled header box."""
        padding = (cls.WIDTH - len(title) - 2) // 2
        top = f"{cls.TOP_LEFT}{cls.HORIZONTAL * cls.WIDTH}{cls.TOP_RIGHT}"
        middle = f"{cls.VERTICAL}{' ' * padding}{cls.BOLD}{title}{cls.RESET}{' ' * (cls.WIDTH - padding - len(title))}{cls.VERTICAL}"
        bottom = f"{cls.BOTTOM_LEFT}{cls.HORIZONTAL * cls.WIDTH}{cls.BOTTOM_RIGHT}"
        print(f"\n{cls.CYAN}{top}{cls.RESET}")
        print(f"{cls.CYAN}{middle}{cls.RESET}")
        print(f"{cls.CYAN}{bottom}{cls.RESET}\n")

    @classmethod
    def print_section(cls, title: str, content: str, color: str = None):
        """Print a section with title and content."""
        color = color or cls.GREEN
        print(f"{color}{cls.BOLD}{'─' * 40}")
        print(f"  {title}")
        print(f"{'─' * 40}{cls.RESET}")
        print(content)
        print()

    @classmethod
    def print_step(cls, step_num: int, description: str):
        """Print a step indicator."""
        print(f"{cls.YELLOW}{cls.BOLD}[Step {step_num}]{cls.RESET} {description}")


# =============================================================================
# EXAMPLE CLINICAL DATA
# =============================================================================

def create_example_clinical_data() -> Dict[str, Any]:
    """
    Create an example clinical case for demonstration.

    This represents a typical ophthalmology patient case with:
    - Demographics (age, gender)
    - Admission/Discharge dates
    - Diagnoses
    - Clinical history
    - Differential diagnosis

    Returns:
        Dictionary containing patient clinical data
    """
    return {
        "Age": "67",
        "Gender": "Male",
        "AdmissionDate": "2024-03-15",
        "DischargeDate": "2024-03-18",
        "AdmissionDiagnosis": "Suspected primary open-angle glaucoma (POAG), right eye",
        "DischargeDiagnosis": "Primary open-angle glaucoma (POAG), moderate stage, right eye; Early cataract, both eyes",
        "ClinicalHistory": """Chief Complaint: Progressive vision loss in right eye over 6 months, occasional headaches.

Present Illness: 67-year-old male presenting with gradual peripheral vision loss in the right eye noticed over the past 6 months. Patient reports difficulty seeing objects to the side while driving. Occasional mild headaches, no eye pain or redness. No history of trauma.

Past Medical History:
- Type 2 Diabetes Mellitus (well-controlled, HbA1c 6.8%)
- Hypertension (on amlodipine 5mg daily)
- No previous eye surgeries

Family History: Mother had glaucoma, diagnosed at age 70.

Ocular Examination:
- Visual Acuity: OD 20/40, OS 20/25
- Intraocular Pressure: OD 28 mmHg, OS 18 mmHg
- Gonioscopy: Open angles bilaterally (Grade 4 Shaffer)
- Fundoscopy: OD - Increased cup-to-disc ratio (0.7), inferior notching, OS - CDR 0.4, healthy rim
- Visual Field (Humphrey 24-2): OD - Superior arcuate defect, OS - Within normal limits
- OCT RNFL: OD - Inferior thinning, OS - Normal""",
        "DifferentialDiagnosis": "1. Primary open-angle glaucoma (most likely)\n2. Normal tension glaucoma\n3. Secondary glaucoma due to diabetes\n4. Ocular hypertension"
    }


def create_example_question() -> str:
    """
    Create an example clinical question for the case.

    Returns:
        Clinical question string
    """
    return """Based on the patient's clinical presentation with elevated IOP, optic disc changes,
and visual field defects, what is the recommended initial treatment approach for managing
this patient's primary open-angle glaucoma? What are the target IOP goals and monitoring
schedule that should be followed?"""


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def run_clinical_dialogue_demo(
    llm_name: str = LLMModelName.DEEPSEEK_CHAT,
    verbose: bool = True
):
    """
    Run the clinical dialogue demonstration.

    This function demonstrates the complete EyeRAG pipeline:
    1. Prepare clinical data and question
    2. Execute the LightRAG distillation graph
    3. Display all intermediate results and final response

    Args:
        llm_name: Name of the LLM to use for response generation
        verbose: Whether to show detailed progress
    """
    fmt = DisplayFormatter()

    # Print welcome header
    fmt.print_header("EyeRAG Clinical Dialogue Demonstration")

    print(f"{fmt.DIM}This demonstration shows how EyeRAG uses LightRAG with context")
    print(f"distillation to answer clinical ophthalmology questions.{fmt.RESET}\n")

    # ==========================================================================
    # STEP 1: Prepare Clinical Data
    # ==========================================================================
    fmt.print_step(1, "Preparing Clinical Patient Data")

    clinical_data = create_example_clinical_data()
    clinical_data_str = json.dumps(clinical_data, indent=2, ensure_ascii=False)

    fmt.print_section("CLINICAL DATA", clinical_data_str, fmt.BLUE)

    # ==========================================================================
    # STEP 2: Define Clinical Question
    # ==========================================================================
    fmt.print_step(2, "Defining Clinical Question")

    question = create_example_question()
    fmt.print_section("CLINICAL QUESTION", question.strip(), fmt.MAGENTA)

    # ==========================================================================
    # STEP 3: Prepare Graph Input
    # ==========================================================================
    fmt.print_step(3, "Preparing LightRAG Distillation Graph Input")

    input_data = {
        'question': question,
        'clinical_data': clinical_data_str,
        'responding_llm': llm_name,
        'messages': [question],
        'question_id': 'demo_001',
        'temperature': 0.0,
        'lightrag_mode': DEFAULT_LIGHTRAG_MODE,  # 'hybrid' mode
    }

    print(f"{fmt.DIM}Input configuration:")
    print(f"  - LLM: {llm_name}")
    print(f"  - LightRAG Mode: {DEFAULT_LIGHTRAG_MODE}")
    print(f"  - Temperature: 0.0 (deterministic){fmt.RESET}\n")

    # ==========================================================================
    # STEP 4: Execute LightRAG Distillation Pipeline
    # ==========================================================================
    fmt.print_step(4, "Executing LightRAG Distillation Pipeline")

    print(f"\n{fmt.YELLOW}Pipeline stages:")
    print("  1. Rewrite Question    - Optimize query for retrieval")
    print("  2. Retrieve Context    - LightRAG hybrid search (KB + entities)")
    print("  3. Distill Context     - Extract and score relevant points")
    print(f"  4. Generate Response   - Create evidence-based answer{fmt.RESET}\n")

    print(f"{fmt.BOLD}Running pipeline...{fmt.RESET}\n")

    try:
        result = execute_agent_graph(
            agent=lightrag_distill_graph,
            inputs=input_data,
            verbose=verbose
        )
    except Exception as e:
        print(f"\n{fmt.BOLD}Error during execution:{fmt.RESET} {str(e)}")
        print(f"\n{fmt.DIM}Note: Make sure you have:")
        print("  1. Set up API keys in .env file")
        print("  2. Initialized the LightRAG knowledge base")
        print(f"  3. Network connectivity for LLM API calls{fmt.RESET}")
        return None

    # ==========================================================================
    # STEP 5: Display Retrieved Context
    # ==========================================================================
    fmt.print_step(5, "Retrieved Medical Knowledge Context")

    context = result.get('context', 'No context retrieved')
    if isinstance(context, list):
        context_display = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context)])
    else:
        context_display = str(context) if context else 'No context retrieved'

    # Truncate if too long for display
    if len(context_display) > 2000:
        context_display = context_display[:2000] + "\n\n... [truncated for display]"

    fmt.print_section("RETRIEVED CONTEXT (Distilled)", context_display, fmt.CYAN)

    # ==========================================================================
    # STEP 6: Display Final Response
    # ==========================================================================
    fmt.print_step(6, "Final Clinical Response")

    response = result.get('response', 'No response generated')
    fmt.print_section("CLINICAL RESPONSE", response, fmt.GREEN)

    # ==========================================================================
    # Summary
    # ==========================================================================
    fmt.print_header("Demonstration Complete")

    print(f"{fmt.BOLD}Summary:{fmt.RESET}")
    print(f"  - Clinical data processed: {len(clinical_data)} fields")
    print(f"  - Question length: {len(question)} characters")
    print(f"  - Context retrieved: {'Yes' if context else 'No'}")
    print(f"  - Response generated: {'Yes' if response else 'No'}")
    print(f"  - Response length: {len(response)} characters\n")

    return result


def run_custom_dialogue(
    clinical_data: Dict[str, Any],
    question: str,
    llm_name: str = LLMModelName.DEEPSEEK_CHAT
) -> Dict[str, Any]:
    """
    Run a custom clinical dialogue with user-provided data.

    This is a simplified interface for programmatic use.

    Args:
        clinical_data: Dictionary with clinical fields:
            - Age, Gender, AdmissionDate, DischargeDate
            - AdmissionDiagnosis, DischargeDiagnosis
            - ClinicalHistory, DifferentialDiagnosis
        question: The clinical question to answer
        llm_name: Name of the LLM to use

    Returns:
        Dictionary with 'response' and 'context' keys

    Example:
        >>> clinical_data = {
        ...     "Age": "55",
        ...     "Gender": "Female",
        ...     "ClinicalHistory": "Progressive vision loss...",
        ...     "DischargeDiagnosis": "Diabetic retinopathy"
        ... }
        >>> question = "What treatment options are available?"
        >>> result = run_custom_dialogue(clinical_data, question)
        >>> print(result['response'])
    """
    clinical_data_str = json.dumps(clinical_data, indent=2, ensure_ascii=False)

    input_data = {
        'question': question,
        'clinical_data': clinical_data_str,
        'responding_llm': llm_name,
        'messages': [question],
        'question_id': 'custom',
        'temperature': 0.0,
        'lightrag_mode': DEFAULT_LIGHTRAG_MODE,
    }

    result = execute_agent_graph(
        agent=lightrag_distill_graph,
        inputs=input_data,
        verbose=False
    )

    return result


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="EyeRAG Clinical Dialogue Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (DeepSeek)
    python -m eye_rag.tools.demo_clinical_dialogue

    # Run with a specific LLM
    python -m eye_rag.tools.demo_clinical_dialogue --llm gpt-4o

    # Run quietly (less verbose)
    python -m eye_rag.tools.demo_clinical_dialogue --quiet

Available LLMs:
    - deepseek-chat (default)
    - gpt-4o
    - gpt-4o-mini
    - gemini-2.0-flash
    - claude-sonnet-4
        """
    )

    parser.add_argument(
        '--llm', '-l',
        type=str,
        default=LLMModelName.DEEPSEEK_CHAT,
        help='LLM model to use for response generation'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    # Run the demonstration
    run_clinical_dialogue_demo(
        llm_name=args.llm,
        verbose=not args.quiet
    )
