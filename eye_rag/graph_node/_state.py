"""
EyeRAG Graph State and Execution Utilities.

This module defines the state schema for EyeRAG graph workflows
and provides utilities for executing and visualizing graphs.
"""

import operator
import sys
from typing import Annotated, TypedDict, List, Dict, Any


class EyeRAGGraphState(TypedDict):
    """State schema for EyeRAG graph workflows."""

    # Input fields
    question: str
    clinical_data: str
    general_query: str
    temperature: float

    # Query processing
    rewritten_questions: List[str]
    responding_llm: str

    # Response fields
    response: str
    follow_up_questions: List[str]
    messages: Annotated[list, operator.add]

    # Retrieval fields
    retrieved_information: List[str]
    is_retrieval_relevant: bool
    retrieval_relevance_explanation: str
    filtered_context: str | list
    context: str

    # Processing state
    initial_response_file_path: str
    critique: str
    with_hallucination: bool

    # Planning fields
    plan: List[str]
    past_steps: List[str]
    mapping: Dict[str, str]
    curr_context: str
    curr_state: str

    # RAG configuration
    lightrag_mode: str
    question_id: str
    medgraphrag_mode: str


class ProgressDisplay:
    """Handles formatted console output for graph execution progress."""

    # Box drawing characters
    BOX_TOP_LEFT = "┌"
    BOX_TOP_RIGHT = "┐"
    BOX_BOTTOM_LEFT = "└"
    BOX_BOTTOM_RIGHT = "┘"
    BOX_HORIZONTAL = "─"
    BOX_VERTICAL = "│"
    BOX_T_RIGHT = "├"
    BOX_T_LEFT = "┤"

    # Status indicators
    CHECKMARK = "✓"
    ARROW = "→"
    BULLET = "•"

    def __init__(self, width: int = 70, use_color: bool = True):
        """
        Initialize the progress display.

        Args:
            width: Display width in characters
            use_color: Whether to use ANSI color codes
        """
        self.width = width
        self.use_color = use_color and sys.stdout.isatty()

    def _color(self, text: str, code: str) -> str:
        """Apply ANSI color code if enabled."""
        if self.use_color:
            return f"\033[{code}m{text}\033[0m"
        return text

    def dim(self, text: str) -> str:
        """Dim text."""
        return self._color(text, "2")

    def bold(self, text: str) -> str:
        """Bold text."""
        return self._color(text, "1")

    def green(self, text: str) -> str:
        """Green text."""
        return self._color(text, "32")

    def cyan(self, text: str) -> str:
        """Cyan text."""
        return self._color(text, "36")

    def yellow(self, text: str) -> str:
        """Yellow text."""
        return self._color(text, "33")

    def horizontal_line(self, char: str = None) -> str:
        """Create a horizontal line."""
        char = char or self.BOX_HORIZONTAL
        return char * self.width

    def header(self, title: str) -> None:
        """Print a header box."""
        padding = (self.width - len(title) - 2) // 2
        top = f"{self.BOX_TOP_LEFT}{self.horizontal_line()}{self.BOX_TOP_RIGHT}"
        middle = f"{self.BOX_VERTICAL}{' ' * padding}{self.bold(title)}{' ' * (self.width - padding - len(title))}{self.BOX_VERTICAL}"
        bottom = f"{self.BOX_BOTTOM_LEFT}{self.horizontal_line()}{self.BOX_BOTTOM_RIGHT}"
        print(f"\n{self.cyan(top)}")
        print(self.cyan(middle))
        print(f"{self.cyan(bottom)}\n")

    def step(self, step_num: int, state_name: str) -> None:
        """Print a step completion indicator."""
        indicator = self.green(f"{self.CHECKMARK}")
        step_label = self.dim(f"Step {step_num}")
        state = self.bold(state_name) if state_name else self.dim("(unnamed)")
        print(f"  {indicator} {step_label} {self.ARROW} {state}")

    def response_preview(self, response: str, max_length: int = 150) -> None:
        """Print a preview of the response."""
        if not response:
            return

        # Truncate and clean the response
        preview = response[:max_length].replace('\n', ' ').strip()
        if len(response) > max_length:
            preview += "..."

        print(f"\n  {self.BULLET} {self.dim('Response preview:')}")
        print(f"    {self.yellow(preview)}")

    def summary(self, total_steps: int, has_response: bool) -> None:
        """Print execution summary."""
        line = self.dim(self.horizontal_line(self.BOX_HORIZONTAL))
        print(f"\n{line}")
        status = self.green("Complete") if has_response else self.yellow("No response")
        print(f"  {self.bold('Execution:')} {status} ({total_steps} steps)")
        print(f"{line}\n")


def execute_agent_graph(
    agent,
    inputs: Dict[str, Any],
    recursion_limit: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Execute the agent graph and stream results with progress display.

    Args:
        agent: The compiled LangGraph agent
        inputs: Input dictionary for the graph
        recursion_limit: Maximum recursion depth
        verbose: Whether to show progress display

    Returns:
        Dictionary containing 'response' and optionally 'context'
    """
    config = {"recursion_limit": recursion_limit}
    result: Dict[str, Any] = {'response': ''}

    display = ProgressDisplay() if verbose else None

    if display:
        display.header("EyeRAG Pipeline")

    step_count = 0
    for idx, event in enumerate(agent.stream(inputs, config=config, stream_mode="values")):
        step_count = idx + 1

        # Capture context
        if "context" in event:
            result['context'] = event['context']

        # Capture response
        if "response" in event:
            result['response'] = event["response"]

        # Display progress
        curr_state = event.get("curr_state", "")
        if display and curr_state:
            display.step(step_count, curr_state)

    # Show response preview and summary
    if display:
        if result.get('response'):
            display.response_preview(result['response'])
        display.summary(step_count, bool(result.get('response')))

    return result


def generate_graph_image(graph, out_graph_path: str = 'workflow_graph.png') -> bool:
    """
    Generate and save a workflow graph visualization.

    Args:
        graph: The compiled LangGraph agent
        out_graph_path: Output path for the PNG image

    Returns:
        True if successful, False otherwise
    """
    try:
        mermaid_png = graph.get_graph(xray=True).draw_mermaid_png()
        with open(out_graph_path, 'wb') as f:
            f.write(mermaid_png)
        print(f"Workflow graph saved to {out_graph_path}")
        return True
    except Exception as e:
        print(f"Error saving graph: {e}")
        return False
