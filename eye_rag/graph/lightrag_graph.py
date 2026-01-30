from langgraph.graph import StateGraph, END
from eye_rag.graph_node import (
    EyeRAGGraphState, answer_question_with_context, rewrite_question,
)
from eye_rag.graph_node.lightrag import retrieve_medical_guide_rewritten_questions


workflow = StateGraph(EyeRAGGraphState)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("retrieve_medical_guide", retrieve_medical_guide_rewritten_questions)
workflow.add_node("generate_response", answer_question_with_context)

workflow.set_entry_point("rewrite_question")
workflow.add_edge("rewrite_question", "retrieve_medical_guide")
workflow.add_edge("retrieve_medical_guide", "generate_response")
workflow.add_edge("generate_response", END)
graph = workflow.compile()


