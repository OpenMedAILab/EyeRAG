"""
Hypothetical RAG pipeline graph using FAISS.

This graph implements a hypothetical document RAG pipeline:
1. Generate direct answer with clinical data (creates hypothetical document)
2. Retrieve medical guide using FAISS naive RAG with hypothetical question
3. Generate final response with context
"""

from langgraph.graph import StateGraph, END
from eye_rag.graph_node import (
    EyeRAGGraphState,
    answer_question_with_context,
    answer_question_with_clinical_data,
)

# Use FAISS-based naive RAG (default)
from eye_rag.graph_node.naive_rag import retrieve_medical_guide_hypothetical_question


workflow = StateGraph(EyeRAGGraphState)
workflow.add_node('direct_answer', answer_question_with_clinical_data)
workflow.add_node("retrieve_medical_guide", retrieve_medical_guide_hypothetical_question)
workflow.add_node("generate_response", answer_question_with_context)

workflow.set_entry_point("direct_answer")
workflow.add_edge("direct_answer", "retrieve_medical_guide")
workflow.add_edge("retrieve_medical_guide", "generate_response")
workflow.add_edge("generate_response", END)
graph = workflow.compile()
