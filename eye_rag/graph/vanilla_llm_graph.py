from langgraph.graph import StateGraph, END
from eye_rag.graph_node import EyeRAGGraphState
from eye_rag.graph_node.answer_question import answer_question_with_clinical_data

workflow = StateGraph(EyeRAGGraphState)
workflow.add_node("generate_response", answer_question_with_clinical_data)

workflow.set_entry_point("generate_response")
workflow.add_edge("generate_response", END)
graph = workflow.compile()
