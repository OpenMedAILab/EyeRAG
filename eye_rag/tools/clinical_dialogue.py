from eye_rag.qa.patient_data import get_all_question_ids
from eye_rag.llm import LLMModelName
from eye_rag.tools.eval_eye_rag import eval_rank, Analyze

response_llm_names = [
    LLMModelName.DEEPSEEK_CHAT,
    # LLMModelName.GEMINI_2_5_FLASH,
    # LLMModelName.GPT_4o,
    # LLMModelName.CLAUDE_SONNET_4,
    # LLMModelName.GROK_4,
    # LLMModelName.LLAMA_3_70B,
]


llm_answer_type = [
    'LLM_Response',
    'LLM_NaiveRAG_Response',
    'LLM_HypotheticalRAG_Response',
    'LLM_HierarchicalIndexRAG_Response',
    "LLM_LightRAG_Hybrid_Distillation_Response",
]

out_dir = f'RESULTS/eye_rag'
my_tool = Analyze(
        json_response_dir=out_dir,
        response_llm_names=response_llm_names,
        llm_answer_type=llm_answer_type,
)
question_ids = get_all_question_ids()[:2]

my_tool.answer_question(question_ids=question_ids)


ranking_llm_names =  [
    # LLMModelName.LLAMA_3_70B,
    # LLMModelName.GPT_4o,
    # LLMModelName.GEMINI_2_0_FLASH,
    LLMModelName.DEEPSEEK_CHAT,
]

eval_rank(
    json_response_dir=out_dir,
    compare_llm_answer_type=llm_answer_type,
    response_llm_names=response_llm_names,
    question_ids=question_ids,
    ranking_llm_names=ranking_llm_names,
)
