import time
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models.chat_models import BaseChatModel

from config import LLM_RESPONSE_TEMPERATURE

from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# Global variable to control max_tokens for all LLMs
GLOBAL_MAX_TOKENS = 5000

# Model mappings for OpenRouter
OPENROUTER_MODEL_MAPPING = {
    # OpenAI models
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "gpt-4o": "openai/gpt-4o",

    # DeepSeek models
    "deepseek-chat": "deepseek/deepseek-chat",

    # Google Gemini models
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-2.5-flash": "google/gemini-2.5-flash",

    # Anthropic Claude models
    "claude-3-5-haiku-20241022": "anthropic/claude-3-5-haiku",
    "claude-3-5-sonnet-latest": "anthropic/claude-3-5-sonnet",
    "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",

    # xAI Grok models
    "grok-3-mini": "x-ai/grok-3",
    "grok-3": "x-ai/grok-3",
    "grok-4-0709": "x-ai/grok-4",

    # Meta Llama models
    "Llama-3.3-8B-Instruct": "meta-llama/llama-3.3-8b-instruct",
    "Llama-3.3-70B-Instruct": "meta-llama/llama-3.3-70b-instruct",
}

DISPLAY_RESPONSE = True


class LLMModelName:
    GPT_4o_MINI = "gpt-4o-mini"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4o = "gpt-4o"

    DEEPSEEK_CHAT = "deepseek-chat"

    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"

    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_5_SONNET_LATEST = "claude-3-5-sonnet-latest"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"

    GROK_3_MINI = "grok-3-mini"
    GROK_3 = "grok-3"
    GROK_4 = "grok-4-0709"

    LLAMA_3_8B = "Llama-3.3-8B-Instruct"
    LLAMA_3_70B = "Llama-3.3-70B-Instruct"


official_name_mapping = {
    LLMModelName.GPT_4o: "GPT-4o",
    LLMModelName.GEMINI_2_5_FLASH: "Gemini 2.5 Flash",
    LLMModelName.GEMINI_2_0_FLASH: "Gemini 2.0 Flash",
    LLMModelName.GROK_4: "Grok 4",
    LLMModelName.LLAMA_3_70B: "Llama 3.3 70B",
    LLMModelName.CLAUDE_SONNET_4: "Claude Sonnet 4",
    LLMModelName.DEEPSEEK_CHAT: "DeepSeek-V2.5",
}


class BaseLLM:
    """
    Base class for Large Language Models, providing common functionalities
    like client initialization and response generation using OpenRouter API.
    """

    def __init__(self, system_prompt: str = '', model: str = ''):
        self.client = self.init_client()
        self.model = model
        self.system_prompt = system_prompt
        self.display_response = DISPLAY_RESPONSE

    @staticmethod
    def init_client():
        """Initializes OpenRouter client using OPENROUTER_API_KEY from environment."""
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

        return OpenAI(
            base_url=openrouter_base_url,
            api_key=openrouter_api_key,
        )

    def disable_display(self):
        """Disables printing LLM responses to console."""
        self.display_response = False

    def generate_response(self, query: str) -> tuple[str, float]:
        """
        Generates a text response from the LLM based on the system prompt and user query.

        Args:
            query: The user's input query.

        Returns:
            A tuple containing the generated answer string and response time in seconds.
        """
        start_time = time.time()
        answer = ""
        try:
            if not self.client:
                raise ValueError("LLM client not initialized. Call init_client().")

            openrouter_model = OPENROUTER_MODEL_MAPPING.get(self.model, self.model)

            response = self.client.chat.completions.create(
                model=openrouter_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
                stream=False,
                temperature=LLM_RESPONSE_TEMPERATURE,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response from LLM (model: {self.model}): {e}")
            answer = "Error: Could not generate response due to an API issue."

        end_time = time.time()
        response_time = end_time - start_time

        if self.display_response:
            print(f"Execution Time: {response_time:.2f} seconds")
            print(answer)
        return answer, response_time


class OpenRouterLLM(BaseLLM):
    """Specific implementation of LLM using OpenRouter API."""

    def __init__(self, system_prompt: str = '', model: str = ''):
        super().__init__(system_prompt=system_prompt, model=model)


def get_chat_llm(
    model: str = 'deepseek-chat',
    temperature: float = LLM_RESPONSE_TEMPERATURE,
    max_tokens=None
) -> BaseChatModel:
    """
    Returns a LangChain ChatOpenAI instance configured for OpenRouter.

    Args:
        model: The model name to use
        temperature: The temperature for response generation
        max_tokens: Maximum number of tokens to generate (default: uses GLOBAL_MAX_TOKENS)

    Returns:
        A LangChain chat model instance
    """

    if model == LLMModelName.DEEPSEEK_CHAT:
        return get_deepseek_llm(model=model, temperature=temperature, max_tokens=max_tokens)

    if max_tokens is None:
        max_tokens = GLOBAL_MAX_TOKENS

    openrouter_model = OPENROUTER_MODEL_MAPPING.get(model, model)

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

    kwargs = {
        "temperature": temperature,
        "model": openrouter_model,
        "api_key": openrouter_api_key,
        "base_url": openrouter_base_url,
    }

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    kwargs["default_headers"] = {
        "HTTP-Referer": "https://github.com/EyeRAG",
        "X-Title": "EyeRAG",
    }

    return ChatOpenAI(**kwargs)


def get_openai_embedding():
    """Creates OpenAI embeddings instance configured for OpenRouter."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
        model="text-embedding-3-small"
    )

def get_deepseek_llm(model=LLMModelName.DEEPSEEK_CHAT, temperature=LLM_RESPONSE_TEMPERATURE,
                     max_tokens=None):
    """Initializes a DeepSeek LLM."""
    # Get DeepSeek API key from environment
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set.")

    # Use global max_tokens if not specified locally
    if max_tokens is None:
        max_tokens = GLOBAL_MAX_TOKENS

    return ChatDeepSeek(
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        api_key=deepseek_api_key,
    )

# def get_deepseek_llm(
#     model=LLMModelName.DEEPSEEK_CHAT,
#     temperature=LLM_RESPONSE_TEMPERATURE,
#     max_tokens=None
# ):
#     """Initializes a DeepSeek LLM."""
#     return get_chat_llm(temperature=temperature, model=model, max_tokens=max_tokens)


def get_open_ai_llm(
    model: str = LLMModelName.GPT_3_5_TURBO,
    temperature: float = LLM_RESPONSE_TEMPERATURE,
    max_tokens=None
) -> BaseChatModel:
    """Initializes an OpenAI LLM via OpenRouter."""
    return get_chat_llm(temperature=temperature, model=model, max_tokens=max_tokens)


def get_gemini_llm(
    model: str = LLMModelName.GEMINI_2_0_FLASH,
    temperature: float = LLM_RESPONSE_TEMPERATURE,
    max_tokens=None
) -> BaseChatModel:
    """Initializes a Google Gemini LLM via OpenRouter."""
    return get_chat_llm(temperature=temperature, model=model, max_tokens=max_tokens)


def get_llama_llm(
    model: str = LLMModelName.LLAMA_3_8B,
    temperature: float = LLM_RESPONSE_TEMPERATURE,
    max_tokens=None
) -> BaseChatModel:
    """Initializes a Llama LLM via OpenRouter."""
    return get_chat_llm(temperature=temperature, model=model, max_tokens=max_tokens)


def get_claude_llm(
    model: str = LLMModelName.CLAUDE_3_5_HAIKU,
    temperature: float = LLM_RESPONSE_TEMPERATURE,
    max_tokens=None
) -> BaseChatModel:
    """Initializes an Anthropic Claude LLM via OpenRouter."""
    return get_chat_llm(temperature=temperature, model=model, max_tokens=max_tokens)


def get_grok_llm(
    model: str = LLMModelName.GROK_3_MINI,
    temperature: float = LLM_RESPONSE_TEMPERATURE,
    max_tokens=None
) -> BaseChatModel:
    """Initializes an xAI Grok LLM via OpenRouter."""
    return get_chat_llm(temperature=temperature, model=model, max_tokens=max_tokens)
