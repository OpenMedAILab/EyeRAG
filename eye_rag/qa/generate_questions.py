import json
import time

from eye_rag.llm import BaseLLM, LLMModelName
from eye_rag.qa.patient_data import patient_data_to_str

PATIENT_SYSTEM_PROMPT = (
    'You are a patient with an eye disease. Based on the diagnosis provided, generate exactly three clear and concise questions regarding your condition. Organize these questions as a JSON-formatted list of strings. Provide only the JSON output; do not include any explanations, thoughts, or additional commentary. Each list item must be exactly one complete question, with no multiple questions combined.'
    )

class PatientLLM(BaseLLM):
    """
    LLM specialized in generating patient-like questions based on medical data.
    Expected to return responses in JSON format.
    """

    def __init__(self, model: str = ''):
        super().__init__(system_prompt=PATIENT_SYSTEM_PROMPT, model=model)

    @staticmethod
    def init_client():
        """
        Initialize the specific LLM client for PatientLLM.
        Example for OpenAI:
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        """
        raise NotImplementedError("PatientLLM.init_client must be implemented with your specific LLM client.")

    def generate_response(self, query: str) -> tuple[dict, float]:
        """
        Generates a JSON response from the LLM, specifically configured for JSON output.

        Args:
            query (str): The user's input query, expecting a JSON response format.

        Returns:
            tuple[dict, float]: A tuple containing the parsed JSON dictionary and response time.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON.
        """
        start_time = time.time()
        response_content = ""
        try:
            if not self.client:
                raise ValueError("LLM client not initialized. Call init_client() in subclass.")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
                response_format={'type': 'json_object'},
                stream=False,
                temperature=0.2,
                max_tokens=2000, # Re-enable as needed from config
            )
            response_content = response.choices[0].message.content
            answer = json.loads(response_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON. Response: '{response_content}'. Error: {e}")
        except NotImplementedError as e:
            print(f"Error: {e}. Returning mock JSON response.")
            answer = {
                "questions": ["Mock Question 1 (Client Not Initialized)?", "Mock Question 2?", "Mock Question 3?"]}
        except Exception as e:
            print(f"Error generating response in PatientLLM (model: {self.model}): {e}")
            answer = {"error": "Could not generate response due to an API issue."}

        end_time = time.time()
        response_time = end_time - start_time

        if self.display_response:
            print(f"Execution Time: {response_time:.2f} seconds")
            print(json.dumps(answer, indent=2, ensure_ascii=False))
        return answer, response_time

    def generate_questions(self, patient_data: dict) -> list[str]:
        """
        Generates a list of relevant medical questions for the patient based on their data.
        The LLM is instructed to return a JSON object with a 'questions' key.

        Args:
            patient_data (dict): Dictionary containing the patient's medical information.

        Returns:
            list[str]: A list of generated questions.
        """
        patient_data_json_str = patient_data_to_str(patient_data)

        prompt = f"""
        Based on the following patient data, please generate three distinct and relevant medical questions
        that a patient might ask about their condition, diagnosis, or treatment.
        The questions should be clear, concise, and directly related to the provided information.

        Output Format:
        {{
            "questions": [
                "Question 1?",
                "Question 2?",
                "Question 3?"
            ]
        }}

        --- Patient Data ---
        {patient_data_json_str}
        """

        response_dict, _ = self.generate_response(query=prompt)
        questions = response_dict.get('questions', [])
        return questions


