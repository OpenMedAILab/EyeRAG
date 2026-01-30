import json
from datetime import datetime
import pandas as pd

from eye_rag.utils import load_csv

TOTAL_NUM_QUESTION = 270
NUM_DISEASE = 3

def get_clinical_data_for_query(patient_data):
    """remove irrelated information from patient_data for questioning
    :param patient_data:
        # ['QuestionID', 'patient_id', '入院日期', '出院日期', '年龄', '性别', '入院诊断', '出院诊断',
    # '主诉+现病史+既往史及手术外伤史+专科情况', '鉴别诊断', 'Question',
    # 'Question_Chinese', 'Question', 'llm_name', 'llm_model_name',
    # 'system_prompt', 'user_prompt', 'response', 'response_time']

    # pop_keys = ['Question', 'Question_Chinese', 'patient_id', 'QuestionID'], ...
    :return:
    """
    clinical_data = {}
    # keys = [ '年龄', '性别', '入院日期', '出院日期','入院诊断', '出院诊断', '主诉+现病史+既往史及手术外伤史+专科情况', '鉴别诊断',]
    keys = ['Age', 'Gender', 'AdmissionDate', 'DischargeDate', 'AdmissionDiagnosis', 'DischargeDiagnosis', 'ClinicalHistory',
            'DifferentialDiagnosis', ]

    clinical_data = {}
    for key in keys:
        value = patient_data.get(key)
        # Use pd.notna() for cleaner validation
        if value and value != "NaN" and pd.notna(value) and value != 'NaT':
            clinical_data[key] = value

    return clinical_data


def get_question(patient_data):
    return patient_data.get("question") or patient_data.get("Question", "")

def get_response(patient_data):
    return patient_data.get("response", "")


def show_patient_data(patient_data, K=1):
    # Print the first few entries to verify (optional)
    # For large datasets, printing the whole dict might be too much.
    # We'll print a JSON representation for better readability if it's small.
    if patient_data:
        print(f"\nFirst {K} entries of the loaded data:")
        # Convert to JSON string for pretty printing
        json_output = {}
        count = 0
        for key, value in patient_data.items():
            if count < K:
                json_output[key] = value
                count += 1
            else:
                break
        print(json.dumps(json_output, indent=4, ensure_ascii=False))
    else:
        print("No data was loaded or an error occurred.")


def get_question_ids_per_disease(fold, num_questions):
    assert 0 <= fold < NUM_DISEASE
    N = int(TOTAL_NUM_QUESTION / NUM_DISEASE)
    sequence1 = list(range(1,N+1)[fold::NUM_DISEASE][:num_questions])
    sequence2 = list(range(N+1,N*2+1)[fold::NUM_DISEASE][:num_questions])
    sequence3 = list(range(N*2+1,TOTAL_NUM_QUESTION+1)[fold::NUM_DISEASE][:num_questions])
    question_ids = sequence1 + sequence2 + sequence3
    return question_ids

def get_question_ids_first_k(num_questions, start_id=1,):
    assert 0 < start_id < num_questions
    question_ids = list(range(start_id,num_questions+1))
    return question_ids

def get_all_question_ids():
    question_ids = list(range(1, TOTAL_NUM_QUESTION + 1))
    return question_ids

def get_external_data_question_ids():
    question_ids = list(range(TOTAL_NUM_QUESTION+1, TOTAL_NUM_QUESTION + 31))
    return question_ids

def patient_data_to_str(patient_data: dict) -> str:
    """
    Converts a patient data dictionary into a human-readable JSON string.

    Args:
        patient_data (dict): Dictionary containing patient's medical information.

    Returns:
        str: JSON string of the patient data, pretty-printed and preserving non-ASCII characters.
    """
    return json.dumps(patient_data, indent=2, ensure_ascii=False)


def generate_general_query(clinical_data: str, question: str) -> str:
    """
    Merge patient clinical data and user question into a single query string for general retrieval.
    Args:
        clinical_data (str): Patient's clinical data.
        question (str): User's query.
    Returns:
        str: Merged query string.
    """
    general_query = f"Patient Data: {clinical_data}; User Query: {question}"
    return general_query


def init_patient_data(data_file):
    loaded_patient_data = load_dialog_data(data_file)
    # show_patient_data(loaded_patient_data)
    return loaded_patient_data


def load_dialog_data(file_path):
    """
    Loads data from a CSV file into a dictionary.
    The 'ID' column is used as the key for the main dictionary.
    The remaining columns for each row are stored as a nested dictionary,
    which serves as the value for the main dictionary.

    Args:
        file_path (str): The path to the CSV file.
    Returns:
        dict: A dictionary where keys are 'ID' values and values are
              dictionaries of the remaining row information.
    """
    try:
        # Load the CSV file into a pandas DataFrame
        df = load_csv(file_path)
        # print(f"Successfully loaded data from '{file_path}'.")

        # Fill NaN values with empty strings before processing
        df = df.fillna('')

        # Initialize an empty dictionary to store the processed data
        patient_data = {}

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            # Get the 'ID' value to use as the key
            ID = row['ID']

            # Create a dictionary for the remaining information in the row
            # Exclude the 'ID' column from this nested dictionary
            other_info = row.drop('ID').to_dict()
            other_info = {k: v for k, v in other_info.items() if v != ''}

            # Convert any Timestamp objects within other_info to string format
            # This prevents TypeError when serializing to JSON
            for key, value in other_info.items():
                if isinstance(value, pd.Timestamp):
                    other_info[key] = value.isoformat()  # Convert to ISO 8601 string
                elif isinstance(value, datetime):  # Also handle standard datetime objects if any
                    other_info[key] = value.isoformat()

            # Optional filtering (currently commented out)
            # if other_info['出入院科室'] != '眼科':
            #     continue
            new_infor = {}
            for key, value in other_info.items():
                # Use pd.notna() for cleaner validation
                if value and value != "NaN" and pd.notna(value) and value != 'NaT':
                    new_infor[key] = value

            # Add the entry to the main patient_data dictionary
            patient_data[ID] = new_infor

        return patient_data

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return {}
    except KeyError:
        print("Error: The 'ID' column was not found in the CSV file. Please ensure the column name is correct.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}
