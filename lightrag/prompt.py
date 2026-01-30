from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"  # For chinese medical guide, change to English if for international medical guide
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# By gemini
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "medical_condition",      # Specific diseases or disorders mentioned in the text (e.g., primary open-angle glaucoma, neovascular AMD)
    "anatomical_structure",   # Anatomical parts of the eye or body (e.g., cornea, retina, optic nerve)
    "diagnostic_procedure",   # Tests, examinations, or imaging techniques (e.g., optical coherence tomography, gonioscopy, visual fields)
    "treatment_method",       # Specific procedures or therapies (e.g., vitrectomy, laser photocoagulation, cataract surgery)
    "medication",             # Specific drugs or classes of drugs (e.g., ranibizumab, aflibercept, corticosteroids)
    "clinical_finding",       # Symptoms or observable signs (e.g., elevated intraocular pressure, drusen, retinal hemorrhages)
    "clinical_outcome",       # Results or complications of a condition or treatment (e.g., improved visual acuity, disease progression, endophthalmitis)
    "risk_factor",            # Factors that increase disease likelihood (e.g., age, diabetes, hypertension)
    "patient_population",     # Specific groups of patients being discussed (e.g., pediatric patients, pregnant patients, adults)
    "medical_device",         # Physical instruments, implants, or equipment (e.g., contact lenses, intraocular lenses, stents)
]

PROMPTS["DEFAULT_USER_PROMPT"] = "n/a"


PROMPTS["entity_extraction"] = """---Goal---
Given a text document that is a part of a medical guide entitled {title} and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name
- entity_type: One of the following types: [{entity_types}]
- entity_description: Provide a comprehensive description of the entity's attributes and activities *based solely on the information present in the input text*. **Do not infer or hallucinate information not explicitly stated.** If the text provides insufficient information to create a comprehensive description, state "Description not available in text."
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Entity_types: [{entity_types}]
Text:
{input_text}
######################
Output:"""
#


PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Medical Guide Title: "Primary Open-Angle Glaucoma Preferred Practice Pattern"
Text Chunk:
```
Risk factors for primary open-angle glaucoma include advanced age, elevated intraocular pressure, and family history. The diagnosis section covers comprehensive eye examination including gonioscopy and visual field testing. Management strategies involve both medical therapy and surgical interventions when indicated.
```

Output:
("entity"{tuple_delimiter}"Primary Open-Angle Glaucoma"{tuple_delimiter}"medical_condition"{tuple_delimiter}"Primary Open-Angle Glaucoma is the primary disease discussed in this guide."){record_delimiter}
("entity"{tuple_delimiter}"Age"{tuple_delimiter}"risk_factor"{tuple_delimiter}"Advanced age is listed as a risk factor for primary open-angle glaucoma."){record_delimiter}
("entity"{tuple_delimiter}"Elevated intraocular pressure"{tuple_delimiter}"clinical_finding"{tuple_delimiter}"Elevated intraocular pressure is noted as a clinical finding associated with glaucoma."){record_delimiter}
("entity"{tuple_delimiter}"Gonioscopy"{tuple_delimiter}"diagnostic_procedure"{tuple_delimiter}"Gonioscopy is a diagnostic procedure mentioned for glaucoma assessment."){record_delimiter}
("entity"{tuple_delimiter}"Medical therapy"{tuple_delimiter}"treatment_method"{tuple_delimiter}"Medical therapy is described as part of the management strategy."){record_delimiter}
("relationship"{tuple_delimiter}"Primary Open-Angle Glaucoma"{tuple_delimiter}"Age"{tuple_delimiter}"Advanced age increases the risk of Primary Open-Angle Glaucoma."{tuple_delimiter}"clinical_relation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Primary Open-Angle Glaucoma"{tuple_delimiter}"Gonioscopy"{tuple_delimiter}"Gonioscopy is used to help diagnose Primary Open-Angle Glaucoma."{tuple_delimiter}"diagnosed_by"{tuple_delimiter}10){record_delimiter}
("content_keywords"{tuple_delimiter}"glaucoma, risk factors, gonioscopy, medical therapy"){completion_delimiter}
#############################""",

    """Example 2:

Medical Guide Title: "Age-Related Macular Degeneration Preferred Practice Pattern"
Text Chunk:
```
The epidemiology of age-related macular degeneration shows increasing prevalence with age. Prevention strategies focus on lifestyle modifications and nutritional supplements. Treatment recommendations differ between dry and wet AMD, with anti-VEGF therapy being the standard care for neovascular AMD.
```

Output:
("entity"{tuple_delimiter}"Age-Related Macular Degeneration"{tuple_delimiter}"medical_condition"{tuple_delimiter}"Age-Related Macular Degeneration is the primary disease addressed in this guide."){record_delimiter}
("entity"{tuple_delimiter}"Increasing prevalence with age"{tuple_delimiter}"risk_factor"{tuple_delimiter}"Older age is associated with higher prevalence of AMD."){record_delimiter}
("entity"{tuple_delimiter}"Nutritional supplements"{tuple_delimiter}"treatment_method"{tuple_delimiter}"Nutritional supplements are mentioned as part of prevention strategies."){record_delimiter}
("entity"{tuple_delimiter}"Anti-VEGF therapy"{tuple_delimiter}"medication"{tuple_delimiter}"Anti-VEGF therapy is the standard treatment for neovascular (wet) AMD."){record_delimiter}
("relationship"{tuple_delimiter}"Age-Related Macular Degeneration"{tuple_delimiter}"Anti-VEGF therapy"{tuple_delimiter}"Anti-VEGF therapy is recommended for neovascular AMD."{tuple_delimiter}"treatment_for"{tuple_delimiter}10){record_delimiter}
("content_keywords"{tuple_delimiter}"AMD, prevention, anti-VEGF, supplements"){completion_delimiter}
#############################""",

    """Example 3:

Medical Guide Title: "Cataract in the Adult Eye Preferred Practice Pattern"
Text Chunk:
```
Preoperative evaluation includes assessment of visual symptoms and comprehensive eye examination. The surgical techniques section covers phacoemulsification and intraocular lens selection. Postoperative care involves monitoring for complications and visual rehabilitation.
```

Output:
("entity"{tuple_delimiter}"Cataract"{tuple_delimiter}"medical_condition"{tuple_delimiter}"Cataract is the condition addressed in this guide."){record_delimiter}
("entity"{tuple_delimiter}"Phacoemulsification"{tuple_delimiter}"treatment_method"{tuple_delimiter}"Phacoemulsification is a surgical technique for cataract removal."){record_delimiter}
("entity"{tuple_delimiter}"Intraocular lens"{tuple_delimiter}"medical_device"{tuple_delimiter}"Intraocular lens selection is discussed as part of surgical planning."){record_delimiter}
("entity"{tuple_delimiter}"Visual rehabilitation"{tuple_delimiter}"clinical_outcome"{tuple_delimiter}"Visual rehabilitation is an expected postoperative outcome."){record_delimiter}
("relationship"{tuple_delimiter}"Cataract"{tuple_delimiter}"Phacoemulsification"{tuple_delimiter}"Phacoemulsification is a primary surgical treatment for cataract."{tuple_delimiter}"treatment_for"{tuple_delimiter}10){record_delimiter}
("content_keywords"{tuple_delimiter}"cataract, phacoemulsification, intraocular lens, postoperative care"){completion_delimiter}
#############################"""
]



PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
---Data---
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS["entity_continue_extraction"] = """
MANY entities and relationships were missed in the last extraction. Please find only the missing entities and relationships from previous text.

---Remember Steps---

1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name
- entity_type: One of the following types: [{entity_types}]
- entity_description: Provide a comprehensive description of the entity's attributes and activities *based solely on the information present in the input text*. **Do not infer or hallucinate information not explicitly stated.** If the text provides insufficient information to create a comprehensive description, state "Description not available in text."
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

---Output---

Add new entities and relations below using the same format, and do not include entities and relations that have been previously extracted. :\n
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---Goal---'

It appears some entities may have still been missed.

---Output---

Answer ONLY by `YES` OR `NO` if there are still entities that need to be added.
""".strip()

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to user query about Knowledge Graph and Document Chunks provided in JSON format below.


---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both current query and the conversation history if provided. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

---Conversation History---
{history}

---Knowledge Graph and Document Chunks---
{context_data}

---RESPONSE GUIDELINES---
**1. Content & Adherence:**
- Strictly adhere to the provided context from the Knowledge Base. Do not invent, assume, or include any information not present in the source data.
- If the answer cannot be found in the provided context, state that you do not have enough information to answer.
- Ensure the response maintains continuity with the conversation history.

**2. Formatting & Language:**
- Format the response using markdown with appropriate section headings.
- The response language must in the same language as the user's question.
- Target format and length: {response_type}

**3. Citations / References:**
- At the end of the response, under a "References" section, each citation must clearly indicate its origin (KG or DC).
- The maximum number of citations is 5, including both KG and DC.
- Use the following formats for citations:
  - For a Knowledge Graph Entity: `[KG] <entity_name>`
  - For a Knowledge Graph Relationship: `[KG] <entity1_name> - <entity2_name>`
  - For a Document Chunk: `[DC] <file_path_or_document_name>`

---USER CONTEXT---
- Additional user prompt: {user_prompt}


Response:"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "What is the long-term prognosis for my left eye diabetic retinopathy?"

Output:
{
  "high_level_keywords": ["Diabetic retinopathy prognosis", "Long-term visual outcome", "Disease progression"],
  "low_level_keywords": ["diabetic retinopathy", "macular edema", "proliferative changes", "visual acuity", ]
}

""",
    """Example 2:

Query: "I had cataract surgery in my right eye last week and now have blurred vision and eye pain — could this be an infection?"

Output:
{
  "high_level_keywords": ["Postoperative complications", "Cataract surgery recovery", "Infection concern"],
  "low_level_keywords": ["blurred vision", "eye pain", "endophthalmitis", "decreased vision", ]
}

"""
    
    """Example 3:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",

    """Example 4:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided provided in JSON format below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

---Document Chunks(DC)---
{content_data}

---RESPONSE GUIDELINES---
**1. Content & Adherence:**
- Strictly adhere to the provided context from the Knowledge Base. Do not invent, assume, or include any information not present in the source data.
- If the answer cannot be found in the provided context, state that you do not have enough information to answer.
- Ensure the response maintains continuity with the conversation history.

**2. Formatting & Language:**
- Format the response using markdown with appropriate section headings.
- The response language must match the user's question language.
- Target format and length: {response_type}

**3. Citations / References:**
- At the end of the response, under a "References" section, cite a maximum of 5 most relevant sources used.
- Use the following formats for citations: `[DC] <file_path_or_document_name>`

---USER CONTEXT---
- Additional user prompt: {user_prompt}


Response:"""
