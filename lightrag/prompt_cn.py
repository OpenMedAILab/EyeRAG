from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "Chinese"  # 中文医学指南
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# 实体类型（中文）
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "疾病",           # 文本中提到的具体疾病或病症（如：原发性开角型青光眼、新生血管性AMD）
    "解剖结构",       # 眼部或身体的解剖部位（如：角膜、视网膜、视神经）
    "诊断方法",       # 检查、检验或影像技术（如：光学相干断层扫描、前房角镜检查、视野检查）
    "治疗方法",       # 具体的手术或治疗方式（如：玻璃体切除术、激光光凝术、白内障手术）
    "药物",           # 具体药物或药物类别（如：雷珠单抗、阿柏西普、皮质类固醇）
    "临床表现",       # 症状或可观察到的体征（如：眼压升高、玻璃膜疣、视网膜出血）
    "临床结局",       # 疾病或治疗的结果或并发症（如：视力改善、疾病进展、眼内炎）
    "危险因素",       # 增加疾病可能性的因素（如：年龄、糖尿病、高血压）
    "患者人群",       # 讨论的特定患者群体（如：儿童患者、孕妇、成人）
    "医疗器械",       # 物理器械、植入物或设备（如：隐形眼镜、人工晶体、支架）
]

PROMPTS["DEFAULT_USER_PROMPT"] = "无"


PROMPTS["entity_extraction"] = """---目标---
给定一份来自名为《{title}》医学指南的文本文档和一个实体类型列表，从文本中识别出所有这些类型的实体，以及所识别实体之间的所有关系。
使用{language}作为输出语言。

---步骤---
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name：实体名称，使用与输入文本相同的语言。如果是英文，首字母大写
- entity_type：以下类型之一：[{entity_types}]
- entity_description：*仅根据输入文本中存在的信息*，提供实体属性和活动的综合描述。**不要推断或虚构文本中未明确说明的信息。**如果文本提供的信息不足以创建综合描述，请注明"文本中无相关描述"。
每个实体的格式为：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤1识别的实体中，找出所有*明确相关*的实体对（source_entity, target_entity）。
对于每对相关实体，提取以下信息：
- source_entity：源实体的名称，如步骤1所识别
- target_entity：目标实体的名称，如步骤1所识别
- relationship_description：解释为什么认为源实体和目标实体相互关联
- relationship_strength：表示源实体和目标实体之间关系强度的数值分数
- relationship_keywords：一个或多个高层次关键词，概括关系的整体性质，侧重于概念或主题而非具体细节
每个关系的格式为：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 识别概括整个文本主要概念、主题或话题的高层次关键词。这些关键词应该捕捉文档中呈现的总体思想。
内容级关键词的格式为：("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以{language}返回输出，作为步骤1和步骤2中识别的所有实体和关系的单一列表。使用**{record_delimiter}**作为列表分隔符。

5. 完成后，输出{completion_delimiter}

######################
---示例---
######################
{examples}

#############################
---实际数据---
######################
实体类型：[{entity_types}]
文本：
{input_text}
######################
输出："""


PROMPTS["entity_extraction_examples"] = [
    """示例1：

医学指南标题：《原发性开角型青光眼诊疗规范》
文本片段：
```
原发性开角型青光眼的危险因素包括高龄、眼压升高和家族史。诊断部分涵盖全面眼科检查，包括前房角镜检查和视野检查。治疗策略包括药物治疗和必要时的手术干预。
```

输出：
("entity"{tuple_delimiter}"原发性开角型青光眼"{tuple_delimiter}"疾病"{tuple_delimiter}"原发性开角型青光眼是本指南讨论的主要疾病。"){record_delimiter}
("entity"{tuple_delimiter}"高龄"{tuple_delimiter}"危险因素"{tuple_delimiter}"高龄被列为原发性开角型青光眼的危险因素。"){record_delimiter}
("entity"{tuple_delimiter}"眼压升高"{tuple_delimiter}"临床表现"{tuple_delimiter}"眼压升高是与青光眼相关的临床表现。"){record_delimiter}
("entity"{tuple_delimiter}"前房角镜检查"{tuple_delimiter}"诊断方法"{tuple_delimiter}"前房角镜检查是用于青光眼评估的诊断方法。"){record_delimiter}
("entity"{tuple_delimiter}"药物治疗"{tuple_delimiter}"治疗方法"{tuple_delimiter}"药物治疗被描述为治疗策略的一部分。"){record_delimiter}
("relationship"{tuple_delimiter}"原发性开角型青光眼"{tuple_delimiter}"高龄"{tuple_delimiter}"高龄增加了原发性开角型青光眼的风险。"{tuple_delimiter}"危险因素关联"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"原发性开角型青光眼"{tuple_delimiter}"前房角镜检查"{tuple_delimiter}"前房角镜检查用于帮助诊断原发性开角型青光眼。"{tuple_delimiter}"诊断依据"{tuple_delimiter}10){record_delimiter}
("content_keywords"{tuple_delimiter}"青光眼，危险因素，前房角镜检查，药物治疗"){completion_delimiter}
#############################""",

    """示例2：

医学指南标题：《年龄相关性黄斑变性诊疗规范》
文本片段：
```
年龄相关性黄斑变性的流行病学显示患病率随年龄增长而升高。预防策略侧重于生活方式调整和营养补充剂。干性和湿性AMD的治疗建议不同，抗VEGF治疗是新生血管性AMD的标准治疗方案。
```

输出：
("entity"{tuple_delimiter}"年龄相关性黄斑变性"{tuple_delimiter}"疾病"{tuple_delimiter}"年龄相关性黄斑变性是本指南讨论的主要疾病。"){record_delimiter}
("entity"{tuple_delimiter}"患病率随年龄增长"{tuple_delimiter}"危险因素"{tuple_delimiter}"高龄与AMD患病率升高相关。"){record_delimiter}
("entity"{tuple_delimiter}"营养补充剂"{tuple_delimiter}"治疗方法"{tuple_delimiter}"营养补充剂被提及为预防策略的一部分。"){record_delimiter}
("entity"{tuple_delimiter}"抗VEGF治疗"{tuple_delimiter}"药物"{tuple_delimiter}"抗VEGF治疗是新生血管性（湿性）AMD的标准治疗方案。"){record_delimiter}
("relationship"{tuple_delimiter}"年龄相关性黄斑变性"{tuple_delimiter}"抗VEGF治疗"{tuple_delimiter}"抗VEGF治疗被推荐用于新生血管性AMD。"{tuple_delimiter}"治疗方案"{tuple_delimiter}10){record_delimiter}
("content_keywords"{tuple_delimiter}"AMD，预防，抗VEGF，补充剂"){completion_delimiter}
#############################""",

    """示例3：

医学指南标题：《成人白内障诊疗规范》
文本片段：
```
术前评估包括视觉症状评估和全面眼科检查。手术技术部分涵盖超声乳化术和人工晶体选择。术后护理包括监测并发症和视力康复。
```

输出：
("entity"{tuple_delimiter}"白内障"{tuple_delimiter}"疾病"{tuple_delimiter}"白内障是本指南讨论的疾病。"){record_delimiter}
("entity"{tuple_delimiter}"超声乳化术"{tuple_delimiter}"治疗方法"{tuple_delimiter}"超声乳化术是白内障摘除的手术技术。"){record_delimiter}
("entity"{tuple_delimiter}"人工晶体"{tuple_delimiter}"医疗器械"{tuple_delimiter}"人工晶体选择作为手术规划的一部分进行讨论。"){record_delimiter}
("entity"{tuple_delimiter}"视力康复"{tuple_delimiter}"临床结局"{tuple_delimiter}"视力康复是预期的术后结果。"){record_delimiter}
("relationship"{tuple_delimiter}"白内障"{tuple_delimiter}"超声乳化术"{tuple_delimiter}"超声乳化术是白内障的主要手术治疗方法。"{tuple_delimiter}"治疗方案"{tuple_delimiter}10){record_delimiter}
("content_keywords"{tuple_delimiter}"白内障，超声乳化术，人工晶体，术后护理"){completion_delimiter}
#############################"""
]



PROMPTS[
    "summarize_entity_descriptions"
] = """你是一个负责生成以下数据综合摘要的助手。
给定一个或两个实体，以及一个描述列表，所有描述都与同一个实体或实体组相关。
请将所有这些内容合并为一个单一的综合描述。确保包含从所有描述中收集的信息。
如果提供的描述相互矛盾，请解决矛盾并提供一个单一、连贯的摘要。
确保以第三人称书写，并包含实体名称以便我们拥有完整的上下文。
使用{language}作为输出语言。

#######
---数据---
实体：{entity_name}
描述列表：{description_list}
#######
输出：
"""

PROMPTS["entity_continue_extraction"] = """
上次提取中遗漏了许多实体和关系。请仅从之前的文本中找出遗漏的实体和关系。

---请记住步骤---

1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name：实体名称，使用与输入文本相同的语言。如果是英文，首字母大写
- entity_type：以下类型之一：[{entity_types}]
- entity_description：*仅根据输入文本中存在的信息*，提供实体属性和活动的综合描述。**不要推断或虚构文本中未明确说明的信息。**如果文本提供的信息不足以创建综合描述，请注明"文本中无相关描述"。
每个实体的格式为：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤1识别的实体中，找出所有*明确相关*的实体对（source_entity, target_entity）。
对于每对相关实体，提取以下信息：
- source_entity：源实体的名称，如步骤1所识别
- target_entity：目标实体的名称，如步骤1所识别
- relationship_description：解释为什么认为源实体和目标实体相互关联
- relationship_strength：表示源实体和目标实体之间关系强度的数值分数
- relationship_keywords：一个或多个高层次关键词，概括关系的整体性质，侧重于概念或主题而非具体细节
每个关系的格式为：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 识别概括整个文本主要概念、主题或话题的高层次关键词。这些关键词应该捕捉文档中呈现的总体思想。
内容级关键词的格式为：("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以{language}返回输出，作为步骤1和步骤2中识别的所有实体和关系的单一列表。使用**{record_delimiter}**作为列表分隔符。

5. 完成后，输出{completion_delimiter}

---输出---

请使用相同格式在下方添加新的实体和关系，不要包含已经提取过的实体和关系：
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---目标---

看起来可能还有一些实体被遗漏了。

---输出---

请仅回答"是"或"否"，表示是否还有需要添加的实体。
""".strip()

PROMPTS["fail_response"] = (
    "抱歉，我无法回答这个问题。[无上下文]"
)

PROMPTS["rag_response"] = """---角色---

你是一个根据下方JSON格式提供的知识图谱和文档片段回答用户问题的助手。


---目标---

基于知识库生成简洁的回答，遵循回答规则，同时考虑当前查询和对话历史（如果提供）。总结所提供知识库中的所有信息，并结合与知识库相关的通用知识。不要包含知识库未提供的信息。

---对话历史---
{history}

---知识图谱和文档片段---
{context_data}

---回答准则---
**1. 内容与遵循：**
- 严格遵循知识库提供的上下文。不要编造、假设或包含源数据中不存在的任何信息。
- 如果在提供的上下文中找不到答案，请说明您没有足够的信息来回答。
- 确保回答与对话历史保持连贯。

**2. 格式与语言：**
- 使用markdown格式化回答，包含适当的章节标题。
- 回答语言必须与用户问题的语言相同。
- 目标格式和长度：{response_type}

**3. 引用/参考：**
- 在回答末尾的"参考文献"部分，每个引用必须清楚标明其来源（KG或DC）。
- 最多引用5条，包括KG和DC。
- 使用以下格式引用：
  - 知识图谱实体：`[KG] <实体名称>`
  - 知识图谱关系：`[KG] <实体1名称> - <实体2名称>`
  - 文档片段：`[DC] <文件路径或文档名称>`

---用户上下文---
- 附加用户提示：{user_prompt}


回答："""

PROMPTS["keywords_extraction"] = """---角色---
你是一个专业的关键词提取专家，专门为检索增强生成（RAG）系统分析用户查询。你的目的是识别用户查询中的高层次和低层次关键词，这些关键词将用于有效的文档检索。

---目标---
给定一个用户查询，你的任务是提取两种不同类型的关键词：
1. **high_level_keywords**：用于总体概念或主题，捕捉用户的核心意图、主题领域或所提问题的类型。
2. **low_level_keywords**：用于具体实体或细节，识别具体的实体、专有名词、专业术语、产品名称或具体项目。

---说明与约束---
1. **输出格式**：你的输出必须是有效的JSON对象，不包含其他内容。不要包含任何解释性文本、markdown代码块（如```json）或JSON前后的任何其他文本。它将直接由JSON解析器解析。
2. **信息来源**：所有关键词必须明确来源于用户查询，高层次和低层次关键词类别都必须包含内容。
3. **简洁且有意义**：关键词应该是简洁的词语或有意义的短语。当多个词代表单一概念时，优先使用多词短语。例如，从"苹果公司的最新财务报告"中，你应该提取"最新财务报告"和"苹果公司"，而不是"最新"、"财务"、"报告"和"苹果"。
4. **处理边缘情况**：对于过于简单、模糊或无意义的查询（如"你好"、"好的"、"asdfghjkl"），你必须返回两个关键词类型都为空列表的JSON对象。

---示例---
{examples}

---实际数据---
用户查询：{query}

---输出---
"""

PROMPTS["keywords_extraction_examples"] = [
    """示例1：

查询："我左眼糖尿病视网膜病变的长期预后如何？"

输出：
{
  "high_level_keywords": ["糖尿病视网膜病变预后", "长期视力结果", "疾病进展"],
  "low_level_keywords": ["糖尿病视网膜病变", "黄斑水肿", "增殖性改变", "视力"]
}

""",
    """示例2：

查询："我上周右眼做了白内障手术，现在视力模糊还有眼痛——这会是感染吗？"

输出：
{
  "high_level_keywords": ["术后并发症", "白内障手术恢复", "感染担忧"],
  "low_level_keywords": ["视力模糊", "眼痛", "眼内炎", "视力下降"]
}

""",

    """示例3：

查询："国际贸易如何影响全球经济稳定？"

输出：
{
  "high_level_keywords": ["国际贸易", "全球经济稳定", "经济影响"],
  "low_level_keywords": ["贸易协定", "关税", "货币汇率", "进口", "出口"]
}

""",

    """示例4：

查询："教育在减少贫困中起什么作用？"

输出：
{
  "high_level_keywords": ["教育", "减贫", "社会经济发展"],
  "low_level_keywords": ["学校入学", "识字率", "职业培训", "收入不平等"]
}

""",
]

PROMPTS["naive_rag_response"] = """---角色---

你是一个根据下方JSON格式提供的文档片段回答用户问题的助手。

---目标---

基于文档片段生成简洁的回答，遵循回答规则，同时考虑对话历史和当前查询。总结所提供文档片段中的所有信息，并结合与文档片段相关的通用知识。不要包含文档片段未提供的信息。

---文档片段(DC)---
{content_data}

---回答准则---
**1. 内容与遵循：**
- 严格遵循知识库提供的上下文。不要编造、假设或包含源数据中不存在的任何信息。
- 如果在提供的上下文中找不到答案，请说明您没有足够的信息来回答。
- 确保回答与对话历史保持连贯。

**2. 格式与语言：**
- 使用markdown格式化回答，包含适当的章节标题。
- 回答语言必须与用户问题的语言相同。
- 目标格式和长度：{response_type}

**3. 引用/参考：**
- 在回答末尾的"参考文献"部分，最多引用5个最相关的来源。
- 使用以下格式引用：`[DC] <文件路径或文档名称>`

---用户上下文---
- 附加用户提示：{user_prompt}


回答："""