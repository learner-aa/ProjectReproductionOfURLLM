"""
Prompt 模板管理

集中管理所有 LLM Prompt 模板，包括:
- Prompt I: 物品属性提取 (COT)
- Prompt II: 用户画像增强推荐
"""


# ============================================================
# Prompt I: 物品属性提取
# ============================================================

PROMPT_I_ATTRIBUTE_EXTRACTION_EN = """You will serve as an assistant to help me extract attributes of the following item.
When given an item, you should first introduce the item briefly, then extract its key attributes as a list.

Item Title: {title}
Item Description: {description}
Item Category: {category}

Please respond in the following format:
INTRODUCTION: <brief introduction of the item>
ATTRIBUTES: ["attribute1", "attribute2", "attribute3", ...]
"""

PROMPT_I_ATTRIBUTE_EXTRACTION_ZH = """请根据以下物品的标题和描述，提取该物品的关键属性标签。

物品标题: {title}
物品描述: {description}
物品类别: {category}

请按以下格式回复:
介绍: <物品的简要介绍>
属性: ["属性1", "属性2", "属性3", ...]
"""

# 批量提取 prompt (用于 API 调用节省 token)
PROMPT_I_BATCH = """You are an item attribute extraction assistant.
For each item below, provide a brief introduction and extract key attributes as a JSON list.

{items_block}

Respond in JSON format:
{{
  "results": [
    {{"id": "item_001", "intro": "...", "attributes": ["attr1", "attr2"]}},
    ...
  ]
}}
"""


# ============================================================
# Prompt II: 用户画像增强推荐
# ============================================================

PROMPT_II_RECOMMEND_BASE = """Instruction: Based on the user's interaction history and profile information, recommend a new {target_domain} item that the user would likely enjoy.

Input:
{user_profile_text}

User Interaction History (chronological, newest last):
{interaction_sequence}

Please recommend ONE new {target_domain} item.
Output:"""

PROMPT_II_RECOMMEND_WITH_PROFILE = """Instruction: You are a recommendation assistant. Based on the user's profile, preferences, and interaction history, recommend a new {target_domain} item.

Input:
=== User Profile ===
{user_profile_text}

=== Interaction History ===
{interaction_sequence}

=== Preference Summary ===
- Preferred attributes: {preferred_attributes}
- Preferred categories: {preferred_categories}
- Similar items the user may like: {similar_items}

Please recommend ONE specific {target_domain} item title that matches the user's preferences.
Output:"""

PROMPT_II_RECOMMEND_COT = """Instruction: You are a cross-domain recommendation assistant. Analyze the user's interaction history and preferences step by step, then recommend a new {target_domain} item.

Input:
=== User Profile ===
{user_profile_text}

=== Interaction History ===
{interaction_sequence}

Think step by step:
1. What are the user's main interests based on their history?
2. What attributes and categories does the user prefer?
3. Based on these preferences, what {target_domain} item would be a good recommendation?

Please output ONLY the recommended item title after "Output:".
Output:"""


# ============================================================
# 用户画像文本模板
# ============================================================

PROFILE_TEMPLATE_COMPACT = """Total interactions: {total_count}
Domain distribution: {domain_x_name}({x_count}), {domain_y_name}({y_count})
Recent items: {recent_items}"""

PROFILE_TEMPLATE_DETAILED = """User Behavior Summary:
- Total interactions: {total_count}
- {domain_x_name} domain: {x_count} items
- {domain_y_name} domain: {y_count} items
- Cross-domain ratio: {cross_ratio:.1%}

Top Preferred Attributes: {top_attributes}
Top Preferred Categories: {top_categories}

Similar Items (based on embedding): {similar_items}

Recent Interaction Sequence:
{recent_sequence}"""


# ============================================================
# 辅助函数
# ============================================================

def format_interaction_sequence(
    sequence: list,
    item_metadata: dict,
    max_display: int = 10,
) -> str:
    """
    将交互序列格式化为文本。

    Args:
        sequence: [{"item_id": str, "domain": str}, ...] 或 [item_id, ...]
        item_metadata: {item_id: {"title": str, "domain": str, ...}}
        max_display: 最多显示的交互数

    Returns:
        格式化文本
    """
    display_seq = sequence[-max_display:]
    lines = []
    for item in display_seq:
        if isinstance(item, dict):
            item_id = item["item_id"]
            domain = item.get("domain", "?")
        else:
            item_id = item
            domain = item_metadata.get(item_id, {}).get("domain", "?")

        title = item_metadata.get(item_id, {}).get("title", item_id)
        lines.append(f"  [{domain}] {title}")

    return "\n".join(lines)


def format_attributes(attr_list: list, top_k: int = 10) -> str:
    """格式化属性列表为文本"""
    if not attr_list:
        return "None"
    if isinstance(attr_list[0], (list, tuple)):
        # [(attr, count), ...] 格式
        top = attr_list[:top_k]
        return ", ".join(f"{attr}({count})" for attr, count in top)
    return ", ".join(attr_list[:top_k])


def get_prompt_template(template_name: str) -> str:
    """
    根据名称获取 prompt 模板。

    Args:
        template_name: 模板名称, 如 "attribute_en", "recommend_base", "recommend_cot"

    Returns:
        prompt 模板字符串
    """
    templates = {
        "attribute_en": PROMPT_I_ATTRIBUTE_EXTRACTION_EN,
        "attribute_zh": PROMPT_I_ATTRIBUTE_EXTRACTION_ZH,
        "attribute_batch": PROMPT_I_BATCH,
        "recommend_base": PROMPT_II_RECOMMEND_BASE,
        "recommend_profile": PROMPT_II_RECOMMEND_WITH_PROFILE,
        "recommend_cot": PROMPT_II_RECOMMEND_COT,
        "profile_compact": PROFILE_TEMPLATE_COMPACT,
        "profile_detailed": PROFILE_TEMPLATE_DETAILED,
    }
    if template_name not in templates:
        raise ValueError(f"未知模板: {template_name}, 可选: {list(templates.keys())}")
    return templates[template_name]
