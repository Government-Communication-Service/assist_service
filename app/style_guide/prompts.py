"""Prompt text for the style guide checker's LLM calls."""

from typing import Dict, List


def format_checker_context(context: str | None) -> str:
    """Format the assistant's optional context for the validation prompts.

    The assistant can say what the text is, who it is for, or what to focus on. It is guidance for the
    checker, not text to check.

    Args:
        context: The context field from the check_style_guide call, or None.

    Returns:
        A <context> block, or "" when there is no context.
    """
    if not context or not context.strip():
        return ""
    return (
        "\n\nCONTEXT FROM THE ASSISTANT (about the document and how to check it; use it to judge scope, "
        "do not check it):\n"
        f"<context>\n{context.strip()}\n</context>\n"
    )


def create_llm_prompt_for_deterministic_false(document: str, rules: List[Dict], checker_context: str = "") -> str:
    """Create a focused prompt for LLM validation of deterministic=false rules.

    This prompt is designed to reduce false positives by:
    1. Understanding the specific scope and purpose of each rule
    2. Requiring violations to match the exact context described in Rule Details
    3. Explicitly warning against over-generalization
    4. Prioritizing precision over recall

    Args:
        document: The text to check.
        rules: The deterministic=false rules to check it against.
        checker_context: Context from the assistant (see format_checker_context), or "".
    """
    prompt = """You are an expert GOV.UK style guide auditor. Your job is NOT to find violations but to find CLEAR
violations that unambiguously break the specific guidance in Rule Details.

FUNDAMENTAL APPROACH:
- Read each Rule Details carefully to understand its SPECIFIC PURPOSE and SCOPE
- Only flag violations where the document's usage clearly contradicts what Rule Details prescribe
- Rules often have narrow application domains - do NOT over-generalize the rule's scope
- If there's any reasonable interpretation where the document is correct, do NOT flag it

CRITICAL WARNINGS AGAINST FALSE POSITIVES:
1. SCOPE MISMATCH - Understand WHAT the rule actually applies to:
   - "Titles" rule applies to PAGE TITLES/HEADINGS, NOT every sentence with capital letters or dashes
   - "Summaries" rule applies to META DESCRIPTIONS and page summaries, NOT regular body text sentences
   - "Maths content" rule applies to mathematical notation, NOT to date ranges, account numbers, or ID numbers
   - "Numbers" rule applies to how numbers are written in text, NOT formatting of phone/account numbers
   - "Telephone numbers" rule applies to phone numbers ONLY, not account numbers or reference codes
   - Do NOT assume a rule name applies to anything containing a related word
2. DOCUMENT STRUCTURE - Recognize what parts of the document are what:
   - PAGE TITLES are standalone headers at the start of sections (typically short, descriptive)
   - SUMMARIES are meta descriptions (HTML meta tags) or explicit introductory summaries
     * They are typically placed at the very beginning of a document
     * They summarize the entire page content
     * Regular sentences in body text are NOT summaries, even if they're descriptive
     * A long sentence in the middle or end of a document is NOT a summary
   - BODY TEXT is the main content - most rules about titles/summaries don't apply here
   - Account numbers, sort codes, reference numbers are NOT telephone numbers
   - Date ranges like "2024-2025" are body content, not titles or math expressions
   - If text appears in the middle or end of a document, it's almost certainly NOT a title or summary
   - If something is clearly NOT in the scope defined by Rule Details, don't flag it
3. HALLUCINATING REQUIREMENTS: Do NOT add requirements not explicitly stated in Rule Details
   - Rule Details are the COMPLETE specification - nothing more is required
   - If Rule Details don't mention a constraint, that constraint does NOT exist
   - Examples in Rule Details show WHAT APPLIES - use them to understand scope

EVALUATION CHECKLIST FOR EACH RULE:
□ What is this rule specifically about? (What's its purpose/domain?)
□ What context must apply for the rule to be violated? (Where does it apply?)
□ Does the document text actually appear in that specific context?
□ Is this text clearly a case where the rule should apply?
□ Does the text explicitly violate what Rule Details prescribe?
If NO to questions 2-4, do NOT flag it.

RULES TO CHECK:
"""
    for i, rule in enumerate(rules, 1):
        prompt += f"\n{'-' * 70}\nRULE {i}: {rule.get('rule_id')}\n"
        prompt += f"Name: {rule.get('rule')}\n"
        prompt += f"Details: {rule.get('details')}\n"

    prompt += checker_context
    prompt += f"\n{'-' * 70}\nDOCUMENT TO CHECK:\n{'-' * 70}\n\n"
    prompt += document
    return prompt


def create_llm_validation_prompt(document: str, rules_with_occurrences: List[Dict], checker_context: str = "") -> str:
    """Create a dynamic prompt for LLM validation.

    Args:
        document: The full document text
        rules_with_occurrences: List of rules with their found occurrences
        checker_context: Context from the assistant (see format_checker_context), or "".

    Returns:
        Formatted prompt string
    """
    prompt = """You are an expert GOV.UK style guide checker. Your task is to validate whether specific occurrences of
text in a document violate style guide rules.

CRITICAL INSTRUCTIONS:
1. The "Rule Details" field is your PRIMARY SOURCE OF TRUTH - read it carefully for each rule
2. The "Matched text" shows what was actually found in the document (which may have different casing)
3. For case-insensitive rules, the matched text can appear in ANY casing - this is NOT automatically a violation
4. A match is ONLY a violation if it breaks the guidance in the Rule Details given its context
5. Rule Details often contain contextual guidance (e.g., "uppercase when referring to X, lowercase otherwise")
6. Use the provided context (current sentence and preceding sentence) to understand the meaning and intent
7. If the Rule Details say "use X in context A, use Y in context B", determine which context applies
8. Just because text was matched does NOT mean it's a violation - evaluate against Rule Details

EXAMPLES OF CORRECT USAGE (NOT VIOLATIONS):
- Rule: "Access to Work - Upper case when referring to the programme, otherwise use lower case"
  - "access to work" in "people need access to work" = CORRECT (general meaning, lowercase is right)
  - "Access to Work" in "Apply for Access to Work funding" = CORRECT (the programme, uppercase is right)
  - "access to work" in "Apply for access to work funding" = VIOLATION (should be uppercase for programme)

Only include violations where the text usage clearly breaks the Rule Details guidance. If the matched text is being
used correctly according to the Rule Details, do NOT include it as a violation.

RULES TO CHECK:\n\n"""

    # Add each rule with its details
    for i, rule_info in enumerate(rules_with_occurrences, 1):
        prompt += f"\nRULE {i}:\n"
        prompt += f"Rule ID: {rule_info['rule_id']}\n"
        prompt += f"Rule Name: {rule_info['rule_title']}\n"
        prompt += f"Rule Details: {rule_info['rule_details']}\n"
        prompt += f"Case Sensitive: {rule_info['case_sensitive']}\n"
        prompt += "\nOccurrences found in document:\n"

        for j, occurrence in enumerate(rule_info["occurrences"], 1):
            matched = occurrence["matched_text"]
            british = occurrence.get("british_spelling")
            suggestion = f" (British spelling: '{british}')".format() if british else ""
            prompt += f"  {j}. Matched text: '{matched}'{suggestion}\n"
            if occurrence.get("preceding_sentence"):
                prompt += f"     Preceding sentence: {occurrence['preceding_sentence']}\n"
            prompt += f"     Current sentence: {occurrence['sentence']}\n"

        prompt += "\n"

    prompt += checker_context
    prompt += "\n\nFULL DOCUMENT FOR CONTEXT:\n\n"
    prompt += document
    return prompt
