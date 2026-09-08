"""Prompts for conversation compaction.

The compaction call deliberately reuses the *chat* system prompt and the exact message
prefix already sent for the turn, so the whole history is a cheap prompt-cache read rather
than a full-price re-read. That is why the instruction below is phrased as a final user
turn: overriding `system` would discard the cached system blocks and, with them, the
cached conversation behind them.
"""

CONVERSATION_COMPACTION_INSTRUCTION = """Your task now is different: do not reply to the previous message.

Instead, write a summary of this entire conversation so far. The summary will replace the
earlier messages in your context on future turns, so treat it as the only record that will
survive. Someone reading only your summary must be able to continue the conversation
without noticing anything is missing.

Everything in the system prompt, up to and including the line "Assist will now be connected
to a user.", is sent again unchanged on every future turn. Do not restate any of it in the
summary — your persona, capabilities, tool availability, and general instructions are already
covered and need no repeating. The summary is only for what happened
in the conversation itself.

If a <conversation-summary> block from an earlier compaction appears in the history above, it
will not be sent again after this — only the summary you write now survives. Fold its content
into your new summary rather than assuming it will still be there.

Preserve, in as much detail as the material requires:
- What the user is trying to achieve, including any longer-term goal behind individual requests.
- Decisions taken and options explicitly rejected, with the reasons given.
- Concrete specifics: names, dates, figures, statistics, quotations, URLs, file and document
  names, and any wording the user has settled on.
- Content that has been drafted or agreed, in enough fidelity to be reused or continued.
- Instructions the user gave about style, tone, format, audience or constraints, which
  continue to apply.
- Anything still outstanding: open questions, unfinished tasks, agreed next steps.

Guidance:
- Be comprehensive rather than brief. Losing a detail is far more costly than length here.
- Organise it under headings so it can be scanned.
- Record facts and drafted content as they are, without re-assessing or improving them.
- Do not invent anything that was not in the conversation, and do not note what you have omitted.

Output only the summary. No preamble, no sign-off, and do not address the user."""


def build_compaction_summary_message(summary: str) -> str:
    """Wrap a stored summary as the text of the synthetic user message that replaces history.

    Tagged so the model can tell a summary of earlier turns from something the user typed.
    """
    return (
        "<conversation-summary>\n"
        "The earlier part of this conversation has been summarised to save space. "
        "This summary replaces those messages and is a faithful record of them. "
        "Treat everything in it as established context, and continue the conversation "
        "as though the original messages were still present. Any messages after this "
        "summary are the verbatim recent turns.\n\n"
        f"{summary}\n"
        "</conversation-summary>"
    )
