"""Collapse a message-level ROI export into one row per user->assistant turn.

Usage:
    uv run build_turns.py <input.csv> <output.parquet>
"""

import sys

import polars as pl


def build_turns(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(~((pl.col("role") == "user") & (pl.col("tokens") == 0)))

    df = df.sort(["chat_id", "message_created_at"]).with_columns(
        turn_id=(pl.col("role") == "user").cum_sum().over("chat_id") - 1
    )

    is_user = pl.col("role") == "user"
    is_assistant = pl.col("role") == "assistant"

    turns = df.group_by(["chat_id", "turn_id"], maintain_order=True).agg(
        user_id=pl.col("user_id").first(),
        chat_created_at=pl.col("chat_created_at").first(),
        title=pl.col("title").first(),
        from_open_chat=pl.col("from_open_chat").first(),
        use_rag=pl.col("use_rag").first(),
        use_gov_uk_search_api=pl.col("use_gov_uk_search_api").first(),
        use_smart_targets=pl.col("use_smart_targets").first(),
        central_document_count=pl.col("central_document_count").first(),
        non_central_document_count=pl.col("non_central_document_count").first(),
        message_id=pl.col("message_id").filter(is_assistant).first(),
        message_created_at=pl.col("message_created_at").filter(is_assistant).first(),
        input_tokens=pl.col("tokens").filter(is_user).sum(),
        output_tokens=pl.col("tokens").filter(is_assistant).sum(),
        any_message_deleted=pl.col("message_deleted_at").is_not_null().any(),
        user_content_length=pl.col("content_length").filter(is_user).sum(),
        assistant_content_length=pl.col("content_length").filter(is_assistant).sum(),
        user_content_enhanced_with_rag_length=pl.col("content_enhanced_with_rag_length").filter(is_user).sum(),
        assistant_content_enhanced_with_rag_length=pl.col("content_enhanced_with_rag_length")
        .filter(is_assistant)
        .sum(),
        user_summary_length=pl.col("summary_length").filter(is_user).sum(),
        assistant_summary_length=pl.col("summary_length").filter(is_assistant).sum(),
        citation_length=pl.col("citation_length").filter(is_assistant).first(),
        sources_length=pl.col("sources_length").filter(is_assistant).first(),
        llm_model=pl.col("llm_model").filter(is_assistant).first(),
    )

    turns = turns.sort(["chat_id", "message_created_at"]).with_columns(
        assistant_gap_to_next_seconds=(
            pl.col("message_created_at").shift(-1).over("chat_id") - pl.col("message_created_at")
        ).dt.total_seconds()
    )

    return turns


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.csv> <output.parquet>", file=sys.stderr)
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    df = pl.read_csv(input_path, try_parse_dates=True)
    turns = build_turns(df)
    turns.write_parquet(output_path)
    print(f"Wrote {turns.height} turns to {output_path}")


if __name__ == "__main__":
    main()
