"""Counterfactual: cost without any conversation-prefix caching vs. caching every turn.

Usage:
    uv run caching_counterfactual.py <input.csv> [cutoff]

`cutoff` (optional, e.g. "2026-09-11 11:00:00") drops rows before that
timestamp - use it for exports that span the pre-fix message-summarisation
period. Omit it for exports that are already clean.

Model
-----
Each user message row is one LLM call. `tokens` on a user row is the
uncached input tokens for that call; `cache_read_tokens` / `cache_write_tokens`
are the actual (historical) cache activity for that call. The true total
input size for the call is reconstructed as:

    total_input = tokens + cache_read_tokens + cache_write_tokens

"Without caching": every turn pays full input price on `total_input`.

"With caching on every turn": a turn is a cache *hit* on the previous turn's
prefix iff the gap since the previous turn's assistant reply is < 5 minutes
AND `tokens` grew monotonically vs. the previous turn (a compaction reset
breaks monotonicity, as does the first turn of a chat). On a hit, the
previous turn's `total_input` is read from cache and the growth beyond it
is written; on a miss, the whole `total_input` is written fresh.

Pricing (Claude Sonnet 5 on Bedrock): $2/$10 per MTok in/out,
cache write = 1.25x input, cache read = 0.1x input.
"""

import sys
from pathlib import Path

import polars as pl

INPUT_PRICE = 2.0 / 1_000_000
OUTPUT_PRICE = 10.0 / 1_000_000
CACHE_WRITE_PRICE = INPUT_PRICE * 1.25
CACHE_READ_PRICE = INPUT_PRICE * 0.1

GAP_THRESHOLD_SECONDS = 5 * 60


def build_turns(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(~((pl.col("role") == "user") & (pl.col("tokens") == 0)))

    df = df.sort(["chat_id", "message_created_at"]).with_columns(
        turn_id=(pl.col("role") == "user").cum_sum().over("chat_id") - 1
    )

    is_user = pl.col("role") == "user"
    is_assistant = pl.col("role") == "assistant"

    turns = df.group_by(["chat_id", "turn_id"], maintain_order=True).agg(
        user_created_at=pl.col("message_created_at").filter(is_user).first(),
        assistant_created_at=pl.col("message_created_at").filter(is_assistant).first(),
        user_tokens=pl.col("tokens").filter(is_user).sum(),
        assistant_tokens=pl.col("tokens").filter(is_assistant).sum(),
        cache_read_tokens=pl.col("cache_read_tokens").filter(is_user).sum(),
        cache_write_tokens=pl.col("cache_write_tokens").filter(is_user).sum(),
    )

    return turns.filter(pl.col("user_created_at").is_not_null())


def add_counterfactuals(turns: pl.DataFrame) -> pl.DataFrame:
    turns = turns.sort(["chat_id", "turn_id"]).with_columns(
        total_input=(
            pl.col("user_tokens") + pl.col("cache_read_tokens").fill_null(0) + pl.col("cache_write_tokens").fill_null(0)
        )
    )

    turns = turns.with_columns(
        prev_assistant_created_at=pl.col("assistant_created_at").shift(1).over("chat_id"),
        prev_total_input=pl.col("total_input").shift(1).over("chat_id"),
    )

    turns = turns.with_columns(
        gap_seconds=(pl.col("user_created_at") - pl.col("prev_assistant_created_at")).dt.total_seconds()
    )

    # Gate on total_input, not the raw `tokens` field: `tokens` is contaminated
    # by the *current* compaction-triggered caching system, which can write a
    # big chunk to cache_write_tokens on the turn compaction is triggered
    # (tokens drops) before the compacted context actually takes effect one
    # turn later (when total_input itself actually falls). total_input growing
    # is the real signal that this turn is a valid continuation of the prefix.
    is_hit = (
        pl.col("prev_total_input").is_not_null()
        & (pl.col("gap_seconds") < GAP_THRESHOLD_SECONDS)
        & (pl.col("total_input") >= pl.col("prev_total_input"))
    )

    turns = turns.with_columns(
        cache_hit=is_hit,
        sim_cache_read=pl.when(is_hit).then(pl.col("prev_total_input")).otherwise(0),
        sim_cache_write=pl.when(is_hit)
        .then((pl.col("total_input") - pl.col("prev_total_input")).clip(lower_bound=0))
        .otherwise(pl.col("total_input")),
    )

    turns = turns.with_columns(
        cost_without_caching=(pl.col("total_input") * INPUT_PRICE + pl.col("assistant_tokens") * OUTPUT_PRICE),
        cost_with_caching=(
            pl.col("sim_cache_read") * CACHE_READ_PRICE
            + pl.col("sim_cache_write") * CACHE_WRITE_PRICE
            + pl.col("assistant_tokens") * OUTPUT_PRICE
        ),
    )

    return turns


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} <input.csv> [cutoff]", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    cutoff = sys.argv[2] if len(sys.argv) == 3 else None

    df = pl.read_csv(
        input_path,
        try_parse_dates=True,
        columns=[
            "chat_id",
            "message_created_at",
            "role",
            "tokens",
            "cache_write_tokens",
            "cache_read_tokens",
        ],
    )

    if cutoff is not None:
        before = df.height
        df = df.filter(pl.col("message_created_at") >= pl.lit(cutoff).str.to_datetime())
        print(f"Filtered {before - df.height} rows before {cutoff} ({df.height} remain)")

    turns = build_turns(df)
    turns = add_counterfactuals(turns)

    n_turns = turns.height
    n_hits = turns.select(pl.col("cache_hit").sum()).item()

    total_without = turns.select(pl.col("cost_without_caching").sum()).item()
    total_with = turns.select(pl.col("cost_with_caching").sum()).item()

    print(f"\nTurns analyzed: {n_turns}")
    print(f"Simulated cache hit rate: {n_hits / n_turns:.1%} ({n_hits}/{n_turns})")
    print(f"\nCost without any caching:      ${total_without:,.2f}")
    print(f"Cost with caching every turn:  ${total_with:,.2f}")
    print(f"Savings:                       ${total_without - total_with:,.2f} ({(1 - total_with / total_without):.1%})")

    output_path = f"../../data/ignored/caching_counterfactual_turns_{Path(input_path).stem}.parquet"
    turns.write_parquet(output_path)
    print(f"\nWrote per-turn detail to {output_path}")


if __name__ == "__main__":
    main()
