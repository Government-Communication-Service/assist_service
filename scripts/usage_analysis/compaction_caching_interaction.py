"""How does the compaction threshold interact with per-turn caching?

Usage:
    uv run compaction_caching_interaction.py <turns.parquet>

(<turns.parquet> is the output of caching_counterfactual.py.)

Compaction mechanics (as specified, not fit from data):
  - A turn triggers compaction when this turn's own total input size
    (accumulated history + this turn) exceeds `threshold`.
  - The triggering turn itself still runs at full (uncompacted) size -
    compaction runs async alongside that turn's response.
  - The *next* turn's baseline drops to `COMPACTED_FLOOR` (12k) + whatever
    that next turn adds on top.
  - Compaction invalidates the cache: the turn right after a trigger is
    always a miss, regardless of gap or size, because the content sent to
    the model changed structurally (summary instead of full history).

To vary the threshold we need each turn's *organic* growth - the tokens it
adds regardless of compaction - separated from the real historical
total_input trajectory (which was generated under whatever threshold
production currently uses, ~100k). We recover it by undoing the real
resets: growth is the plain turn-over-turn delta, except immediately after
a real reset (total_input fell), where we back it out assuming the real
system *also* floors at COMPACTED_FLOOR. That's an approximation for the
~1-2% of turns right after a real compaction, not for anything else.
"""

import sys

import polars as pl

INPUT_PRICE = 2.0 / 1_000_000
OUTPUT_PRICE = 10.0 / 1_000_000
CACHE_WRITE_PRICE = INPUT_PRICE * 1.25
CACHE_READ_PRICE = INPUT_PRICE * 0.1
GAP_THRESHOLD_SECONDS = 5 * 60

COMPACTED_FLOOR = 12_000


def compute_organic_growth(turns: pl.DataFrame) -> pl.DataFrame:
    turns = turns.sort(["chat_id", "turn_id"]).with_columns(
        prev_total_input=pl.col("total_input").shift(1).over("chat_id"),
    )
    turns = turns.with_columns(
        growth=pl.when(pl.col("prev_total_input").is_null())
        .then(pl.col("total_input"))
        .when(pl.col("total_input") >= pl.col("prev_total_input"))
        .then(pl.col("total_input") - pl.col("prev_total_input"))
        .otherwise(pl.col("total_input") - COMPACTED_FLOOR)
        .clip(lower_bound=0)
    )
    return turns


def simulate(turns: pl.DataFrame, threshold: float | None, caching: bool) -> float:
    """threshold=None means compaction never triggers."""
    total_cost = 0.0
    for chat in turns.partition_by("chat_id"):
        baseline = 0
        prompt_cache_total = None
        prev_reply_time = None
        cache_valid = True  # gets invalidated the turn after a trigger
        for row in chat.sort("turn_id").iter_rows(named=True):
            sim_total_input = baseline + row["growth"]
            gap = None if prev_reply_time is None else (row["user_created_at"] - prev_reply_time).total_seconds()

            can_read = (
                caching
                and cache_valid
                and prompt_cache_total is not None
                and gap is not None
                and gap < GAP_THRESHOLD_SECONDS
                and sim_total_input >= prompt_cache_total
            )

            if can_read:
                growth_to_write = max(sim_total_input - prompt_cache_total, 0)
                cost = prompt_cache_total * CACHE_READ_PRICE + growth_to_write * CACHE_WRITE_PRICE
            elif caching:
                cost = sim_total_input * CACHE_WRITE_PRICE
            else:
                cost = sim_total_input * INPUT_PRICE

            if caching:
                prompt_cache_total = sim_total_input

            cost += row["assistant_tokens"] * OUTPUT_PRICE
            total_cost += cost

            triggered = threshold is not None and sim_total_input > threshold
            baseline = COMPACTED_FLOOR if triggered else sim_total_input
            cache_valid = not triggered
            prev_reply_time = row["assistant_created_at"]

    return total_cost


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <turns.parquet>", file=sys.stderr)
        sys.exit(1)

    turns = compute_organic_growth(pl.read_parquet(sys.argv[1]))

    thresholds: list[float | None] = [
        80_000,
        100_000,
        125_000,
        150_000,
        200_000,
        None,  # never compact
    ]

    print(f"{'threshold':>10}  {'no caching':>12}  {'with caching':>13}  {'saving':>8}")
    for t in thresholds:
        label = "never" if t is None else f"{t:,.0f}"
        no_cache = simulate(turns, t, caching=False)
        with_cache = simulate(turns, t, caching=True)
        saving = 1 - with_cache / no_cache
        print(f"{label:>10}  ${no_cache:>11,.2f}  ${with_cache:>12,.2f}  {saving:>7.1%}")


if __name__ == "__main__":
    main()
