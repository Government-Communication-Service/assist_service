"""Explore whether skipping cache writes under some condition beats always writing.

Usage:
    uv run caching_policy_exploration.py <turns.parquet>

(<turns.parquet> is the output of caching_counterfactual.py.)

Unlike caching_counterfactual.py's independent per-adjacent-pair model, this
runs a proper sequential simulation per chat: a decision to skip a write
doesn't just cost that turn, it changes what the *next* turn can read, so
policies have to be evaluated turn-by-turn with carried-forward state.

Finding: every policy tried below (skip late at night, skip once a chat gets
long, skip once total_input crosses a size threshold, skip when this turn's
growth dwarfs the cached prefix so far) is worse than always writing. Reason:
skipping a write doesn't eliminate the cost of caching that content - it
defers it. If a later turn in the same chat does write, the deferred tokens
get swept into that write's `growth` and pay the cache-write premium anyway,
on top of the plain-input price already paid when skipped (double payment).
Skipping only wins for the minority of turns whose content is *never*
subsumed into a later write - and "always write" already prices that
minority in via its miss rate. There's no cheaply observable signal that
identifies those turns in advance: P(a write pays off) is ~55-70% and
roughly flat across hour of day, weekday, and turn position; it does fall
with the size of this turn's own growth (52% in the top growth quartile vs
69% in the bottom), but the payoff on a hit scales with size too, so gating
on growth size still loses (tested below).
"""

import sys

import polars as pl

INPUT_PRICE = 2.0 / 1_000_000
OUTPUT_PRICE = 10.0 / 1_000_000
CACHE_WRITE_PRICE = INPUT_PRICE * 1.25
CACHE_READ_PRICE = INPUT_PRICE * 0.1
GAP_THRESHOLD_SECONDS = 5 * 60


def simulate(turns: pl.DataFrame, write_decision) -> float:
    """write_decision(row, turn_index, anchor_total_or_None, growth_or_total) -> bool.

    `anchor_total` is the size of whatever's currently cached for this chat
    (None if nothing valid is cached right now); `growth_or_total` is the
    amount that would need writing - the increment beyond the anchor on a
    cache hit, or the whole turn on a miss.
    """
    total_cost = 0.0
    for chat in turns.partition_by("chat_id"):
        anchor_total = None
        prev_reply_time = None
        for i, row in enumerate(chat.sort("turn_id").iter_rows(named=True)):
            total_input = row["total_input"]
            gap = None if prev_reply_time is None else (row["user_created_at"] - prev_reply_time).total_seconds()
            # Gate on total_input, not the raw `tokens` field - see
            # caching_counterfactual.py's add_counterfactuals for why.
            can_read = (
                anchor_total is not None
                and gap is not None
                and gap < GAP_THRESHOLD_SECONDS
                and total_input >= anchor_total
            )

            pending = max(total_input - anchor_total, 0) if can_read else total_input
            write_now = write_decision(row, i, anchor_total, pending)

            if can_read:
                cost = anchor_total * CACHE_READ_PRICE + pending * (CACHE_WRITE_PRICE if write_now else INPUT_PRICE)
            else:
                cost = pending * (CACHE_WRITE_PRICE if write_now else INPUT_PRICE)

            if write_now:
                anchor_total = total_input
            elif not can_read:
                anchor_total = None

            cost += row["assistant_tokens"] * OUTPUT_PRICE
            total_cost += cost
            prev_reply_time = row["assistant_created_at"]

    return total_cost


def stop_after_turn(n: int):
    def policy(row, i, anchor_total, pending, n=n):
        return i < n

    return policy


def stop_after_size(threshold: int):
    def policy(row, i, anchor_total, pending, threshold=threshold):
        return (anchor_total or 0) < threshold and row["total_input"] < threshold

    return policy


def skip_during_hours(start: int, end: int):
    def policy(row, i, anchor_total, pending, start=start, end=end):
        h = row["assistant_created_at"].hour
        in_window = (start <= h < end) if start < end else (h >= start or h < end)
        return not in_window

    return policy


def skip_when_growth_exceeds_anchor(cap: float):
    def policy(row, i, anchor_total, pending, cap=cap):
        if not anchor_total:
            return True
        return (pending / anchor_total) <= cap

    return policy


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <turns.parquet>", file=sys.stderr)
        sys.exit(1)

    turns = pl.read_parquet(sys.argv[1])

    always_write = simulate(turns, lambda row, i, anchor_total, pending: True)
    print(f"Always write (baseline):        ${always_write:,.2f}\n")

    def report(label: str, cost: float) -> None:
        direction = "better" if cost < always_write else "worse"
        print(f"{label}: ${cost:,.2f} ({direction} by ${abs(always_write - cost):,.2f})")

    for n in [5, 10, 15, 20]:
        report(f"Stop permanently after turn {n:>2}          ", simulate(turns, stop_after_turn(n)))

    print()
    for thresh in [50_000, 100_000, 150_000, 200_000]:
        report(f"Stop permanently once total_input >= {thresh:>7,}", simulate(turns, stop_after_size(thresh)))

    print()
    for start, end in [(22, 6), (23, 7)]:
        report(
            f"Never write {start:02d}:00-{end:02d}:00 (non-sticky)  ", simulate(turns, skip_during_hours(start, end))
        )

    print()
    for cap in [0.5, 1.0, 1.5, 2.0, 3.0]:
        report(f"Skip write when growth/anchor > {cap}       ", simulate(turns, skip_when_growth_exceeds_anchor(cap)))


if __name__ == "__main__":
    main()
