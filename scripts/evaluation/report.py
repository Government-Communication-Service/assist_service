import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Evaluation report")


@app.cell
def _():
    import json
    import os
    import pathlib

    import altair as alt
    import marimo as mo
    import pandas as pd

    alt.data_transformers.disable_max_rows()
    return alt, json, mo, os, pathlib, pd


@app.cell
def _(mo, os, pathlib):
    default_results_dir = pathlib.Path(__file__).parent.parent.parent / "data/ignored/evaluation/results"
    results_dir = pathlib.Path(os.environ.get("EVAL_RESULTS_DIR", default_results_dir))
    runs = sorted(path.parent.parent.name for path in results_dir.glob("*/collated/turns.json"))
    mo.stop(
        not runs,
        mo.md(
            f"No collated runs in `{results_dir}`. Run `uv run run_eval.py ...` then `uv run collate.py --run-id ...`."
        ),
    )
    default_run = os.environ.get("EVAL_RUN")
    run_picker = mo.ui.dropdown(options=runs, value=default_run if default_run in runs else runs[-1], label="Run")
    mo.md(f"# Evaluation report\n\nResults in `{results_dir}`.").callout()
    return results_dir, run_picker, runs


@app.cell
def _(json, pd, results_dir, run_picker):
    run_dir = results_dir / run_picker.value
    raw = pd.DataFrame(json.loads((run_dir / "collated/turns.json").read_text()))
    windows = json.loads((run_dir / "collated/windows.json").read_text())
    raw["sent_at"] = pd.to_datetime(raw["sent_at"])
    for _column in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"):
        raw[_column] = raw[_column].fillna(0)
    raw["total_input_tokens"] = raw["input_tokens"] + raw["cache_read_tokens"] + raw["cache_write_tokens"]
    raw["cache_read_share"] = raw["cache_read_tokens"] / raw["total_input_tokens"].where(raw["total_input_tokens"] > 0)
    raw["failed"] = raw["error"].notna()

    # A resumed run can hold several attempts at the same (env, case, repeat); analyse the latest chat of each.
    _latest_chat = (
        raw.sort_values("sent_at").groupby(["env", "case_id", "repeat"])["chat_uuid"].last().rename("latest_chat")
    )
    _joined = raw.join(_latest_chat, on=["env", "case_id", "repeat"])
    _is_latest = (_joined["chat_uuid"] == _joined["latest_chat"]) | (
        _joined["chat_uuid"].isna() & _joined["latest_chat"].isna()
    )
    turns = raw[_is_latest.values].copy()
    envs = sorted(turns["env"].unique())
    return envs, raw, run_dir, turns, windows


@app.cell
def _(mo, run_picker, turns, windows, raw):
    _modes = sorted({window["flag_mode"] for window in windows.values()})
    mo.vstack(
        [
            run_picker,
            mo.md(
                f"**{len(turns)}** turns analysed ({len(raw) - len(turns)} from superseded attempts ignored) · "
                f"environments: {', '.join(f'`{e}`' for e in sorted(windows))} · flag mode: {', '.join(_modes)} · "
                f"**{int(raw['failed'].sum())}** failed turns in total"
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Whole-run cost

        Not affected by the filters below. Each environment's cost is **answer cost** (`message.completion_cost`
        of the assistant rows, i.e. what the answering model calls cost) plus **other calls** (everything in
        `llm_internal_response` over the eval window that is not a `chat_loop` call). On an environment with the
        `purpose` column the answer calls also appear in `llm_internal_response` as `chat_loop`, hence the exclusion.
        On an environment without the `purpose` column (the old pipeline), every row in the window is taken to be an
        other call, which assumes it records only its background calls there. Check this against a few rows.
        """
    )
    return


@app.cell
def _(pd, raw, windows):
    _rows, _purpose_rows = [], []
    for _env, _window in windows.items():
        _by_purpose = pd.DataFrame(_window["by_purpose"])
        _by_purpose["env"] = _env
        _purpose_rows.append(_by_purpose)
        _window_cost = _by_purpose["cost"].sum()
        _loop_cost = _by_purpose.loc[_by_purpose["purpose"] == "chat_loop", "cost"].sum()
        _env_turns = raw[raw["env"] == _env]
        _answer_cost = _env_turns["answer_cost"].sum()
        _case_runs = _env_turns.groupby(["case_id", "repeat"]).ngroups
        _rows.append(
            {
                "env": _env,
                "answer cost": _answer_cost,
                "other calls cost": _window_cost - _loop_cost,
                "total cost": _answer_cost + _window_cost - _loop_cost,
                "other share": (_window_cost - _loop_cost) / (_answer_cost + _window_cost - _loop_cost),
                "case runs": _case_runs,
                "total cost / case run": (_answer_cost + _window_cost - _loop_cost) / max(_case_runs, 1),
                "window llm_internal_response cost": _window_cost,
            }
        )
    cost_summary = pd.DataFrame(_rows).set_index("env")
    purpose_breakdown = pd.concat(_purpose_rows, ignore_index=True)[
        ["env", "purpose", "calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "cost"]
    ]
    return cost_summary, purpose_breakdown


@app.cell
def _(cost_summary, mo, purpose_breakdown):
    mo.vstack(
        [
            mo.ui.table(
                cost_summary.reset_index(),
                selection=None,
                format_mapping={
                    "answer cost": "${:.2f}".format,
                    "other calls cost": "${:.2f}".format,
                    "total cost": "${:.2f}".format,
                    "other share": "{:.0%}".format,
                    "total cost / case run": "${:.3f}".format,
                    "window llm_internal_response cost": "${:.2f}".format,
                },
            ),
            mo.md("**`llm_internal_response` over the window, by purpose**"),
            mo.ui.table(purpose_breakdown, selection=None, format_mapping={"cost": "${:.3f}".format}),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("## Per-turn comparison\n\nFilters apply to everything below this point.")
    return


@app.cell
def _(mo, turns):
    scenario_filter = mo.ui.multiselect(
        options=sorted(turns["scenario"].unique()), value=sorted(turns["scenario"].unique()), label="Scenarios"
    )
    repeat_filter = mo.ui.multiselect(
        options=[int(r) for r in sorted(turns["repeat"].unique())],
        value=[int(r) for r in sorted(turns["repeat"].unique())],
        label="Repeats",
    )
    ok_only = mo.ui.checkbox(label="Exclude failed turns", value=True)
    mo.hstack([scenario_filter, repeat_filter, ok_only], justify="start")
    return ok_only, repeat_filter, scenario_filter


@app.cell
def _(ok_only, repeat_filter, scenario_filter, turns):
    df = turns[turns["scenario"].isin(scenario_filter.value) & turns["repeat"].isin(repeat_filter.value)]
    if ok_only.value:
        df = df[~df["failed"]]
    return (df,)


@app.cell
def _(df, mo):
    def p50(series):
        return series.quantile(0.5)

    def p95(series):
        return series.quantile(0.95)

    summary = (
        df.groupby("env")
        .agg(
            turns=("case_id", "size"),
            answer_cost_mean=("answer_cost", "mean"),
            input_tokens_mean=("total_input_tokens", "mean"),
            output_tokens_mean=("output_tokens", "mean"),
            cache_read_share=("cache_read_share", "mean"),
            tool_calls_mean=("n_tool_calls", "mean"),
            ttft_p50=("time_to_first_token_s", p50),
            ttft_p95=("time_to_first_token_s", p95),
            total_time_p50=("total_time_s", p50),
            total_time_p95=("total_time_s", p95),
        )
        .reset_index()
    )
    mo.ui.table(
        summary,
        selection=None,
        format_mapping={
            "answer_cost_mean": "${:.4f}".format,
            "input_tokens_mean": "{:,.0f}".format,
            "output_tokens_mean": "{:,.0f}".format,
            "cache_read_share": "{:.0%}".format,
            "tool_calls_mean": "{:.2f}".format,
            "ttft_p50": "{:.1f}s".format,
            "ttft_p95": "{:.1f}s".format,
            "total_time_p50": "{:.1f}s".format,
            "total_time_p95": "{:.1f}s".format,
        },
    )
    return p50, p95, summary


@app.cell
def _(alt, df, mo):
    _by_scenario = df.groupby(["scenario", "env"], as_index=False).agg(
        cost=("answer_cost", "mean"), cache=("cache_read_share", "mean"), tokens=("total_input_tokens", "mean")
    )
    _bars = lambda field, title, fmt: (  # noqa: E731
        alt.Chart(_by_scenario, title=title)
        .mark_bar()
        .encode(
            x=alt.X("scenario:N", title=None),
            xOffset="env:N",
            y=alt.Y(f"{field}:Q", title=None, axis=alt.Axis(format=fmt)),
            color="env:N",
            tooltip=["scenario", "env", alt.Tooltip(f"{field}:Q", format=fmt if fmt != "%" else ".1%")],
        )
        .properties(width=300, height=240)
    )
    mo.hstack(
        [
            _bars("cost", "Mean answer cost per turn ($)", "$.4f"),
            _bars("tokens", "Mean input tokens per turn (incl. cached, all calls)", ",.0f"),
            _bars("cache", "Mean cache-read share of input tokens", "%"),
        ]
    )
    return


@app.cell
def _(alt, df, mo):
    _boxes = lambda field, title: (  # noqa: E731
        alt.Chart(df[["scenario", "env", field]], title=title)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("scenario:N", title=None),
            xOffset="env:N",
            y=alt.Y(f"{field}:Q", title="seconds"),
            color="env:N",
        )
        .properties(width=500, height=260)
    )
    mo.hstack([_boxes("time_to_first_token_s", "Time to first token"), _boxes("total_time_s", "Total time")])
    return


@app.cell
def _(alt, df, mo, pd):
    _long = df[df["scenario"] == "long_chat"]
    if _long.empty:
        _out = mo.md("_No `long_chat` turns in the current selection._")
    else:
        _long = _long.sort_values(["env", "case_id", "repeat", "turn_index"]).copy()
        _long["cumulative_cost"] = _long.groupby(["env", "case_id", "repeat"])["answer_cost"].cumsum()
        _scale = _long.groupby(["env", "turn_index"], as_index=False).agg(
            cumulative_cost=("cumulative_cost", "mean"),
            input_tokens=("total_input_tokens", "mean"),
            cost=("answer_cost", "mean"),
        )
        _line = lambda field, title, fmt: (  # noqa: E731
            alt.Chart(_scale, title=title)
            .mark_line(point=True)
            .encode(
                x=alt.X("turn_index:O", title="turn"),
                y=alt.Y(f"{field}:Q", title=None, axis=alt.Axis(format=fmt)),
                color="env:N",
                tooltip=["env", "turn_index", alt.Tooltip(f"{field}:Q", format=fmt)],
            )
            .properties(width=300, height=240)
        )
        _out = mo.vstack(
            [
                mo.md("### Long chat: how cost scales with conversation length"),
                mo.hstack(
                    [
                        _line("cost", "Answer cost of each turn ($)", "$.4f"),
                        _line("cumulative_cost", "Cumulative answer cost ($)", "$.3f"),
                        _line("input_tokens", "Input tokens of each turn (incl. cached)", ",.0f"),
                    ]
                ),
            ]
        )
    _out  # noqa: B018
    return


@app.cell
def _(envs, mo, os):
    # EVAL_BASELINE sets the default baseline (the other environment becomes the candidate).
    _baseline = os.environ.get("EVAL_BASELINE")
    if _baseline not in envs:
        _baseline = next((e for e in envs if "prod" in e), envs[0])
    baseline = mo.ui.dropdown(options=envs, value=_baseline, label="Baseline")
    candidate = mo.ui.dropdown(
        options=envs, value=next((e for e in envs if e != _baseline), envs[-1]), label="Candidate"
    )
    mo.vstack([mo.md("### Paired comparison, turn by turn"), mo.hstack([baseline, candidate], justify="start")])
    return baseline, candidate


@app.cell
def _(alt, baseline, candidate, df, mo, pd):
    _keys = ["scenario", "case_id", "repeat", "turn_index"]
    _metrics = ["answer_cost", "total_time_s", "time_to_first_token_s", "total_input_tokens"]
    _a = df[df["env"] == baseline.value][_keys + _metrics]
    _b = df[df["env"] == candidate.value][_keys + _metrics]
    pairs = _a.merge(_b, on=_keys, suffixes=("_base", "_cand"))
    if pairs.empty or baseline.value == candidate.value:
        _out = mo.md("_No matched turns for this pair._")
    else:
        _ratio = (
            pairs.groupby("scenario")
            .apply(
                lambda g: pd.Series(
                    {
                        "matched turns": len(g),
                        "median cost ratio": (g["answer_cost_cand"] / g["answer_cost_base"]).median(),
                        "median total-time ratio": (g["total_time_s_cand"] / g["total_time_s_base"]).median(),
                        "median TTFT ratio": (
                            g["time_to_first_token_s_cand"] / g["time_to_first_token_s_base"]
                        ).median(),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )

        def _scatter(base_col, cand_col, title):
            _top = float(max(pairs[base_col].max(), pairs[cand_col].max()))
            _diag = pd.DataFrame({"x": [0, _top], "y": [0, _top]})
            return (
                (
                    alt.Chart(pairs, title=title)
                    .mark_circle(size=60, opacity=0.7)
                    .encode(
                        x=alt.X(f"{base_col}:Q", title=baseline.value),
                        y=alt.Y(f"{cand_col}:Q", title=candidate.value),
                        color="scenario:N",
                        tooltip=_keys + [base_col, cand_col],
                    )
                    + alt.Chart(_diag).mark_line(color="grey", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
                )
                .properties(width=360, height=320)
                .interactive()
            )

        _out = mo.vstack(
            [
                mo.md(
                    "Points below the dashed line are cheaper / faster on the candidate. "
                    "Ratios are candidate ÷ baseline. "
                    "Answer cost only; background calls are in the whole-run table above."
                ),
                mo.hstack(
                    [
                        _scatter("answer_cost_base", "answer_cost_cand", "Answer cost per turn ($)"),
                        _scatter("total_time_s_base", "total_time_s_cand", "Total time per turn (s)"),
                    ]
                ),
                mo.ui.table(
                    _ratio,
                    selection=None,
                    format_mapping={
                        "median cost ratio": "{:.2f}".format,
                        "median total-time ratio": "{:.2f}".format,
                        "median TTFT ratio": "{:.2f}".format,
                    },
                ),
            ]
        )
    _out  # noqa: B018
    return (pairs,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Quality

        Reads `judgments.jsonl` in the run directory if it exists. One JSON object per line:

        - `{"kind": "acceptable", "env": ..., "case_id": ..., "repeat": ..., "turn_index": ...,
          "acceptable": true, "reason": ...}`
        - `{"kind": "pairwise", "case_id": ..., "repeat": ..., "turn_index": ...,
          "winner": "<env name>" | "tie", "reason": ...}`
        """
    )
    return


@app.cell
def _(alt, df, json, mo, pd, run_dir):
    _path = run_dir / "judgments.jsonl"
    if not _path.exists():
        _out = mo.md(f"_No judgments yet (`{_path.name}` not found)._")
    else:
        _judgments = pd.DataFrame([json.loads(line) for line in _path.read_text().splitlines() if line.strip()])
        _keys = ["case_id", "repeat", "turn_index"]
        _parts = []
        _acceptable = _judgments[_judgments["kind"] == "acceptable"]
        if not _acceptable.empty:
            _merged = df.merge(_acceptable[["env"] + _keys + ["acceptable"]], on=["env"] + _keys)
            _rate = _merged.groupby(["scenario", "env"], as_index=False).agg(
                acceptable=("acceptable", "mean"), turns=("acceptable", "size")
            )
            _parts.append(
                alt.Chart(_rate, title="Share of answers judged acceptable")
                .mark_bar()
                .encode(
                    x=alt.X("scenario:N", title=None),
                    xOffset="env:N",
                    y=alt.Y("acceptable:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                    color="env:N",
                    tooltip=["scenario", "env", alt.Tooltip("acceptable:Q", format=".0%"), "turns"],
                )
                .properties(width=420, height=260)
            )
        _pairwise = _judgments[_judgments["kind"] == "pairwise"]
        if not _pairwise.empty:
            _scenario = df[["scenario"] + _keys].drop_duplicates()
            _wins = _pairwise.merge(_scenario, on=_keys).groupby(["scenario", "winner"], as_index=False).size()
            # Environments keep the default palette (as in the other charts); ties sit between them in light grey.
            _envs = sorted(w for w in _wins["winner"].unique() if w != "tie")
            _order = [*_envs[:1], "tie", *_envs[1:]]
            _wins["stack_order"] = _wins["winner"].map({w: i for i, w in enumerate(_order)})
            _parts.append(
                alt.Chart(_wins, title="Pairwise winners")
                .mark_bar()
                .encode(
                    x=alt.X("scenario:N", title=None),
                    y=alt.Y("size:Q", title="turns", stack="normalize", axis=alt.Axis(format="%")),
                    color=alt.Color(
                        "winner:N",
                        sort=_order,
                        scale=alt.Scale(domain=_order, range=["#4c78a8", "#d9d9d9", "#f58518"][: len(_order)]),
                    ),
                    order=alt.Order("stack_order:Q", sort="descending"),
                    tooltip=["scenario", "winner", "size"],
                )
                .properties(width=420, height=260)
            )
        _out = mo.hstack(_parts) if _parts else mo.md("_`judgments.jsonl` has no usable rows._")
    _out  # noqa: B018
    return


@app.cell
def _(mo):
    mo.md("## Response browser")
    return


@app.cell
def _(df, mo):
    case_picker = mo.ui.dropdown(
        options=sorted(df["case_id"].unique()), value=sorted(df["case_id"].unique())[0], label="Case"
    )
    repeat_picker = mo.ui.dropdown(
        options=[int(r) for r in sorted(df["repeat"].unique())], value=int(df["repeat"].min()), label="Repeat"
    )
    mo.hstack([case_picker, repeat_picker], justify="start")
    return case_picker, repeat_picker


@app.cell
def _(case_picker, envs, mo, repeat_picker, turns):
    _case = turns[(turns["case_id"] == case_picker.value) & (turns["repeat"] == repeat_picker.value)]
    _blocks = []
    for _turn_index in sorted(_case["turn_index"].unique()):
        _turn = _case[_case["turn_index"] == _turn_index]
        _first = _turn.iloc[0]
        _columns = []
        for _env in envs:
            _row = _turn[_turn["env"] == _env]
            if _row.empty:
                _columns.append(mo.md(f"**{_env}**\n\n_not run_"))
                continue
            _r = _row.iloc[0]
            _cache_share = _r["cache_read_share"]
            _cache_share_shown = 0 if _cache_share != _cache_share else _cache_share  # NaN when no input tokens
            _facts = (
                f"cost ${_r['answer_cost'] or 0:.4f} · input {int(_r['total_input_tokens']):,} tok "
                f"(cache read {_cache_share_shown:.0%}) · "
                f"TTFT {_r['time_to_first_token_s']}s · total {_r['total_time_s']}s · "
                f"tools: {', '.join(_r['tools']) or 'none'}"
            )
            _error = f"\n\n**ERROR:** {_r['error']}" if _r["failed"] else ""
            _columns.append(mo.md(f"**{_env}**\n\n_{_facts}_{_error}\n\n{_r['response_text']}"))
        _blocks.append(
            mo.vstack(
                [
                    mo.md(f"#### Turn {_turn_index + 1}: {_first['query']}"),
                    mo.accordion({"Expected answer": mo.md(_first["expected_answer"])}),
                    mo.hstack(_columns, align="start", widths="equal"),
                ]
            )
        )
    mo.vstack(_blocks)
    return


if __name__ == "__main__":
    app.run()
