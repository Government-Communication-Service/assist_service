# ruff: noqa: B018, N803, N806
# B018: bare expressions render cell output in marimo notebooks
# N803/N806: uppercase args/vars pass module-level constants between cells
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import datetime as dt

    import editdistance
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    import polars as pl
    import statsmodels.formula.api as smf
    from plotly.subplots import make_subplots
    from scipy.cluster.hierarchy import linkage
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    return (
        AgglomerativeClustering,
        KMeans,
        PCA,
        StandardScaler,
        dt,
        editdistance,
        ff,
        go,
        linkage,
        make_subplots,
        mo,
        np,
        pd,
        pl,
        px,
        silhouette_score,
        smf,
    )


@app.cell(hide_code=True)
def _(pl):
    turns_all = pl.read_parquet("../../data/ignored/turns_5k_active.parquet")
    return (turns_all,)


@app.cell(hide_code=True)
def _(dt, mo, turns_all):
    min_date = turns_all["message_created_at"].min().date()
    max_date = turns_all["message_created_at"].max().date()

    date_filter = mo.ui.date_range(
        start=min_date,
        stop=max_date,
        value=(dt.date(2026, 8, 3), dt.date(2026, 8, 28)),
        label="Date range (filters on assistant message_created_at)",
    )
    date_filter
    return (date_filter,)


@app.cell(hide_code=True)
def _(date_filter, pl, turns_all):
    turns = turns_all.filter(
        pl.col("message_created_at").dt.date().is_between(date_filter.value[0], date_filter.value[1])
    )
    # turns
    return (turns,)


@app.cell(hide_code=True)
def _(mo):
    # Fixed categorical color order (dataviz skill palette) - slot 1 = input, slot 2 = output
    INPUT_COLOR = "#2a78d6"
    OUTPUT_COLOR = "#eb6834"
    mo.md("""
    ## Overall average tokens per message

    Throughout this analysis, 'input tokens' includes all tokens fed into the
    model input. This tends to increase over time during a chat, because the
    history of the chat is fed into the model in each turn.

    Average input tokens is a lot larger than median, suggesting a long tail
    of very large chats, or chats enhanced by a large amount of context.
    """)
    return INPUT_COLOR, OUTPUT_COLOR


@app.cell(hide_code=True)
def _(pl, turns):
    overall = turns.select(
        avg_input_tokens=pl.col("input_tokens").mean().round(0).cast(pl.Int64),
        avg_output_tokens=pl.col("output_tokens").mean().round(0).cast(pl.Int64),
        median_input_tokens=pl.col("input_tokens").median().round(0).cast(pl.Int64),
        median_output_tokens=pl.col("output_tokens").median().round(0).cast(pl.Int64),
        n_turns=pl.len(),
    )
    overall
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Average tokens over time

    No obvious trends over August
    """)
    return


@app.cell(hide_code=True)
def _(INPUT_COLOR, OUTPUT_COLOR, make_subplots, pl, turns):
    by_day = (
        turns.with_columns(day=pl.col("message_created_at").dt.date())
        .group_by("day")
        .agg(
            avg_input_tokens=pl.col("input_tokens").mean(),
            avg_output_tokens=pl.col("output_tokens").mean(),
            median_input_tokens=pl.col("input_tokens").median(),
            median_output_tokens=pl.col("output_tokens").median(),
        )
        .sort("day")
    )

    by_day_long = by_day.unpivot(
        index="day",
        on=[
            "avg_input_tokens",
            "avg_output_tokens",
            "median_input_tokens",
            "median_output_tokens",
        ],
        variable_name="metric",
        value_name="tokens",
    ).with_columns(
        series=pl.when(pl.col("metric").str.contains("input_tokens")).then(pl.lit("input")).otherwise(pl.lit("output")),
        stat=pl.when(pl.col("metric").str.starts_with("avg")).then(pl.lit("mean")).otherwise(pl.lit("median")),
    )

    fig_time = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Input tokens", "Output tokens"))

    for series_name, color, subplot_row in [
        ("input", INPUT_COLOR, 1),
        ("output", OUTPUT_COLOR, 2),
    ]:
        for stat_name, dash in [("mean", "solid"), ("median", "dot")]:
            trace_df = by_day_long.filter((pl.col("series") == series_name) & (pl.col("stat") == stat_name))
            fig_time.add_scatter(
                x=trace_df["day"],
                y=trace_df["tokens"],
                mode="lines",
                name=f"{series_name}, {stat_name}",
                line={"color": color, "width": 2, "dash": dash},
                legendgroup=series_name,
                row=subplot_row,
                col=1,
            )

    fig_time.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
    )
    fig_time.update_yaxes(title_text="tokens")
    fig_time
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Average tokens by feature flag

    The apparent large token usage of Smart Targets is misleading. In this sample
    the chats that use Smart Targets also use Gov.UK search.
    """)
    return


@app.cell(hide_code=True)
def _(INPUT_COLOR, OUTPUT_COLOR, pl, px, turns):
    features = turns.with_columns(
        use_rag=pl.col("use_rag") == "t",
        use_gov_uk_search_api=pl.col("use_gov_uk_search_api") == "t",
        use_smart_targets=pl.col("use_smart_targets") == "t",
        has_personal_docs=pl.col("non_central_document_count") > 0,
        is_prebuilt_prompt=pl.col("from_open_chat") == "f",
    )

    feature_cols = [
        "use_rag",
        "gov_uk_search",
        "smart_targets",
        "personal_docs",
        "prebuilt_prompt",
    ]
    feature_source_col = {
        "use_rag": "use_rag",
        "gov_uk_search": "use_gov_uk_search_api",
        "smart_targets": "use_smart_targets",
        "personal_docs": "has_personal_docs",
        "prebuilt_prompt": "is_prebuilt_prompt",
    }

    feature_rows = []
    for col in feature_cols:
        source_col = feature_source_col[col]
        grouped = features.group_by(source_col).agg(
            avg_input_tokens=pl.col("input_tokens").mean(),
            avg_output_tokens=pl.col("output_tokens").mean(),
            n=pl.len(),
        )
        for row in grouped.iter_rows(named=True):
            feature_rows.append(
                {
                    "feature": col,
                    "active": bool(row[source_col]) if row[source_col] is not None else None,
                    "avg_input_tokens": row["avg_input_tokens"],
                    "avg_output_tokens": row["avg_output_tokens"],
                    "n": row["n"],
                }
            )

    by_feature = pl.DataFrame(feature_rows)
    by_feature_long = by_feature.unpivot(
        index=["feature", "active", "n"],
        on=["avg_input_tokens", "avg_output_tokens"],
        variable_name="metric",
        value_name="avg_tokens",
    ).with_columns(active_label=pl.col("active").cast(pl.Utf8))

    fig_feature = px.bar(
        by_feature_long.sort("feature"),
        x="active_label",
        y="avg_tokens",
        color="metric",
        facet_col="feature",
        facet_col_spacing=0.06,
        barmode="group",
        color_discrete_map={
            "avg_input_tokens": INPUT_COLOR,
            "avg_output_tokens": OUTPUT_COLOR,
        },
    )
    fig_feature.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_feature.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        legend_title_text="",
        margin={"t": 80},
    )
    fig_feature
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Average tokens by turn number

    Later turns with more accumulated context have more input tokens.
    """)
    return


@app.cell(hide_code=True)
def _(INPUT_COLOR, OUTPUT_COLOR, pl, px, turns):
    turn_bucket_order = [str(n) for n in range(1, 11)] + ["11+"]

    by_turn = (
        turns.with_columns(
            turn_bucket=pl.when(pl.col("turn_id") <= 10).then(pl.col("turn_id").cast(pl.Utf8)).otherwise(pl.lit("11+"))
        )
        .group_by("turn_bucket")
        .agg(
            avg_input_tokens=pl.col("input_tokens").mean(),
            avg_output_tokens=pl.col("output_tokens").mean(),
            n=pl.len(),
        )
    )

    by_turn_long = by_turn.unpivot(
        index=["turn_bucket", "n"],
        on=["avg_input_tokens", "avg_output_tokens"],
        variable_name="metric",
        value_name="avg_tokens",
    )

    fig_turn = px.bar(
        by_turn_long,
        x="turn_bucket",
        y="avg_tokens",
        color="metric",
        barmode="group",
        category_orders={"turn_bucket": turn_bucket_order},
        color_discrete_map={
            "avg_input_tokens": INPUT_COLOR,
            "avg_output_tokens": OUTPUT_COLOR,
        },
    )
    fig_turn.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        legend_title_text="",
        xaxis_title="turn number in chat",
    )
    fig_turn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Distribution of tokens per message
    """)
    return


@app.cell(hide_code=True)
def _(INPUT_COLOR, OUTPUT_COLOR, make_subplots, pl, turns):
    # doubling thresholds (200, 400, 800, ...) are all positive, so unlike a raw
    # histogram bin edge at 0 they render fine on a true log x-axis - which also
    # gives clean, evenly-spaced ticks instead of long rotated bucket-range labels
    def survival_curve(col: str) -> pl.DataFrame:
        max_val = turns[col].max()
        n_total = turns.height
        thresholds = [200]
        while thresholds[-1] < max_val:
            thresholds.append(thresholds[-1] * 2)
        probabilities = [turns.filter(pl.col(col) >= t).height / n_total for t in thresholds]
        return pl.DataFrame({"threshold": thresholds, "probability": probabilities})

    def format_tokens(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:g}M"
        if n >= 1_000:
            return f"{n / 1_000:g}k"
        return str(n)

    input_survival = survival_curve("input_tokens")
    output_survival = survival_curve("output_tokens")

    fig_hist = make_subplots(rows=2, cols=1, subplot_titles=("Input tokens", "Output tokens"))
    fig_hist.add_scatter(
        x=input_survival["threshold"],
        y=input_survival["probability"],
        mode="lines+markers",
        line={"color": INPUT_COLOR, "shape": "hv"},
        name="input",
        row=1,
        col=1,
    )
    fig_hist.add_scatter(
        x=output_survival["threshold"],
        y=output_survival["probability"],
        mode="lines+markers",
        line={"color": OUTPUT_COLOR, "shape": "hv"},
        name="output",
        row=2,
        col=1,
    )
    fig_hist.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        showlegend=False,
    )
    for survival, hist_subplot_row in [(input_survival, 1), (output_survival, 2)]:
        fig_hist.update_xaxes(
            type="log",
            tickvals=survival["threshold"],
            ticktext=[format_tokens(t) for t in survival["threshold"]],
            row=hist_subplot_row,
            col=1,
        )
    fig_hist.update_xaxes(title_text="tokens (threshold)", row=2, col=1)
    fig_hist.update_yaxes(title_text="P(tokens >= threshold)", tickformat=".0%")
    fig_hist
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Chats by number of turns (cumulative: proportion with *at least* that many)

    More than a quarter of chats have at least five turns, but only 10%
    of chats have at least 10 turns
    """)
    return


@app.cell(hide_code=True)
def _(INPUT_COLOR, pl, px, turns):
    chat_turn_bucket_order = [str(n) for n in range(1, 11)] + ["11+"]

    turns_per_chat = turns.group_by("chat_id").agg(n_turns=pl.len())
    n_chats_total = turns_per_chat.height

    by_chat_turns = pl.DataFrame(
        {
            "turn_count_bucket": chat_turn_bucket_order,
            "proportion": [turns_per_chat.filter(pl.col("n_turns") >= k).height / n_chats_total for k in range(1, 11)]
            + [turns_per_chat.filter(pl.col("n_turns") >= 11).height / n_chats_total],
        }
    )

    fig_chat_turns = px.bar(
        by_chat_turns,
        x="turn_count_bucket",
        y="proportion",
        category_orders={"turn_count_bucket": chat_turn_bucket_order},
        color_discrete_sequence=[INPUT_COLOR],
    )
    fig_chat_turns.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis_title="turns in chat",
        yaxis_title="proportion of chats with >= that many turns",
    )
    fig_chat_turns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Regression: drivers of input token usage

    Not easy to interpret. I fit a model of main effects plus all interactions
    of features. The interaction of RAG + Gov.UK search + smart targets increases
    input tokens, but the single effect of Smart targets is close to zero. This
    indicates that Smart targets don't on their own add a lot of tokens.

    Each additional document adds about 2,700 tokens to each message. Gov.UK
    search adds more than 8,000 tokens to each message.
    """)
    return


@app.cell(hide_code=True)
def _(pd, pl, smf, turns):
    reg_turn_bucket_order = [str(n) for n in range(1, 11)] + ["11+"]

    # use_gov_uk_search_api and use_smart_targets never appear with use_rag=0 in this
    # data - not a code-level dependency, but likely behavioral: RAG is ticked on by
    # default in the app while gov.uk search/smart targets are opt-in extras, so
    # users who bother enabling them tend to be the ones who also want RAG. That
    # leaves 3 of the 8 factorial cells for a full
    # use_rag*use_gov_uk_search*use_smart_targets interaction with zero observations,
    # which makes the full interaction model rank-deficient. Model the
    # actually-observed feature combinations as one categorical instead.
    reg_df = turns.with_columns(
        turn_bucket=pl.when(pl.col("turn_id") <= 10).then(pl.col("turn_id").cast(pl.Utf8)).otherwise(pl.lit("11+")),
        feature_combo=pl.concat_str(
            [
                pl.when(pl.col("use_rag") == "t").then(pl.lit("rag")).otherwise(pl.lit("none")),
                pl.when(pl.col("use_gov_uk_search_api") == "t").then(pl.lit("+gov_uk")).otherwise(pl.lit("")),
                pl.when(pl.col("use_smart_targets") == "t").then(pl.lit("+smart_targets")).otherwise(pl.lit("")),
            ]
        ),
    ).to_pandas()
    reg_df["turn_bucket"] = pd.Categorical(reg_df["turn_bucket"], categories=reg_turn_bucket_order, ordered=True)

    reg_model = smf.ols(
        "input_tokens ~ C(turn_bucket, Treatment(reference='1'))"
        " + C(feature_combo, Treatment(reference='rag'))"
        " + non_central_document_count",
        data=reg_df,
    ).fit()

    reg_ci = reg_model.conf_int()
    coef_df = pl.DataFrame(
        {
            "term": reg_model.params.index.tolist(),
            "estimate": reg_model.params.values.tolist(),
            "ci_low": reg_ci[0].values.tolist(),
            "ci_high": reg_ci[1].values.tolist(),
        }
    )
    return coef_df, reg_df, reg_model


@app.cell(hide_code=True)
def _(INPUT_COLOR, coef_df, go):
    coef_df_sorted = coef_df.sort("estimate")

    fig_coef = go.Figure()
    fig_coef.add_trace(
        go.Scatter(
            x=coef_df_sorted["estimate"],
            y=coef_df_sorted["term"],
            error_x={
                "type": "data",
                "symmetric": False,
                "array": coef_df_sorted["ci_high"] - coef_df_sorted["estimate"],
                "arrayminus": coef_df_sorted["estimate"] - coef_df_sorted["ci_low"],
            },
            mode="markers",
            marker={"color": INPUT_COLOR, "size": 8},
        )
    )
    fig_coef.add_vline(x=0, line_dash="dot", line_color="#898781")
    fig_coef.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis_title="effect on input tokens",
        yaxis_title="",
    )
    fig_coef
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Feature combo usage frequency and total cost impact

    Effect size alone doesn't tell you how much a feature combo actually costs -
    a big per-turn effect on a rarely-used combo may matter less than a small
    effect on a common one. Total impact = effect (vs. the `rag`-only baseline)
    x number of turns using that combo.
    """)
    return


@app.cell(hide_code=True)
def _(INPUT_COLOR, pl, px, reg_df, reg_model):
    combo_effects = {
        term.split("[T.")[-1].rstrip("]"): estimate
        for term, estimate in reg_model.params.items()
        if "feature_combo" in term
    }
    combo_effects["rag"] = 0.0  # reference level, effect defined relative to it

    by_combo = (
        pl.from_pandas(reg_df[["feature_combo"]])
        .group_by("feature_combo")
        .agg(n=pl.len())
        .with_columns(
            proportion=pl.col("n") / pl.col("n").sum(),
            effect=pl.col("feature_combo").replace_strict(combo_effects),
        )
        .with_columns(total_impact=pl.col("effect") * pl.col("n"))
    )

    fig_combo_freq = px.bar(
        by_combo.sort("proportion", descending=True),
        x="feature_combo",
        y="proportion",
        color_discrete_sequence=[INPUT_COLOR],
    )
    fig_combo_freq.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis_title="feature combo",
        yaxis_title="proportion of turns",
    )
    fig_combo_freq
    return (by_combo,)


@app.cell(hide_code=True)
def _(OUTPUT_COLOR, by_combo, px):
    fig_combo_impact = px.bar(
        by_combo.sort("total_impact", descending=True),
        x="feature_combo",
        y="total_impact",
        color_discrete_sequence=[OUTPUT_COLOR],
    )
    fig_combo_impact.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis_title="feature combo",
        yaxis_title="total excess input tokens (effect x n turns)",
    )
    fig_combo_impact
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Document upload usage frequency and total cost impact, by number of docs

    The regression treats `non_central_document_count` as continuous (one
    coefficient = tokens per doc). Total impact per bucket = that coefficient x
    the actual number of docs uploaded, summed over turns in the bucket.
    """)
    return


@app.cell(hide_code=True)
def _(INPUT_COLOR, pl, px, reg_model, turns):
    doc_bucket_order = [str(n) for n in range(0, 11)] + ["11+"]
    doc_coef = reg_model.params["non_central_document_count"]

    by_doc_count = (
        turns.with_columns(
            doc_bucket=pl.when(pl.col("non_central_document_count") <= 10)
            .then(pl.col("non_central_document_count").cast(pl.Utf8))
            .otherwise(pl.lit("11+"))
        )
        .group_by("doc_bucket")
        .agg(
            n=pl.len(),
            total_docs=pl.col("non_central_document_count").sum(),
        )
        .with_columns(
            proportion=pl.col("n") / pl.col("n").sum(),
            total_impact=pl.col("total_docs") * doc_coef,
        )
    )

    fig_doc_freq = px.bar(
        by_doc_count,
        x="doc_bucket",
        y="proportion",
        category_orders={"doc_bucket": doc_bucket_order},
        color_discrete_sequence=[INPUT_COLOR],
    )
    fig_doc_freq.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis_title="docs uploaded",
        yaxis_title="proportion of turns",
    )
    fig_doc_freq
    return by_doc_count, doc_bucket_order


@app.cell(hide_code=True)
def _(OUTPUT_COLOR, by_doc_count, doc_bucket_order, px):
    fig_doc_impact = px.bar(
        by_doc_count,
        x="doc_bucket",
        y="total_impact",
        category_orders={"doc_bucket": doc_bucket_order},
        color_discrete_sequence=[OUTPUT_COLOR],
    )
    fig_doc_impact.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis_title="docs uploaded",
        yaxis_title="total excess input tokens (coef x total docs in bucket)",
    )
    fig_doc_impact
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Probability next turn arrives within 5 minutes, by turn number

    This analysis is a strong indicator of the VFM of 5-minute caching.
    Caching trades a 25% premium on cache writes for a 90% reduction in cost
    for cache reads.
    If P(next message within 5 minutes) is greater than about 0.25, caching will
    have a positive payoff.
    """)
    return


@app.cell(hide_code=True)
def _(INPUT_COLOR, OUTPUT_COLOR, go, pl, turns):
    prob_turn_bucket_order = [str(n) for n in range(1, 11)] + ["11+"]
    prob_turn_bucket_rank = {bucket: i for i, bucket in enumerate(prob_turn_bucket_order)}

    next_turn_prob = (
        turns.with_columns(
            turn_bucket=pl.when(pl.col("turn_id") <= 10).then(pl.col("turn_id").cast(pl.Utf8)).otherwise(pl.lit("11+")),
            # a null gap means there's no next message at all (chat ended) - treat as a
            # cache miss, not as missing data, so it pulls the probability down
            within_5_min=(pl.col("assistant_gap_to_next_seconds") <= 300).fill_null(False),
            has_next_turn=pl.col("assistant_gap_to_next_seconds").is_not_null(),
        )
        .group_by("turn_bucket")
        .agg(
            prob_within_5_min=pl.col("within_5_min").mean(),
            prob_has_next_turn=pl.col("has_next_turn").mean(),
            n=pl.len(),
        )
        .sort(pl.col("turn_bucket").replace_strict(prob_turn_bucket_rank))
    )

    fig_next_turn_prob = go.Figure()
    fig_next_turn_prob.add_scatter(
        x=next_turn_prob["turn_bucket"],
        y=next_turn_prob["prob_within_5_min"],
        mode="lines+markers",
        name="P(next turn within 5 min)",
        line={"color": INPUT_COLOR},
    )
    fig_next_turn_prob.add_scatter(
        x=next_turn_prob["turn_bucket"],
        y=next_turn_prob["prob_has_next_turn"],
        mode="lines+markers",
        name="P(there is ever a next turn)",
        line={"color": OUTPUT_COLOR},
    )
    fig_next_turn_prob.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis_title="turn number in chat",
        yaxis_title="probability",
        legend_title_text="",
    )
    fig_next_turn_prob
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Chat clustering (feature-vector approach)

    Summarize each chat into a fixed set of features (turn count, token
    totals/means, gap stats, feature usage) and cluster with k-means. This is
    a first pass - a true sequence-level clustering (e.g. DTW on the raw
    per-turn (input, output, gap) sequence) is a natural follow-up, since it
    doesn't collapse the *order* of turns into summary stats the way this
    does. Chats with only 1 turn are excluded - there's no sequence to
    cluster.
    """)
    return


@app.cell(hide_code=True)
def _(pl, turns, turns_all):
    # duration is a whole-chat-lifetime property, so measure it from turns_all
    # (full unfiltered history for these chats) rather than the date-windowed
    # `turns` - otherwise a chat that started before the window would have its
    # true duration truncated at the window edge
    duration_stats = (
        turns_all.filter(pl.col("chat_id").is_in(turns["chat_id"].unique().implode()))
        .group_by("chat_id")
        .agg(
            duration_days=(pl.col("message_created_at").max() - pl.col("message_created_at").min()).dt.total_seconds()
            / 86400
        )
        .with_columns(
            duration_gt_1_day=(pl.col("duration_days") > 1).cast(pl.Int8),
            duration_gt_1_week=(pl.col("duration_days") > 7).cast(pl.Int8),
            duration_gt_1_month=(pl.col("duration_days") > 30).cast(pl.Int8),
            duration_gt_6_months=(pl.col("duration_days") > 182).cast(pl.Int8),
        )
    )

    chat_features = (
        turns.with_columns(
            within_5_min=(pl.col("assistant_gap_to_next_seconds") <= 300).fill_null(False),
        )
        .group_by("chat_id")
        .agg(
            n_turns=pl.len(),
            total_input_tokens=pl.col("input_tokens").sum(),
            total_output_tokens=pl.col("output_tokens").sum(),
            mean_input_tokens=pl.col("input_tokens").mean(),
            mean_output_tokens=pl.col("output_tokens").mean(),
            mean_gap_seconds=pl.col("assistant_gap_to_next_seconds").mean(),
            frac_within_5_min=pl.col("within_5_min").mean(),
            max_docs=pl.col("non_central_document_count").max(),
            used_rag=(pl.col("use_rag") == "t").any(),
            used_gov_uk_search=(pl.col("use_gov_uk_search_api") == "t").any(),
            used_smart_targets=(pl.col("use_smart_targets") == "t").any(),
        )
        .filter(pl.col("n_turns") >= 2)
        .with_columns(
            used_rag=pl.col("used_rag").cast(pl.Int8),
            used_gov_uk_search=pl.col("used_gov_uk_search").cast(pl.Int8),
            used_smart_targets=pl.col("used_smart_targets").cast(pl.Int8),
        )
        .join(duration_stats, on="chat_id")
        .with_columns(log_duration_days=(pl.col("duration_days") + 1).log())
        # group_by row order isn't guaranteed - sort explicitly so KMeans (which
        # is sensitive to row order even with a fixed random_state) is reproducible
        .sort("chat_id")
    )
    return (chat_features,)


@app.cell(hide_code=True)
def _(chat_features):
    # grand totals across all clustered (2+ turn) chats - the denominator for
    # "what share of all tokens does this cluster account for"
    grand_total_input_tokens = chat_features["total_input_tokens"].sum()
    grand_total_output_tokens = chat_features["total_output_tokens"].sum()
    return grand_total_input_tokens, grand_total_output_tokens


@app.cell(hide_code=True)
def _():
    cluster_feature_cols = [
        "n_turns",
        "total_input_tokens",
        "total_output_tokens",
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_gap_seconds",
        "frac_within_5_min",
        "max_docs",
        "used_rag",
        "used_gov_uk_search",
        "used_smart_targets",
    ]
    # duration is excluded from clustering itself (see the ablation check below)
    # but kept available for the profile tables, since it's still worth
    # describing per cluster even though it doesn't help form the clusters
    profile_extra_cols = [
        "duration_days",
        "duration_gt_1_day",
        "duration_gt_1_week",
        "duration_gt_1_month",
        "duration_gt_6_months",
    ]
    return cluster_feature_cols, profile_extra_cols


@app.cell(hide_code=True)
def _(AgglomerativeClustering, StandardScaler, chat_features, cluster_feature_cols, silhouette_score):
    duration_ablation_cols = [*cluster_feature_cols, "duration_days", "log_duration_days"]

    X_no_duration = StandardScaler().fit_transform(chat_features.select(cluster_feature_cols).to_numpy())
    duration_flag_cols = [
        *cluster_feature_cols,
        "duration_gt_1_day",
        "duration_gt_1_week",
        "duration_gt_1_month",
        "duration_gt_6_months",
    ]
    X_with_duration_flags = StandardScaler().fit_transform(chat_features.select(duration_flag_cols).to_numpy())
    X_with_log_duration = StandardScaler().fit_transform(chat_features.select(duration_ablation_cols).to_numpy())

    ablation_rows = []
    for X_variant, variant_name in [
        (X_no_duration, "no duration"),
        (X_with_duration_flags, "duration threshold flags"),
        (X_with_log_duration, "log(duration_days)"),
    ]:
        for ablation_k in range(3, 7):
            sil = silhouette_score(
                X_variant, AgglomerativeClustering(n_clusters=ablation_k, linkage="ward").fit_predict(X_variant)
            )
            ablation_rows.append({"variant": variant_name, "k": ablation_k, "silhouette": round(sil, 3)})
    return X_no_duration, ablation_rows


@app.cell(hide_code=True)
def _(ablation_rows, mo, pl):
    ablation_table = pl.DataFrame(ablation_rows).pivot("k", index="variant", values="silhouette")

    mo.md(f"""
    **Does adding duration help or hurt clustering?** Agglomerative (Ward)
    silhouette score at k=3-6, with three variants of the duration feature:

    {ablation_table}

    Every duration variant scores lower than leaving it out entirely, at
    every k tried - even after log-transforming `duration_days` to tame its
    heavy right skew (few chats span hundreds of days, most span minutes).
    Duration looks genuinely counterproductive here, not just a scaling
    artifact: chats of similar duration don't seem to share the same profile
    on the other features, so adding it just dilutes the separation the
    other features already provide. Clustering below uses **no duration
    features** - duration is reported per cluster in the profile tables
    purely as a description, not as an input.
    """)
    return


@app.cell(hide_code=True)
def _(KMeans, X_no_duration, silhouette_score):
    km_solutions = {}
    for solution_k in range(3, 7):
        km_labels_k = KMeans(n_clusters=solution_k, random_state=0, n_init=10).fit_predict(X_no_duration)
        km_solutions[solution_k] = km_labels_k

    best_k = max(km_solutions, key=lambda k: silhouette_score(X_no_duration, km_solutions[k]))
    cluster_labels = km_solutions[best_k]
    feature_matrix = X_no_duration
    return best_k, cluster_labels, feature_matrix, km_solutions


@app.cell(hide_code=True)
def _(km_solutions, mo, np, silhouette_score, feature_matrix):
    mo.md(f"""
    K-means solutions at k=3-6 (not just the silhouette-optimal one):

    {
        chr(10).join(
            f"- k={k}: silhouette={silhouette_score(feature_matrix, labels):.3f}, "
            f"sizes={sorted(np.bincount(labels), reverse=True)}"
            for k, labels in km_solutions.items()
        )
    }
    """)
    return


@app.cell(hide_code=True)
def _(PCA, chat_features, cluster_labels, feature_matrix, pl, px):
    # fixed categorical order, dataviz skill palette slots 1-8
    CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

    pca_coords = PCA(n_components=2, random_state=0).fit_transform(feature_matrix)

    cluster_plot_df = chat_features.select("chat_id").with_columns(
        pc1=pca_coords[:, 0],
        pc2=pca_coords[:, 1],
        cluster=pl.Series([str(c) for c in cluster_labels]),
    )

    n_clusters_seen = cluster_plot_df["cluster"].n_unique()
    fig_clusters = px.scatter(
        cluster_plot_df,
        x="pc1",
        y="pc2",
        color="cluster",
        category_orders={"cluster": [str(c) for c in range(n_clusters_seen)]},
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.5,
    )
    fig_clusters.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title_text="cluster",
    )
    fig_clusters
    return


@app.cell(hide_code=True)
def _(
    chat_features,
    cluster_feature_cols,
    cluster_labels,
    grand_total_input_tokens,
    grand_total_output_tokens,
    pl,
    profile_extra_cols,
    turns,
):
    # a per-chat median gap is a bad statistic here: most chats have only 1-2
    # gaps (1,031 of ~3,400 chats have exactly one), so a chat's own "median"
    # is often just a single coin-flip value between a quick reply and a
    # next-day return, and averaging those noisy per-chat medians across
    # chats doesn't reconstruct anything meaningful. Pool the raw turn-level
    # gaps within each cluster instead and take one mean/median over all of
    # them - a statistic actually stable at the sample sizes involved.
    chat_cluster_map = chat_features.select("chat_id").with_columns(cluster=pl.Series(cluster_labels).cast(pl.Utf8))

    pooled_gap_stats = (
        turns.join(chat_cluster_map, on="chat_id")
        .filter(pl.col("assistant_gap_to_next_seconds").is_not_null())
        .group_by("cluster")
        .agg(
            pooled_mean_gap_minutes=pl.col("assistant_gap_to_next_seconds").mean() / 60,
            pooled_median_gap_minutes=pl.col("assistant_gap_to_next_seconds").median() / 60,
        )
    )

    profile_cols = [*cluster_feature_cols, *profile_extra_cols]
    cluster_profile = (
        chat_features.select(profile_cols)
        .with_columns(cluster=pl.Series(cluster_labels).cast(pl.Utf8))
        .group_by("cluster")
        .agg([pl.col(c).mean() for c in profile_cols] + [pl.len().alias("n_chats")])
        .join(pooled_gap_stats, on="cluster")
        .with_columns(
            n_turns=pl.col("n_turns").round(1),
            # cluster's share of tokens across the whole clustered dataset:
            # (per-chat mean total) x (n chats in cluster) recovers the
            # cluster's grand total, then divide by the dataset grand total
            pct_input_tokens=(pl.col("total_input_tokens") * pl.col("n_chats") / grand_total_input_tokens * 100).round(
                1
            ),
            pct_output_tokens=(
                pl.col("total_output_tokens") * pl.col("n_chats") / grand_total_output_tokens * 100
            ).round(1),
            total_input_tokens=pl.col("total_input_tokens").round(0).cast(pl.Int64),
            total_output_tokens=pl.col("total_output_tokens").round(0).cast(pl.Int64),
            mean_input_tokens=pl.col("mean_input_tokens").round(0).cast(pl.Int64),
            mean_output_tokens=pl.col("mean_output_tokens").round(0).cast(pl.Int64),
            pooled_mean_gap_minutes=pl.col("pooled_mean_gap_minutes").round(1),
            pooled_median_gap_minutes=pl.col("pooled_median_gap_minutes").round(1),
            pct_within_5_min=(pl.col("frac_within_5_min") * 100).round(1),
            max_docs=pl.col("max_docs").round(1),
            pct_used_rag=(pl.col("used_rag") * 100).round(1),
            pct_used_gov_uk_search=(pl.col("used_gov_uk_search") * 100).round(1),
            pct_used_smart_targets=(pl.col("used_smart_targets") * 100).round(1),
            duration_days=pl.col("duration_days").round(1),
            pct_duration_gt_1_day=(pl.col("duration_gt_1_day") * 100).round(1),
            pct_duration_gt_1_week=(pl.col("duration_gt_1_week") * 100).round(1),
            pct_duration_gt_1_month=(pl.col("duration_gt_1_month") * 100).round(1),
            pct_duration_gt_6_months=(pl.col("duration_gt_6_months") * 100).round(1),
        )
        .drop(
            "mean_gap_seconds",
            "frac_within_5_min",
            "used_rag",
            "used_gov_uk_search",
            "used_smart_targets",
            "duration_gt_1_day",
            "duration_gt_1_week",
            "duration_gt_1_month",
            "duration_gt_6_months",
        )
        .sort("cluster")
    )
    cluster_profile = cluster_profile.select(
        ["cluster", "n_chats", *[c for c in cluster_profile.columns if c not in ("cluster", "n_chats")]]
    )
    cluster_profile
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Reading the clusters** (`chat_features` is explicitly sorted by
    `chat_id` before clustering - `KMeans(random_state=0)` is only
    reproducible if the row order feeding it is also fixed, since row order
    affects which points the seeded initialization picks first):

    - **Cluster 0** (by far the largest group, ~3,000 chats) - plain
      `rag`-only, short (~5 turns), fast-moving chats. This is the default
      "typical" chat.
    - **Cluster 1** (a small group, ~74 chats) - the "all the toppings"
      power-user group: near-universal RAG + gov.uk search and 100%
      smart-targets usage, plus by far the highest total/mean input tokens
      and the slowest mean gap between turns (chats that get resumed hours
      later rather than continued immediately).
    - **Cluster 2** (~247 chats) - long-running chats (~29 turns) with very
      high total input tokens and above-average doc uploads - RAG-only but
      clearly power-users of the conversation itself rather than the extra
      features.
    - **Cluster 3** (a small group, ~35 chats) - uses none of RAG/gov.uk
      search/smart-targets at all, short chats, by far the fastest mean gap.

    This is a feature-vector summary, though - it collapses each chat's
    *order* of turns into fixed stats. The sequence-analysis section below is
    a more literal treatment of "chat as sequence."
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Agglomerative clustering (feature-vector), for comparison

    K-means tends toward similarly-sized, roughly spherical clusters because
    it minimizes within-cluster variance around a centroid - which is a poor
    fit when the real groups are very different sizes (as they clearly are
    here: one dominant "typical chat" cluster and a few small, distinct
    minority groups). Ward-linkage agglomerative clustering builds clusters
    bottom-up by merging the pair that most increases within-cluster
    variance the least, which doesn't have the same built-in pull toward
    equal sizes - worth checking whether it actually gives more balanced
    groups here, or whether the size imbalance is a genuine feature of the
    data rather than a k-means artifact.
    """)
    return


@app.cell(hide_code=True)
def _(feature_matrix, ff, linkage):
    fig_dendrogram = ff.create_dendrogram(feature_matrix, linkagefun=lambda x: linkage(x, method="ward"))
    fig_dendrogram.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis_title="chat (leaves, one per chat - no labels at this sample size)",
        yaxis_title="Ward linkage distance",
        showlegend=False,
    )
    fig_dendrogram.update_xaxes(showticklabels=False)
    fig_dendrogram
    return


@app.cell(hide_code=True)
def _(AgglomerativeClustering, feature_matrix, silhouette_score):
    agg_solutions = {}
    for agg_solution_k in range(3, 7):
        agg_solutions[agg_solution_k] = AgglomerativeClustering(n_clusters=agg_solution_k, linkage="ward").fit_predict(
            feature_matrix
        )

    agg_best_k = max(agg_solutions, key=lambda k: silhouette_score(feature_matrix, agg_solutions[k]))
    agg_cluster_labels = agg_solutions[agg_best_k]
    return agg_best_k, agg_cluster_labels, agg_solutions


@app.cell(hide_code=True)
def _(agg_best_k, agg_solutions, best_k, cluster_labels, feature_matrix, mo, np, silhouette_score):
    kmeans_sizes = sorted(np.bincount(cluster_labels), reverse=True)
    agg_sizes_at_best = sorted(np.bincount(agg_solutions[agg_best_k]), reverse=True)

    mo.md(f"""
    Agglomerative (Ward) solutions at k=3-6, same format as the k-means ones
    above, for direct comparison:

    {
        chr(10).join(
            f"- k={k}: silhouette={silhouette_score(feature_matrix, labels):.3f}, "
            f"sizes={sorted(np.bincount(labels), reverse=True)}"
            for k, labels in agg_solutions.items()
        )
    }

    Cluster sizes at each method's own silhouette-optimal k, largest to
    smallest:

    - **k-means** (k={best_k}): {kmeans_sizes}
    - **agglomerative** (k={agg_best_k}): {agg_sizes_at_best}
    """)
    return


@app.cell(hide_code=True)
def _(
    agg_cluster_labels,
    chat_features,
    cluster_feature_cols,
    grand_total_input_tokens,
    grand_total_output_tokens,
    pl,
    profile_extra_cols,
    turns,
):
    agg_chat_cluster_map = chat_features.select("chat_id").with_columns(
        cluster=pl.Series(agg_cluster_labels).cast(pl.Utf8)
    )

    agg_pooled_gap_stats = (
        turns.join(agg_chat_cluster_map, on="chat_id")
        .filter(pl.col("assistant_gap_to_next_seconds").is_not_null())
        .group_by("cluster")
        .agg(
            pooled_mean_gap_minutes=pl.col("assistant_gap_to_next_seconds").mean() / 60,
            pooled_median_gap_minutes=pl.col("assistant_gap_to_next_seconds").median() / 60,
        )
    )

    agg_profile_cols = [*cluster_feature_cols, *profile_extra_cols]
    agg_cluster_profile = (
        chat_features.select(agg_profile_cols)
        .with_columns(cluster=pl.Series(agg_cluster_labels).cast(pl.Utf8))
        .group_by("cluster")
        .agg([pl.col(c).mean() for c in agg_profile_cols] + [pl.len().alias("n_chats")])
        .join(agg_pooled_gap_stats, on="cluster")
        .with_columns(
            n_turns=pl.col("n_turns").round(1),
            # cluster's share of tokens across the whole clustered dataset:
            # (per-chat mean total) x (n chats in cluster) recovers the
            # cluster's grand total, then divide by the dataset grand total
            pct_input_tokens=(pl.col("total_input_tokens") * pl.col("n_chats") / grand_total_input_tokens * 100).round(
                1
            ),
            pct_output_tokens=(
                pl.col("total_output_tokens") * pl.col("n_chats") / grand_total_output_tokens * 100
            ).round(1),
            total_input_tokens=pl.col("total_input_tokens").round(0).cast(pl.Int64),
            total_output_tokens=pl.col("total_output_tokens").round(0).cast(pl.Int64),
            mean_input_tokens=pl.col("mean_input_tokens").round(0).cast(pl.Int64),
            mean_output_tokens=pl.col("mean_output_tokens").round(0).cast(pl.Int64),
            pooled_mean_gap_minutes=pl.col("pooled_mean_gap_minutes").round(1),
            pooled_median_gap_minutes=pl.col("pooled_median_gap_minutes").round(1),
            pct_within_5_min=(pl.col("frac_within_5_min") * 100).round(1),
            max_docs=pl.col("max_docs").round(1),
            pct_used_rag=(pl.col("used_rag") * 100).round(1),
            pct_used_gov_uk_search=(pl.col("used_gov_uk_search") * 100).round(1),
            pct_used_smart_targets=(pl.col("used_smart_targets") * 100).round(1),
            duration_days=pl.col("duration_days").round(1),
            pct_duration_gt_1_day=(pl.col("duration_gt_1_day") * 100).round(1),
            pct_duration_gt_1_week=(pl.col("duration_gt_1_week") * 100).round(1),
            pct_duration_gt_1_month=(pl.col("duration_gt_1_month") * 100).round(1),
            pct_duration_gt_6_months=(pl.col("duration_gt_6_months") * 100).round(1),
        )
        .drop(
            "mean_gap_seconds",
            "frac_within_5_min",
            "used_rag",
            "used_gov_uk_search",
            "used_smart_targets",
            "duration_gt_1_day",
            "duration_gt_1_week",
            "duration_gt_1_month",
            "duration_gt_6_months",
        )
        .sort("cluster")
    )
    agg_cluster_profile = agg_cluster_profile.select(
        ["cluster", "n_chats", *[c for c in agg_cluster_profile.columns if c not in ("cluster", "n_chats")]]
    )
    agg_cluster_profile
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cross-tabulation: k-means vs agglomerative, both at k=4

    Every chat gets two labels - its k-means cluster and its agglomerative
    (Ward) cluster - so we can see directly how much the two methods agree,
    rather than just comparing size distributions.
    """)
    return


@app.cell(hide_code=True)
def _(agg_solutions, chat_features, km_solutions, pl):
    cross_tab_df = chat_features.select("chat_id").with_columns(
        kmeans_cluster=pl.Series(km_solutions[4]).cast(pl.Utf8),
        agglomerative_cluster=pl.Series(agg_solutions[4]).cast(pl.Utf8),
    )

    cross_tab = (
        cross_tab_df.group_by(["kmeans_cluster", "agglomerative_cluster"])
        .agg(n=pl.len())
        .pivot("agglomerative_cluster", index="kmeans_cluster", values="n")
        .fill_null(0)
        .sort("kmeans_cluster")
        .rename({c: f"agg_{c}" for c in ["0", "1", "2", "3"]})
        .select(["kmeans_cluster", "agg_0", "agg_1", "agg_2", "agg_3"])
    )
    cross_tab
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The five-way "consensus" split

    Three of the four clusters are identical between k-means and
    agglomerative with zero cross-contamination. The only disagreement is
    k-means' cluster 2, which agglomerative splits into two. Treat the
    (kmeans_cluster, agglomerative_cluster) pair itself as the group label -
    only 5 of the 16 possible combinations are ever actually observed, since
    3 of the 4 rows in the cross-tab are pure - so this "product partition"
    is a genuine 5-way split, not an arbitrary relabeling.
    """)
    return


@app.cell(hide_code=True)
def _(agg_solutions, chat_features, km_solutions, pl):
    consensus_map = chat_features.select("chat_id").with_columns(
        consensus_cluster=pl.Series(km_solutions[4]).cast(pl.Utf8) + "|" + pl.Series(agg_solutions[4]).cast(pl.Utf8)
    )
    return (consensus_map,)


@app.cell(hide_code=True)
def _(
    chat_features,
    cluster_feature_cols,
    consensus_map,
    grand_total_input_tokens,
    grand_total_output_tokens,
    pl,
    profile_extra_cols,
    turns,
):
    consensus_pooled_gap_stats = (
        turns.join(consensus_map, on="chat_id")
        .filter(pl.col("assistant_gap_to_next_seconds").is_not_null())
        .group_by("consensus_cluster")
        .agg(
            pooled_mean_gap_minutes=pl.col("assistant_gap_to_next_seconds").mean() / 60,
            pooled_median_gap_minutes=pl.col("assistant_gap_to_next_seconds").median() / 60,
        )
    )

    consensus_profile_cols = [*cluster_feature_cols, *profile_extra_cols]
    consensus_cluster_profile = (
        chat_features.select(["chat_id", *consensus_profile_cols])
        .join(consensus_map, on="chat_id")
        .group_by("consensus_cluster")
        .agg([pl.col(c).mean() for c in consensus_profile_cols] + [pl.len().alias("n_chats")])
        .join(consensus_pooled_gap_stats, on="consensus_cluster")
        .with_columns(
            n_turns=pl.col("n_turns").round(1),
            # cluster's share of tokens across the whole clustered dataset:
            # (per-chat mean total) x (n chats in cluster) recovers the
            # cluster's grand total, then divide by the dataset grand total
            pct_input_tokens=(pl.col("total_input_tokens") * pl.col("n_chats") / grand_total_input_tokens * 100).round(
                1
            ),
            pct_output_tokens=(
                pl.col("total_output_tokens") * pl.col("n_chats") / grand_total_output_tokens * 100
            ).round(1),
            total_input_tokens=pl.col("total_input_tokens").round(0).cast(pl.Int64),
            total_output_tokens=pl.col("total_output_tokens").round(0).cast(pl.Int64),
            mean_input_tokens=pl.col("mean_input_tokens").round(0).cast(pl.Int64),
            mean_output_tokens=pl.col("mean_output_tokens").round(0).cast(pl.Int64),
            pooled_mean_gap_minutes=pl.col("pooled_mean_gap_minutes").round(1),
            pooled_median_gap_minutes=pl.col("pooled_median_gap_minutes").round(1),
            pct_within_5_min=(pl.col("frac_within_5_min") * 100).round(1),
            max_docs=pl.col("max_docs").round(1),
            pct_used_rag=(pl.col("used_rag") * 100).round(1),
            pct_used_gov_uk_search=(pl.col("used_gov_uk_search") * 100).round(1),
            pct_used_smart_targets=(pl.col("used_smart_targets") * 100).round(1),
            duration_days=pl.col("duration_days").round(1),
            pct_duration_gt_1_day=(pl.col("duration_gt_1_day") * 100).round(1),
            pct_duration_gt_1_week=(pl.col("duration_gt_1_week") * 100).round(1),
            pct_duration_gt_1_month=(pl.col("duration_gt_1_month") * 100).round(1),
            pct_duration_gt_6_months=(pl.col("duration_gt_6_months") * 100).round(1),
        )
        .drop(
            "mean_gap_seconds",
            "frac_within_5_min",
            "used_rag",
            "used_gov_uk_search",
            "used_smart_targets",
            "duration_gt_1_day",
            "duration_gt_1_week",
            "duration_gt_1_month",
            "duration_gt_6_months",
        )
        .sort("n_chats", descending=True)
    )
    consensus_cluster_profile = consensus_cluster_profile.select(
        [
            "consensus_cluster",
            "n_chats",
            *[c for c in consensus_cluster_profile.columns if c not in ("consensus_cluster", "n_chats")],
        ]
    )
    consensus_cluster_profile
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Browse turns by consensus cluster

    Same turn-level dataset as at the top of the notebook, with
    `consensus_cluster` joined on - use the table's built-in filter/sort to
    pull up examples from a specific cluster.
    """)
    return


@app.cell(hide_code=True)
def _(consensus_map, turns):
    turns_with_consensus_cluster = turns.join(consensus_map, on="chat_id", how="left")
    turns_with_consensus_cluster
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Linear turn-number model with interactions

    An alternative specification to the dummy-per-turn model above: treat turn
    number as a single continuous slope and interact it with the feature combo
    and doc count, to see whether their per-turn effect grows or shrinks as a
    chat gets longer.

    This isn't easy to interpret and not useful.
    """)
    return


@app.cell(hide_code=True)
def _(pl, reg_df, smf):
    # shift turn_id so turn 1 (the actual minimum in the data) is the reference,
    # rather than turn 0 which never occurs - makes the intercept interpretable
    # as the effect at the first turn, not an extrapolation outside the data
    reg_model_linear = smf.ols(
        "input_tokens ~ I(turn_id - 1) * C(feature_combo, Treatment(reference='rag'))"
        " + I(turn_id - 1) * non_central_document_count",
        data=reg_df,
    ).fit()

    reg_linear_ci = reg_model_linear.conf_int()
    coef_df_linear = pl.DataFrame(
        {
            "term": reg_model_linear.params.index.tolist(),
            "estimate": reg_model_linear.params.values.tolist(),
            "ci_low": reg_linear_ci[0].values.tolist(),
            "ci_high": reg_linear_ci[1].values.tolist(),
        }
    )
    return (coef_df_linear,)


@app.cell(hide_code=True)
def _(INPUT_COLOR, coef_df_linear, go):
    coef_df_linear_sorted = coef_df_linear.sort("estimate")

    fig_coef_linear = go.Figure()
    fig_coef_linear.add_trace(
        go.Scatter(
            x=coef_df_linear_sorted["estimate"],
            y=coef_df_linear_sorted["term"],
            error_x={
                "type": "data",
                "symmetric": False,
                "array": coef_df_linear_sorted["ci_high"] - coef_df_linear_sorted["estimate"],
                "arrayminus": coef_df_linear_sorted["estimate"] - coef_df_linear_sorted["ci_low"],
            },
            mode="markers",
            marker={"color": INPUT_COLOR, "size": 8},
        )
    )
    fig_coef_linear.add_vline(x=0, line_dash="dot", line_color="#898781")
    fig_coef_linear.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis_title="effect on input tokens",
        yaxis_title="",
    )
    fig_coef_linear
    return


# @app.cell(hide_code=True, disabled=True)
# def _(mo):
#     mo.md("""
#     ## Sequence analysis (optimal matching, TraMineR-style)

#     This is closer to what "sequence clustering" usually means in social
#     sequence analysis (TraMineR in R): discretize each turn into a small
#     alphabet of states, represent a chat as a *string* of states in order,
#     then measure dissimilarity between two chats via **optimal matching**
#     (edit distance - the cost to transform one sequence into the other via
#     substitutions/insertions/deletions) rather than via summary statistics.
#     Clusters are found by hierarchical clustering on the resulting distance
#     matrix.

#     Simplifications for this first pass:

#     - **Alphabet of 4 states**, from a 2x2 split of each turn: input tokens
#       above/below the overall median (`H`/`L`) x whether the *next* turn
#       arrives within 5 min (`F`ast) or not (`S`low/chat ends) - reusing the
#       same 5-minute cache threshold as the earlier probability chart.
#     - **Equal substitution/indel costs** (i.e. plain edit distance, via
#       `editdistance`) - TraMineR supports data-driven substitution costs, but
#       equal costs are its own standard default and a reasonable starting
#       point.
#     - **Subsampled to 400 chats** (of the 2+ turn chats) - optimal matching is
#       O(n^2) in the number of chats and O(L^2) in sequence length per pair, so
#       the full ~3,400-chat set would be slow to iterate on interactively.
#     """)
#     return


@app.cell(hide_code=True, disabled=True)
def _(pl, turns):
    state_seq_sample_size = 400

    turn_states = turns.with_columns(
        token_level=pl.when(pl.col("input_tokens") > pl.col("input_tokens").median())
        .then(pl.lit("H"))
        .otherwise(pl.lit("L")),
        speed_level=pl.when((pl.col("assistant_gap_to_next_seconds") <= 300).fill_null(False))
        .then(pl.lit("F"))
        .otherwise(pl.lit("S")),
    ).with_columns(state=pl.col("token_level") + pl.col("speed_level"))

    chat_sequences = (
        turn_states.sort(["chat_id", "turn_id"])
        .group_by("chat_id", maintain_order=True)
        .agg(sequence=pl.col("state").str.join(""), n_turns=pl.len())
        .filter(pl.col("n_turns") >= 2)
        .sample(n=state_seq_sample_size, seed=0)
    )
    return (chat_sequences,)


@app.cell(hide_code=True, disabled=True)
def _(AgglomerativeClustering, chat_sequences, editdistance, np, silhouette_score):
    sequences = chat_sequences["sequence"].to_list()
    seq_lengths = [len(s) for s in sequences]
    n_seqs = len(sequences)

    # normalize by the pair's longer sequence length - plain edit distance is
    # dominated by length differences (a 150-turn chat is "far" from
    # everything just by virtue of being long), which drowns out the actual
    # state pattern; this is TraMineR's standard normalization for OM
    om_distance = np.zeros((n_seqs, n_seqs))
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            d = editdistance.eval(sequences[i], sequences[j]) / max(seq_lengths[i], seq_lengths[j])
            om_distance[i, j] = d
            om_distance[j, i] = d

    om_silhouette_by_k = {
        k: silhouette_score(
            om_distance,
            AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(om_distance),
            metric="precomputed",
        )
        for k in range(2, 9)
    }
    om_best_k = max(om_silhouette_by_k, key=om_silhouette_by_k.get)

    om_cluster_labels = AgglomerativeClustering(
        n_clusters=om_best_k, metric="precomputed", linkage="average"
    ).fit_predict(om_distance)
    return om_best_k, om_cluster_labels, om_silhouette_by_k


# @app.cell(hide_code=True, disabled=True)
# def _(mo, om_best_k, om_silhouette_by_k):
#     mo.md(f"""
#     Best silhouette score at k={om_best_k} (scores by k: {om_silhouette_by_k}).

#     **Caveat worth being upfront about**: even normalized, silhouette still
#     picks k=2 - and that split is just "the 1-2 extreme-length outlier chats"
#     vs. "everyone else," which isn't an interesting finding. Length keeps
#     dominating even after normalization, because a handful of very long
#     chats sit far from the bulk of the sample under any edit-distance metric.
#     Looking at k=3 or k=4 instead (still high silhouette, just not the global
#     max) surfaces something more useful: alongside the length-outlier
#     cluster, a genuine shape difference shows up - one cluster of
#     medium-length chats with a highly regular `LFLFLFLF...` rhythm (small,
#     fast-turnaround messages back and forth), distinct from the bulk of
#     short chats that don't have room to show any rhythm at all. The index
#     plot below is the more trustworthy artifact here - eyeball it rather than
#     fully trusting the silhouette-selected k.
#     """)
#     return


@app.cell(hide_code=True, disabled=True)
def _(chat_sequences, go, np, om_cluster_labels, pl):
    # index plot (TraMineR's seqIplot): one row per chat, one column per turn
    # position, cell color = state at that turn - rows sorted by cluster so
    # each cluster's characteristic shape is visible as a block
    STATE_CODES = {"LS": 0, "LF": 1, "HS": 2, "HF": 3}
    STATE_LABELS = ["low tokens, slow", "low tokens, fast", "high tokens, slow", "high tokens, fast"]
    STATE_COLORS = ["#2a78d6", "#86b6ef", "#eb6834", "#f3b190"]
    n_states = len(STATE_COLORS)

    # equal-width flat bands so each integer state maps to one solid color,
    # rather than plotly's default smooth interpolation between colors
    state_colorscale = []
    for state_i, state_color in enumerate(STATE_COLORS):
        state_colorscale.append([state_i / n_states, state_color])
        state_colorscale.append([(state_i + 1) / n_states, state_color])

    index_plot_df = chat_sequences.with_columns(cluster=pl.Series(om_cluster_labels)).sort(["cluster", "sequence"])

    max_len = index_plot_df["sequence"].str.len_chars().max() // 2
    grid = np.full((index_plot_df.height, max_len), np.nan)
    for row_i, seq in enumerate(index_plot_df["sequence"].to_list()):
        for char_i in range(0, len(seq), 2):
            grid[row_i, char_i // 2] = STATE_CODES[seq[char_i : char_i + 2]]

    fig_index_plot = go.Figure(
        data=go.Heatmap(
            z=grid,
            zmin=-0.5,
            zmax=n_states - 0.5,
            colorscale=state_colorscale,
            showscale=True,
            colorbar={"tickvals": list(range(n_states)), "ticktext": STATE_LABELS},
        )
    )
    fig_index_plot.update_layout(
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis_title="turn number",
        yaxis_title="chat (sorted by cluster)",
    )
    fig_index_plot
    return


if __name__ == "__main__":
    app.run()
