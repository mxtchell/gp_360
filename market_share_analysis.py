from __future__ import annotations
import ast
from dataclasses import dataclass
import json
from types import SimpleNamespace
from typing import Dict, List, Optional

from skill_framework import SkillInput, SkillVisualization, skill, SkillParameter, SkillOutput, SuggestedQuestion, \
    ParameterDisplayDescription
from skill_framework.skills import ExportData
from skill_framework.layouts import wire_layout

from ar_analytics import MarketShareBreakdown, MSBTemplateParameterSetup, ArUtils
from ar_analytics.defaults import market_share_analysis_config, default_table_layout

import jinja2
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Default manufacturer for "my" market share queries
DEFAULT_MANUFACTURER = "Apex International"

# Extended parameter guidance with default manufacturer
EXTENDED_PARAMETER_GUIDANCE = (
    (market_share_analysis_config.parameter_guidance or "") +
    f" IMPORTANT: When users ask for 'my' market share, 'our' market share, or refer to their own company's performance, "
    f"use manufacturer='{DEFAULT_MANUFACTURER}' in other_filters."
)

# Custom description that excludes impact/what-if questions
CUSTOM_DESCRIPTION = (
    "Analyze CURRENT market share position and trends. Shows share by category, region, brand. "
    "DO NOT use this skill for 'impact' or 'what if' questions about price changes - use FP&A What-If Analysis instead."
)

CUSTOM_LIMITATIONS = (
    "This skill shows CURRENT market share only. "
    "DO NOT use for: 'what will be the impact', 'what if we increase prices', 'how would price changes affect share'. "
    "For impact/what-if scenarios, use FP&A What-If Analysis skill."
)

@skill(
    name=market_share_analysis_config.name,
    llm_name=market_share_analysis_config.llm_name,
    description=CUSTOM_DESCRIPTION,
    capabilities=market_share_analysis_config.capabilities,
    limitations=CUSTOM_LIMITATIONS,
    example_questions=market_share_analysis_config.example_questions,
    parameter_guidance=EXTENDED_PARAMETER_GUIDANCE,
    parameters=[
        SkillParameter(
            name="metric",
            constrained_to="metrics",
            is_multi=False
        ),
        SkillParameter(
            name="growth_type",
            is_multi=False,
            constrained_values=["Y/Y", "P/P"],
            description="Growth type either Y/Y or P/P",
            default_value="Y/Y"
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            is_multi=True
        ),
        SkillParameter(
            name="limit_n",
            description="limit the number of values by this number",
            default_value=20
        ),
        SkillParameter(
            name="periods",
            constrained_to="date_filter",
            is_multi=True,
            description="If provided by the user, list time periods in a format 'q2 2023', '2021', 'jan 2023', 'mat nov 2022', 'mat q1 2021', 'ytd q4 2022', 'ytd 2023', 'ytd', 'mat', '<no_period_provided>' or '<since_launch>'. Use knowledge about today's date to handle relative periods and open ended periods. If given a range, for example 'last 3 quarters, 'between q3 2022 to q4 2023' etc, enumerate the range into a list of valid dates. Don't include natural language words or phrases, only valid dates like 'q3 2023', '2022', 'mar 2020', 'ytd sep 2021', 'mat q4 2021', 'ytd q1 2022', 'ytd 2021', 'ytd', 'mat', '<no_period_provided>' or '<since_launch>' etc."
        ),
        SkillParameter(
            name="global_view",
            parameter_type="code",
            default_value=""
        ),
        SkillParameter(
            name="market_view",
            parameter_type="code",
            default_value=""
        ),
        SkillParameter(
            name="include_drivers",
            parameter_type="code",
            default_value=True
        ),
        SkillParameter(
            name="market_cols",
            parameter_type="code",
            default_value=""
        ),
        SkillParameter(
            name="impact_calcs",
            parameter_type="code",
            default_value=""
        ),
        SkillParameter(
            name="decomposition_display_config",
            parameter_type="code",
            default_value=""
        ),
        SkillParameter(
            name="subject_metric_config",
            parameter_type="code",
            default_value=""
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt being used for max response.",
            default_value=market_share_analysis_config.max_prompt
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt being used for detailed insights.",
            default_value=market_share_analysis_config.insight_prompt
        ),
        SkillParameter(
            name="table_viz_layout",
            parameter_type="visualization",
            description="Table Viz Layout",
            default_value=default_table_layout
        ),
        SkillParameter(
            name="scenario",
            parameter_type="code",
            description="Scenario filter - locked to actuals",
            default_value="actuals"
        )
    ]
)
def market_share_analysis(parameters: SkillInput):
    print(f"Skill received following parameters: {parameters}")
    param_dict = {"periods": [], "metric": None, "limit_n": 20, "growth_type": "Y/Y", "other_filters": [], "global_view": [], "market_view": [],
                  "include_drivers": True, "market_cols": [], "impact_calcs": {}, "decomposition_display_config": {}, "subject_metric_config": {},
                  "scenario": "actuals"}
    
    code_params = ["global_view", "market_view", "market_cols", "impact_calcs", "decomposition_display_config", "subject_metric_config"]

    # Update param_dict with values from parameters.arguments if they exist
    for key in param_dict:
        if hasattr(parameters.arguments, key) and getattr(parameters.arguments, key) is not None:
            param_dict[key] = getattr(parameters.arguments, key)
            if key in code_params and isinstance(param_dict[key], str) and param_dict[key]:
                try:
                    param_dict[key] = json.loads(param_dict[key])
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for Python-style dict/list syntax (single quotes)
                    try:
                        param_dict[key] = ast.literal_eval(param_dict[key])
                        logger.info(f"Parsed parameter {key} using ast.literal_eval")
                    except (ValueError, SyntaxError):
                        logger.error(f"Error decoding parameter: {key}")
                        param_dict[key] = {}

    # Scenario is locked to "actuals" - don't allow override from parameters
    param_dict["scenario"] = "actuals"

    if str(param_dict["growth_type"]).lower() not in ["y/y", 'p/p']:
        param_dict["growth_type"] = "Y/Y"

    env = SimpleNamespace(**param_dict)
    MSBTemplateParameterSetup(env=env)
    env.msa = MarketShareBreakdown.from_env(env=env)

    result_dfs = env.msa.run_from_env()
    print(result_dfs.keys())
    tables = env.msa.get_display_tables()
    param_info = [ParameterDisplayDescription(key=k, value=v) for k, v in env.msa.paramater_display_infomation.items()]

    share_metric_label = env.msa.share_metric_label
    include_drivers = env.msa.include_drivers
    metric_drivers_labels = env.msa.metric_drivers_labels
    subject_metric_drivers = env.msa.subject_metric_drivers
    decomposition_metric_drivers = env.msa.decomposition_metric_drivers

    insights_dfs = [env.msa.subject_facts, env.msa.bottom_peers_facts, env.msa.top_peers_facts, env.msa.bottom_breakouts_facts, env.msa.top_breakouts_facts, env.msa.metric_driver_challenges_facts, env.msa.df_notes]
    followups = env.msa.suggestions

    viz, insights, final_prompt, export_data = render_layout(tables,
                                                            env.msa.title,
                                                            env.msa.subtitle,
                                                            insights_dfs,
                                                            env.msa.warning_message,
                                                            parameters.arguments.max_prompt,
                                                            parameters.arguments.insight_prompt, 
                                                            parameters.arguments.table_viz_layout,
                                                            share_metric_label, 
                                                            include_drivers,
                                                            metric_drivers_labels,
                                                            subject_metric_drivers,
                                                            decomposition_metric_drivers)

    return SkillOutput(
        final_prompt=final_prompt,
        narrative=None,
        visualizations=viz,
        parameter_display_descriptions=param_info,
        followup_questions=[SuggestedQuestion(label=f.get("label"), question=f.get("question")) for f in followups if
                            f.get("label")],
        export_data=[ExportData(name=name, data=df) for name, df in export_data.items()]
    )

def get_data(
        tab_name: str, 
        df: pd.DataFrame, 
        ignore_cols: List[str] = [], 
        highlight_col: str = None, 
        followup_col: str = None, 
        sparkline_col: str = None
    ):

    dim_member_col = f"Share by {tab_name}"
    has_subject = highlight_col in df.columns
    is_grouping = 'is_collapsible' in df.columns and df['is_collapsible'].any()

    def get_row_data(
            row: pd.Series, 
            is_child: bool = False
        ) -> List[Dict | str]:

        new_row = []
        is_subject = has_subject and bool(row[highlight_col])
        click_followup = row.get(followup_col)

        for col, val in row.items():
            # Skip the is_subject column from output
            if col in ignore_cols:
                continue

            if col == sparkline_col:
                val = {"sparkLineData": val}
            else:   
                if pd.isna(val):
                    val = 'N/A'

                if is_grouping and col == dim_member_col:
                    val = val.strip().replace("-", "")
                    if is_child: # hack to add better looking indentation 
                        four_space_indent = "    "
                        val = f"{four_space_indent}{four_space_indent}{val}"

                if col == dim_member_col and click_followup != "" and not is_subject:
                    val = {"value": val, "style": {"text-decoration": "underline"}}
            new_row.append(val)

        row_info = {"data": new_row}
        if click_followup:
            row_info["onClick"] = {"args": click_followup, "event": "askQuestion"}
        if is_subject:
            row_info["style"] = {'background-color': '#FFF0BE'}

        return row_info

    data = []

    # reset index, ordering already determined by skill
    df = df.reset_index(drop=True)
    index = 0

    while index < len(df):

        row = df.iloc[index]

        if ('is_collapsible' in row and row['is_collapsible'] 
            and 'parent_dim_member' in row and row['parent_dim_member'] is None):

            parent_row_data = get_row_data(row)
            children = []

            child_row = df.iloc[index + 1] if index + 1 < len(df) else None

            while child_row is not None and child_row['parent_dim_member'] is not None:
                children.append(get_row_data(child_row, is_child=True))
                index += 1
                child_row = df.iloc[index + 1] if index + 1 < len(df) else None

            parent_row_data["group"] = children

            data.append(parent_row_data)

        else:
            data.append(get_row_data(row))

        index += 1

    return data

def get_table_layout_vars_msa(
        tab_name: str, 
        df: pd.DataFrame, 
        share_metric_label: str,
        include_drivers: bool,
        metric_drivers_labels: Dict[str, str],
        subject_metric_drivers: Dict[str, List[str]],
        decomposition_metric_drivers: Dict[str, List[str]],
        ignore_cols=[], 
        highlight_col="is_subject", 
        followup_col="msg", 
        sparkline_col="sparkline"
    ):
    """
    Generates table layout variables from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary containing the table layout variables.
            - "data" (list): A list of lists representing the table data.
            - "col_defs" (list): A list of dictionaries representing the column definitions.
    """
    ignore_cols = ignore_cols or []
    # add the highlight and followup columns to the ignore list if they are provided
    if highlight_col:
        ignore_cols.append(highlight_col)
    if followup_col:
        ignore_cols.append(followup_col)

    table_vars = {}
    dim_member_col = f"Share by {tab_name}"
    data = get_data(tab_name, df, ignore_cols=ignore_cols, highlight_col=highlight_col, followup_col=followup_col, sparkline_col=sparkline_col)
    col_defs = []
    columns = [col for col in list(df.columns) if col not in ignore_cols]

    # create a reverse mapping of all list values to the key
    subject_metric_driver_metrics_reverse = {metric_drivers_labels[item]: k for k, v in subject_metric_drivers.items() for item in v}
    decomposition_metric_driver_metrics_reverse = {metric_drivers_labels[item]: k for k, v in decomposition_metric_drivers.items() for item in v}

    for col in columns:

        group = share_metric_label
        if col in subject_metric_driver_metrics_reverse:
            group = subject_metric_driver_metrics_reverse[col]
        elif col in decomposition_metric_driver_metrics_reverse:
            group = decomposition_metric_driver_metrics_reverse[col]

        if col == sparkline_col:
            col_defs.append({"name": sparkline_col, "sparkLineOptions": {"colors": ["blue"]}, "group": group})
        elif col == dim_member_col:
            col_defs.append({"name": col, "style": {"textAlign": "left", "white-space": "pre"}, "group": group})
        else:
            col_defs.append({"name": col, "group": group})

    table_vars["data"] = data
    table_vars["col_defs"] = col_defs
    return table_vars

def render_layout(
        tables: Dict[str, pd.DataFrame],
        title: str,
        subtitle: str,
        insights_dfs: List[pd.DataFrame],
        warnings: str,
        max_prompt: str,
        insight_prompt: str, 
        viz_layout: str,
        share_metric_label: str,
        include_drivers: bool,
        metric_drivers_labels: Dict[str, str],
        subject_metric_drivers: Dict[str, List[str]],
        decomposition_metric_drivers: Dict[str, List[str]]
    ):
    facts = []
    for i_df in insights_dfs:
        facts.append(i_df.to_dict(orient='records'))

    insight_template = jinja2.Template(insight_prompt).render(**{"facts": facts})
    max_response_prompt = jinja2.Template(max_prompt).render(**{"facts": facts})

    # adding insights
    ar_utils = ArUtils()
    insights = ar_utils.get_llm_response(insight_template)
    viz_list = []
    export_data = {}

    general_vars = {
        "headline": title if title else "Total",
        "sub_headline": subtitle or "Market Share Analysis",
        "hide_growth_warning": False if warnings else True,
        "exec_summary": insights if insights else "No Insights.",
        "warning": warnings,
        "hide_footer": True
    }

    viz_layout = json.loads(viz_layout)

    for name, table in tables.items():
        export_data[name] = table
        # dim_note = find_footnote(footnotes, table)
        # hide_footer = False if dim_note else True

        table_vars = get_table_layout_vars_msa(
            name, 
            table, 
            share_metric_label,
            include_drivers,
            metric_drivers_labels,
            subject_metric_drivers,
            decomposition_metric_drivers,
            ignore_cols=["parent_dim_member", "is_collapsible", "followup_nl"],
            highlight_col="is_subject",
            followup_col="followup_nl",
            sparkline_col="sparkline"
        )
        # table_vars["hide_footer"] = hide_footer
        rendered = wire_layout(viz_layout, {**general_vars, **table_vars})
        viz_list.append(SkillVisualization(title=name, layout=rendered))

    return viz_list, insights, max_response_prompt, export_data
