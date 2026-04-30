from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import jinja2
from ar_analytics import AdvanceTrend, TrendTemplateParameterSetup, ArUtils
from ar_analytics.defaults import trend_analysis_config, default_trend_chart_layout, default_table_layout, \
    get_table_layout_vars, default_ppt_trend_chart_layout, default_ppt_table_layout
from skill_framework import SkillVisualization, skill, SkillParameter, SkillInput, SkillOutput, \
    ParameterDisplayDescription
from skill_framework.layouts import wire_layout
from skill_framework.preview import preview_skill
from skill_framework.skills import ExportData

RUNNING_LOCALLY = False

logger = logging.getLogger(__name__)

# Metric type definitions for formatting
CURRENCY_MILLIONS_METRICS = [
    'gross_revenue', 'net_revenue', 'revenue', 'sales', 'cogs', 'cost_of_goods_sold',
    'gross_profit', 'brand_contribution_margin', 'operating_income', 'ebitda', 'ebit'
]
PERCENTAGE_METRICS = [
    'margin', 'growth', 'share', 'rate', 'percent', 'pct', 'ratio', 'yield'
]
PRICE_METRICS = [
    'price', 'asp', 'average_selling_price', 'unit_price', 'cost_per_unit'
]


def get_metric_format_type(metric_name):
    """Determine the format type for a metric based on its name."""
    if not metric_name:
        return 'number'

    metric_lower = metric_name.lower().replace(' ', '_')

    # Check for percentage metrics
    for pct_metric in PERCENTAGE_METRICS:
        if pct_metric in metric_lower:
            return 'percentage'

    # Check for price metrics
    for price_metric in PRICE_METRICS:
        if price_metric in metric_lower:
            return 'price'

    # Check for large currency metrics (display in millions)
    for currency_metric in CURRENCY_MILLIONS_METRICS:
        if currency_metric in metric_lower:
            return 'currency_millions'

    return 'number'


def apply_chart_formatting(charts):
    """Apply smart formatting to chart data based on metric type.

    - Currency metrics (revenue, profit, etc.): Scale to millions, format as $X.XM
    - Percentage metrics: Format as X.X%
    - Price metrics: Format as $X.XX
    - Other: Format with commas
    """
    for chart_name, vars_dict in charts.items():
        metric_name = vars_dict.get('absolute_metric_name', '') or chart_name
        format_type = get_metric_format_type(metric_name)

        logger.info(f"Formatting chart '{chart_name}': metric={metric_name}, format_type={format_type}")

        # Process each series type (absolute, growth, difference)
        for prefix in ['absolute_', 'growth_', 'difference_']:
            series_key = f'{prefix}series'
            y_axis_key = f'{prefix}y_axis'

            if series_key not in vars_dict:
                continue

            # Determine format based on prefix and metric type
            if prefix == 'growth_':
                # Growth is always percentage
                current_format = 'percentage'
            elif prefix == 'difference_':
                # Difference uses same format as absolute but could be negative
                current_format = format_type
            else:
                current_format = format_type

            # Scale data and set formats
            if current_format == 'currency_millions':
                # Scale series data to millions
                for series in vars_dict[series_key]:
                    if isinstance(series, dict) and 'data' in series:
                        series['data'] = [
                            round(val / 1_000_000, 2) if val is not None and isinstance(val, (int, float)) else val
                            for val in series['data']
                        ]
                        # Update tooltip format
                        series['tooltip'] = {'pointFormat': '<b>{series.name}</b>: ${point.y:,.2f}M<br/>'}

                # Update Y-axis format
                if y_axis_key in vars_dict:
                    y_axes = vars_dict[y_axis_key] if isinstance(vars_dict[y_axis_key], list) else [vars_dict[y_axis_key]]
                    for axis in y_axes:
                        if isinstance(axis, dict):
                            axis['labels'] = axis.get('labels', {})
                            axis['labels']['format'] = '${value:,.1f}M'

            elif current_format == 'percentage':
                # Format as percentage
                for series in vars_dict[series_key]:
                    if isinstance(series, dict) and 'data' in series:
                        # Scale to percentage if values are decimals (0.xx)
                        series['data'] = [
                            round(val * 100, 2) if val is not None and isinstance(val, (int, float)) and abs(val) < 1 else val
                            for val in series['data']
                        ]
                        series['tooltip'] = {'pointFormat': '<b>{series.name}</b>: {point.y:,.1f}%<br/>'}

                if y_axis_key in vars_dict:
                    y_axes = vars_dict[y_axis_key] if isinstance(vars_dict[y_axis_key], list) else [vars_dict[y_axis_key]]
                    for axis in y_axes:
                        if isinstance(axis, dict):
                            axis['labels'] = axis.get('labels', {})
                            axis['labels']['format'] = '{value:,.1f}%'

            elif current_format == 'price':
                # Format as currency (not scaled)
                for series in vars_dict[series_key]:
                    if isinstance(series, dict):
                        series['tooltip'] = {'pointFormat': '<b>{series.name}</b>: ${point.y:,.2f}<br/>'}

                if y_axis_key in vars_dict:
                    y_axes = vars_dict[y_axis_key] if isinstance(vars_dict[y_axis_key], list) else [vars_dict[y_axis_key]]
                    for axis in y_axes:
                        if isinstance(axis, dict):
                            axis['labels'] = axis.get('labels', {})
                            axis['labels']['format'] = '${value:,.2f}'

            else:
                # Default number format with commas
                for series in vars_dict[series_key]:
                    if isinstance(series, dict):
                        series['tooltip'] = {'pointFormat': '<b>{series.name}</b>: {point.y:,.0f}<br/>'}

                if y_axis_key in vars_dict:
                    y_axes = vars_dict[y_axis_key] if isinstance(vars_dict[y_axis_key], list) else [vars_dict[y_axis_key]]
                    for axis in y_axes:
                        if isinstance(axis, dict):
                            axis['labels'] = axis.get('labels', {})
                            axis['labels']['format'] = '{value:,.0f}'

    return charts

@skill(
    name=trend_analysis_config.name,
    llm_name=trend_analysis_config.llm_name,
    description=trend_analysis_config.description,
    capabilities=trend_analysis_config.capabilities,
    limitations=trend_analysis_config.limitations,
    example_questions=trend_analysis_config.example_questions,
    parameter_guidance=trend_analysis_config.parameter_guidance,
    parameters=[
        SkillParameter(
            name="periods",
            constrained_to="date_filter",
            is_multi=True,
            description="If provided by the user, list time periods in a format 'q2 2023', '2021', 'jan 2023', 'mat nov 2022', 'mat q1 2021', 'ytd q4 2022', 'ytd 2023', 'ytd', 'mat', '<no_period_provided>' or '<since_launch>'. Use knowledge about today's date to handle relative periods and open ended periods. If given a range, for example 'last 3 quarters, 'between q3 2022 to q4 2023' etc, enumerate the range into a list of valid dates. Don't include natural language words or phrases, only valid dates like 'q3 2023', '2022', 'mar 2020', 'ytd sep 2021', 'mat q4 2021', 'ytd q1 2022', 'ytd 2021', 'ytd', 'mat', '<no_period_provided>' or '<since_launch>' etc."
        ),
        SkillParameter(
            name="metrics",
            is_multi=True,
            constrained_to="metrics"
        ),
        SkillParameter(
            name="limit_n",
            description="limit the number of values by this number",
            default_value=10
        ),
        SkillParameter(
            name="breakouts",
            is_multi=True,
            constrained_to="dimensions",
            description="breakout dimension(s) for analysis."
        ),
        SkillParameter(
            name="time_granularity",
            is_multi=False,
            constrained_to="date_dimensions",
            description="time granularity provided by the user. only add if explicitly stated by user."
        ),
        SkillParameter(
            name="growth_type",
            constrained_to=None,
            constrained_values=["Y/Y", "P/P", "None"],
            description="Growth type either Y/Y, P/P, or None"
        ),
        SkillParameter(
            name="compare_metrics",
            is_multi=True,
            constrained_to=None,
            constrained_values=["forecast", "budget"],
            description="Compare actuals against forecast and/or budget scenarios. When specified, 'scenario' is automatically added as a breakout."
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            is_multi=True
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt being used for max response.",
            default_value=trend_analysis_config.max_prompt
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt being used for detailed insights.",
            default_value=trend_analysis_config.insight_prompt
        ),
        SkillParameter(
            name="table_viz_layout",
            parameter_type="visualization",
            description="Table Viz Layout",
            default_value=default_table_layout
        ),
        SkillParameter(
            name="chart_viz_layout",
            parameter_type="visualization",
            description="Chart Viz Layout",
            default_value=default_trend_chart_layout
        ),
        SkillParameter(
            name="chart_ppt_layout",
            parameter_type="visualization",
            description="chart slide Viz Layout",
            default_value=default_ppt_trend_chart_layout
        ),
        SkillParameter(
            name="table_ppt_export_viz_layout",
            parameter_type="visualization",
            description="table slide Viz Layout",
            default_value=default_ppt_table_layout
        )
    ]
)
def trend(parameters: SkillInput):
    print(f"Skill received following parameters: {parameters.arguments}")
    param_dict = {"periods": [], "metrics": None, "limit_n": 10, "breakouts": [], "growth_type": None, "other_filters": [], "time_granularity": None, "compare_metrics": []}

    # Update param_dict with values from parameters.arguments if they exist
    for key in param_dict:
        if hasattr(parameters.arguments, key) and getattr(parameters.arguments, key) is not None:
            param_dict[key] = getattr(parameters.arguments, key)

    # Filter out invalid compare_metrics values (null, None, empty strings)
    valid_compare_metrics = [
        v for v in (param_dict["compare_metrics"] or [])
        if v and str(v).lower() not in ('null', 'none', '')
    ]

    # If compare_metrics specified with valid values, auto-add 'scenario' to breakouts
    if valid_compare_metrics:
        if param_dict["breakouts"] is None:
            param_dict["breakouts"] = ["scenario"]
        elif "scenario" not in param_dict["breakouts"]:
            param_dict["breakouts"] = list(param_dict["breakouts"]) + ["scenario"]
    else:
        # REQUIREMENT: Force filter to 'actuals' scenario unless user specifies budget or forecast
        # Check if scenario filter already exists in other_filters
        existing_scenario_filter = False
        if param_dict["other_filters"]:
            for f in param_dict["other_filters"]:
                if isinstance(f, dict) and f.get('dim', '').lower() == 'scenario':
                    existing_scenario_filter = True
                    break

        # Add scenario = 'actuals' filter if not already present
        if not existing_scenario_filter:
            if param_dict["other_filters"] is None:
                param_dict["other_filters"] = []
            param_dict["other_filters"] = list(param_dict["other_filters"]) + [{'dim': 'scenario', 'op': '=', 'val': ['actuals']}]
            logger.info("Auto-added scenario='actuals' filter (no compare_metrics specified)")

    env = SimpleNamespace(**param_dict)
    TrendTemplateParameterSetup(env=env)
    env.trend = AdvanceTrend.from_env(env=env)
    df = env.trend.run_from_env()
    param_info = [ParameterDisplayDescription(key=k, value=v) for k, v in env.trend.paramater_display_infomation.items()]
    tables = [env.trend.display_dfs.get("Metrics Table")]

    insights_dfs = [env.trend.df_notes, env.trend.facts, env.trend.top_facts, env.trend.bottom_facts]

    charts = env.trend.get_dynamic_layout_chart_vars()

    # Apply smart formatting based on metric type
    charts = apply_chart_formatting(charts)

    viz, slides, insights, final_prompt = render_layout(charts,
                                                tables,
                                                env.trend.title,
                                                env.trend.subtitle,
                                                insights_dfs,
                                                env.trend.warning_message,
                                                parameters.arguments.max_prompt,
                                                parameters.arguments.insight_prompt,
                                                parameters.arguments.table_viz_layout,
                                                parameters.arguments.chart_viz_layout,
                                                parameters.arguments.chart_ppt_layout,
                                                parameters.arguments.table_ppt_export_viz_layout)

    display_charts = env.trend.display_charts

    return SkillOutput(
        final_prompt=final_prompt,
        narrative=None,
        visualizations=viz,
        ppt_slides=slides,
        parameter_display_descriptions=param_info,
        followup_questions=[],
        export_data=[ExportData(name="Metrics Table", data=tables[0]),
                     *[ExportData(name=chart, data=display_charts[chart].get("df")) for chart in display_charts.keys()]]
    )

def map_chart_variables(chart_vars, prefix):
    """
    Maps prefixed chart variables to generic variable names expected by the layout.

    Args:
        chart_vars: Dictionary containing all chart variables with prefixes
        prefix: The prefix to extract (e.g., 'absolute_', 'growth_', 'difference_')

    Returns:
        Dictionary with mapped variables using generic names
    """
    suffixes = ['series', 'x_axis_categories', 'y_axis', 'metric_name', 'meta_df_id']

    mapped_vars = {}

    for suffix in suffixes:
        prefixed_key = f"{prefix}{suffix}"
        if prefixed_key in chart_vars:
            mapped_vars[suffix] = chart_vars[prefixed_key]

    if 'footer' in chart_vars:
        mapped_vars['footer'] = chart_vars['footer']
    if 'hide_footer' in chart_vars:
        mapped_vars['hide_footer'] = chart_vars['hide_footer']

    return mapped_vars

def render_layout(charts, tables, title, subtitle, insights_dfs, warnings, max_prompt, insight_prompt, table_viz_layout, chart_viz_layout, chart_ppt_layout, table_ppt_export_viz_layout):
    facts = []
    for i_df in insights_dfs:
        facts.append(i_df.to_dict(orient='records'))

    insight_template = jinja2.Template(insight_prompt).render(**{"facts": facts})
    max_response_prompt = jinja2.Template(max_prompt).render(**{"facts": facts})

    # adding insights
    ar_utils = ArUtils()
    insights = ar_utils.get_llm_response(insight_template)

    tab_vars = {"headline": title if title else "Total",
                "sub_headline": subtitle or "Trend Analysis",
                "hide_growth_warning": False if warnings else True,
                "exec_summary": insights if insights else "No Insight.",
                "warning": warnings}

    viz = []
    slides = []
    for name, chart_vars in charts.items():
        chart_vars["footer"] = f"*{chart_vars['footer']}" if chart_vars.get('footer') else "No additional info."
        rendered = wire_layout(json.loads(chart_viz_layout), {**tab_vars, **chart_vars})
        viz.append(SkillVisualization(title=name, layout=rendered))

        prefixes = ["absolute_", "growth_", "difference_"]

        for prefix in prefixes:
            if (prefix in ["growth_", "difference_"] and
                chart_vars.get("hide_growth_chart", False)):
                continue

            try:
                mapped_vars = map_chart_variables(chart_vars, prefix)
                slide = wire_layout(json.loads(chart_ppt_layout), {**tab_vars, **mapped_vars})
                slides.append(slide)
            except Exception as e:
                logger.error(f"Error rendering chart ppt slide for prefix '{prefix}' in chart '{name}': {e}")

    table_vars = get_table_layout_vars(tables[0])
    table = wire_layout(json.loads(table_viz_layout), {**tab_vars, **table_vars})
    viz.append(SkillVisualization(title="Metrics Table", layout=table))

    if table_ppt_export_viz_layout is not None:
        try: 
            table_slide = wire_layout(json.loads(table_ppt_export_viz_layout), {**tab_vars, **table_vars})
            slides.append(table_slide)
        except Exception as e:
            logger.error(f"Error rendering table ppt slide: {e}")
    else:
        slides.append(table)

    return viz, slides, insights, max_response_prompt

if __name__ == '__main__':
    skill_input: SkillInput = trend.create_input(arguments={
        'metrics': ["sales", "volume"],
        'periods': ["2021", "2022"],
        'growth_type': "Y/Y",
        "other_filters": [{"dim": "brand", "op": "=", "val": ["barilla"]}]
    })
    out = trend(skill_input)
    preview_skill(trend, out)