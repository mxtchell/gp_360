from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import jinja2
from ar_analytics import BreakoutAnalysis, BreakoutAnalysisTemplateParameterSetup, ArUtils
from ar_analytics.defaults import dimension_breakout_config, default_table_layout, get_table_layout_vars, \
    default_bridge_chart_viz, default_ppt_table_layout
from ar_analytics.helpers.df_meta_util import apply_metadata_to_layout_element
from skill_framework import SkillInput, SkillVisualization, skill, SkillParameter, SkillOutput, SuggestedQuestion, \
    ParameterDisplayDescription
from skill_framework.layouts import wire_layout
from skill_framework.preview import preview_skill
from skill_framework.skills import ExportData

logger = logging.getLogger(__name__)

@skill(
    name=dimension_breakout_config.name,
    llm_name=dimension_breakout_config.llm_name,
    description=dimension_breakout_config.description,
    capabilities=dimension_breakout_config.capabilities,
    limitations=dimension_breakout_config.limitations,
    example_questions=dimension_breakout_config.example_questions,
    parameter_guidance=dimension_breakout_config.parameter_guidance,
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
            name="growth_type",
            constrained_to=None,
            constrained_values=["Y/Y", "P/P", "None"],
            description="Growth type either Y/Y, P/P, or None"
        ),
        SkillParameter(
            name="other_filters",
            is_multi=True,
            constrained_to="filters"
        ),
        SkillParameter(
            name="growth_trend",
            constrained_to=None,
            constrained_values=["fastest growing", "highest growing", "highest declining", "fastest declining",
                                "smallest overall", "biggest overall"],
            description="indicates the trend type (fastest, highest, overall size) within a specified growth metric (year over year, period over period) for entities being analyzed."
        ),
        SkillParameter(
            name="calculated_metric_filters",
            description='This parameter allows filtering based on computed values like growth, delta, or share. The computed values are only available for metrics selected for this analysis. The available computations are growth, delta and share. It accepts a list of conditions, where each condition is a dictionary with:  metric: The metric being filtered. computation: The computation (growth, delta, share) operator: The comparison operator (">", "<", ">=", "<=", "between", "=="). value: The numeric threshold for filtering. If using "between", provide a list [min, max]. scale: the scale of value (percentage, bps, absolute)'
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt being used for max response.",
            default_value=dimension_breakout_config.max_prompt
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt being used for detailed insights.",
            default_value=dimension_breakout_config.insight_prompt
        ),
        SkillParameter(
            name="table_viz_layout",
            parameter_type="visualization",
            description="Table Viz Layout",
            default_value=default_table_layout
        ),
        SkillParameter(
            name="bridge_chart_viz_layout",
            parameter_type="visualization",
            description="Bridge Chart Viz Layout",
            default_value=default_bridge_chart_viz
        ),
        SkillParameter(
            name="table_ppt_layout",
            parameter_type="visualization",
            description="Table PPT Layout",
            default_value=default_ppt_table_layout
        )
    ]
)
def simple_breakout(parameters: SkillInput):
    param_dict = {"periods": [], "metrics": None, "limit_n": 10, "breakouts": None, "growth_type": None, "other_filters": [], "growth_trend": None, "calculated_metric_filters": None}
    print(f"Skill received following parameters: {parameters.arguments}")
    # Update param_dict with values from parameters.arguments if they exist
    for key in param_dict:
        if hasattr(parameters.arguments, key) and getattr(parameters.arguments, key) is not None:
            param_dict[key] = getattr(parameters.arguments, key)

    env = SimpleNamespace(**param_dict)
    BreakoutAnalysisTemplateParameterSetup(env=env)
    env.ba = BreakoutAnalysis.from_env(env=env)
    df = env.ba.run_from_env()

    tables = env.ba.get_display_tables()
    param_info = [ParameterDisplayDescription(key=k, value=v) for k, v in env.ba.paramater_display_infomation.items()]

    insights_dfs = [env.ba.df_notes, env.ba.breakout_facts, env.ba.subject_facts]
    followups = env.ba.get_suggestions()

    viz, slides, insights, final_prompt, export_data = render_layout(tables,
                                                            env.ba.get_display_bridge_charts(),
                                                            env.ba.title,
                                                            env.ba.subtitle,
                                                            insights_dfs,
                                                            env.ba.warning_message,
                                                            env.ba.footnotes,
                                                            parameters.arguments.max_prompt,
                                                            parameters.arguments.insight_prompt,
                                                            parameters.arguments.table_viz_layout,
                                                            parameters.arguments.bridge_chart_viz_layout,
                                                            parameters.arguments.table_ppt_layout)

    return SkillOutput(
        final_prompt=final_prompt,
        narrative=None,
        visualizations=viz,
        ppt_slides=slides,
        parameter_display_descriptions=param_info,
        followup_questions=[SuggestedQuestion(label=f.get("label"), question=f.get("question")) for f in followups if f.get("label")],
        export_data=[*[ExportData(name=name, id=df.max_metadata.get_id(), data=df) for name, df in export_data.items()], *[ExportData(name="Breakout Table Raw", id=df.max_metadata.get_id(), data=df)]] 
    )

def find_footnote(footnotes, df):
    footnotes = footnotes or {}
    dim_note = None
    for col in df.columns:
        if col in footnotes:
            dim_note = footnotes.get(col)
            break
    return dim_note

def render_layout(tables, bridge_chart_data, title, subtitle, insights_dfs, warnings, footnotes, max_prompt, insight_prompt, viz_layout, bridge_chart_viz_layout, table_ppt_layout):
    facts = []
    for i_df in insights_dfs:
        facts.append(i_df.to_dict(orient='records'))

    insight_template = jinja2.Template(insight_prompt).render(**{"facts": facts})
    max_response_prompt = jinja2.Template(max_prompt).render(**{"facts": facts})

    # adding insights
    ar_utils = ArUtils()
    insights = ar_utils.get_llm_response(insight_template)
    viz_list = []
    slides = []
    export_data = {}

    general_vars = {"headline": title if title else "Total",
					"sub_headline": subtitle or "Breakout Analysis",
					"hide_growth_warning": False if warnings else True,
					"exec_summary": insights if insights else "No Insights.",
					"warning": warnings}

    viz_layout = json.loads(viz_layout)

    for name, table in tables.items():
        export_data[name] = table
        dim_note = find_footnote(footnotes, table)
        hide_footer = False if dim_note else True
        table_vars = get_table_layout_vars(table)
        table_vars["hide_footer"] = hide_footer
        table_vars["footer"] = f"*{dim_note.strip()}" if dim_note else "No additional info."
        meta_viz_layout = apply_metadata_to_layout_element(viz_layout, "DataTable0",
                                                           {"sourceDataframeId": table.max_metadata.get_id()})
        rendered = wire_layout(meta_viz_layout, {**general_vars, **table_vars})
        viz_list.append(SkillVisualization(title=name, layout=rendered))
        if table_ppt_layout is not None:
            slide = wire_layout(json.loads(table_ppt_layout), {**general_vars, **table_vars})
            slides.append(slide)
        else:
            slides.append(rendered)

    if bridge_chart_data is not None:
        for chart_name, chart_info in bridge_chart_data.items():
        
            bridge_viz_layout = json.loads(bridge_chart_viz_layout)
            meta_viz_layout = apply_metadata_to_layout_element(bridge_viz_layout, "HighchartsChart0",
                                                            {"sourceDataframeId": chart_info["sourceDataframeId"]})
            rendered = wire_layout(meta_viz_layout, {**general_vars, **table_vars, **chart_info})
            viz_list.append(SkillVisualization(title=chart_name, layout=rendered))

    return viz_list, slides, insights, max_response_prompt, export_data

if __name__ == '__main__':
    skill_input: SkillInput = simple_breakout.create_input(arguments={'metrics': ["sales", "volume", "sales_share"], 'breakouts': ["brand", "manufacturer"], 'periods': ["2022"], 'growth_type': "Y/Y", 'other_filters': []})
    out = simple_breakout(skill_input)
    preview_skill(simple_breakout, out)