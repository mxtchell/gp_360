from __future__ import annotations

import json
import logging
import pandas as pd
import numpy as np

import jinja2
from ar_analytics import ArUtils
from ar_analytics.helpers.utils import get_dataset_id
from answer_rocket import AnswerRocketClient
from skill_framework import (
    SkillVisualization, skill, SkillParameter, SkillInput, SkillOutput,
    ParameterDisplayDescription
)
from skill_framework.skills import ExportData
from skill_framework.layouts import wire_layout

logger = logging.getLogger(__name__)

# Default prompts
DEFAULT_INSIGHT_PROMPT = """
Analyze the following what-if scenario results for {{ facts[0].metric }}:

Scenario: {{ facts[0].scenario }}
Breakout by: {{ facts[0].breakout }}

{% for row in facts[0].results %}
- {{ row }}
{% endfor %}

Provide a brief analysis covering:
1. Overall {{ facts[0].metric }} impact magnitude and direction
2. Which segments are most/least affected and why
3. Key drivers of the changes
4. Business implications and recommended actions

Use a professional finance tone. Be concise (3-4 sentences).
"""

DEFAULT_MAX_PROMPT = DEFAULT_INSIGHT_PROMPT

# Import layout at the end to avoid circular dependency
WHATIF_LAYOUT = """{
    "layoutJson": {
        "type": "Document",
        "gap": "0px",
        "style": {
            "backgroundColor": "#ffffff",
            "width": "100%",
            "height": "max-content",
            "padding": "15px",
            "gap": "15px"
        },
        "children": [
            {
                "name": "FlexContainer_Header",
                "type": "FlexContainer",
                "children": "",
                "minHeight": "80px",
                "direction": "column",
                "style": {
                    "backgroundColor": "#3b82f6",
                    "padding": "20px",
                    "borderRadius": "8px",
                    "marginBottom": "20px"
                },
                "label": "FlexContainer-Header"
            },
            {
                "name": "Header_Title",
                "type": "Header",
                "children": "",
                "text": "COGS What-If Analysis",
                "style": {
                    "fontSize": "24px",
                    "fontWeight": "bold",
                    "color": "#ffffff",
                    "textAlign": "left",
                    "margin": "0"
                },
                "parentId": "FlexContainer_Header",
                "label": "Header-Main_Title"
            },
            {
                "name": "Header_Subtitle",
                "type": "Header",
                "children": "",
                "text": "Impact of Price Changes on COGS",
                "style": {
                    "fontSize": "16px",
                    "fontWeight": "normal",
                    "color": "#e5e7eb",
                    "textAlign": "left",
                    "marginTop": "5px"
                },
                "parentId": "FlexContainer_Header",
                "label": "Header-Subtitle"
            },
            {
                "name": "HighchartsChart0",
                "type": "HighchartsChart",
                "children": "",
                "minHeight": "400px",
                "options": {
                    "chart": {
                        "type": "column",
                        "backgroundColor": "#f8fafc"
                    },
                    "title": {
                        "text": "COGS Forecasted vs Estimated",
                        "style": {
                            "fontSize": "20px"
                        }
                    },
                    "xAxis": {
                        "categories": ["Snack Bars", "Biscuits", "Cakes and Pastries", "Chocolate"],
                        "title": {
                            "text": "Category"
                        }
                    },
                    "yAxis": {
                        "title": {
                            "text": "COGS"
                        },
                        "labels": {
                            "format": "${value:,.0f}"
                        }
                    },
                    "series": [
                        {
                            "name": "COGS Forecasted",
                            "data": [1640740, 4441940, 1634330, 3289790],
                            "color": "#5DADE2"
                        },
                        {
                            "name": "COGS Estimated",
                            "data": [1655090, 4470820, 1641080, 3330580],
                            "color": "#8E44AD"
                        }
                    ],
                    "credits": {
                        "enabled": false
                    },
                    "legend": {
                        "enabled": true,
                        "align": "center",
                        "verticalAlign": "bottom",
                        "layout": "horizontal"
                    },
                    "plotOptions": {
                        "column": {
                            "dataLabels": {
                                "enabled": false
                            }
                        }
                    },
                    "tooltip": {
                        "pointFormat": "<b>{series.name}</b>: ${point.y:,.0f}"
                    }
                },
                "label": "HighchartsChart-COGS",
                "extraStyles": "border-radius: 8px"
            },
            {
                "name": "DataTable0",
                "type": "DataTable",
                "children": "",
                "columns": [
                    {"name": "Category"},
                    {"name": "COGS Forecasted"},
                    {"name": "COGS Estimated"},
                    {"name": "Change"},
                    {"name": "Material Forecasted"},
                    {"name": "Material Estimated"},
                    {"name": "Material Change"},
                    {"name": "Cocoa Forecasted"},
                    {"name": "Cocoa Estimated"},
                    {"name": "Cocoa Change"}
                ],
                "data": [
                    ["Snack Bars", "$1.64M", "$1.66M", "0.88%", "$1.15M", "$1.16M", "1.25%", "$267.13M", "$301.49M", "5.0%"],
                    ["Biscuits", "$4.44M", "$4.47M", "0.65%", "$2.89M", "$2.92M", "1.0%", "$577.45M", "$606.33M", "5.0%"]
                ],
                "label": "DataTable-COGS"
            }
        ]
    },
    "inputVariables": [
        {
            "name": "chart_title",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.title.text"
                }
            ]
        },
        {
            "name": "chart_categories",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.xAxis.categories"
                }
            ]
        },
        {
            "name": "chart_data_series",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.series"
                }
            ]
        },
        {
            "name": "data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "DataTable0",
                    "fieldName": "data"
                }
            ]
        },
        {
            "name": "col_defs",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "DataTable0",
                    "fieldName": "columns"
                }
            ]
        }
    ]
}"""

# Component name mappings for different metrics
PRICE_COL_MAPPING = {
    # COGS components
    "material": "Material",
    "labor": "Labor",
    "overheads": "Overheads",
    "logistics": "Logistics",
    "sugar": "% of Sugar",
    "cocoa": "% of Cocoa",
    "wheat": "% of Wheat",
    "other_materials": "% Others",
    # Marketing components
    "digital": "Digital",
    "traditional": "Traditional",
    "trade": "Trade",
    "brand": "Brand"
}

# Metric display name mapping for user-friendly error messages
METRIC_DISPLAY_NAMES = {
    "cogs": "COGS",
    "marketing_spend": "Marketing Spend",
    "marketing_expense": "Marketing Expense",
    "marketing": "Marketing",
    "revenue": "Revenue",
    "gross_profit": "Gross Profit",
    "net_sales": "Net Sales",
}

def _format_metric_name(metric: str) -> str:
    """Format raw metric name to user-friendly display name."""
    metric_lower = metric.lower()
    if metric_lower in METRIC_DISPLAY_NAMES:
        return METRIC_DISPLAY_NAMES[metric_lower]
    # Fallback: title case with underscores replaced by spaces
    return metric.replace('_', ' ').title()


@skill(
    name="FP&A What-If Analysis",
    llm_name="What-If Scenario Analysis",
    description="Analyze the impact of cost or spend changes on key financial metrics like COGS, Marketing Spend, or other configurable metrics. Model future projection scenarios by applying percentage changes to cost components and see forecasted vs estimated impacts by dimension.",
    capabilities="Financial what-if scenario analysis: Model impact of cost changes on metrics like COGS (material, labor, overheads, logistics, commodities) or Marketing Spend (digital, traditional, trade, brand). Supports future projection scenarios - forecast how metric changes would impact financials. Shows forecasted vs estimated values with detailed breakdown.",
    limitations="Requires a breakout dimension for analysis. Requires at least one component change in price_change_scenario.",
    example_questions="What would be the impact of a 5% increase in cocoa price on COGS? How would a 10% increase in digital marketing spend affect total marketing by region? What if we increase trade marketing by 15% next quarter? Model a scenario where labor costs rise 8% - what's the COGS impact by category?",
    parameter_guidance="Specify the metric to analyze (cogs, marketing_spend, etc.). Provide cost component changes in price_change_scenario as JSON like {'cocoa': 0.05} for 5% increase, or {'digital': 0.10, 'trade': 0.15} for multiple changes. Values are decimal percentages (0.05 = 5%). Use for future projections by specifying expected cost increases. For 'next quarter' use Q1 2026.",
    parameters=[
        SkillParameter(
            name="metric",
            is_multi=False,
            constrained_to="metrics",
            description="The metric to analyze (e.g., 'cogs', 'marketing_spend'). Determines which cost components are available.",
            default_value="cogs"
        ),
        SkillParameter(
            name="periods",
            constrained_to="date_filter",
            is_multi=True,
            description="Time period for analysis or future projection. Use 'Q1 2026' for 'next quarter'. Examples: 'Q3 2024', 'Q1 2026'."
        ),
        SkillParameter(
            name="breakout",
            is_multi=False,
            constrained_to="dimensions",
            description="Breakout dimension for analysis (e.g., 'category', 'region')",
            default_value="category"
        ),
        SkillParameter(
            name="price_change_scenario",
            description="JSON object with component changes as decimal percentages. For COGS: 'cocoa', 'sugar', 'wheat', 'other_materials', 'material', 'labor', 'overheads', 'logistics'. For Marketing: 'digital', 'traditional', 'trade', 'brand'. Example: {'cocoa': 0.05} or {'digital': 0.15, 'trade': 0.10}."
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            is_multi=True,
            description="Additional filters (region, category, etc.)"
        ),
        SkillParameter(
            name="whatif_layout",
            parameter_type="visualization",
            description="Layout for COGS What-If Analysis",
            default_value=WHATIF_LAYOUT
        ),
        SkillParameter(
            name="table_name",
            parameter_type="code",
            description="Table/view name for COGS data query",
            default_value=""
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt being used for max response.",
            default_value=DEFAULT_MAX_PROMPT
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt being used for detailed insights.",
            default_value=DEFAULT_INSIGHT_PROMPT
        )
    ]
)
def whatif_analysis(parameters: SkillInput):
    print(f"Skill received following parameters: {parameters.arguments}")

    # Parse parameters
    metric = parameters.arguments.metric if hasattr(parameters.arguments, 'metric') and parameters.arguments.metric else 'cogs'
    periods = parameters.arguments.periods if hasattr(parameters.arguments, 'periods') else []
    breakout = parameters.arguments.breakout if hasattr(parameters.arguments, 'breakout') else 'category'
    other_filters = parameters.arguments.other_filters if hasattr(parameters.arguments, 'other_filters') else []
    whatif_layout = parameters.arguments.whatif_layout if hasattr(parameters.arguments, 'whatif_layout') else WHATIF_LAYOUT
    table_name = parameters.arguments.table_name if hasattr(parameters.arguments, 'table_name') and parameters.arguments.table_name else None

    # Parse price change scenario
    price_scenario = {}
    if hasattr(parameters.arguments, 'price_change_scenario') and parameters.arguments.price_change_scenario:
        try:
            if isinstance(parameters.arguments.price_change_scenario, dict):
                price_scenario = parameters.arguments.price_change_scenario
            else:
                price_scenario = json.loads(parameters.arguments.price_change_scenario)
            # Map to display names
            price_scenario = {PRICE_COL_MAPPING.get(k, k): float(v) for k, v in price_scenario.items()}
        except Exception as e:
            logger.error(f"Error parsing price scenario: {e}")
            return SkillOutput(
                final_prompt="Error parsing price_change_scenario parameter. Must be valid JSON.",
                narrative="Error: Invalid price_change_scenario format. Use format like {'cocoa': 0.05} for 5% increase.",
                visualizations=[],
                parameter_display_descriptions=[]
            )

    if not price_scenario:
        return SkillOutput(
            final_prompt="No price changes specified.",
            narrative="Error: You must specify at least one cost change in price_change_scenario parameter.",
            visualizations=[],
            parameter_display_descriptions=[]
        )

    # Get AnswerRocketClient
    try:
        client = AnswerRocketClient()
    except Exception as e:
        logger.error(f"Failed to initialize AnswerRocketClient: {e}")
        return SkillOutput(
            final_prompt=f"Failed to initialize client: {str(e)}",
            narrative=f"Error: {str(e)}",
            visualizations=[],
            parameter_display_descriptions=[]
        )

    # Create analysis engine
    analyzer = WhatIfAnalysisEngine(
        client=client,
        metric=metric,
        periods=periods,
        breakout=breakout,
        filters=other_filters,
        price_scenario=price_scenario,
        table_name=table_name
    )

    # Run analysis
    try:
        results_df = analyzer.run()
    except Exception as e:
        logger.error(f"Error running what-if analysis: {e}", exc_info=True)
        return SkillOutput(
            final_prompt=f"Error running analysis: {str(e)}",
            narrative=f"Error: {str(e)}",
            visualizations=[],
            parameter_display_descriptions=[]
        )

    # Create visualization data
    chart_data = analyzer.create_chart_data(results_df)
    table_data = analyzer.create_table_data(results_df)

    # Generate insights using LLM
    ar_utils = ArUtils()

    # Format metric name for display
    metric_display = metric.upper().replace('_', ' ')

    # Build facts for prompt template
    facts = [{
        'metric': metric_display,
        'scenario': ', '.join([f'{k}: {v:+.1%}' for k, v in price_scenario.items()]),
        'breakout': breakout,
        'results': results_df.to_dict(orient='records')
    }]

    # Use prompts from platform
    insight_prompt_rendered = jinja2.Template(parameters.arguments.insight_prompt).render(facts=facts)
    max_response_prompt = jinja2.Template(parameters.arguments.max_prompt).render(facts=facts)

    insights = ar_utils.get_llm_response(insight_prompt_rendered)

    # Prepare layout variables
    layout_vars = {
        "chart_title": f"{metric_display}: Forecasted vs Estimated",
        "chart_categories": chart_data['categories'],
        "chart_data_series": chart_data['series'],
        "data": table_data['data'],
        "col_defs": table_data['columns']
    }

    # Wire the layout
    rendered = wire_layout(json.loads(whatif_layout), layout_vars)

    # Create parameter display descriptions
    param_info = [
        ParameterDisplayDescription(key="", value=f"Breakout: {breakout}"),
        ParameterDisplayDescription(key="", value=f"Period: {', '.join(periods) if periods else 'Not specified'}")
    ]

    # Add filters to parameter display
    for f in other_filters:
        dim = f.get('dim') or f.get('col') or f.get('attribute', '')
        val = f.get('val') or f.get('values', '')
        if isinstance(val, list):
            val = ', '.join(val)
        if dim and val:
            # Capitalize dimension name for display
            dim_label = dim.replace('_', ' ').title()
            param_info.append(ParameterDisplayDescription(key="", value=f"{dim_label}: {val}"))

    for k, v in price_scenario.items():
        formatted_val = f"{v:+.1%}"
        param_info.append(ParameterDisplayDescription(key="", value=f"{k}: {formatted_val}"))

    return SkillOutput(
        final_prompt=max_response_prompt,
        narrative=insights,
        visualizations=[SkillVisualization(title=f"{metric_display} What-If Analysis", layout=rendered)],
        parameter_display_descriptions=param_info,
        followup_questions=[],
        export_data=[
            ExportData(name=f"{metric_display} What-If Analysis", data=results_df)
        ]
    )


class WhatIfAnalysisEngine:
    """Engine for running what-if scenario analysis on financial metrics"""

    def __init__(self, client, metric, periods, breakout, filters, price_scenario, table_name=None):
        self.client = client
        self.metric = metric.lower()
        self.periods = periods
        self.breakout = breakout
        self.filters = filters
        self.price_scenario = price_scenario
        self.table_name = table_name

        self.forecasted_col = "Forecasted"
        self.estimated_col = "Estimated"
        self.change_col = "Change"

        # Get database_id and dataset_id from platform context (inherited from copilot)
        self.dataset_id = get_dataset_id()
        dataset = self.client.data.get_dataset(dataset_id=self.dataset_id)
        self.database_id = dataset.database.database_id

        # Get table name from dataset's fact entity if not provided
        if not self.table_name:
            domain_entity = next((x for x in dataset.domain_objects if x.type == "factEntity"), None)
            if domain_entity and hasattr(domain_entity, 'db_table'):
                self.table_name = domain_entity.db_table
            else:
                self.table_name = getattr(dataset, 'name', None) or 'data'

    def run(self):
        """Run the what-if analysis and return results DataFrame"""

        # Pull base metric data from database
        base_df = self._pull_metric_data()

        # Calculate metric breakdown by dimension
        forecasted_df = self._calculate_breakouts(base_df)

        # Recalculate metric with price changes
        estimated_df = self._recalculate_metric(forecasted_df, self.price_scenario)

        # Merge and calculate changes
        results_df = self._merge_and_calculate_changes(forecasted_df, estimated_df)

        return results_df

    def _pull_metric_data(self):
        """Pull metric data from database using SQL query"""

        # Build filter clause
        filter_clauses = []
        for f in self.filters:
            dim = f.get('dim') or f.get('col')
            op = f.get('op', '=')
            val = f.get('val')

            if dim and val:
                if isinstance(val, list):
                    if len(val) == 1:
                        filter_clauses.append(f"UPPER({dim}) {op} UPPER('{val[0]}')")
                    else:
                        val_str = ", ".join([f"UPPER('{v}')" for v in val])
                        filter_clauses.append(f"UPPER({dim}) IN ({val_str})")
                elif isinstance(val, str):
                    filter_clauses.append(f"UPPER({dim}) {op} UPPER('{val}')")
                else:
                    filter_clauses.append(f"{dim} {op} {val}")

        filter_clause = " AND " + " AND ".join(filter_clauses) if filter_clauses else ""

        # Parse period to date range (using same logic as metric_drivers)
        if self.periods and len(self.periods) > 0:
            period_str = self.periods[0]
            start_date, end_date = self._parse_period_to_date_range(period_str)
            logger.info(f"Parsed period '{period_str}' to date range: {start_date} to {end_date}")
        else:
            raise ValueError("Period is required but was not provided")

        # Query metric by breakout dimension
        query = f"""
        SELECT {self.breakout}, SUM({self.metric}) as {self.metric}
        FROM {self.table_name}
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        {filter_clause}
        GROUP BY {self.breakout}
        """

        logger.info(f"{self.metric.upper()} query: {query}")
        result = self.client.data.execute_sql_query(
            database_id=self.database_id,
            sql_query=query,
            row_limit=10000
        )

        df = result.df if hasattr(result, 'df') else None
        if df is None or df.empty:
            friendly_metric = _format_metric_name(self.metric)
            raise ValueError(
                f"No {friendly_metric} data available for {self.periods[0]}. "
                f"Please try a different time period or check your filter selections."
            )

        return df

    def _parse_period_to_date_range(self, period_str):
        """Convert period string to date range for SQL query"""
        from dateutil.parser import parse

        if not period_str:
            raise ValueError("Period is required but was not provided")

        period_lower = period_str.lower().strip()

        # Handle quarters (Q1 2024, Q2 2025, etc.)
        if period_lower.startswith('q'):
            parts = period_str.split()
            quarter = int(parts[0][1])  # Extract quarter number
            year = int(parts[1])

            quarter_map = {
                1: ('01-01', '03-31'),
                2: ('04-01', '06-30'),
                3: ('07-01', '09-30'),
                4: ('10-01', '12-31')
            }
            start_month_day, end_month_day = quarter_map[quarter]
            return f"{year}-{start_month_day}", f"{year}-{end_month_day}"

        # Handle single months (January 2025, Jan 2025, 2025-01, etc.)
        try:
            parsed_date = parse(period_str, fuzzy=True)
            year = parsed_date.year
            month = parsed_date.month

            # Get last day of month
            if month == 12:
                last_day = 31
            elif month in [4, 6, 9, 11]:
                last_day = 30
            elif month == 2:
                # Check for leap year
                if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                    last_day = 29
                else:
                    last_day = 28
            else:
                last_day = 31

            return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day}"
        except:
            # If can't parse, return as-is
            return period_str, period_str

    def _get_metric_config(self):
        """Get breakdown configuration based on metric type"""
        if self.metric == 'cogs':
            return {
                'components': {
                    "Material": 0.60,
                    "Labor": 0.22,
                    "Overheads": 0.08,
                    "Logistics": 0.10
                },
                'sub_components': {
                    "% of Sugar": 0.20,
                    "% of Cocoa": 0.25,
                    "% of Wheat": 0.20,
                    "% Others": 0.35
                },
                'sub_component_parent': 'Material'
            }
        elif self.metric in ('marketing_spend', 'marketing'):
            return {
                'components': {
                    "Digital": 0.35,
                    "Traditional": 0.25,
                    "Trade": 0.25,
                    "Brand": 0.15
                },
                'sub_components': {},
                'sub_component_parent': None
            }
        else:
            # Generic fallback - apply changes directly to metric
            return {
                'components': {},
                'sub_components': {},
                'sub_component_parent': None
            }

    def _calculate_breakouts(self, df):
        """Calculate metric breakdown by cost components"""
        breakout_df = df.copy()
        config = self._get_metric_config()

        # Calculate each cost component as percentage of metric
        for component, pct in config['components'].items():
            breakout_df[component] = breakout_df[self.metric] * pct

        # Calculate sub-components if applicable
        if config['sub_component_parent'] and config['sub_components']:
            parent_col = config['sub_component_parent']
            for sub_comp, pct in config['sub_components'].items():
                breakout_df[sub_comp] = breakout_df[parent_col] * pct

        return breakout_df

    def _recalculate_metric(self, df, price_changes):
        """Recalculate metric with price changes applied"""
        estimated_df = df.copy()
        config = self._get_metric_config()

        component_cols = list(config['components'].keys())
        sub_component_cols = list(config['sub_components'].keys())

        # Apply sub-component price changes first
        for sub_comp in sub_component_cols:
            if sub_comp in price_changes:
                estimated_df[sub_comp] = estimated_df[sub_comp] * (1 + price_changes[sub_comp])

        # Recalculate parent component as sum of sub-components if applicable
        if config['sub_component_parent'] and sub_component_cols:
            estimated_df[config['sub_component_parent']] = estimated_df[sub_component_cols].sum(axis=1)

        # Apply component price changes
        for component in component_cols:
            if component in price_changes and component != config['sub_component_parent']:
                estimated_df[component] = estimated_df[component] * (1 + price_changes[component])

        # Recalculate total metric
        if component_cols:
            estimated_df[self.metric] = estimated_df[component_cols].sum(axis=1)
        else:
            # Direct percentage change on metric if no components
            for key, pct in price_changes.items():
                estimated_df[self.metric] = estimated_df[self.metric] * (1 + pct)
                break  # Only apply first change for simple metrics

        return estimated_df

    def _merge_and_calculate_changes(self, forecasted_df, estimated_df):
        """Merge forecasted and estimated, calculate changes"""
        config = self._get_metric_config()
        metric_upper = self.metric.upper()

        result_data = []

        for idx, row in forecasted_df.iterrows():
            category = row[self.breakout]
            est_row = estimated_df.iloc[idx]

            row_data = {self.breakout: category}

            # Add metric columns
            row_data[f"{metric_upper}_Forecasted"] = row[self.metric]
            row_data[f"{metric_upper}_Estimated"] = est_row[self.metric]
            row_data[f"{metric_upper}_Change"] = (est_row[self.metric] - row[self.metric]) / row[self.metric] if row[self.metric] != 0 else 0

            # Add component columns
            for component in config['components'].keys():
                if component in row:
                    row_data[f"{component}_Forecasted"] = row[component]
                    row_data[f"{component}_Estimated"] = est_row[component]
                    row_data[f"{component}_Change"] = (est_row[component] - row[component]) / row[component] if row[component] != 0 else 0

            # Add sub-component columns
            for sub_comp in config['sub_components'].keys():
                if sub_comp in row:
                    col_name = sub_comp.replace("% of ", "").replace(" ", "_")
                    row_data[f"{col_name}_Forecasted"] = row[sub_comp]
                    row_data[f"{col_name}_Estimated"] = est_row[sub_comp]
                    row_data[f"{col_name}_Change"] = (est_row[sub_comp] - row[sub_comp]) / row[sub_comp] if row[sub_comp] != 0 else 0

            result_data.append(row_data)

        return pd.DataFrame(result_data)

    def create_chart_data(self, df):
        """Create Highcharts column chart data from results DataFrame"""
        metric_upper = self.metric.upper()

        categories = df[self.breakout].tolist()
        forecasted_data = df[f"{metric_upper}_Forecasted"].tolist()
        estimated_data = df[f"{metric_upper}_Estimated"].tolist()

        return {
            "categories": categories,
            "series": [
                {"name": f"{metric_upper} Forecasted", "data": forecasted_data, "color": "#5DADE2"},
                {"name": f"{metric_upper} Estimated", "data": estimated_data, "color": "#8E44AD"}
            ]
        }

    def create_table_data(self, df):
        """Create DataTable data from results DataFrame"""
        config = self._get_metric_config()
        metric_upper = self.metric.upper()

        # Build dynamic columns
        columns = [{"name": self.breakout.title()}]
        columns.extend([
            {"name": f"{metric_upper} Forecasted"},
            {"name": f"{metric_upper} Estimated"},
            {"name": "Change"}
        ])

        # Add first component columns if available
        component_keys = list(config['components'].keys())
        if component_keys:
            first_comp = component_keys[0]
            columns.extend([
                {"name": f"{first_comp} Forecasted"},
                {"name": f"{first_comp} Estimated"},
                {"name": f"{first_comp} Change"}
            ])

        data = []
        for _, row in df.iterrows():
            row_data = [
                row[self.breakout],
                f"${row[f'{metric_upper}_Forecasted']/1000000:.2f}M",
                f"${row[f'{metric_upper}_Estimated']/1000000:.2f}M",
                f"{row[f'{metric_upper}_Change']:.2%}"
            ]

            # Add first component data if available
            if component_keys:
                first_comp = component_keys[0]
                row_data.extend([
                    f"${row[f'{first_comp}_Forecasted']/1000000:.2f}M",
                    f"${row[f'{first_comp}_Estimated']/1000000:.2f}M",
                    f"{row[f'{first_comp}_Change']:.2%}"
                ])

            data.append(row_data)

        return {"columns": columns, "data": data}


if __name__ == '__main__':
    skill_input: SkillInput = whatif_analysis.create_input(arguments={
        'periods': ['Q3 2024'],
        'breakout': 'category',
        'price_change_scenario': {'cocoa': 0.05}
    })
    out = whatif_analysis(skill_input)
    print(out.narrative)
