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
Analyze the following COGS what-if scenario results:

Scenario: {{ facts[0].scenario }}
Breakout by: {{ facts[0].breakout }}

{% for row in facts[0].results %}
- {{ row }}
{% endfor %}

Provide a brief analysis covering:
1. Overall COGS impact magnitude and direction across categories
2. Which categories are most/least affected and why
3. Which cost components drive the largest changes
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

# Hardcoded constants for COGS breakdown columns
PRICE_COL_MAPPING = {
    "material": "Material",
    "labor": "Labor",
    "overheads": "Overheads",
    "logistics": "Logistics",
    "sugar": "% of Sugar",
    "cocoa": "% of Cocoa",
    "wheat": "% of Wheat",
    "other_materials": "% Others"
}


@skill(
    name="FP&A What-If Analysis",
    llm_name="COGS What-If Scenario Analysis",
    description="Analyze the impact of commodity cost changes on COGS (Cost of Goods Sold). Shows how changes in material costs (cocoa, sugar, wheat, other materials) or operational costs (labor, overheads, logistics) affect total COGS by category.",
    capabilities="COGS scenario analysis: Analyze impact of commodity price changes (cocoa, sugar, wheat, other materials) or operational cost changes (labor, overheads, logistics) on cost of goods sold by category. Shows forecasted vs estimated values with detailed breakdown by material and cost components.",
    limitations="Requires category as breakout dimension. Supports only COGS metric. Requires at least one cost component change in price_change_scenario.",
    example_questions="What would be the impact of a 5% increase in cocoa price on COGS? How would a 10% increase in labor costs affect COGS by category? What if sugar and wheat prices both increase by 3%?",
    parameter_guidance="IMPORTANT: Always use 'category' as breakout dimension for COGS analysis. Provide commodity or operational cost changes in price_change_scenario as JSON like {'cocoa': 0.05} for 5% increase in cocoa, or {'labor': 0.10, 'sugar': 0.03} for multiple changes. Values should be decimal percentages (0.05 = 5%).",
    parameters=[
        SkillParameter(
            name="periods",
            constrained_to="date_filter",
            is_multi=True,
            description="Time period for analysis (e.g., 'Q3 2024', 'Jul 2024 to Sep 2024')"
        ),
        SkillParameter(
            name="breakout",
            is_multi=False,
            constrained_to="dimensions",
            description="Breakout dimension - must be 'category' for COGS analysis",
            default_value="category"
        ),
        SkillParameter(
            name="price_change_scenario",
            description="JSON object with cost changes as decimal percentages. Keys: 'cocoa', 'sugar', 'wheat', 'other_materials' for commodities; 'material', 'labor', 'overheads', 'logistics' for cost components. Example: {'cocoa': 0.05, 'sugar': 0.03} for 5% cocoa and 3% sugar increase."
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            is_multi=True,
            description="Additional filters (region, brand, etc.)"
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
    periods = parameters.arguments.periods if hasattr(parameters.arguments, 'periods') else []
    breakout = parameters.arguments.breakout if hasattr(parameters.arguments, 'breakout') else 'category'
    other_filters = parameters.arguments.other_filters if hasattr(parameters.arguments, 'other_filters') else []
    whatif_layout = parameters.arguments.whatif_layout if hasattr(parameters.arguments, 'whatif_layout') else WHATIF_LAYOUT
    table_name = parameters.arguments.table_name if hasattr(parameters.arguments, 'table_name') and parameters.arguments.table_name else None

    # Force category as breakout for COGS
    if breakout.lower() != 'category':
        breakout = 'category'

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

    # Build facts for prompt template
    facts = [{
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
        "chart_title": "COGS: Forecasted vs Estimated",
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
        visualizations=[SkillVisualization(title="COGS What-If Analysis", layout=rendered)],
        parameter_display_descriptions=param_info,
        followup_questions=[],
        export_data=[
            ExportData(name="COGS What-If Analysis", data=results_df)
        ]
    )


class WhatIfAnalysisEngine:
    """Engine for running COGS what-if scenario analysis"""

    def __init__(self, client, periods, breakout, filters, price_scenario, table_name=None):
        self.client = client
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
        """Run the COGS what-if analysis and return results DataFrame"""

        # Pull base COGS data from database
        base_df = self._pull_cogs_data()

        # Calculate COGS breakdown by category
        forecasted_df = self._calculate_category_breakouts_from_cogs(base_df)

        # Recalculate COGS with price changes
        estimated_df = self._recalculate_cogs(forecasted_df, self.price_scenario)

        # Merge and calculate changes
        results_df = self._merge_and_calculate_changes(forecasted_df, estimated_df)

        return results_df

    def _pull_cogs_data(self):
        """Pull COGS data from database using SQL query"""

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

        # Query COGS by category
        query = f"""
        SELECT {self.breakout}, SUM(cogs) as cogs
        FROM {self.table_name}
        WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
        {filter_clause}
        GROUP BY {self.breakout}
        """

        logger.info(f"COGS query: {query}")
        result = self.client.data.execute_sql_query(
            database_id=self.database_id,
            sql_query=query,
            row_limit=10000
        )

        df = result.df if hasattr(result, 'df') else None
        if df is None or df.empty:
            raise ValueError(f"No COGS data found for period {self.periods[0]}")

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

    def _get_cogs_breakdown(self):
        """Get COGS breakdown percentages by category"""
        return {
            "Material": {"Biscuits": 0.65, "Chocolate": 0.62, "Snack Bars": 0.70, "Cakes and Pastries": 0.55},
            "Labor": {"Biscuits": 0.20, "Chocolate": 0.22, "Snack Bars": 0.18, "Cakes and Pastries": 0.25},
            "Overheads": {"Biscuits": 0.08, "Chocolate": 0.06, "Snack Bars": 0.05, "Cakes and Pastries": 0.09},
            "Logistics": {"Biscuits": 0.07, "Chocolate": 0.10, "Snack Bars": 0.07, "Cakes and Pastries": 0.11}
        }

    def _get_cogs_commodity_breakdown(self):
        """Get commodity breakdown percentages within materials by category"""
        return {
            "% of Sugar": {"Biscuits": 0.15, "Chocolate": 0.25, "Snack Bars": 0.10, "Cakes and Pastries": 0.30},
            "% of Cocoa": {"Biscuits": 0.20, "Chocolate": 0.40, "Snack Bars": 0.25, "Cakes and Pastries": 0.15},
            "% of Wheat": {"Biscuits": 0.30, "Chocolate": 0.00, "Snack Bars": 0.05, "Cakes and Pastries": 0.10},
            "% Others": {"Biscuits": 0.35, "Chocolate": 0.35, "Snack Bars": 0.60, "Cakes and Pastries": 0.45}
        }

    def _calculate_category_breakouts_from_cogs(self, df):
        """Calculate COGS breakdown by cost components for each category"""

        breakout_df = df.copy()

        # Get breakdown percentages
        cogs_breakdown = self._get_cogs_breakdown()
        commodity_breakdown = self._get_cogs_commodity_breakdown()

        # Calculate each cost component
        for component, category_pcts in cogs_breakdown.items():
            breakout_df[component] = breakout_df.apply(
                lambda row: row['cogs'] * category_pcts.get(row[self.breakout], 0),
                axis=1
            )

        # Calculate commodity breakdown within materials
        for commodity, category_pcts in commodity_breakdown.items():
            breakout_df[commodity] = breakout_df.apply(
                lambda row: row['Material'] * category_pcts.get(row[self.breakout], 0),
                axis=1
            )

        return breakout_df

    def _recalculate_cogs(self, df, price_changes):
        """Recalculate COGS with price changes applied"""

        estimated_df = df.copy()

        # Get commodity columns
        commodity_cols = ["% of Sugar", "% of Cocoa", "% of Wheat", "% Others"]
        cogs_component_cols = ["Material", "Labor", "Overheads", "Logistics"]

        # Apply commodity price changes first
        for commodity in commodity_cols:
            if commodity in price_changes:
                estimated_df[commodity] = estimated_df[commodity] * (1 + price_changes[commodity])

        # Recalculate Material as sum of commodities
        estimated_df["Material"] = estimated_df[commodity_cols].sum(axis=1)

        # Apply cost component price changes
        for component in cogs_component_cols:
            if component in price_changes and component != "Material":
                estimated_df[component] = estimated_df[component] * (1 + price_changes[component])

        # Recalculate total COGS
        estimated_df["cogs"] = estimated_df[cogs_component_cols].sum(axis=1)

        return estimated_df

    def _merge_and_calculate_changes(self, forecasted_df, estimated_df):
        """Merge forecasted and estimated, calculate changes"""

        # Create multi-index columns structure
        result_data = []

        for idx, row in forecasted_df.iterrows():
            category = row[self.breakout]
            est_row = estimated_df.iloc[idx]

            row_data = {self.breakout: category}

            # Add COGS columns
            row_data["COGS_Forecasted"] = row["cogs"]
            row_data["COGS_Estimated"] = est_row["cogs"]
            row_data["COGS_Change"] = (est_row["cogs"] - row["cogs"]) / row["cogs"] if row["cogs"] != 0 else 0

            # Add Material columns
            row_data["Material_Forecasted"] = row["Material"]
            row_data["Material_Estimated"] = est_row["Material"]
            row_data["Material_Change"] = (est_row["Material"] - row["Material"]) / row["Material"] if row["Material"] != 0 else 0

            # Add commodity columns
            for commodity in ["% of Sugar", "% of Cocoa", "% of Wheat", "% Others"]:
                col_name = commodity.replace("% of ", "").replace(" ", "_")
                row_data[f"{col_name}_Forecasted"] = row[commodity]
                row_data[f"{col_name}_Estimated"] = est_row[commodity]
                row_data[f"{col_name}_Change"] = (est_row[commodity] - row[commodity]) / row[commodity] if row[commodity] != 0 else 0

            result_data.append(row_data)

        return pd.DataFrame(result_data)

    def create_chart_data(self, df):
        """Create Highcharts column chart data from results DataFrame"""

        categories = df[self.breakout].tolist()
        forecasted_data = df["COGS_Forecasted"].tolist()
        estimated_data = df["COGS_Estimated"].tolist()

        return {
            "categories": categories,
            "series": [
                {"name": "COGS Forecasted", "data": forecasted_data, "color": "#5DADE2"},
                {"name": "COGS Estimated", "data": estimated_data, "color": "#8E44AD"}
            ]
        }

    def create_table_data(self, df):
        """Create DataTable data from results DataFrame"""

        columns = [
            {"name": self.breakout.title()},
            {"name": "COGS Forecasted"},
            {"name": "COGS Estimated"},
            {"name": "Change"},
            {"name": "Material Forecasted"},
            {"name": "Material Estimated"},
            {"name": "Material Change"},
            {"name": "Cocoa Forecasted"},
            {"name": "Cocoa Estimated"},
            {"name": "Cocoa Change"}
        ]

        data = []
        for _, row in df.iterrows():
            data.append([
                row[self.breakout],
                f"${row['COGS_Forecasted']/1000000:.2f}M",
                f"${row['COGS_Estimated']/1000000:.2f}M",
                f"{row['COGS_Change']:.2%}",
                f"${row['Material_Forecasted']/1000000:.2f}M",
                f"${row['Material_Estimated']/1000000:.2f}M",
                f"{row['Material_Change']:.2%}",
                f"${row['Cocoa_Forecasted']/1000000:.2f}M",
                f"${row['Cocoa_Estimated']/1000000:.2f}M",
                f"{row['Cocoa_Change']:.2%}"
            ])

        return {"columns": columns, "data": data}


if __name__ == '__main__':
    skill_input: SkillInput = whatif_analysis.create_input(arguments={
        'periods': ['Q3 2024'],
        'breakout': 'category',
        'price_change_scenario': {'cocoa': 0.05}
    })
    out = whatif_analysis(skill_input)
    print(out.narrative)
