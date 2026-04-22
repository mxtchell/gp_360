from __future__ import annotations

import json
import logging
import pandas as pd
import numpy as np
from types import SimpleNamespace
from typing import Dict, List, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta

import jinja2
from ar_analytics import ArUtils
from ar_analytics.helpers.utils import get_dataset_id
from answer_rocket import AnswerRocketClient
from skill_framework import (
    SkillVisualization, skill, SkillParameter, SkillInput, SkillOutput,
    ParameterDisplayDescription
)
from skill_framework.layouts import wire_layout
from skill_framework.skills import ExportData

logger = logging.getLogger(__name__)

# Default prompts
DEFAULT_MAX_PROMPT = """
Based on the following trend analysis facts:
{% for fact_list in facts %}
{% for fact in fact_list %}
- {{ fact }}
{% endfor %}
{% endfor %}

Provide a concise executive summary (2-3 sentences) highlighting the key trends.
"""

DEFAULT_INSIGHT_PROMPT = """
Analyze the following trend data:
{% for fact_list in facts %}
{% for fact in fact_list %}
- {{ fact }}
{% endfor %}
{% endfor %}

Provide detailed insights covering:
1. Overall trend direction and magnitude
2. Key peaks and valleys
3. Comparison vs forecast/budget (if applicable)
4. Seasonal patterns or anomalies
5. Actionable recommendations

Format in clear markdown with bullet points.
"""

# Chart layout for trend visualization
TREND_CHART_LAYOUT = """{
    "layoutJson": {
        "type": "Document",
        "gap": "0px",
        "style": {
            "backgroundColor": "#ffffff",
            "width": "100%",
            "height": "max-content",
            "padding": "15px",
            "gap": "20px"
        },
        "children": [
            {
                "name": "CardContainer0",
                "type": "CardContainer",
                "children": "",
                "minHeight": "80px",
                "rows": 2,
                "columns": 1,
                "style": {
                    "border-radius": "11.911px",
                    "background": "#2563EB",
                    "padding": "10px",
                    "fontFamily": "Arial"
                },
                "hidden": false
            },
            {
                "name": "Header0",
                "type": "Header",
                "children": "",
                "text": "Trend Analysis",
                "style": {
                    "fontSize": "20px",
                    "fontWeight": "700",
                    "color": "#ffffff",
                    "textAlign": "left"
                },
                "parentId": "CardContainer0",
                "hidden": false
            },
            {
                "name": "Paragraph0",
                "type": "Paragraph",
                "children": "",
                "text": "Time Series Analysis",
                "style": {
                    "fontSize": "15px",
                    "fontWeight": "normal",
                    "color": "#fafafa"
                },
                "parentId": "CardContainer0",
                "hidden": false
            },
            {
                "name": "HighchartsChart0",
                "type": "HighchartsChart",
                "minHeight": "400px",
                "options": {
                    "chart": {
                        "type": "line"
                    },
                    "title": {
                        "text": ""
                    },
                    "xAxis": {
                        "categories": [],
                        "title": {"text": ""}
                    },
                    "yAxis": {
                        "title": {"text": ""}
                    },
                    "series": [],
                    "credits": {"enabled": false},
                    "legend": {
                        "enabled": true,
                        "align": "center",
                        "verticalAlign": "bottom"
                    },
                    "plotOptions": {
                        "line": {
                            "dataLabels": {"enabled": false},
                            "marker": {"enabled": true, "radius": 4}
                        }
                    },
                    "tooltip": {
                        "shared": true,
                        "crosshairs": true
                    }
                },
                "hidden": false
            },
            {
                "name": "FlexContainer0",
                "type": "FlexContainer",
                "children": "",
                "minHeight": "150px",
                "style": {
                    "borderRadius": "11.911px",
                    "box-shadow": "0px 0px 8.785px 0px rgba(0, 0, 0, 0.10) inset",
                    "padding": "15px",
                    "fontFamily": "Arial",
                    "backgroundColor": "#edf2f7",
                    "border-left": "4px solid #3b82f6"
                },
                "direction": "column",
                "hidden": false
            },
            {
                "name": "Markdown0",
                "type": "Markdown",
                "children": "",
                "text": "Insights will appear here...",
                "style": {
                    "fontSize": "14px",
                    "color": "#000000"
                },
                "parentId": "FlexContainer0"
            },
            {
                "name": "DataTable0",
                "type": "DataTable",
                "children": "",
                "columns": [],
                "data": [],
                "styles": {
                    "td": {"vertical-align": "middle"}
                }
            }
        ]
    },
    "inputVariables": [
        {
            "name": "headline",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "Header0", "fieldName": "text"}]
        },
        {
            "name": "sub_headline",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "Paragraph0", "fieldName": "text"}]
        },
        {
            "name": "chart_categories",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "HighchartsChart0", "fieldName": "options.xAxis.categories"}]
        },
        {
            "name": "chart_y_axis",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "HighchartsChart0", "fieldName": "options.yAxis"}]
        },
        {
            "name": "chart_data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "HighchartsChart0", "fieldName": "options.series"}]
        },
        {
            "name": "exec_summary",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "Markdown0", "fieldName": "text"}]
        },
        {
            "name": "data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "DataTable0", "fieldName": "data"}]
        },
        {
            "name": "col_defs",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "DataTable0", "fieldName": "columns"}]
        }
    ]
}"""


def format_number(value, is_currency=True, decimals=2):
    """Format numbers with M/K/B abbreviations"""
    if pd.isna(value) or not isinstance(value, (int, float)):
        return str(value) if not pd.isna(value) else "N/A"

    abs_value = abs(value)

    if abs_value >= 1_000_000_000:
        formatted = f"{value / 1_000_000_000:.{decimals}f}B"
    elif abs_value >= 1_000_000:
        formatted = f"{value / 1_000_000:.{decimals}f}M"
    elif abs_value >= 1_000:
        formatted = f"{value / 1_000:.{decimals}f}K"
    else:
        formatted = f"{value:.{decimals}f}"

    if is_currency:
        formatted = f"${formatted}"

    return formatted


def format_display_name(name):
    """Format technical names to display names"""
    if not name:
        return name

    special_cases = {
        'region_l1': 'Region L1',
        'region_l2': 'Secondary Region',
        'market_type_1': 'Market Type 1',
        'customer_type': 'Customer Type',
        'gross_revenue': 'Gross Revenue',
        'net_revenue': 'Net Revenue',
        'gross_profit': 'Gross Profit',
        'brand_contribution_margin': 'Brand Contribution Margin',
        'units_carton': 'Units (Carton)',
        'brand': 'Product Brand',
        'category': 'Product Category',
    }

    if name.lower() in special_cases:
        return special_cases[name.lower()]

    return name.replace('_', ' ').title()


class TrendAnalysis:
    """Trend Analysis with scenario comparisons"""

    def __init__(self, client, metrics, periods, breakouts=None, time_granularity='month',
                 growth_type='Y/Y', compare_metrics=None, other_filters=None,
                 top_n=10, table_name=None):
        self.client = client
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.periods = periods if isinstance(periods, list) else [periods]
        self.breakouts = breakouts if isinstance(breakouts, list) else [breakouts] if breakouts else []
        self.time_granularity = time_granularity or 'month'
        self.growth_type = growth_type
        self.compare_metrics = compare_metrics or []  # ['forecast', 'budget']
        self.other_filters = other_filters or []
        self.top_n = top_n
        self.table_name = table_name

        # Get database_id and dataset_id from platform context
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

        self.results = {}
        self.facts = []
        self.title = ""
        self.subtitle = ""
        self.date_column = 'end_date'  # or start_date depending on dataset

        logger.info(f"TrendAnalysis initialized: database_id={self.database_id}, table_name={self.table_name}")

    def parse_period_to_date_range(self, period_str):
        """Convert period string to date range for SQL query"""
        from dateutil.parser import parse

        if not period_str:
            return None, None

        period_lower = period_str.lower().strip()

        # Handle year (2024, 2025)
        if period_lower.isdigit() and len(period_lower) == 4:
            year = int(period_lower)
            return f"{year}-01-01", f"{year}-12-31"

        # Handle quarters
        if period_lower.startswith('q'):
            parts = period_str.split()
            quarter = int(parts[0][1])
            year = int(parts[1])

            quarter_map = {
                1: ('01-01', '03-31'),
                2: ('04-01', '06-30'),
                3: ('07-01', '09-30'),
                4: ('10-01', '12-31')
            }
            start_month_day, end_month_day = quarter_map[quarter]
            return f"{year}-{start_month_day}", f"{year}-{end_month_day}"

        # Handle single months
        try:
            parsed_date = parse(period_str, fuzzy=True)
            year = parsed_date.year
            month = parsed_date.month

            if month == 12:
                last_day = 31
            elif month in [4, 6, 9, 11]:
                last_day = 30
            elif month == 2:
                if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                    last_day = 29
                else:
                    last_day = 28
            else:
                last_day = 31

            return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day}"
        except:
            return period_str, period_str

    def get_date_trunc_expression(self):
        """Get SQL date truncation expression based on granularity"""
        granularity_map = {
            'day': f"DATE({self.date_column})",
            'week': f"DATE_TRUNC('week', {self.date_column})",
            'month': f"DATE_TRUNC('month', {self.date_column})",
            'quarter': f"DATE_TRUNC('quarter', {self.date_column})",
            'year': f"DATE_TRUNC('year', {self.date_column})"
        }
        return granularity_map.get(self.time_granularity.lower(), f"DATE_TRUNC('month', {self.date_column})")

    def build_filter_clause(self):
        """Build SQL WHERE clause from filters"""
        clauses = []

        for filter_dict in self.other_filters:
            dim = filter_dict.get('dim') or filter_dict.get('col')
            op = filter_dict.get('op', '=')
            val = filter_dict.get('val')

            if dim and val:
                if isinstance(val, list):
                    if len(val) == 1:
                        clauses.append(f"UPPER({dim}) {op} UPPER('{val[0]}')")
                    else:
                        val_str = ", ".join([f"UPPER('{v}')" for v in val])
                        clauses.append(f"UPPER({dim}) IN ({val_str})")
                elif isinstance(val, str):
                    clauses.append(f"UPPER({dim}) {op} UPPER('{val}')")
                else:
                    clauses.append(f"{dim} {op} {val}")

        return " AND " + " AND ".join(clauses) if clauses else ""

    def query_trend_data(self):
        """Query trend data with scenario support"""
        logger.info(f"Querying trend data for metrics: {self.metrics}")

        filter_clause = self.build_filter_clause()
        date_trunc = self.get_date_trunc_expression()
        metric_cols = ", ".join([f"SUM({m}) as {m}" for m in self.metrics])

        # Parse period range
        start_date, end_date = None, None
        for period in self.periods:
            s, e = self.parse_period_to_date_range(period)
            if s and (start_date is None or s < start_date):
                start_date = s
            if e and (end_date is None or e > end_date):
                end_date = e

        if not start_date or not end_date:
            raise ValueError("Could not parse period range")

        # Query with scenario column
        query = f"""
        SELECT
            {date_trunc} as period,
            scenario,
            {metric_cols}
        FROM {self.table_name}
        WHERE {self.date_column} BETWEEN '{start_date}' AND '{end_date}'
        {filter_clause}
        GROUP BY {date_trunc}, scenario
        ORDER BY period
        """

        logger.info(f"Trend query: {query}")
        result = self.client.data.execute_sql_query(
            database_id=self.database_id,
            sql_query=query,
            row_limit=10000
        )
        df = result.df if hasattr(result, 'df') else pd.DataFrame()

        if df.empty:
            raise ValueError(f"No data found for periods {self.periods}")

        # Pivot by scenario
        df = self._pivot_scenario(df)

        return df, start_date, end_date

    def _pivot_scenario(self, df):
        """Pivot dataframe by scenario column"""
        if 'scenario' not in df.columns:
            return df

        metric_cols = self.metrics
        pivoted = df.pivot_table(
            index=['period'],
            values=metric_cols,
            columns='scenario',
            aggfunc='sum'
        ).reset_index()

        # Flatten multi-level columns
        pivoted.columns = ['_'.join(col).strip() if isinstance(col, tuple) and col[-1] else col[0] for col in pivoted.columns]

        # Rename actuals columns to base metric name
        rename_dict = {f"{m}_actuals": m for m in metric_cols}
        pivoted = pivoted.rename(columns=rename_dict)

        return pivoted

    def calculate_metrics(self, df):
        """Calculate variance metrics for forecast/budget comparisons"""
        result_df = df.copy()

        for metric in self.metrics:
            for comp in self.compare_metrics:
                comp_col = f"{metric}_{comp}"
                if comp_col in result_df.columns:
                    # Calculate variance
                    result_df[f'{metric}_{comp}_var'] = result_df[metric] - result_df[comp_col]
                    result_df[f'{metric}_{comp}_var_pct'] = (result_df[metric] - result_df[comp_col]) / result_df[comp_col]
                    result_df[f'{metric}_{comp}_var_pct'] = result_df[f'{metric}_{comp}_var_pct'].replace([np.inf, -np.inf], np.nan)

        return result_df

    def create_chart_data(self, df):
        """Create chart data for visualization"""
        metric = self.metrics[0]
        metric_display = format_display_name(metric)

        # Format periods for x-axis
        categories = []
        for p in df['period']:
            if pd.isna(p):
                categories.append('N/A')
            elif hasattr(p, 'strftime'):
                if self.time_granularity == 'month':
                    categories.append(p.strftime('%b %Y'))
                elif self.time_granularity == 'quarter':
                    q = (p.month - 1) // 3 + 1
                    categories.append(f'Q{q} {p.year}')
                elif self.time_granularity == 'year':
                    categories.append(str(p.year))
                else:
                    categories.append(p.strftime('%Y-%m-%d'))
            else:
                categories.append(str(p))

        # Build series
        series = []

        # Actuals series
        if metric in df.columns:
            actuals_data = []
            for val in df[metric]:
                if pd.isna(val):
                    actuals_data.append(None)
                else:
                    actuals_data.append(round(val / 1_000_000, 2))

            series.append({
                'name': f'{metric_display} (Actuals)',
                'data': actuals_data,
                'color': '#5DADE2',
                'type': 'line'
            })

        # Forecast/Budget series
        for comp in self.compare_metrics:
            comp_col = f"{metric}_{comp}"
            if comp_col in df.columns:
                comp_data = []
                for val in df[comp_col]:
                    if pd.isna(val):
                        comp_data.append(None)
                    else:
                        comp_data.append(round(val / 1_000_000, 2))

                color = '#F8C471' if comp == 'forecast' else '#82E0AA'
                series.append({
                    'name': f'{metric_display} ({comp.title()})',
                    'data': comp_data,
                    'color': color,
                    'type': 'line',
                    'dashStyle': 'dash'
                })

        # Calculate Y-axis max
        all_values = []
        for s in series:
            all_values.extend([v for v in s['data'] if v is not None])
        max_val = max(all_values) if all_values else 100

        import math
        y_max = math.ceil(max_val / 100) * 100

        return {
            'chart_categories': categories,
            'chart_data': series,
            'chart_y_axis': {
                'title': {'text': metric_display},
                'min': 0,
                'max': y_max,
                'labels': {'format': '${value}M'}
            }
        }

    def create_table_data(self, df):
        """Create formatted table data"""
        metric = self.metrics[0]
        metric_display = format_display_name(metric)

        columns = [{'name': 'Period'}]
        col_mapping = [('period', 'Period')]

        # Actuals
        if metric in df.columns:
            columns.append({'name': f'{metric_display} (Actuals)'})
            col_mapping.append((metric, f'{metric_display} (Actuals)'))

        # Forecast/Budget columns
        for comp in self.compare_metrics:
            comp_col = f"{metric}_{comp}"
            var_pct_col = f"{metric}_{comp}_var_pct"

            if comp_col in df.columns:
                columns.append({'name': f'{metric_display} ({comp.title()})'})
                col_mapping.append((comp_col, f'{metric_display} ({comp.title()})'))

            if var_pct_col in df.columns:
                columns.append({'name': f'vs {comp.title()} %'})
                col_mapping.append((var_pct_col, f'vs {comp.title()} %'))

        # Build data rows
        data = []
        for _, row in df.iterrows():
            row_data = []
            for col_name, display_name in col_mapping:
                val = row.get(col_name)

                if col_name == 'period':
                    if pd.isna(val):
                        row_data.append('N/A')
                    elif hasattr(val, 'strftime'):
                        if self.time_granularity == 'month':
                            row_data.append(val.strftime('%b %Y'))
                        else:
                            row_data.append(val.strftime('%Y-%m-%d'))
                    else:
                        row_data.append(str(val))
                elif 'pct' in col_name.lower() or '%' in display_name:
                    if pd.isna(val):
                        row_data.append('N/A')
                    else:
                        row_data.append(f"{val*100:.2f}%")
                else:
                    row_data.append(format_number(val, is_currency=True))

            data.append(row_data)

        return {'columns': columns, 'data': data}

    def run_analysis(self):
        """Run complete trend analysis"""
        logger.info("Starting trend analysis")

        # Remove 'scenario' from breakouts if present
        self.breakouts = [b for b in self.breakouts if b and b.lower() != 'scenario']

        df, start_date, end_date = self.query_trend_data()
        df = self.calculate_metrics(df)

        self.results = {
            'df': df,
            'chart': self.create_chart_data(df),
            'table': self.create_table_data(df)
        }

        # Generate facts
        metric = self.metrics[0]
        metric_display = format_display_name(metric)

        if metric in df.columns and len(df) > 0:
            first_val = df[metric].iloc[0]
            last_val = df[metric].iloc[-1]
            change = last_val - first_val
            change_pct = (change / first_val) * 100 if first_val != 0 else 0

            self.facts.append({
                'fact': f"{metric_display} changed from {format_number(first_val)} to {format_number(last_val)} ({change_pct:+.1f}%)",
                'category': 'trend'
            })

            max_val = df[metric].max()
            min_val = df[metric].min()
            self.facts.append({
                'fact': f"Peak: {format_number(max_val)}, Low: {format_number(min_val)}",
                'category': 'range'
            })

        # Set title/subtitle
        filters_desc = []
        for f in self.other_filters:
            dim = f.get('dim') or f.get('col', '')
            val = f.get('val', '')
            if isinstance(val, list):
                val = ', '.join(val)
            if dim and val:
                filters_desc.append(f"{val.title()}")

        self.title = ', '.join(filters_desc) if filters_desc else "Total"

        # Format date range for subtitle
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            self.subtitle = f"Trend by {self.time_granularity} • {start_dt.strftime('%B %Y')} to {end_dt.strftime('%B %Y')}"
        except:
            self.subtitle = f"Trend by {self.time_granularity}"

        logger.info("Analysis complete")
        return self


@skill(
    name="Trend Analysis",
    llm_name="Trend Analysis with Scenario Comparison",
    description="Analyze metric trends over time with optional comparison to forecast or budget scenarios.",
    capabilities="Time series trend analysis. Multiple time granularities (day, week, month, quarter, year). Scenario variance analysis vs Forecast and Budget. Line chart visualization with multiple series. Period-over-period comparisons.",
    limitations="Requires 'scenario' column in dataset with values: actuals, budget, forecast. Requires date columns for period filtering.",
    example_questions="Show me the revenue trend for 2024. What is the gross revenue trend for EMEA by month? Compare actual vs forecast revenue for Q2 2024. Show quarterly revenue trends for the biscuits category.",
    parameter_guidance="Select metrics to analyze. Specify period range. Choose time granularity (month, quarter, year). Select compare_metrics (forecast, budget) for variance analysis. Add filters as needed.",
    parameters=[
        SkillParameter(
            name="periods",
            constrained_to="date_filter",
            is_multi=True,
            description="Time periods for analysis (e.g., '2024', 'Q2 2024', 'Jan 2024 to Jun 2024')"
        ),
        SkillParameter(
            name="metrics",
            is_multi=True,
            constrained_to="metrics",
            description="Metrics to analyze (e.g., gross_revenue, net_revenue)"
        ),
        SkillParameter(
            name="breakouts",
            is_multi=True,
            constrained_to="dimensions",
            description="Dimensions for breakout analysis (optional)"
        ),
        SkillParameter(
            name="time_granularity",
            constrained_to="date_dimensions",
            description="Time granularity: day, week, month, quarter, year",
            default_value="month"
        ),
        SkillParameter(
            name="growth_type",
            constrained_values=["Y/Y", "P/P", "None"],
            description="Growth comparison type",
            default_value="None"
        ),
        SkillParameter(
            name="compare_metrics",
            is_multi=True,
            constrained_values=["forecast", "budget"],
            description="Scenario comparisons to include (forecast, budget)"
        ),
        SkillParameter(
            name="other_filters",
            is_multi=True,
            constrained_to="filters",
            description="Additional filters (e.g., region=EMEA, category=biscuits)"
        ),
        SkillParameter(
            name="limit_n",
            description="Number of results to display",
            default_value=10
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt for executive summary",
            default_value=""
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt for detailed insights",
            default_value=""
        ),
        SkillParameter(
            name="chart_viz_layout",
            parameter_type="visualization",
            description="Layout for trend chart",
            default_value=TREND_CHART_LAYOUT
        ),
        SkillParameter(
            name="table_name",
            parameter_type="code",
            description="Table/view name (inherited from dataset if not provided)",
            default_value=""
        )
    ]
)
def trend(parameters: SkillInput):
    """Execute Trend Analysis with scenario comparisons"""

    logger.info(f"Skill received parameters: {parameters.arguments}")

    # Extract parameters
    periods = getattr(parameters.arguments, 'periods', [])
    metrics = getattr(parameters.arguments, 'metrics', [])
    breakouts = getattr(parameters.arguments, 'breakouts', [])
    time_granularity = getattr(parameters.arguments, 'time_granularity', 'month')
    growth_type = getattr(parameters.arguments, 'growth_type', 'None')
    compare_metrics = getattr(parameters.arguments, 'compare_metrics', [])
    other_filters = getattr(parameters.arguments, 'other_filters', [])
    top_n = int(getattr(parameters.arguments, 'limit_n', 10) or 10)
    table_name = getattr(parameters.arguments, 'table_name', None)
    if table_name == "":
        table_name = None

    # Get prompts from platform or use defaults
    max_prompt = parameters.arguments.max_prompt if hasattr(parameters.arguments, 'max_prompt') and parameters.arguments.max_prompt else DEFAULT_MAX_PROMPT
    insight_prompt = parameters.arguments.insight_prompt if hasattr(parameters.arguments, 'insight_prompt') and parameters.arguments.insight_prompt else DEFAULT_INSIGHT_PROMPT
    viz_layout = getattr(parameters.arguments, 'chart_viz_layout', TREND_CHART_LAYOUT)

    # Validate required parameters
    if not periods:
        return SkillOutput(
            final_prompt="Please specify a time period for analysis.",
            narrative="**Missing Parameter**: Period is required.",
            visualizations=[],
            warnings=["Period parameter is required"]
        )

    if not metrics:
        return SkillOutput(
            final_prompt="Please specify at least one metric to analyze.",
            narrative="**Missing Parameter**: Metric is required.",
            visualizations=[],
            warnings=["Metric parameter is required"]
        )

    # Initialize client
    try:
        client = AnswerRocketClient()
        ar_utils = ArUtils()
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        return SkillOutput(
            final_prompt=f"Failed to initialize client: {str(e)}",
            warnings=[str(e)]
        )

    # Run analysis
    analysis = TrendAnalysis(
        client=client,
        metrics=metrics,
        periods=periods,
        breakouts=breakouts,
        time_granularity=time_granularity or 'month',
        growth_type=growth_type,
        compare_metrics=compare_metrics or [],
        other_filters=other_filters,
        top_n=top_n,
        table_name=table_name
    )

    try:
        analysis.run_analysis()
    except ValueError as e:
        logger.error(f"Analysis failed: {e}")
        return SkillOutput(
            final_prompt=f"Analysis could not be completed: {str(e)}",
            narrative=f"**Error**: {str(e)}",
            visualizations=[],
            warnings=[str(e)]
        )

    # Generate insights
    facts_list = [pd.DataFrame(analysis.facts)]
    insight_template = jinja2.Template(insight_prompt).render(facts=[facts_list])
    max_response_prompt = jinja2.Template(max_prompt).render(facts=[facts_list])

    try:
        insights = ar_utils.get_llm_response(insight_template)
    except:
        insights = "Analysis complete. Review the trend charts and tables for detailed findings."

    # Create visualization
    result = analysis.results
    metric_display = format_display_name(metrics[0])

    general_vars = {
        "headline": analysis.title,
        "sub_headline": analysis.subtitle,
        "exec_summary": insights
    }

    layout_vars = {
        **general_vars,
        **result['chart'],
        'data': result['table']['data'],
        'col_defs': result['table']['columns']
    }

    rendered = wire_layout(json.loads(viz_layout), layout_vars)
    viz_list = [SkillVisualization(title=metric_display, layout=rendered)]

    # Parameter display
    param_info = [
        ParameterDisplayDescription(key="", value=f"Metrics: {', '.join([format_display_name(m) for m in metrics])}"),
    ]

    # Add filter info
    filter_parts = []
    for f in other_filters:
        dim = f.get('dim') or f.get('col') or ''
        val = f.get('val') or ''
        if isinstance(val, list):
            val = ', '.join(val)
        if dim and val:
            filter_parts.append(f"{val} ({format_display_name(dim)})")

    if filter_parts:
        param_info.append(ParameterDisplayDescription(key="", value=f"Filter: {' and '.join(filter_parts)}"))

    if breakouts:
        param_info.append(ParameterDisplayDescription(key="", value=f"Breakouts: {', '.join([format_display_name(b) for b in breakouts if b])}"))

    param_info.append(ParameterDisplayDescription(key="", value=f"Period: {', '.join(periods)}"))
    param_info.append(ParameterDisplayDescription(key="", value=f"Granularity: {time_granularity or 'month'}"))

    if compare_metrics:
        comp_display = ' and '.join([f"vs. {c.title()}" for c in compare_metrics])
        param_info.append(ParameterDisplayDescription(key="", value=f"Variance: {comp_display}"))

    return SkillOutput(
        final_prompt=max_response_prompt,
        narrative=insights,
        visualizations=viz_list,
        parameter_display_descriptions=param_info,
        export_data=[ExportData(name=metric_display, data=result['df'])]
    )
