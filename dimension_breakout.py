from __future__ import annotations

import json
import logging
import pandas as pd
import numpy as np
from types import SimpleNamespace
from typing import Dict, List, Optional

import jinja2
from ar_analytics import ArUtils
from ar_analytics.helpers.utils import get_dataset_id
from answer_rocket import AnswerRocketClient
from skill_framework import (
    SkillInput, SkillVisualization, skill, SkillParameter, SkillOutput,
    SuggestedQuestion, ParameterDisplayDescription
)
from skill_framework.layouts import wire_layout
from skill_framework.skills import ExportData

logger = logging.getLogger(__name__)

# Default prompts
DEFAULT_MAX_PROMPT = """
Based on the following breakout analysis facts:
{% for fact_list in facts %}
{% for fact in fact_list %}
- {{ fact }}
{% endfor %}
{% endfor %}

Provide a concise executive summary (2-3 sentences) highlighting the most significant findings.
"""

DEFAULT_INSIGHT_PROMPT = """
Analyze the following dimension breakout data:
{% for fact_list in facts %}
{% for fact in fact_list %}
- {{ fact }}
{% endfor %}
{% endfor %}

Provide detailed insights covering:
1. Top performers and underperformers
2. Key variance drivers vs forecast/budget
3. Period-over-period trends

Format the insights using bullet points only. Do NOT use tables or markdown tables. Keep response to 100-150 words.
"""

# Layout template for table with chart
TABLE_WITH_CHART_LAYOUT = """{
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
                "text": "Dimension Breakout",
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
                "text": "Breakout Analysis",
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
                        "type": "bar"
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
                        "bar": {
                            "dataLabels": {"enabled": false}
                        }
                    },
                    "tooltip": {
                        "pointFormat": "<b>{series.name}</b>: {point.formatted}"
                    }
                },
                "hidden": false
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


class DimensionBreakoutAnalysis:
    """Dimension Breakout Analysis with scenario comparisons"""

    def __init__(self, client, metrics, periods, breakouts, growth_type='Y/Y',
                 compare_metrics=None, other_filters=None, top_n=10,
                 growth_trend=None, table_name=None):
        self.client = client
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.periods = periods if isinstance(periods, list) else [periods]
        self.breakouts = breakouts if isinstance(breakouts, list) else [breakouts] if breakouts else []
        self.growth_type = growth_type
        self.compare_metrics = compare_metrics or []  # ['forecast', 'budget']
        self.other_filters = other_filters or []
        self.top_n = top_n
        self.growth_trend = growth_trend
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
        self.format_dict = {}
        self.current_start_date = None
        self.current_end_date = None
        self.prior_start_date = None
        self.prior_end_date = None

        logger.info(f"DimensionBreakoutAnalysis initialized: database_id={self.database_id}, table_name={self.table_name}")

    def parse_period_to_date_range(self, period_str):
        """Convert period string to date range for SQL query"""
        from dateutil.parser import parse

        if not period_str:
            raise ValueError("Period is required")

        period_lower = period_str.lower().strip()

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

    def get_prior_period_dates(self, start_date, end_date):
        """Calculate prior period dates based on growth_type"""
        from dateutil.relativedelta import relativedelta
        from datetime import datetime

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        if self.growth_type == 'Y/Y':
            prior_start = (start_dt - relativedelta(years=1)).strftime('%Y-%m-%d')
            prior_end = (end_dt - relativedelta(years=1)).strftime('%Y-%m-%d')
        else:  # P/P
            delta = end_dt - start_dt
            prior_end = (start_dt - relativedelta(days=1)).strftime('%Y-%m-%d')
            prior_start = (start_dt - relativedelta(days=delta.days + 1)).strftime('%Y-%m-%d')

        return prior_start, prior_end

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

    def query_breakout_data(self, breakout_dim):
        """Query data for a specific breakout dimension with scenario pivot"""
        logger.info(f"Querying breakout data for dimension: {breakout_dim}")

        filter_clause = self.build_filter_clause()
        metric_cols = ", ".join([f"SUM({m}) as {m}" for m in self.metrics])

        # Parse current period
        start_date, end_date = self.parse_period_to_date_range(self.periods[0])
        self.current_start_date = start_date
        self.current_end_date = end_date

        # Query current period with all scenarios
        current_query = f"""
        SELECT {breakout_dim}, scenario, {metric_cols}
        FROM {self.table_name}
        WHERE end_date BETWEEN '{start_date}' AND '{end_date}'
        {filter_clause}
        GROUP BY {breakout_dim}, scenario
        """

        logger.info(f"Current query: {current_query}")
        result = self.client.data.execute_sql_query(
            database_id=self.database_id,
            sql_query=current_query,
            row_limit=10000
        )
        current_df = result.df if hasattr(result, 'df') else pd.DataFrame()

        if current_df.empty:
            raise ValueError(f"No data found for period {self.periods[0]}")

        # Pivot by scenario
        current_df = self._pivot_scenario(current_df, breakout_dim)

        # Query prior period if growth_type specified
        prior_df = None
        if self.growth_type and self.growth_type != 'None':
            prior_start, prior_end = self.get_prior_period_dates(start_date, end_date)
            self.prior_start_date = prior_start
            self.prior_end_date = prior_end

            prior_query = f"""
            SELECT {breakout_dim}, SUM({self.metrics[0]}) as {self.metrics[0]}
            FROM {self.table_name}
            WHERE scenario = 'actuals'
            AND end_date BETWEEN '{prior_start}' AND '{prior_end}'
            {filter_clause}
            GROUP BY {breakout_dim}
            """

            logger.info(f"Prior query: {prior_query}")
            result = self.client.data.execute_sql_query(
                database_id=self.database_id,
                sql_query=prior_query,
                row_limit=10000
            )
            prior_df = result.df if hasattr(result, 'df') else None

        return current_df, prior_df, breakout_dim

    def _pivot_scenario(self, df, breakout_dim):
        """Pivot dataframe by scenario column"""
        if 'scenario' not in df.columns:
            return df

        metric_cols = self.metrics
        pivoted = df.pivot_table(
            index=[breakout_dim],
            values=metric_cols,
            columns='scenario'
        ).reset_index()

        # Flatten multi-level columns
        pivoted.columns = ['_'.join(col).strip() if isinstance(col, tuple) and col[-1] else col[0] for col in pivoted.columns]

        # Rename actuals columns to base metric name
        rename_dict = {f"{m}_actuals": m for m in metric_cols}
        pivoted = pivoted.rename(columns=rename_dict)

        return pivoted

    def calculate_metrics(self, current_df, prior_df, breakout_dim):
        """Calculate change metrics and variances"""
        df = current_df.copy()
        metric = self.metrics[0]
        metric_display = format_display_name(metric)

        # Merge prior period data
        if prior_df is not None and not prior_df.empty:
            df = pd.merge(
                df, prior_df,
                on=breakout_dim,
                how='left',
                suffixes=('', '_prev')
            )

            # Calculate change metrics
            prev_col = f"{metric}_prev"
            if prev_col in df.columns:
                df[f'{metric}_change'] = df[metric] - df[prev_col]
                df[f'{metric}_change_pct'] = (df[metric] - df[prev_col]) / df[prev_col]
                df[f'{metric}_change_pct'] = df[f'{metric}_change_pct'].replace([np.inf, -np.inf], np.nan)

        # Calculate forecast/budget variances
        for comp in self.compare_metrics:
            comp_col = f"{metric}_{comp}"
            if comp_col in df.columns:
                df[f'{metric}_{comp}_var'] = df[metric] - df[comp_col]
                df[f'{metric}_{comp}_var_pct'] = (df[metric] - df[comp_col]) / df[comp_col]
                df[f'{metric}_{comp}_var_pct'] = df[f'{metric}_{comp}_var_pct'].replace([np.inf, -np.inf], np.nan)

        # Sort based on growth_trend
        sort_col = metric
        ascending = False

        if self.growth_trend:
            trend_lower = self.growth_trend.lower()
            if 'declining' in trend_lower:
                sort_col = f'{metric}_change_pct' if f'{metric}_change_pct' in df.columns else metric
                ascending = True
            elif 'growing' in trend_lower:
                sort_col = f'{metric}_change_pct' if f'{metric}_change_pct' in df.columns else metric
                ascending = False
            elif 'smallest' in trend_lower:
                ascending = True
            elif 'biggest' in trend_lower:
                ascending = False

        df = df.sort_values(by=sort_col, ascending=ascending, na_position='last')
        df = df.head(self.top_n)

        # Add rank
        df['rank'] = range(1, len(df) + 1)

        return df

    def create_display_table(self, df, breakout_dim):
        """Create formatted display table"""
        metric = self.metrics[0]
        metric_display = format_display_name(metric)
        dim_display = format_display_name(breakout_dim)

        columns = [{'name': dim_display}]
        display_data = []

        # Build columns list
        col_mapping = [(breakout_dim, dim_display)]

        # Current period
        if metric in df.columns:
            columns.append({'name': f'{metric_display} (Current)'})
            col_mapping.append((metric, f'{metric_display} (Current)'))

        # Previous period
        prev_col = f"{metric}_prev"
        if prev_col in df.columns:
            columns.append({'name': f'{metric_display} (Previous)'})
            col_mapping.append((prev_col, f'{metric_display} (Previous)'))

        # Change
        change_col = f'{metric}_change'
        if change_col in df.columns:
            columns.append({'name': f'{metric_display} (Change)'})
            col_mapping.append((change_col, f'{metric_display} (Change)'))

        # Change %
        change_pct_col = f'{metric}_change_pct'
        if change_pct_col in df.columns:
            columns.append({'name': f'{metric_display} (Change %)'})
            col_mapping.append((change_pct_col, f'{metric_display} (Change %)'))

        # Forecast/Budget columns
        for comp in self.compare_metrics:
            comp_title = comp.title()
            comp_col = f"{metric}_{comp}"
            var_pct_col = f"{metric}_{comp}_var_pct"

            if comp_col in df.columns:
                columns.append({'name': f'{metric_display} ({comp_title})'})
                col_mapping.append((comp_col, f'{metric_display} ({comp_title})'))

            if var_pct_col in df.columns:
                columns.append({'name': f'{metric_display} vs {comp_title} %'})
                col_mapping.append((var_pct_col, f'{metric_display} vs {comp_title} %'))

        # Build data rows
        for _, row in df.iterrows():
            row_data = []
            for col_name, display_name in col_mapping:
                val = row.get(col_name)

                if col_name == breakout_dim:
                    row_data.append(str(val) if not pd.isna(val) else 'N/A')
                elif 'pct' in col_name.lower() or '%' in display_name:
                    if pd.isna(val):
                        row_data.append('N/A')
                    else:
                        row_data.append(f"{val*100:.2f}%")
                else:
                    row_data.append(format_number(val, is_currency=True))

            display_data.append(row_data)

        return {'columns': columns, 'data': display_data}

    def create_chart_data(self, df, breakout_dim):
        """Create chart data for visualization"""
        metric = self.metrics[0]
        metric_display = format_display_name(metric)

        categories = df[breakout_dim].tolist()

        # Scale values to millions
        current_data = []
        previous_data = []

        for _, row in df.iterrows():
            cat = row[breakout_dim]

            # Current value
            current_val = row.get(metric, 0)
            if pd.isna(current_val):
                current_val = 0
            current_scaled = current_val / 1_000_000
            current_data.append({
                'name': cat,
                'y': current_scaled,
                'formatted': format_number(current_val)
            })

            # Previous value
            prev_col = f"{metric}_prev"
            if prev_col in row:
                prev_val = row.get(prev_col, 0)
                if pd.isna(prev_val):
                    prev_val = 0
                prev_scaled = prev_val / 1_000_000
                previous_data.append({
                    'name': cat,
                    'y': prev_scaled,
                    'formatted': format_number(prev_val)
                })

        # Build series
        series = [{
            'name': f'{metric_display} (Current)',
            'data': current_data,
            'color': '#5DADE2'
        }]

        if previous_data:
            series.append({
                'name': f'{metric_display} (Previous)',
                'data': previous_data,
                'color': '#F8C471'
            })

        # Calculate Y-axis max
        all_values = [d['y'] for d in current_data] + [d['y'] for d in previous_data]
        max_val = max(all_values) if all_values else 100

        import math
        y_max = math.ceil(max_val / 100) * 100

        return {
            'chart_categories': categories,
            'chart_data': series,
            'chart_y_axis': {
                'title': {'text': ''},
                'min': 0,
                'max': y_max,
                'labels': {'format': '${value}M'}
            }
        }

    def _format_date_for_display(self, start_date, end_date):
        """Format date range for display in subtitle"""
        from datetime import datetime

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Format as "April 2024 to June 2024" style
        start_str = start_dt.strftime('%B %Y')
        end_str = end_dt.strftime('%B %Y')

        if start_str == end_str:
            return start_str
        return f"{start_str} to {end_str}"

    def run_analysis(self):
        """Run complete breakout analysis"""
        logger.info("Starting dimension breakout analysis")

        # Remove 'scenario' from breakouts if present
        self.breakouts = [b for b in self.breakouts if b.lower() != 'scenario']

        if not self.breakouts:
            raise ValueError("At least one breakout dimension is required")

        for breakout_dim in self.breakouts:
            current_df, prior_df, dim = self.query_breakout_data(breakout_dim)
            calc_df = self.calculate_metrics(current_df, prior_df, dim)
            table_data = self.create_display_table(calc_df, dim)
            chart_data = self.create_chart_data(calc_df, dim)

            self.results[breakout_dim] = {
                'df': calc_df,
                'table': table_data,
                'chart': chart_data
            }

            # Add facts
            metric = self.metrics[0]
            metric_display = format_display_name(metric)
            dim_display = format_display_name(breakout_dim)

            for _, row in calc_df.head(3).iterrows():
                val = row.get(metric, 0)
                self.facts.append({
                    'fact': f"{dim_display} '{row[breakout_dim]}': {format_number(val)}",
                    'category': breakout_dim
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

        # Build subtitle with date ranges
        current_date_str = self._format_date_for_display(self.current_start_date, self.current_end_date)
        self.subtitle = f"Breakout by {format_display_name(self.breakouts[0])} • {current_date_str}"

        if self.growth_type and self.growth_type != 'None' and self.prior_start_date:
            prior_date_str = self._format_date_for_display(self.prior_start_date, self.prior_end_date)
            self.subtitle += f" vs {prior_date_str}"

        logger.info("Analysis complete")
        return self


@skill(
    name="Dimension Breakout",
    llm_name="Dimension Breakout with Scenario Analysis",
    description="Analyze metrics broken out by dimensions with period-over-period comparisons and scenario variance (vs Forecast, vs Budget).",
    capabilities="Dimension breakout analysis with Y/Y or P/P growth comparisons. Scenario variance analysis vs Forecast and Budget. Horizontal bar chart visualization. Top/bottom performer ranking. Multi-dimensional breakout support.",
    limitations="Requires 'scenario' column in dataset with values: actuals, budget, forecast. Requires date columns for period filtering.",
    example_questions="Which brands are performing poorly in EMEA for biscuits? Show me top 10 fastest declining products by revenue. What is the revenue breakout by region vs budget? Which categories missed their forecast in Q2 2024?",
    parameter_guidance="Select metrics to analyze. Choose breakout dimensions (brand, category, region). Specify period for analysis. Select growth type (Y/Y, P/P, None). Choose compare metrics (forecast, budget) for variance analysis. Add filters as needed.",
    parameters=[
        SkillParameter(
            name="periods",
            constrained_to="date_filter",
            is_multi=True,
            description="Time periods for analysis (e.g., 'Q2 2024', 'Jan 2024')"
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
            description="Dimensions for breakout analysis (e.g., brand, category, region)"
        ),
        SkillParameter(
            name="growth_type",
            constrained_values=["Y/Y", "P/P", "None"],
            description="Growth comparison type: Y/Y (year-over-year), P/P (period-over-period), or None",
            default_value="Y/Y"
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
            description="Additional filters (e.g., category=biscuits, region=EMEA)"
        ),
        SkillParameter(
            name="limit_n",
            description="Number of top results to display",
            default_value=10
        ),
        SkillParameter(
            name="growth_trend",
            constrained_values=["fastest growing", "highest growing", "fastest declining", "highest declining", "biggest overall", "smallest overall"],
            description="Sort direction for ranking"
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt for executive summary",
            default_value=DEFAULT_MAX_PROMPT
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt for detailed insights",
            default_value=DEFAULT_INSIGHT_PROMPT
        ),
        SkillParameter(
            name="table_viz_layout",
            parameter_type="visualization",
            description="Layout for visualization",
            default_value=TABLE_WITH_CHART_LAYOUT
        ),
        SkillParameter(
            name="table_name",
            parameter_type="code",
            description="Table/view name (inherited from dataset if not provided)",
            default_value=""
        )
    ]
)
def dimension_breakout(parameters: SkillInput):
    """Execute Dimension Breakout Analysis with scenario comparisons"""

    logger.info(f"Skill received parameters: {parameters.arguments}")

    # Extract parameters
    periods = getattr(parameters.arguments, 'periods', [])
    metrics = getattr(parameters.arguments, 'metrics', [])
    breakouts = getattr(parameters.arguments, 'breakouts', [])
    growth_type = getattr(parameters.arguments, 'growth_type', 'Y/Y')
    compare_metrics = getattr(parameters.arguments, 'compare_metrics', [])
    other_filters = getattr(parameters.arguments, 'other_filters', [])
    top_n = int(getattr(parameters.arguments, 'limit_n', 10) or 10)
    growth_trend = getattr(parameters.arguments, 'growth_trend', None)
    max_prompt = parameters.arguments.max_prompt
    insight_prompt = parameters.arguments.insight_prompt
    viz_layout = getattr(parameters.arguments, 'table_viz_layout', TABLE_WITH_CHART_LAYOUT)
    table_name = getattr(parameters.arguments, 'table_name', None)
    if table_name == "":
        table_name = None

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

    if not breakouts:
        return SkillOutput(
            final_prompt="Please specify at least one breakout dimension.",
            narrative="**Missing Parameter**: Breakout dimension is required.",
            visualizations=[],
            warnings=["Breakout parameter is required"]
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
    analysis = DimensionBreakoutAnalysis(
        client=client,
        metrics=metrics,
        periods=periods,
        breakouts=breakouts,
        growth_type=growth_type,
        compare_metrics=compare_metrics or [],
        other_filters=other_filters,
        top_n=top_n,
        growth_trend=growth_trend,
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
        insights = "Analysis complete. Review the breakout tables and charts for detailed findings."

    # Create visualizations
    viz_list = []
    export_data = {}

    for dim_name, result in analysis.results.items():
        dim_display = format_display_name(dim_name)

        general_vars = {
            "headline": analysis.title,
            "sub_headline": analysis.subtitle
        }

        layout_vars = {
            **general_vars,
            **result['chart'],
            'data': result['table']['data'],
            'col_defs': result['table']['columns']
        }

        rendered = wire_layout(json.loads(viz_layout), layout_vars)
        viz_list.append(SkillVisualization(title=dim_display, layout=rendered))
        export_data[dim_display] = result['df']

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

    param_info.append(ParameterDisplayDescription(key="", value=f"Breakout: {format_display_name(breakouts[0])}"))

    # Add period info with date ranges
    if analysis.current_start_date:
        current_date_str = analysis._format_date_for_display(analysis.current_start_date, analysis.current_end_date)
        param_info.append(ParameterDisplayDescription(key="", value=f"Period: {current_date_str}"))

    if analysis.prior_start_date:
        prior_date_str = analysis._format_date_for_display(analysis.prior_start_date, analysis.prior_end_date)
        param_info.append(ParameterDisplayDescription(key="", value=f"Compare Period: {prior_date_str}"))

    param_info.append(ParameterDisplayDescription(key="", value=f"Growth Type: {growth_type}"))

    if growth_trend:
        param_info.append(ParameterDisplayDescription(key="", value=f"Sort: {growth_trend}"))

    if compare_metrics:
        comp_display = ' and '.join([f"vs. {c.title()}" for c in compare_metrics])
        param_info.append(ParameterDisplayDescription(key="", value=f"Variance: {comp_display}"))

    param_info.append(ParameterDisplayDescription(key="", value=f"Top {top_n}"))

    return SkillOutput(
        final_prompt=max_response_prompt,
        narrative=insights,
        visualizations=viz_list,
        parameter_display_descriptions=param_info,
        export_data=[ExportData(name=name, data=df) for name, df in export_data.items()]
    )
