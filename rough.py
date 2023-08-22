import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import sqlite3
import warnings
from plotly.subplots import make_subplots

warnings.simplefilter(action='ignore', category=UserWarning)

conn = sqlite3.connect("ALL_FAULTS_WITH_JUNE.sqlite3")
query_card = 'SELECT * FROM CARD'

def month_data(df_card_data):
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    df_month = df_card_data['MONTH'].unique()
    # print(df_month)
    # sorted(df_month, key=lambda x: months.index(x.split(",")[0]))
    month_order = sorted(df_month, key=lambda x: (int(x.split(",")[1]), months.index(x.split(",")[0])))

    return month_order


query_raw='SELECT * FROM RAW'
df_raw=pd.read_sql_query(query_raw,conn)
final_month_list = month_data(df_raw)

#
df_raw['MONTH'] = pd.Categorical(df_raw['MONTH'], categories=final_month_list, ordered=True)
#
#
# Filter the data for 'ASSEMBLY ISSUE' and 'COMPONENT FAIL ISSUE'
filtered_data = df_raw[df_raw['FAULT_CATEGORY'].isin(['DRY SOLDER', "SOLDERING ISSUE", "COMPONENT DAMAGE/MISS ISSUE",
                                                  "DRY SOLDER", "WRONG MOUNTING", "REVERSE POLARITY", "SOLDERING ISSUE","LEAD CUT ISSUE",
                                                      "OPERATOR FAULT","COATING ISSUE"])]
#
filtered_data1 = df_raw[df_raw['FAULT_CATEGORY'].isin(["COMPONENT FAIL ISSUE", "CC ISSUE"])]
#
# # Step 1: Calculate the total quantity of products in each respective month
total_quantity_per_month = df_raw.groupby('MONTH')['PRODUCT_NAME'].count()
#
# # Step 2: Calculate the occurrences of each category for table1
table1 = filtered_data1.groupby(['MONTH', 'FAULT_CATEGORY'])['PRODUCT_NAME'].count().unstack(fill_value=0)
# # print(table1)
table1['COMP. FAIL'] = table1.sum(axis=1)
# # print(table1)
#
#
# # Step 3: Group the data month-wise and count the occurrences of each category for table
table = filtered_data.groupby(['MONTH', 'FAULT_CATEGORY'])['PRODUCT_NAME'].count().unstack(fill_value=0)
table['ASSEMBLY ISSUE'] = table[['DRY SOLDER', "SOLDERING ISSUE", "COMPONENT DAMAGE/MISS ISSUE",
                                                  "DRY SOLDER", "WRONG MOUNTING", "REVERSE POLARITY", "SOLDERING ISSUE","LEAD CUT ISSUE",
                                                      "OPERATOR FAULT","COATING ISSUE"]].sum(axis=1)
# # Step 4: Merge the two tables
final_table = pd.merge(table, table1, how="left", on="MONTH")
#
# # Step 5: Calculate the percentage of COMP. FAIL issues out of the total quantity of products in each respective month
final_table['ASSEMBLY ISSUE %'] = round((table["ASSEMBLY ISSUE"] / total_quantity_per_month) * 100, 0)
final_table['COMP. FAIL %'] = round((table1["COMP. FAIL"] / total_quantity_per_month) * 100, 0)

transposed_df = final_table.T
#
transposed_df=transposed_df.tail(2)





final_month_list = month_data(df_card)
month_string = f'{final_month_list[0]}-{final_month_list[-1]}'
month_string = month_string.replace(',', "'")
d = df_card

d['MONTH'] = pd.Categorical(d['MONTH'], categories=final_month_list, ordered=True)

d = d.groupby(['MONTH'])[['TEST_QUANTITY', 'REJECT_QUANTITY']].sum().reset_index()

d['DPT'] = round((d['REJECT_QUANTITY'] / d['TEST_QUANTITY']) * 1000, 0)

del d['TEST_QUANTITY']
del d['REJECT_QUANTITY']
# print(d)
dpt_title = f'DPT COMPARISON OVER MONTHS ({month_string})'

# Create a Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    dcc.Graph(id='bar-graph'),
    dcc.Graph(id='fc'),
    dcc.RangeSlider(
        id='month-slider',
        marks={i: month for i, month in enumerate(d['MONTH'])},
        min=0,
        max=len(d['MONTH']) - 1,
        value=[0, len(d['MONTH']) - 1],
        step=3,
        updatemode='drag'
    )
])


# Callback to update the bar graph based on the range slider values
@app.callback(
    [dash.dependencies.Output('bar-graph', 'figure'),
dash.dependencies.Output('fc', 'figure')],
    [dash.dependencies.Input('month-slider', 'value')]
)
def update_graph(selected_months):
    start_index, end_index = selected_months
    selected_data = d.iloc[start_index:end_index + 1]
    selected_data_fc=transposed_df.iloc[:,start_index:end_index+1]
    # print(selected_data)

    # Create the bar graph

    dpt_title = f'DPT COMPARISON OVER MONTHS ({month_string})'
    fig = go.Figure(data=go.Bar(x=selected_data['MONTH'], y=selected_data['DPT'], text=d['DPT'], textposition='outside',
                                textfont=dict(size=10, family='Arial Black')))

    # Update the layout
    fig.update_layout(
        title=dict(text=dpt_title, font=dict(size=24, family='Arial Black')),
        xaxis_title=dict(text='Months', font=dict(size=18, family='Arial Black')),
        yaxis_title=dict(text='DPT (Defects Per Thousand)', font=dict(size=18, family='Arial Black')),
        showlegend=False
    )
    fig.update_xaxes(tickfont_family='Arial Black', tickangle=315)
    fig.update_yaxes(tickfont_family='Arial Black')

    coefficients = np.polyfit(selected_data.index, selected_data['DPT'], 1)
    trendline_values = np.polyval(coefficients, selected_data.index)
    fig.add_trace(go.Scatter(x=selected_data['MONTH'], y=trendline_values, mode='lines', line=dict(dash='dash', width=3), name='Trendline'))


    #2nd graph

    assembly_x = np.arange(len(selected_data_fc.columns))
    assembly_y = selected_data_fc.loc['ASSEMBLY ISSUE %'].values
    assembly_trendline_coeffs = np.polyfit(assembly_x, assembly_y, 1)

    comp_fail_x = np.arange(len(selected_data_fc.columns))
    comp_fail_y = selected_data_fc.loc['COMP. FAIL %'].values
    comp_fail_trendline_coeffs = np.polyfit(comp_fail_x, comp_fail_y, 1)

    assembly_trendline = assembly_trendline_coeffs[0] * assembly_x + assembly_trendline_coeffs[1]
    comp_fail_trendline = comp_fail_trendline_coeffs[0] * comp_fail_x + comp_fail_trendline_coeffs[1]

    fc_graph=go.Figure()
    fc_graph.add_trace(go.Bar(x=selected_data_fc.columns,
                              y=selected_data_fc.loc['ASSEMBLY ISSUE %'],
                              name='ASSEMBLY ISSUE %',
    marker_color='royalblue',
    text=[f'{val:.0f}%' for val in selected_data_fc.loc['ASSEMBLY ISSUE %']],textposition='outside',textfont=dict(size=10,family='Arial Black'
),showlegend=True))

    fc_graph.add_trace(go.Bar(
    x=selected_data_fc.columns,
    y=selected_data_fc.loc['COMP. FAIL %'],
    name='COMP. FAIL %',
    marker_color='firebrick',
text=[f'{val:.0f}%' for val in selected_data_fc.loc['COMP. FAIL %']],textposition='outside',textfont=dict(size=12,family='Arial Black'
),showlegend=True))

    fc_graph.add_trace(go.Scatter(
    x=selected_data_fc.columns,
    y=assembly_trendline,
    mode='lines',
    name='ASSEMBLY ISSUE % Trendline',
    line=dict(color='blue', width=3, dash='dash'),
    showlegend=False
))
    fc_graph.add_trace(go.Scatter(
    x=selected_data_fc.columns,
    y=comp_fail_trendline,
    mode='lines',
    name='COMP. FAIL % Trendline',
    line=dict(color='red', width=3, dash='dash'),
    showlegend=False
))

    fc_title = f'F1 & F2 FAULT CATEGORY COMPARISON ({month_string})'
    fc_graph.update_layout(
        title=dict(text=fc_title, font=dict(size=24, family='Arial Black')),
        xaxis_title=dict(text='Months', font=dict(size=18, family='Arial Black')),
        yaxis_title=dict(text='Fault Percentage', font=dict(size=18, family='Arial Black')),
        showlegend=True,
        legend=dict(
            orientation='h',  # Horizontal orientation for the legend
            x=0.4,  # Position the legend at the center of the graph horizontally
            y=0.95,  # Position the legend slightly above the graph
        )
    )

    fc_graph.update_xaxes(tickfont_family='Arial Black', tickangle=315)
    fc_graph.update_yaxes(tickfont_family='Arial Black')


    return fig,fc_graph


if __name__ == '__main__':
    app.run_server(debug=True)
