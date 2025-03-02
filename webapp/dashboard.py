import plotly.graph_objs as go
from dash import dash_table
from dash import dcc, html

dash_app_layout = html.Div(
    style={'fontFamily': 'Roboto, sans-serif', 'padding': '30px'},  # Global styling
    children=[
        html.Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap",
            style={'display': 'none'}  # Hide the Link component itself
        ),
        html.Link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css",
            style={'display': 'none'}  # Hide the Link component itself
        ),
        html.Div(
            children=[
                html.H2("Solar PV Detection Performance", style={'textAlign': 'center', 'color': '#333', 'marginBottom': '30px'}),
                html.P(
                    children=[
                        "The ",
                        html.A("PV panel detection app ", href="/"),
                        " combines prediction probability estimates from several neural networks (sub-models) into a stacked model for greater robustness. "
                        "This dashboard displays the performance of each sub-model (see ",
                        html.A("the original paper", href="https://arxiv.org/abs/2309.12214", target="_blank"),
                        " for details on each model) on our user generated labeled dataset and how they are combined to produce our stacked model. "
                    ],
                    style={'textAlign': 'center', 'fontSize': '16px', 'color': '#666', 'max-width':'1000px',  'margin':'0 auto 30px ', 'padding':'10px'}
                ),
            ]
        ),
        
        html.Div(
            children=[
                # Two columns for the charts (using Flexbox for layout)
                html.P(
                    "The graphs below show how True Positive and True Negative rates change as the prediction thresholds varies. "
                    "Our stacked model is effectively picking a set of thresholds for each model that maximizes both the true positive and true negative rates.",
                    style={'textAlign': 'center', 'fontSize': '16px', 'color': '#666', 'max-width':'1000px',  'margin':'20px auto ', 'padding':'10px'}
                ),
                html.Div(
                    children=[
                        html.H4("True Positive Rate Vs Discrimination Threshold", style={'textAlign': 'center', 'color': '#333',}),
                        dcc.Loading(
                            type='circle',
                            children=[dcc.Graph(id='tp-chart', style={'height':'300px'})],
                        ),
                    ],
                    style={
                        'flex': '1 1 35%',  # Flex-grow, flex-shrink, and flex-basis for responsive behavior
                        'padding': '10px',
                        'boxShadow': '0 2px 5px rgba(0, 0, 0, 0.1)',
                        'width': '100%',
                        'max-width': '500px',
                        'min-width': '300px',
                        'marginBottom': '20px',  # Ensure spacing between stacked items on small screens
                    },
                ),
                html.Div(
                    children=[
                        html.H4("True Negative Rate Vs Discrimination Threshold", style={'textAlign': 'center', 'color': '#333'}),
                        dcc.Loading(
                            type='circle',
                            children=[dcc.Graph(id='tn-chart', style={'height':'300px'})],
                        ),
                    ],
                    style={
                        'flex': '1 1 35%',  # Flex-grow, flex-shrink, and flex-basis for responsive behavior
                        'padding': '10px',
                        'boxShadow': '0 2px 5px rgba(0, 0, 0, 0.1)',
                        'width': '100%',
                        'max-width': '500px',
                        'min-width': '300px',
                        'marginBottom': '20px',  # Ensure spacing between stacked items on small screens
                    },
                ),
            ],
            style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap', 'gap': '20px', 'border': '2px solid #D0D0D0', 'borderRadius': '10px', 'max-width': '1200px', 'margin':'0 auto 30px'}
        ),
        
        # Add a table to display logistic regression parameters and accuracy
        html.Div(
            children=[
                # Performance Metrics Table
                html.P(
                    "For the PV detection model, the above submodels are combined by a logistic regression stacked model fit on the feedback data compiled from user's identifying whether panels were present in each image. "
                    "We evaluate how well our stacked model is able to accurately predict the feedback results via cross-validation. "
                    "The first table below displays performance metrics from the latest stacked model evaluation. "
                    "The second table lists the submodel weights and the intercept of the logistic regression stacked model currently in use on the app.",
                    style={'textAlign': 'center', 'fontSize': '16px', 'color': '#333', 'marginTop': '10px', 'marginBottom': '10px', 'max-width': '1000px', 'padding':'10px'}
                ),
                # Button to trigger the rerun of finding optimal parameters
                html.Div(
                    children=[
                        html.Button(
                            "Recalculate Optimal Parameters", 
                            id='rerun-btn', 
                            n_clicks=0, 
                            style={'padding': '10px 20px', 'fontSize': '16px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}
                        ),
                    ],
                    style={'textAlign': 'center', 'marginBottom': '20px', 'width': '100%'}
                ),
                html.Div(
                    children=[
                        html.H4("Optimal Model Performance Metrics", style={'textAlign': 'center', 'color': '#333', 'marginTop': '10px'}),
                        dcc.Loading(
                            type='circle',
                            children=[
                                dash_table.DataTable(
                                    id='performance-metrics-table',
                                    columns=[
                                        {'name': 'Metric', 'id': 'metric'},
                                        {'name': 'Value', 'id': 'value'},
                                    ],
                                    style_table={'width': '60%', 'margin': '30px auto'},
                                    style_header={'backgroundColor': '#f4f4f4', 'fontWeight': 'bold'},
                                    style_cell={'padding': '10px', 'textAlign': 'center', 'fontSize': '14px'},
                                    style_data={'backgroundColor': '#f9f9f9'},
                                ),
                            ],
                        ),
                    ],
                    style={'flex': '1 1 45%', 'padding': '10px', 'width': '100%', 'max-width': '600px', 'boxShadow': '0 2px 5px rgba(0, 0, 0, 0.1)'}
                ),
                # Parameters Table
                html.Div(
                    children=[
                        html.H4("Logistic Regression Parameters", style={'textAlign': 'center', 'color': '#333', 'marginTop': '10px'}),
                        dcc.Loading(
                            type='circle',
                            children=[
                                dash_table.DataTable(
                                    id='params-table',
                                    columns=[
                                        {'name': 'Feature', 'id': 'feature'},
                                        {'name': 'Value', 'id': 'param_value'},
                                    ],
                                    style_table={'width': '60%', 'margin': '30px auto'},
                                    style_header={'backgroundColor': '#f4f4f4', 'fontWeight': 'bold'},
                                    style_cell={'padding': '10px', 'textAlign': 'center', 'fontSize': '14px'},
                                    style_data={'backgroundColor': '#f9f9f9'},
                                ),
                            ],
                        ),
                    ],
                    style={'flex': '1 1 45%', 'padding': '10px', 'width': '100%', 'max-width': '600px', 'boxShadow': '0 2px 5px rgba(0, 0, 0, 0.1)'}
                ),
            ],
            style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap', 'border': '2px solid #D0D0D0', 'borderRadius': '10px', 'max-width': '1200px', 'margin':'0 auto'}
        ),

        html.Div(
            children=[
                html.Span(
                    children=[
                        html.I(className="fas fa-caret-down"),  # Font Awesome downward caret icon (wide V)
                        " Show Model Performance History",
                    ],
                    id='toggle-history-btn',  # Button for toggling visibility
                    style={
                        'padding':'10px',
                        'fontSize': '18px',  # Larger arrow size
                        'color': '#333',  # Dark color for contrast
                        'cursor': 'pointer',  # Pointer cursor on hover
                        'transition': 'transform 0.3s ease-in-out',  # Smooth transition for rotate
                    }
                ),
            ],
            style={'paddingTop': '25px'}  # Add relative positioning to parent
        ),

        
        # History of Accuracy and F1 Score graph (Initially hidden)
        html.Div(
            children=[
                html.P(
                    "The history graph shows how model performance evolves over time as it is retrained with more user feedback data. "
                    "Each point reflects the model's metrics at a specific retraining stage, highlighting improvements or changes as the model adapts to new information.",
                    style={ 'fontSize': '16px', 'color': '#666', 'max-width':'800px', 'marginLeft':'20px'}
                ),
                dcc.Graph(id='history-chart', style={'max-width':'1000px'}),
            ],
            id='history-container',  # Container for history chart
            style={'display': 'none', 'marginTop': '30px'},  # Hidden initially
        ),
    ]
)

def generate_dash_outputs(thresholds, model_metrics, model_performance, performance_data, history_data, metrics_to_show):
        
    # Parameters for parameter table
    params_data = [{'feature': feature, 'param_value': f"{value:0.3f}"} for feature, value in zip(model_performance['features'], model_performance['param_values'])]
    
    # Create figure for True Positive Rate vs Thresholds
    bold_colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7B7A3', '#F6D02F']
    tp_rate_fig = {
        'data': [
            go.Scatter(
                x=thresholds,
                y=model_metrics[model]["TP_rate"],
                mode='lines+markers',
                name=f'{model}',
                line={'width': 3, 'color': bold_colors[i % len(bold_colors)]},
                marker={'size': 8, 'color': bold_colors[i % len(bold_colors)]}
            )
            for i, model in enumerate(model_metrics)
        ],
        'layout': go.Layout(
            title_x=0.5,
            xaxis={'title': 'Threshold', 'tickangle': 45},
            yaxis={'title': 'True Positive Rate'},
            template='plotly_white',
            plot_bgcolor='#f7f7f7',
            paper_bgcolor='#FFFFFF',
            font={'color': '#333'},
            showlegend=True,
            legend={
                'x': 0.1,               # Place the legend at the left side
                'y': 0.1,            # Adjust the position below the plot (negative y value moves it down)
                'xanchor': 'left',    # Anchor the x position to the left
                'yanchor': 'bottom',  # Anchor the y position to the bottom
                'orientation': 'v'    # Horizontal legend (so it’s aligned horizontally at the bottom)
            },
            margin={'t': 20, 'b': 50, 'l': 70, 'r': 50},
        )
    }

    # Create figure for True Negative Rate vs Thresholds
    tn_rate_fig = {
        'data': [
            go.Scatter(
                x=thresholds,
                y=model_metrics[model]["TN_rate"],
                mode='lines+markers',
                name=f'{model}',
                line={'width': 3, 'color': bold_colors[i % len(bold_colors)]},
                marker={'size': 8, 'color': bold_colors[i % len(bold_colors)]}
            )
            for i, model in enumerate(model_metrics)
        ],
        'layout': go.Layout(
            title_x=0.5,
            xaxis={'title': 'Threshold', 'tickangle': 45},
            yaxis={'title': 'True Negative Rate'},
            template='plotly_white',
            plot_bgcolor='#f7f7f7',
            paper_bgcolor='#FFFFFF',
            font={'color': '#333'},
            showlegend=True,
            legend={
                'x': 0.9,               # Place the legend at the left side
                'y': 0.1,            # Adjust the position below the plot (negative y value moves it down)
                'xanchor': 'right',    # Anchor the x position to the left
                'yanchor': 'bottom',  # Anchor the y position to the bottom
                'orientation': 'v'    # Horizontal legend (so it’s aligned horizontally at the bottom)
            },
            margin={'t': 20, 'b': 50, 'l': 70, 'r': 50},
        )
    }
    
    # Make the history fig
    history_colors = ['#4CAF50', '#FF9800', '#2196F3', '#00BCD4', '#F6D02F']

    history_fig = {
        'data': [
            go.Scatter(
                x=[data['date'] for data in history_data if metric_key in data],
                y=[data[metric_key] for data in history_data if metric_key in data],
                mode='lines+markers',
                name=metric_name,
                line={'color': history_colors[i]},
            ) 
            for i, (metric_name, metric_key) in enumerate(metrics_to_show.items())
        ],
        'layout': go.Layout(
            title_x=0.5,
            xaxis={'title': 'Date', 'tickangle': 45},
            yaxis={
                'title': 'Cross-validation Score',
                'range': [0, 1],  # Set the y-axis range to be between 0 and 1
            },
            template='plotly_white',
            plot_bgcolor='#f7f7f7',
            paper_bgcolor='#FFFFFF',
            font={'color': '#333'},
            showlegend=True,
            legend={
                'x': 0.9,               # Place the legend at the left side
                'y': 0.1,            # Adjust the position below the plot (negative y value moves it down)
                'xanchor': 'right',    # Anchor the x position to the left
                'yanchor': 'bottom',  # Anchor the y position to the bottom
                'orientation': 'v'    # Horizontal legend (so it’s aligned horizontally at the bottom)
            },
            margin={'t': 10, 'b': 20, 'l': 50, 'r': 0},
        )
    }

    # Return the figures and the table data
    return tp_rate_fig, tn_rate_fig, performance_data, params_data, history_fig