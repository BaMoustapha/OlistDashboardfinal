#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DASHBOARD SIMPLIFIÉ - ANALYSE DES PERFORMANCES COMMERCIALES OLIST
Application Dash avec filtres interactifs
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration de l'app

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Olist - Analyse des Performances Commerciales"

server = app.server



# Chargement des données
def load_data():
    """Charge toutes les données nécessaires"""
    try:
        # Données principales
        df = pd.read_csv('data/olist_prepared_data.csv')
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
        return df
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        return None


# Chargement des données
df = load_data()

# Couleurs du thème
colors = {
    'primary': '#3498db',
    'success': '#27ae60',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#9b59b6',
    'dark': '#2c3e50',
    'light': '#ecf0f1'
}

# Layout principal
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("📊 Tableau de Bord - Analyse des Performances Olist",
                    style={'color': 'white', 'marginBottom': 10}),
            html.P("Analyse complète des données e-commerce",
                   style={'color': 'white', 'fontSize': 18})
        ], style={'textAlign': 'center'})
    ], style={'backgroundColor': colors['dark'], 'padding': '30px', 'marginBottom': '20px'}),

    # Section des filtres
    html.Div([
        html.Div([
            # Filtre de date
            html.Div([
                html.Label("Période d'analyse:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.DatePickerRange(
                    id='date-filter',
                    start_date=df['order_purchase_timestamp'].min() if df is not None else None,
                    end_date=df['order_purchase_timestamp'].max() if df is not None else None,
                    display_format='DD/MM/YYYY',
                    style={'width': '100%'}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),

            # Filtre de statut
            html.Div([
                html.Label("Statut des commandes:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='status-filter',
                    options=[{'label': 'Tous', 'value': 'all'}] +
                            [{'label': status, 'value': status} for status in
                             df['order_status'].unique()] if df is not None else [],
                    value='all',
                    multi=True,
                    placeholder="Sélectionner les statuts..."
                )
            ], style={'width': '48%', 'display': 'inline-block'})
        ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'marginBottom': '20px',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
    ], style={'padding': '0 20px'}),

    # Navigation
    dcc.Tabs(id='main-tabs', value='overview', children=[
        dcc.Tab(label='🏠 Vue d\'ensemble', value='overview'),
        dcc.Tab(label='📦 Analyse Produits', value='products'),
        dcc.Tab(label='📈 Analyse Temporelle', value='temporal'),
        dcc.Tab(label='⭐ Satisfaction', value='satisfaction'),
        dcc.Tab(label='🚚 Logistique', value='logistics')
    ], style={'fontSize': '16px', 'padding': '0 20px'}),

    # Contenu
    html.Div(id='page-content', style={'padding': '20px', 'backgroundColor': colors['light']})
])


# Callback pour filtrer les données
@app.callback(
    Output('page-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('date-filter', 'start_date'),
     Input('date-filter', 'end_date'),
     Input('status-filter', 'value')]
)
def display_page(tab, start_date, end_date, status_filter):
    if df is None:
        return html.Div([
            html.H3("⚠️ Erreur de chargement des données"),
            html.P("Vérifiez que le fichier '../output/prepared/olist_prepared_data.csv' existe")
        ])

    # Filtrage des données
    filtered_df = df.copy()

    # Filtre par date
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['order_purchase_timestamp'] >= start_date) &
            (filtered_df['order_purchase_timestamp'] <= end_date)
            ]

    # Filtre par statut
    if status_filter and 'all' not in status_filter and isinstance(status_filter, list):
        filtered_df = filtered_df[filtered_df['order_status'].isin(status_filter)]

    if tab == 'overview':
        return create_overview_page(filtered_df)
    elif tab == 'products':
        return create_products_page(filtered_df)
    elif tab == 'temporal':
        return create_temporal_page(filtered_df)
    elif tab == 'satisfaction':
        return create_satisfaction_page(filtered_df)
    elif tab == 'logistics':
        return create_logistics_page(filtered_df)
    else:
        return html.Div()


# PAGE 1: VUE D'ENSEMBLE
def create_overview_page(filtered_df):
    """Crée la page vue d'ensemble"""

    # Calcul des métriques
    total_orders = filtered_df['order_id'].nunique()
    total_customers = filtered_df['customer_unique_id'].nunique()
    total_revenue = filtered_df['total_amount_fcfa'].sum()
    avg_basket = filtered_df.groupby('order_id')['total_amount_fcfa'].sum().mean() if total_orders > 0 else 0
    avg_review = filtered_df['review_score'].mean() if 'review_score' in filtered_df.columns and not filtered_df[
        'review_score'].isna().all() else 0
    conversion_rate = (filtered_df[filtered_df['order_status'] == 'delivered'][
                           'order_id'].nunique() / total_orders * 100) if total_orders > 0 else 0

    return html.Div([
        # KPIs principaux
        html.Div([
            create_kpi_card("Commandes", f"{total_orders:,}", colors['primary'], "📦"),
            create_kpi_card("Clients", f"{total_customers:,}", colors['success'], "👥"),
            create_kpi_card("CA Total", f"{total_revenue / 1e6:.1f}M FCFA", colors['danger'], "💰"),
            create_kpi_card("Panier Moyen", f"{avg_basket:,.0f} FCFA", colors['warning'], "🛒"),
            create_kpi_card("Note Moyenne", f"{avg_review:.2f}/5", colors['info'], "⭐"),
            create_kpi_card("Taux Livraison", f"{conversion_rate:.1f}%", colors['success'], "✅"),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'between', 'marginBottom': '30px'}),

        # Graphiques principaux
        html.Div([
            html.Div([
                dcc.Graph(figure=create_revenue_evolution(filtered_df))
            ], style={'width': '49%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(figure=create_orders_by_status(filtered_df))
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
        ]),

        html.Div([
            html.Div([
                dcc.Graph(figure=create_top_categories(filtered_df))
            ], style={'width': '49%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(figure=create_geographic_distribution(filtered_df))
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
        ], style={'marginTop': '20px'}),
    ])


# PAGE 2: ANALYSE PRODUITS
def create_products_page(filtered_df):
    """Crée la page d'analyse des produits"""

    return html.Div([
        html.H2("Analyse des Produits", style={'textAlign': 'center', 'marginBottom': '30px'}),

        # Sélecteurs
        html.Div([
            html.Label("Nombre de catégories à afficher:"),
            dcc.Slider(
                id='top-n-categories',
                min=5,
                max=20,
                step=5,
                value=10,
                marks={i: str(i) for i in range(5, 21, 5)}
            )
        ], style={'width': '50%', 'margin': 'auto', 'marginBottom': '30px'}),

        # Container pour les graphiques
        html.Div(id='product-charts-container')
    ])


# PAGE 3: ANALYSE TEMPORELLE
def create_temporal_page(filtered_df):
    """Crée la page d'analyse temporelle"""

    return html.Div([
        html.H2("Analyse Temporelle et Saisonnalité", style={'textAlign': 'center', 'marginBottom': '30px'}),

        # Évolution mensuelle
        html.Div([
            dcc.Graph(figure=create_monthly_evolution(filtered_df))
        ]),

        # Patterns hebdomadaires et horaires
        html.Div([
            html.Div([
                dcc.Graph(figure=create_weekly_pattern(filtered_df))
            ], style={'width': '49%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(figure=create_hourly_pattern(filtered_df))
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
        ], style={'marginTop': '20px'}),

        # Heatmap jour/heure
        html.Div([
            html.H3("Heatmap Activité (Jour/Heure)", style={'textAlign': 'center', 'marginTop': '30px'}),
            dcc.Graph(figure=create_activity_heatmap(filtered_df))
        ])
    ])


# PAGE 4: SATISFACTION
def create_satisfaction_page(filtered_df):
    """Crée la page d'analyse de la satisfaction"""

    return html.Div([
        html.H2("Analyse de la Satisfaction Client", style={'textAlign': 'center', 'marginBottom': '30px'}),

        # Distribution des reviews
        html.Div([
            html.Div([
                dcc.Graph(figure=create_review_distribution(filtered_df))
            ], style={'width': '49%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(figure=create_review_by_category(filtered_df))
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
        ]),

        # Corrélation satisfaction/délai
        html.Div([
            html.H3("Impact du Délai sur la Satisfaction", style={'textAlign': 'center', 'marginTop': '30px'}),
            dcc.Graph(figure=create_delivery_satisfaction_correlation(filtered_df))
        ])
    ])


# PAGE 5: LOGISTIQUE
def create_logistics_page(filtered_df):
    """Crée la page d'analyse logistique"""

    avg_delivery = filtered_df['delivery_days'].mean() if 'delivery_days' in filtered_df.columns and not filtered_df[
        'delivery_days'].isna().all() else 0
    late_deliveries = (
                filtered_df['delivery_vs_estimate'] > 0).sum() if 'delivery_vs_estimate' in filtered_df.columns else 0
    on_time_rate = (filtered_df[
                        'delivery_vs_estimate'] <= 0).mean() * 100 if 'delivery_vs_estimate' in filtered_df.columns else 0
    min_delivery = filtered_df['delivery_days'].min() if 'delivery_days' in filtered_df.columns and not filtered_df[
        'delivery_days'].isna().all() else 0
    max_delivery = filtered_df['delivery_days'].max() if 'delivery_days' in filtered_df.columns and not filtered_df[
        'delivery_days'].isna().all() else 0

    return html.Div([
        html.H2("Analyse Logistique", style={'textAlign': 'center', 'marginBottom': '30px'}),

        # KPIs Logistique
        html.Div([
            create_kpi_card("Délai Moyen", f"{avg_delivery:.1f} jours", colors['primary'], "📦"),
            create_kpi_card("Livraisons en Retard", f"{late_deliveries:,}", colors['danger'], "⏰"),
            create_kpi_card("Livraisons à Temps", f"{on_time_rate:.1f}%", colors['success'], "✅"),
            create_kpi_card("Délai Min-Max", f"{min_delivery:.0f}-{max_delivery:.0f} j", colors['info'], "📊"),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),

        # Graphiques
        html.Div([
            html.Div([
                dcc.Graph(figure=create_delivery_time_distribution(filtered_df))
            ], style={'width': '49%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(figure=create_delivery_by_state(filtered_df))
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
        ]),

        # Performance par état
        html.Div([
            html.H3("Performance de Livraison par État", style={'textAlign': 'center', 'marginTop': '30px'}),
            dcc.Graph(figure=create_delivery_performance_map(filtered_df))
        ])
    ])


# Fonctions de création des graphiques
def create_kpi_card(title, value, color, icon):
    """Crée une carte KPI"""
    return html.Div([
        html.Div([
            html.Span(icon, style={'fontSize': '30px', 'marginRight': '10px'}),
            html.H4(title, style={'margin': '10px 0', 'color': colors['dark']})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
        html.H2(value, style={'color': color, 'margin': '10px 0', 'textAlign': 'center'})
    ], style={
        'backgroundColor': 'white',
        'padding': '20px',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'width': '180px',
        'margin': '10px'
    })


def create_revenue_evolution(filtered_df):
    """Graphique évolution du CA"""
    monthly_revenue = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M'))[
        'total_amount_fcfa'].sum()

    fig = px.line(
        x=monthly_revenue.index.astype(str),
        y=monthly_revenue.values / 1e6,
        title="Évolution du Chiffre d'Affaires",
        labels={'x': 'Mois', 'y': 'CA (Millions FCFA)'}
    )
    fig.update_traces(mode='lines+markers')
    return fig


def create_orders_by_status(filtered_df):
    """Graphique des commandes par statut"""
    status_counts = filtered_df['order_status'].value_counts()

    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Distribution des Statuts de Commande",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    return fig


def create_top_categories(filtered_df):
    """Graphique top catégories"""
    top_categories = filtered_df.groupby('product_category_name_english')['total_amount_fcfa'].sum().nlargest(10)

    fig = px.bar(
        x=top_categories.values / 1e6,
        y=top_categories.index,
        orientation='h',
        title="Top 10 Catégories par CA",
        labels={'x': 'CA (Millions FCFA)', 'y': 'Catégorie'}
    )
    return fig


def create_geographic_distribution(filtered_df):
    """Graphique distribution géographique"""
    state_revenue = filtered_df.groupby('customer_state')['total_amount_fcfa'].sum().nlargest(10)

    fig = px.bar(
        x=state_revenue.index,
        y=state_revenue.values / 1e6,
        title="Top 10 États par CA",
        labels={'x': 'État', 'y': 'CA (Millions FCFA)'},
        color=state_revenue.values,
        color_continuous_scale='Viridis'
    )
    return fig


def create_monthly_evolution(filtered_df):
    """Évolution mensuelle détaillée"""
    monthly_stats = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M')).agg({
        'order_id': 'nunique',
        'total_amount_fcfa': 'sum',
        'customer_unique_id': 'nunique'
    })
    monthly_stats.index = monthly_stats.index.astype(str)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Nombre de Commandes', 'Chiffre d\'Affaires', 'Clients Actifs'),
        vertical_spacing=0.1
    )

    fig.add_trace(
        go.Scatter(x=monthly_stats.index, y=monthly_stats['order_id'], mode='lines+markers', name='Commandes'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=monthly_stats.index, y=monthly_stats['total_amount_fcfa'] / 1e6, mode='lines+markers', name='CA'),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=monthly_stats.index, y=monthly_stats['customer_unique_id'], mode='lines+markers', name='Clients'),
        row=3, col=1
    )

    fig.update_layout(height=800, title="Évolution Mensuelle des Métriques Clés", showlegend=False)

    return fig


def create_weekly_pattern(filtered_df):
    """Pattern hebdomadaire"""
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_revenue = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.day_name())[
        'total_amount_fcfa'].sum()
    weekly_revenue = weekly_revenue.reindex(days_order)

    fig = px.bar(
        x=days_order,
        y=weekly_revenue.values / 1e6,
        title="Pattern Hebdomadaire du CA",
        labels={'x': 'Jour', 'y': 'CA (Millions FCFA)'},
        color=weekly_revenue.values,
        color_continuous_scale='Blues'
    )
    return fig


def create_hourly_pattern(filtered_df):
    """Pattern horaire - CORRIGÉ"""
    hourly_orders = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.hour)['order_id'].nunique()

    # Utilisation de line avec fill='tozeroy' au lieu de area
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_orders.index,
        y=hourly_orders.values,
        fill='tozeroy',
        mode='lines',
        name='Commandes',
        line=dict(color='royalblue')
    ))

    fig.update_layout(
        title="Distribution Horaire des Commandes",
        xaxis_title="Heure",
        yaxis_title="Nombre de commandes",
        hovermode='x'
    )

    return fig


def create_activity_heatmap(filtered_df):
    """Heatmap activité jour/heure"""
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    heatmap_data = filtered_df.pivot_table(
        values='total_amount_fcfa',
        index=filtered_df['order_purchase_timestamp'].dt.hour,
        columns=filtered_df['order_purchase_timestamp'].dt.day_name(),
        aggfunc='sum'
    )
    heatmap_data = heatmap_data.reindex(columns=days_order)

    fig = px.imshow(
        heatmap_data.values / 1e6,
        x=days_order,
        y=list(range(24)),
        title="Heatmap CA par Jour et Heure",
        labels={'x': 'Jour', 'y': 'Heure', 'color': 'CA (M FCFA)'},
        color_continuous_scale='YlOrRd'
    )
    return fig


def create_review_distribution(filtered_df):
    """Distribution des notes"""
    if 'review_score' in filtered_df.columns and not filtered_df['review_score'].isna().all():
        review_counts = filtered_df['review_score'].value_counts().sort_index()

        fig = px.bar(
            x=review_counts.index,
            y=review_counts.values,
            title="Distribution des Notes de Satisfaction",
            labels={'x': 'Note', 'y': 'Nombre de commandes'},
            color=review_counts.index,
            color_continuous_scale='RdYlGn'
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Données de review non disponibles", xref="paper", yref="paper", x=0.5, y=0.5)
    return fig


def create_review_by_category(filtered_df):
    """Review moyen par catégorie"""
    if 'review_score' in filtered_df.columns and not filtered_df['review_score'].isna().all():
        category_reviews = filtered_df.groupby('product_category_name_english')['review_score'].mean().nlargest(15)

        fig = px.bar(
            x=category_reviews.values,
            y=category_reviews.index,
            orientation='h',
            title="Top 15 Catégories par Satisfaction",
            labels={'x': 'Note moyenne', 'y': 'Catégorie'},
            color=category_reviews.values,
            color_continuous_scale='RdYlGn'
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Données de review non disponibles", xref="paper", yref="paper", x=0.5, y=0.5)
    return fig


def create_delivery_satisfaction_correlation(filtered_df):
    """Corrélation délai/satisfaction - CORRIGÉ sans trendline"""
    if 'review_score' in filtered_df.columns and 'delivery_days' in filtered_df.columns and not filtered_df[
        'review_score'].isna().all():
        delivery_review = filtered_df.groupby(filtered_df['delivery_days'].round()).agg({
            'review_score': 'mean',
            'order_id': 'count'
        }).reset_index()
        delivery_review = delivery_review[delivery_review['delivery_days'] <= 30]

        fig = px.scatter(
            delivery_review,
            x='delivery_days',
            y='review_score',
            size='order_id',
            title="Impact du Délai de Livraison sur la Satisfaction",
            labels={'delivery_days': 'Délai (jours)', 'review_score': 'Note moyenne', 'order_id': 'Nb commandes'}
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Données insuffisantes", xref="paper", yref="paper", x=0.5, y=0.5)
    return fig


def create_delivery_time_distribution(filtered_df):
    """Distribution des délais de livraison"""
    if 'delivery_days' in filtered_df.columns and not filtered_df['delivery_days'].isna().all():
        delivery_data = filtered_df['delivery_days'].dropna()
        delivery_data = delivery_data[delivery_data <= 60]  # Filtrer les outliers

        fig = px.histogram(
            delivery_data,
            nbins=30,
            title="Distribution des Délais de Livraison",
            labels={'value': 'Délai (jours)', 'count': 'Nombre de commandes'}
        )
        if len(delivery_data) > 0:
            fig.add_vline(x=delivery_data.mean(), line_dash="dash", line_color="red",
                          annotation_text=f"Moyenne: {delivery_data.mean():.1f} jours")
    else:
        fig = go.Figure()
        fig.add_annotation(text="Données de livraison non disponibles", xref="paper", yref="paper", x=0.5, y=0.5)
    return fig


def create_delivery_by_state(filtered_df):
    """Délai moyen par état"""
    if 'delivery_days' in filtered_df.columns and not filtered_df['delivery_days'].isna().all():
        state_delivery = filtered_df.groupby('customer_state')['delivery_days'].mean().nlargest(15)

        fig = px.bar(
            x=state_delivery.index,
            y=state_delivery.values,
            title="Délai Moyen de Livraison par État (Top 15)",
            labels={'x': 'État', 'y': 'Délai moyen (jours)'},
            color=state_delivery.values,
            color_continuous_scale='RdYlGn_r'
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Données de livraison non disponibles", xref="paper", yref="paper", x=0.5, y=0.5)
    return fig


def create_delivery_performance_map(filtered_df):
    """Performance de livraison par état"""
    if 'delivery_days' in filtered_df.columns and 'delivery_vs_estimate' in filtered_df.columns and not filtered_df[
        'delivery_days'].isna().all():
        state_performance = filtered_df.groupby('customer_state').agg({
            'delivery_days': 'mean',
            'delivery_vs_estimate': lambda x: (x <= 0).mean() * 100,
            'order_id': 'count'
        }).reset_index()
        state_performance.columns = ['État', 'Délai moyen', 'Taux à temps (%)', 'Nb commandes']
        state_performance = state_performance.sort_values('Taux à temps (%)', ascending=False).head(20)

        fig = px.scatter(
            state_performance,
            x='Délai moyen',
            y='Taux à temps (%)',
            size='Nb commandes',
            color='Taux à temps (%)',
            text='État',
            title="Performance de Livraison par État",
            color_continuous_scale='RdYlGn',
            size_max=30
        )
        fig.update_traces(textposition='top center')
    else:
        fig = go.Figure()
        fig.add_annotation(text="Données insuffisantes", xref="paper", yref="paper", x=0.5, y=0.5)
    return fig


# Callback pour les graphiques produits
@app.callback(
    Output('product-charts-container', 'children'),
    [Input('top-n-categories', 'value'),
     Input('date-filter', 'start_date'),
     Input('date-filter', 'end_date'),
     Input('status-filter', 'value')]
)
def update_product_charts(top_n, start_date, end_date, status_filter):
    """Met à jour les graphiques produits avec filtres"""

    # Filtrage des données
    filtered_df = df.copy()

    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['order_purchase_timestamp'] >= start_date) &
            (filtered_df['order_purchase_timestamp'] <= end_date)
            ]

    if status_filter and 'all' not in status_filter and isinstance(status_filter, list):
        filtered_df = filtered_df[filtered_df['order_status'].isin(status_filter)]

    # Top catégories par CA
    top_categories = filtered_df.groupby('product_category_name_english')['total_amount_fcfa'].sum().nlargest(top_n)

    fig1 = px.bar(
        x=top_categories.values / 1e6,
        y=top_categories.index,
        orientation='h',
        title=f"Top {top_n} Catégories par CA",
        labels={'x': 'CA (Millions FCFA)', 'y': 'Catégorie'},
        color=top_categories.values,
        color_continuous_scale= 'Viridis'
    )

    # Top catégories par nombre de commandes
    top_orders = filtered_df.groupby('product_category_name_english')['order_id'].nunique().nlargest(top_n)

    fig2 = px.bar(
        x=top_orders.values,
        y=top_orders.index,
        orientation='h',
        title=f"Top {top_n} Catégories par Nombre de Commandes",
        labels={'x': 'Nombre de commandes', 'y': 'Catégorie'},
        color=top_orders.values,
        color_continuous_scale='Blues'
    )

    # Tableau de performance
    category_stats = filtered_df.groupby('product_category_name_english').agg({
        'order_id': 'nunique',
        'total_amount_fcfa': 'sum',
        'review_score': 'mean' if 'review_score' in filtered_df.columns else lambda x: 0
    }).round(2)
    category_stats.columns = ['Commandes', 'CA Total', 'Note Moy']
    category_stats['CA Total'] = category_stats['CA Total'].apply(lambda x: f"{x / 1e6:.2f}M FCFA")
    category_stats = category_stats.nlargest(top_n, 'Commandes')

    table = dash_table.DataTable(
        data=category_stats.reset_index().to_dict('records'),
        columns=[
            {"name": "Catégorie", "id": "product_category_name_english"},
            {"name": "Commandes", "id": "Commandes"},
            {"name": "CA Total", "id": "CA Total"},
            {"name": "Note Moyenne", "id": "Note Moy"}
        ],
        style_cell={'textAlign': 'center'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
        ],
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
    )

    return html.Div([
        html.Div([dcc.Graph(figure=fig1)], style={'width': '49%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=fig2)], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H3("Performance par Catégorie", style={'textAlign': 'center', 'marginTop': '30px'}),
            table
        ], style={'width': '100%', 'marginTop': '20px'})
    ])


# CSS personnalisé
app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    margin: 0;
                    background-color: #f5f6fa;
                }
                .tab {
                    padding: 12px 20px !important;
                    font-weight: 500;
                }
                .tab--selected {
                    background-color: #3498db !important;
                    color: white !important;
                }
                h1, h2, h3 {
                    font-weight: 600;
                }
                .dash-table-container {
                    margin: 20px auto;
                    max-width: 90%;
                }
                .DateRangePickerInput {
                    background-color: white !important;
                    border-radius: 5px;
                }
                .Select-control {
                    border-radius: 5px !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)