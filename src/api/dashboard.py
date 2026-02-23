"""
TimeStone AI - Interactive Dashboard

Streamlit dashboard for visualizing transformation scenarios
and Monte Carlo simulation results.
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# Page config
st.set_page_config(
    page_title="TimeStone AI - Transformation Simulator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load simulation data"""
    try:
        # Load scenarios
        with open("ktz_scenarios.json", 'r') as f:
            scenarios_data = json.load(f)
        
        # Load simulation results
        with open("ktz_simulation_results.json", 'r') as f:
            results_data = json.load(f)
        
        # Load digital twin
        with open("ktz_digital_twin.json", 'r') as f:
            twin_data = json.load(f)
        
        return scenarios_data, results_data, twin_data
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None, None


def main():
    # Header
    st.markdown('<h1 class="main-header">🔮 TimeStone AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict business transformation outcomes before you commit</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    scenarios_data, results_data, twin_data = load_data()
    
    if not all([scenarios_data, results_data, twin_data]):
        st.error("Failed to load data. Please ensure all JSON files exist.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=TimeStone+AI", use_container_width=True)
        st.markdown("### Company Profile")
        st.info(f"""
        **{scenarios_data['company']}**
        
        Industry: {scenarios_data['industry']}
        
        Scenarios Analyzed: {scenarios_data['total_scenarios']}
        
        Simulations Run: {results_data['simulation_parameters']['iterations'] * len(results_data['results']):,}
        """)
        
        st.markdown("### Navigation")
        page = st.radio(
            "Select View:",
            ["📊 Overview", "🎯 Top Scenarios", "📈 All Results", "🔬 Simulation Details"]
        )
    
    # Main content
    if page == "📊 Overview":
        show_overview(scenarios_data, results_data, twin_data)
    elif page == "🎯 Top Scenarios":
        show_top_scenarios(results_data)
    elif page == "📈 All Results":
        show_all_results(results_data)
    elif page == "🔬 Simulation Details":
        show_simulation_details(results_data)


def show_overview(scenarios_data, results_data, twin_data):
    """Show overview dashboard"""
    st.header("Executive Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    results = results_data['results']
    top_scenario = max(results, key=lambda x: x['success_probability'])
    avg_roi = sum(r['mean_roi'] for r in results) / len(results)
    high_confidence = sum(1 for r in results if r['success_probability'] > 0.8)
    
    with col1:
        st.metric(
            "Best Success Probability",
            f"{top_scenario['success_probability']:.1%}",
            delta="Scenario #" + str(top_scenario['scenario_id'])
        )
    
    with col2:
        st.metric(
            "Average ROI",
            f"{avg_roi:.0%}",
            delta=f"{len(results)} scenarios"
        )
    
    with col3:
        st.metric(
            "High Confidence Scenarios",
            high_confidence,
            delta=f"{high_confidence/len(results):.0%} of total"
        )
    
    with col4:
        st.metric(
            "Total Simulations",
            f"{results_data['simulation_parameters']['iterations'] * len(results):,}",
            delta="Monte Carlo"
        )
    
    st.markdown("---")
    
    # ROI Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 ROI Distribution")
        
        roi_data = pd.DataFrame([
            {
                'Scenario': r['scenario_name'][:30] + '...' if len(r['scenario_name']) > 30 else r['scenario_name'],
                'Mean ROI': r['mean_roi'] * 100,
                'Success Probability': r['success_probability']
            }
            for r in results[:20]  # Top 20
        ])
        
        fig = px.scatter(
            roi_data,
            x='Success Probability',
            y='Mean ROI',
            size='Mean ROI',
            color='Success Probability',
            hover_data=['Scenario'],
            title="Success Probability vs ROI",
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Top 10 by Success Probability")
        
        top_10 = sorted(results, key=lambda x: x['success_probability'], reverse=True)[:10]
        
        top_df = pd.DataFrame([
            {
                'Scenario': r['scenario_name'][:40],
                'Success %': r['success_probability'] * 100,
                'Mean ROI %': r['mean_roi'] * 100
            }
            for r in top_10
        ])
        
        fig = px.bar(
            top_df,
            x='Success %',
            y='Scenario',
            orientation='h',
            color='Mean ROI %',
            title="Success Probability (%)",
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def show_top_scenarios(results_data):
    """Show top 3 scenarios in detail"""
    st.header("🏆 TOP-3 Recommended Scenarios")
    
    results = results_data['results']
    top_3 = sorted(results, key=lambda x: x['success_probability'], reverse=True)[:3]
    
    for i, scenario in enumerate(top_3, 1):
        with st.expander(f"🏆 RANK #{i}: {scenario['scenario_name']}", expanded=(i==1)):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Success Probability", f"{scenario['success_probability']:.1%}")
                st.metric("Mean ROI", f"{scenario['mean_roi']:.1%}")
            
            with col2:
                st.metric("Median ROI", f"{scenario['median_roi']:.1%}")
                st.metric("Risk Score", f"{scenario['risk_score']:.3f}")
            
            with col3:
                st.metric("90% CI Lower", f"{scenario['confidence_90_lower']:.1%}")
                st.metric("90% CI Upper", f"{scenario['confidence_90_upper']:.1%}")
            
            # Confidence interval visualization
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                x=[scenario['mean_roi'] * 100],
                name='ROI Distribution',
                marker_color='#667eea',
                boxmean='sd'
            ))
            
            fig.update_layout(
                title=f"ROI Distribution for {scenario['scenario_name']}",
                xaxis_title="ROI (%)",
                height=200,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            if scenario['success_probability'] >= 0.9:
                st.success("✅ **STRONG RECOMMENDATION**: High success probability with excellent ROI potential")
            elif scenario['success_probability'] >= 0.7:
                st.info("⚠️ **PROCEED WITH CAUTION**: Good potential but requires risk management")
            else:
                st.warning("🔴 **PILOT FIRST**: Moderate success probability, recommend small-scale testing")


def show_all_results(results_data):
    """Show all simulation results"""
    st.header("📈 All Simulation Results")
    
    results = results_data['results']
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'ID': r['scenario_id'],
            'Scenario': r['scenario_name'],
            'Success %': f"{r['success_probability']*100:.1f}",
            'Mean ROI %': f"{r['mean_roi']*100:.1f}",
            'Median ROI %': f"{r['median_roi']*100:.1f}",
            '90% CI Lower': f"{r['confidence_90_lower']*100:.1f}",
            '90% CI Upper': f"{r['confidence_90_upper']*100:.1f}",
            'Risk Score': f"{r['risk_score']:.3f}"
        }
        for r in results
    ])
    
    st.dataframe(df, use_container_width=True, height=600)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="timestone_simulation_results.csv",
        mime="text/csv"
    )


def show_simulation_details(results_data):
    """Show simulation methodology details"""
    st.header("🔬 Simulation Methodology")
    
    params = results_data['simulation_parameters']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monte Carlo Parameters")
        st.code(f"""
Iterations per Scenario: {params['iterations']:,}
Total Scenarios: {params['total_scenarios']}
Total Simulations: {params['iterations'] * params['total_scenarios']:,}
Random Seed: {params['random_seed']}
        """)
        
        st.markdown("""
        ### Methodology
        
        TimeStone AI uses **Monte Carlo simulation** to predict transformation outcomes:
        
        1. **Digital Twin Creation**: Synthetic model of company
        2. **Scenario Generation**: 50 transformation hypotheses
        3. **Probabilistic Simulation**: 1,000 iterations per scenario
        4. **Statistical Analysis**: Confidence intervals, risk scores
        5. **Ranking**: TOP-3 by success probability
        """)
    
    with col2:
        st.subheader("Risk Factors Modeled")
        st.markdown("""
        - Revenue impact variance (±30%)
        - Cost reduction uncertainty (±20%)
        - Implementation delays (0-50%)
        - Budget overruns (0-30%)
        - Market response volatility
        - Execution capability constraints
        """)
        
        st.subheader("Success Criteria")
        st.markdown("""
        **High Confidence (>90%)**
        - Low risk
        - Clear ROI path
        - Proven technology
        
        **Medium Confidence (70-90%)**
        - Moderate risk
        - Good ROI potential
        - Some unknowns
        
        **Pilot Recommended (<70%)**
        - Higher risk
        - Test before scale
        - Validate assumptions
        """)


if __name__ == "__main__":
    main()
