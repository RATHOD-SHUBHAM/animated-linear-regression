"""Custom CSS for the Streamlit app."""

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(160deg, #0e1117 0%, #1a1f2e 50%, #0e1117 100%);
    }
    [data-testid="stSidebar"] {
        background-color: #161b26;
        border-right: 1px solid #2a3142;
    }
    [data-testid="stMetric"] {
        background-color: #1c2333;
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid #2a3142;
    }
    [data-testid="stMetricLabel"] {
        color: #41BEE9 !important;
    }
    h1, h2, h3 {
        color: #41BEE9 !important;
    }
    .hero-text {
        color: #b0b8c4;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stTabs"] button {
        color: #41BEE9;
    }
    .live-stats {
        display: flex;
        gap: 2rem;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        background-color: #1c2333;
        border-radius: 10px;
        border: 1px solid #2a3142;
        color: #b0b8c4;
        font-size: 0.95rem;
    }
    .live-stats b {
        color: #41BEE9;
        font-weight: 600;
    }
</style>
"""
