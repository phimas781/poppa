import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Configuration
st.set_page_config(
    page_title="Gwamz Hit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
    }
    .stMetric {
        background-color: #1a1a1a;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_model():
    return joblib.load('gwamz_streams_predictor_v2.joblib')

@st.cache_data
def load_data():
    df = pd.read_csv('gwamz_data.csv', parse_dates=['release_date'])
    df['days_since_release'] = (datetime.now() - df['release_date']).dt.days
    return df

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Initialization Error: {str(e)}")
    st.stop()

# Sidebar inputs
with st.sidebar:
    st.title("Track Parameters")
    release_date = st.date_input("Release Date", datetime.today())
    album_type = st.selectbox("Album Type", ['single', 'album'])
    total_tracks = st.slider("Total Tracks", 1, 20, 1 if album_type == 'single' else 10)
    track_number = st.slider("Track Number", 1, total_tracks, 1)
    markets = st.slider("Available Markets", 1, 200, 185)
    explicit = st.toggle("Explicit Content", True)
    is_remix = st.toggle("Is Remix/Edit/Sped Up", False)
    
    if st.button("Predict Streams", type="primary"):
        st.session_state.predict = True

# Main content
st.title("🎤 Gwamz Hit Predictor")
st.markdown("Predict the streaming performance of new tracks")

# Prediction logic
if 'predict' in st.session_state:
    input_data = {
        'artist_followers': 7937,
        'artist_popularity': 41,
        'album_type': album_type,
        'release_year': release_date.year,
        'total_tracks_in_album': total_tracks,
        'available_markets_count': markets,
        'track_number': track_number,
        'disc_number': 1,
        'explicit': explicit,
        'days_since_release': 0,  # New release
        'is_remix': int(is_remix),
        'is_single': int(album_type == 'single')
    }
    
    try:
        prediction = model.predict(pd.DataFrame([input_data]))[0]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Streams", f"{int(prediction):,}")
        with col2:
            st.metric("Compared to Average", 
                     f"{(prediction/df['streams'].mean()-1)*100:+.1f}%",
                     delta_color="off")
        with col3:
            st.metric("Expected Revenue", 
                     f"${prediction*0.003:,.0f}",
                     "Est. at $0.003 per stream")
        
        # Performance analysis
        st.subheader("Performance Analysis")
        if prediction > 2000000:
            st.success("🔥 Mega Hit Potential (Top 5% of tracks)")
        elif prediction > 1000000:
            st.success("🎵 Strong Hit Potential")
        elif prediction > 500000:
            st.info("👍 Solid Performer")
        else:
            st.warning("💡 Needs Marketing Boost")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Data visualization
st.divider()
st.subheader("Historical Performance")

tab1, tab2 = st.tabs(["Stream Trends", "Track Analysis"])
with tab1:
    fig, ax = plt.subplots(figsize=(12, 4))
    df.set_index('release_date')['streams'].plot(ax=ax, style='o-', color='#1DB954')
    plt.title("Streaming Performance Over Time", pad=20)
    plt.xlabel("Release Date")
    plt.ylabel("Streams (millions)")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x/1e6:.1f}M")
    st.pyplot(fig)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df.groupby('explicit')['streams'].mean(), color='#1DB954')
    with col2:
        st.bar_chart(df.groupby('album_type')['streams'].mean(), color='#1DB954')

# Data explorer
with st.expander("📊 Full Dataset"):
    st.dataframe(df.sort_values('release_date', ascending=False), 
                 height=300,
                 column_config={
                     "streams": st.column_config.NumberColumn(format="%,d")
                 })

# Footer
st.divider()
st.caption("© 2023 Gwamz Analytics | Model v2.0")
