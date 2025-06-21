import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import traceback

# Configure page first
st.set_page_config(
    page_title="Gwamz Analytics",
    page_icon="🎵",
    layout="wide"
)

# Error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            st.code(traceback.format_exc())
            st.stop()
    return wrapper

@handle_errors
@st.cache_resource
def load_model():
    model = joblib.load('gwamz_streams_predictor_v2.joblib')
    # Verify model structure
    if not hasattr(model, 'predict'):
        raise ValueError("Loaded object is not a valid scikit-learn model")
    return model

@handle_errors
@st.cache_data
def load_data():
    df = pd.read_csv('gwamz_data.csv', parse_dates=['release_date'])
    df['days_since_release'] = (datetime.now() - df['release_date']).dt.days
    return df

# Main app
def main():
    st.title("Gwamz Song Performance Predictor")
    
    # Load resources
    try:
        model = load_model()
        df = load_data()
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return

    # Sidebar inputs
    with st.sidebar:
        st.header("Track Parameters")
        release_date = st.date_input("Release Date", datetime.today())
        album_type = st.selectbox("Album Type", ['single', 'album'])
        total_tracks = st.slider("Total Tracks", 1, 20, 1 if album_type == 'single' else 10)
        track_number = st.slider("Track Number", 1, total_tracks, 1)
        markets = st.slider("Available Markets", 1, 200, 185)
        explicit = st.checkbox("Explicit Content", True)
        is_remix = st.checkbox("Is Remix/Edit/Sped Up", False)
        
        if st.button("Predict Streams"):
            try:
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
                    'days_since_release': 0,
                    'is_remix': int(is_remix),
                    'is_single': int(album_type == 'single')
                }
                
                prediction = model.predict(pd.DataFrame([input_data]))[0]
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Streams", f"{int(prediction):,}")
                with col2:
                    avg = df['streams'].mean()
                    diff = (prediction - avg) / avg * 100
                    st.metric("Vs Average", f"{diff:+.1f}%")
                
                # Visualization
                fig, ax = plt.subplots()
                df['streams'].plot(kind='hist', ax=ax, bins=20)
                ax.axvline(prediction, color='red', linestyle='--')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
