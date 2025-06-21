import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import traceback

# Configure page
st.set_page_config(
    page_title="Gwamz Analytics",
    page_icon="🎵",
    layout="wide"
)

# Error handler
def show_error(e):
    st.error("An error occurred")
    with st.expander("Error Details"):
        st.code(f"{type(e).__name__}: {str(e)}")
        st.code(traceback.format_exc())
    st.stop()

# Load resources with verification
def load_model():
    try:
        model = joblib.load('gwamz_streams_predictor_v2.joblib')
        
        # Verify it's a working model
        test_input = pd.DataFrame([{
            'artist_followers': 7937,
            'artist_popularity': 41,
            'album_type': 'single',
            'release_year': 2023,
            'total_tracks_in_album': 1,
            'available_markets_count': 185,
            'track_number': 1,
            'disc_number': 1,
            'explicit': True,
            'days_since_release': 30,
            'is_remix': 0,
            'is_single': 1
        }])
        
        try:
            model.predict(test_input)
        except Exception as e:
            raise ValueError("Model exists but predictions fail") from e
            
        return model
    except FileNotFoundError:
        raise FileNotFoundError("Model file not found. Please ensure gwamz_streams_predictor_v2.joblib is in the same folder")
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}") from e

def load_data():
    try:
        df = pd.read_csv('gwamz_data.csv', parse_dates=['release_date'])
        
        # Verify required columns exist
        required_cols = {'artist_followers', 'artist_popularity', 'album_type', 
                        'release_date', 'streams'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
            
        df['days_since_release'] = (datetime.now() - df['release_date']).dt.days
        return df
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {str(e)}") from e

def main():
    st.title("🎵 Gwamz Song Performance Predictor")
    
    # Load resources
    try:
        with st.spinner("Loading model..."):
            model = load_model()
        with st.spinner("Loading data..."):
            df = load_data()
    except Exception as e:
        show_error(e)
    
    # Input sidebar
    with st.sidebar:
        st.header("Track Parameters")
        release_date = st.date_input("Release Date", datetime.today())
        album_type = st.selectbox("Album Type", ['single', 'album'])
        total_tracks = st.slider("Total Tracks", 1, 20, 1 if album_type == 'single' else 10)
        track_number = st.slider("Track Number", 1, total_tracks, 1)
        markets = st.slider("Available Markets", 1, 200, 185)
        explicit = st.checkbox("Explicit Content", True)
        is_remix = st.checkbox("Is Remix/Edit/Sped Up", False)
        
        predict_clicked = st.button("Predict Streams", type="primary")
    
    # Prediction logic
    if predict_clicked:
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
            
            with st.spinner("Making prediction..."):
                prediction = model.predict(pd.DataFrame([input_data]))[0]
            
            # Display results
            col1, col2 = st.columns(2)
            col1.metric("Predicted Streams", f"{int(prediction):,}")
            
            avg_streams = df['streams'].mean()
            col2.metric("Compared to Average", 
                       f"{(prediction/avg_streams-1)*100:+.1f}%",
                       delta_color="off")
            
            # Visualization
            st.subheader("Performance Context")
            fig, ax = plt.subplots(figsize=(10, 4))
            df['streams'].plot(kind='hist', bins=20, ax=ax, alpha=0.7)
            ax.axvline(prediction, color='red', linestyle='--', label='Prediction')
            ax.set_xlabel("Streams")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            show_error(e)

if __name__ == "__main__":
    main()
