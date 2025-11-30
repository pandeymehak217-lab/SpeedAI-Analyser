import streamlit as st
import pandas as pd
import json
import os
import glob
from pathlib import Path

# --- Configuration & Setup ---
# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Realtime Analyser by Mehak", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- CONSTANTS ---
OUTPUT_DIR = Path("output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Custom Styling (Advanced Dark Theme & Animations) ---
st.markdown("""
<style>
/* 1. ADVANCED DARK THEME BACKGROUND */
.main {
    /* Subtle gradient from a deep blue to a dark purple/black */
    background: linear-gradient(to bottom, #111827, #0A0A0A);
    color: #F9FAFB; /* Light text color */
}

/* ------------------ KEYFRAMES FOR ANIMATION ------------------ */

/* Enhanced Shimmer/Glow Effect for Title (Neon Green Glow) */
@keyframes shimmer {
    0% {text-shadow: 0 0 10px rgba(0, 255, 255, 0.2);}
    50% {text-shadow: 0 0 20px #00FFFF, 0 0 30px rgba(0, 255, 255, 0.7);}
    100% {text-shadow: 0 0 10px rgba(0, 255, 255, 0.2);}
}

/* Subtle Pulse Effect for Metrics */
@keyframes pulse {
    0% {transform: scale(1); box-shadow: 0 0 0 rgba(255, 255, 255, 0.0);}
    50% {transform: scale(1.02); box-shadow: 0 0 15px rgba(0, 255, 255, 0.4);} /* Teal glow */
    100% {transform: scale(1); box-shadow: 0 0 0 rgba(255, 255, 255, 0.0);}
}

/* -------------------- STYLING CLASSES -------------------- */

/* 2. MAIN APP TITLE (Teal/Cyan, Huge, Extra Bold, Animated) */
.main-header-mehek {
    font-size: 100px; /* Even bigger */
    font-weight: 900;
    color: #00FFFF; /* Bright Cyan/Teal */
    text-align: center;
    margin-bottom: 30px;
    padding-top: 20px;
    /* Apply Enhanced Shimmer/Glow Animation */
    animation: shimmer 4s infinite cubic-bezier(0.4, 0, 0.6, 1);
    letter-spacing: 5px;
    text-transform: uppercase;
}

/* 3. ANALYSIS ID TEXT (Bright Green, Strong Contrast) */
.analysis-id-text {
    font-size: 1.5em;
    color: #39FF14; /* Neon Green */
    font-weight: 600;
    text-align: center;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(57, 255, 20, 0.3);
}

/* 4. DETAILED SPEED ANALYSIS HEADER (White Text, Orange Underline) */
.detailed-analysis-header {
    border-bottom: 3px solid #FF9900; /* Bright Orange border */
    padding-bottom: 10px;
    margin-top: 40px;
    color: #F9FAFB; /* Off-White text */
    font-weight: 800;
    font-size: 2.5rem;
    letter-spacing: 1px;
}

/* 5. METRIC VALUES (White/Teal, Bold, Animated Pulse) */
div[data-testid="stMetricValue"] {
    font-size: 36px; /* Bigger value */
    font-weight: 800;
    color: #00FFFF; /* Bright Teal */
    /* Applied to the surrounding container for the card effect */
}
div[data-testid="stMetricLabel"] {
    font-size: 16px;
    color: #FFD700; /* Gold for labels */
    font-weight: 500;
}

/* 6. STYLE METRIC CONTAINERS (Card Effect with Subtle Border and Shadow) */
div[data-testid^="stMetric"] {
    background-color: #1F2937; /* Darker background for the card */
    border-radius: 12px;
    padding: 20px;
    margin-top: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    border: 1px solid #374151;
    /* Apply Pulse Animation to the entire card */
    animation: pulse 2s infinite ease-in-out;
}

/* 7. Enhance Dataframe Look */
.stDataFrame {
    border: 1px solid #374151;
    border-radius: 8px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
}

/* Global H2 Styling */
h2 {
    color: #F9FAFB;
    font-weight: 700;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# --- Function to load the LATEST analysis file ---
def load_latest_analysis_data():
    """Finds the latest JSON analysis file and loads its data."""
    list_of_files = glob.glob(str(OUTPUT_DIR / '*_results.json'))
    
    if not list_of_files:
        return None, "No analysis results found. Please run the analysis_core.py script first."

    latest_file = max(list_of_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        file_id = Path(latest_file).stem.replace('_results', '')
        return data, file_id
    except Exception as e:
        return None, f"Error loading analysis file: {e}"


def display_dashboard_results(data, file_id):
    """Displays the dashboard layout with data, using new CSS classes."""
    
    # --- Advanced Title ---
    st.markdown("<p class='main-header-mehek'>Realtime Analyser by Mehak</p>", unsafe_allow_html=True)
    
    # --- Analysis ID Text ---
    st.markdown(f"<p class='analysis-id-text'>-- Latest Analysis ID: <span>{file_id}</span> --</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 1. Processed Video Player ---
    st.header("🎥 Processed Video Output")
    
    # Use a container for the video to give it a clean border/card look
    with st.container(border=True):
        processed_video_path = OUTPUT_DIR / f"{file_id}_processed_video.mp4"
        if processed_video_path.exists():
            st.video(str(processed_video_path))
        else:
            # Use a placeholder for the missing video for better visual structure
            st.markdown(f"""
                <div style="background-color: #1F2937; padding: 40px; border-radius: 8px; text-align: center; border: 2px dashed #4B5563;">
                    <h3 style="color: #FFD700;">Video File Not Found</h3>
                    <p style="color: #9CA3AF;">Expected file: {file_id}_processed_video.mp4</p>
                    <p style="color: #9CA3AF;">Please ensure the analysis script has completed successfully.</p>
                </div>
            """, unsafe_allow_html=True)

    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 2. Key Metrics (Advanced Cards) ---
    st.header("📊 Key Detection Metrics")
    
    if 'total_objects_per_class' in data and 'all_tracked_objects' in data:
        
        total_unique_vehicles = len(data.get('all_tracked_objects', []))
        frames_processed = data.get('metadata', {}).get('total_frames', 'N/A')
        video_fps = data.get('metadata', {}).get('video_fps', 'N/A')
        time_taken = data.get('metadata', {}).get('analysis_time_seconds', 'N/A')

        # Calculate FPS/Time Metric for a cool 4th card
        if isinstance(frames_processed, int) and isinstance(time_taken, (int, float)) and time_taken > 0:
            processing_fps = round(frames_processed / time_taken, 2)
        else:
            processing_fps = 'N/A'


        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Unique Vehicles", total_unique_vehicles, delta_color="normal")
        with col2:
            st.metric("Frames Processed", frames_processed, delta_color="normal")
        with col3:
            st.metric("Video FPS (Source)", round(video_fps, 2) if isinstance(video_fps, (int, float)) else 'N/A', delta_color="normal")
        with col4:
            st.metric("Processing FPS (App)", processing_fps, delta_color="normal")

        st.markdown("---")

    # --- 3. Detailed Data Table ---
    if 'all_tracked_objects' in data:
        # --- Enhanced Detailed Results Header ---
        st.markdown("<h2 class='detailed-analysis-header'>Detailed Speed Analysis Log</h2>", unsafe_allow_html=True)
        
        df_log = pd.DataFrame(data['all_tracked_objects'])
        
        display_columns = ['track_id', 'class_name', 'avg_speed_kph', 'max_speed_kph', 'total_frames_tracked']
        
        if not df_log.empty:
            # Conditional formatting for the dataframe (High speed detection highlighting)
            def highlight_speed(s):
                if s.name in ['Avg Speed (km/h)', 'Max Speed (km/h)']:
                    # Example threshold: highlight speeds over 80 kph
                    return [f'background-color: {"#8B0000" if v > 80 else ""}' for v in s]
                return [''] * len(s)

            df_display = df_log[display_columns].round(2)
            df_display.columns = ['Vehicle ID', 'Class', 'Avg Speed (km/h)', 'Max Speed (km/h)', 'Frames Tracked']
            
            st.dataframe(
                df_display.style.apply(highlight_speed, axis=0), 
                use_container_width=True, 
                hide_index=True
            )
            
            # Additional detail for advanced data presentation
            st.caption(f"Showing details for {len(df_display)} unique tracked objects. Max Speed highlighting is applied for speeds over 80 km/h.")
        else:
            st.info("No detailed tracking data available.")


# --- Main Application Logic ---
if __name__ == "__main__":
    
    analysis_data, identifier = load_latest_analysis_data()
    
    if analysis_data:
        display_dashboard_results(analysis_data, identifier)
        st.info("To see new results, run the analysis script again: `python analysis_core.py --video <path>`")
    else:
        # Display the main title even if there's an error, for better visual consistency
        st.markdown("<p class='main-header-mehek'>Realtime Analyser by Mehek</p>", unsafe_allow_html=True)
        st.error(f"Analysis Error: {identifier}")
        st.warning("Action needed: Please run the analysis script first to generate data in the 'output' folder.")