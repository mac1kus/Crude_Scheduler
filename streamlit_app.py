import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import re
import math

# Try to import Excel support libraries
try:
    import openpyxl
    EXCEL_SUPPORT = True
except ImportError:
    EXCEL_SUPPORT = False

# Page config
st.set_page_config(
    page_title="Tank Simulation Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .tank-card {
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# State colors
STATE_COLORS = {
    'READY': '#10b981',      # Green
    'FEEDING': '#3b82f6',    # Blue
    'EMPTY': '#ef4444',      # Red
    'FILLING': '#f59e0b',    # Orange
    'FILLED': '#8b5cf6',     # Purple
    'SETTLING': '#eab308',   # Yellow
    'LAB': '#06b6d4',        # Cyan
    'SUSPENDED': '#6b7280',  # Gray
    'MAINTENANCE': '#ec4899', # Pink
    'CLEANING': '#14b8a6',   # Teal
    'RESERVED': '#a855f7'    # Purple
}

# Helper function to read CSV with multiple encoding attempts
def safe_read_csv(filepath, **kwargs):
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            return pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', **kwargs)
        except Exception:
            continue
    # Last resort - try without encoding specification
    try:
        return pd.read_csv(filepath, on_bad_lines='skip', **kwargs)
    except Exception as e:
        st.error(f"Failed to read {os.path.basename(filepath)}: {e}")
        return None

def load_data(folder_path):
    """Load simulation data from CSV and Excel files"""
    crude_mix = {}
    processing_rate_html = None
    try:
        files = os.listdir(folder_path)
    except Exception as e:
        st.error(f"Cannot access folder: {e}")
        return None, None, None, None, {}, None
    
    # Get most recent files (support both CSV and Excel for snapshots)
    log_files = [f for f in files if f.startswith('simulation_log_') and f.endswith('.csv')]
    summary_files = [f for f in files if f.startswith('daily_summary_') and f.endswith('.csv')]
    cargo_files = [f for f in files if f.startswith('cargo_report_') and f.endswith('.csv')]
    
    # Support both CSV and Excel for snapshots (more flexible naming)
    snapshot_files = [f for f in files if 
                     ('snapshot' in f.lower() or 'tank_snapshot' in f.lower()) and 
                     (f.endswith('.csv') or f.endswith('.xlsx') or f.endswith('.xls'))]
    
    if not log_files:
        st.warning("No simulation log files found. Looking for alternative data sources...")
        # If no log files but snapshot files exist, we can still show tank volumes
        if not snapshot_files:
            return None, None, None, None, {}, None
    
    # Get most recent files
    log_file = sorted(log_files)[-1] if log_files else None
    summary_file = sorted(summary_files)[-1] if summary_files else None
    cargo_file = sorted(cargo_files)[-1] if cargo_files else None
    snapshot_file = sorted(snapshot_files)[-1] if snapshot_files else None
    
    # Load log data
    log_df = None
    if log_file:
        log_df = safe_read_csv(os.path.join(folder_path, log_file))
        if log_df is not None:
            try:
                log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'], format='%d/%m/%Y %H:%M', errors='coerce')
            except Exception:
                try:
                    log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'], errors='coerce')
                except Exception as e:
                    st.error(f"Timestamp parsing error: {e}")
            
            # Sort log_df chronologically by timestamp
            if 'Timestamp' in log_df.columns:
                log_df = log_df.sort_values('Timestamp', ascending=True).reset_index(drop=True)
            
            # Parse processing rate from SIM_START event
            sim_start = log_df[log_df['Event'] == 'SIM_START']
            if not sim_start.empty:
                message = sim_start.iloc[0]['Message']
                import re
                rate_match = re.search(r'processing rate:\s*([\d,]+)', str(message))
                if rate_match:
                    try:
                        processing_rate_html = float(rate_match.group(1).replace(',', ''))
                    except:
                        pass
            
            # Parse crude mix from READY_1 events
            ready_events = log_df[log_df['Event'] == 'READY_1']
            if not ready_events.empty:
                # Get the first READY_1 message which has the mix
                message = ready_events.iloc[0]['Message']
                
                # Parse: "Tank X now READY - Mix: [Crude1: 50.0%, Crude2: 25.0%]"
                import re
                match = re.search(r'Mix: \[(.*?)\]', str(message))
                if match:
                    mix_str = match.group(1)
                    # Split by commas and parse each crude
                    for item in mix_str.split(','):
                        item = item.strip()
                        # Parse "Crude Name: 50.0%"
                        parts = item.split(':')
                        if len(parts) == 2:
                            crude_name = parts[0].strip()
                            percentage_str = parts[1].strip().replace('%', '')
                            try:
                                crude_mix[crude_name] = float(percentage_str)
                            except:
                                pass
                    
                    if crude_mix:
                        st.success(f"✅ Loaded crude mix from simulation log: {len(crude_mix)} crude types")
    
    # Load summary data
    summary_df = None
    if summary_file:
        summary_df = safe_read_csv(os.path.join(folder_path, summary_file))
        if summary_df is not None:
            try:
                summary_df['Date'] = pd.to_datetime(summary_df['Date'], format='%d/%m/%Y', errors='coerce')
            except Exception:
                try:
                    summary_df['Date'] = pd.to_datetime(summary_df['Date'], errors='coerce')
                except Exception:
                    pass
    
    # Load cargo data
    cargo_df = None
    if cargo_file:
        cargo_df = safe_read_csv(os.path.join(folder_path, cargo_file))
    
    # Load snapshot data (supports both CSV and Excel) - HORIZONTAL FORMAT
    snapshot_df = None
    if snapshot_file:
        file_path = os.path.join(folder_path, snapshot_file)
        
        # Check if it's an Excel file
        if snapshot_file.endswith(('.xlsx', '.xls')):
            try:
                # Try reading Excel file
                if EXCEL_SUPPORT:
                    snapshot_df = pd.read_excel(file_path, engine='openpyxl' if snapshot_file.endswith('.xlsx') else None)
                else:
                    # Try to install openpyxl on the fly
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openpyxl'])
                    import openpyxl
                    snapshot_df = pd.read_excel(file_path, engine='openpyxl')
                
                st.success(f"✅ Loaded snapshot data from Excel: {snapshot_file}")
            except Exception as e:
                st.error(f"Error reading Excel snapshot file: {e}")
                st.info("Try saving the Excel file as CSV format for better compatibility.")
        else:
            # It's a CSV file
            snapshot_df = safe_read_csv(file_path)
            if snapshot_df is not None:
                st.success(f"✅ Loaded snapshot data from CSV: {snapshot_file}")
        
        # IMPORTANT: Detect horizontal format
        if snapshot_df is not None:
            # Horizontal format detection:
            # First column should be timestamp
            # Subsequent columns are tank data (volumes then statuses)
            first_col = snapshot_df.columns[0]
            
            try:
                # Try to parse first column as datetime
                timestamps = pd.to_datetime(snapshot_df[first_col], format='%d/%m/%Y %H:%M', errors='coerce')
                if timestamps.isna().all():
                    timestamps = pd.to_datetime(snapshot_df[first_col], errors='coerce')
                
                if timestamps.notna().sum() > 0:
                    st.info(f"📊 Detected HORIZONTAL snapshot format: {len(snapshot_df)} time points, {len(snapshot_df.columns)-1} data columns")
                    # Store parsed timestamps
                    snapshot_df['_Timestamp'] = timestamps
            except Exception as e:
                st.warning(f"Could not parse timestamps from first column: {e}")
    
    # Load crude mix from simulation log READY_1 events
    
    return log_df, summary_df, cargo_df, snapshot_df, crude_mix, processing_rate_html

def detect_number_of_tanks(log_df, snapshot_df):
    """Dynamically detect the number of tanks from log and snapshot data"""
    max_tank = 0
    detected_tanks = set()
    
    # Check log_df for tank columns
    if log_df is not None:
        tank_cols = [col for col in log_df.columns if col.startswith('Tank') and col[4:].replace('_Volume', '').isdigit()]
        for col in tank_cols:
            # Handle both Tank17 and Tank17_Volume formats
            tank_num_str = col[4:].replace('_Volume', '').replace('_Status', '')
            if tank_num_str.isdigit():
                tank_num = int(tank_num_str)
                detected_tanks.add(tank_num)
                max_tank = max(max_tank, tank_num)
    
    # Check snapshot_df - HORIZONTAL FORMAT
    if snapshot_df is not None:
        # In horizontal format, we need to count how many numeric columns exist
        # Columns after timestamp are: Tank1_vol, Tank2_vol, ..., TankN_vol, Tank1_status, ...
        # We need to identify where volumes end and statuses begin
        
        first_col = snapshot_df.columns[0]
        num_cols = len(snapshot_df.columns) - 1  # Exclude timestamp column
        
        # Sample first data row to determine tank count
        if len(snapshot_df) > 0:
            first_row = snapshot_df.iloc[0]
            volume_count = 0
            
            # Count how many numeric columns we have (these are volumes)
            for i in range(1, len(snapshot_df.columns)):
                col_name = snapshot_df.columns[i]
                value = first_row[col_name]
                
                if pd.notna(value):
                    try:
                        # Try to convert to float
                        float(str(value).replace(',', ''))
                        volume_count += 1
                    except:
                        # If it's not a number, we've reached status columns
                        break
            
            max_tank = max(max_tank, volume_count)
            st.sidebar.info(f"Detected {volume_count} tank volumes in snapshot data")
    
    # Return detected tanks (no default fallback)
    return max_tank

def get_tank_status(log_df, snapshot_df, timestamp, num_tanks=None):
    """Get tank status at specific timestamp - READS FROM HORIZONTAL SNAPSHOT FORMAT"""
    # Auto-detect number of tanks if not specified
    if num_tanks is None:
        num_tanks = detect_number_of_tanks(log_df, snapshot_df)
    
    tank_status = {}
    
    # Method 1: Try to get status from log_df
    if log_df is not None and not log_df.empty:
        filtered = log_df[log_df['Timestamp'] <= timestamp].copy()
        if not filtered.empty:
            latest_row = filtered.iloc[-1]
            
            # Dynamically find all tank columns in the data
            tank_cols = [col for col in latest_row.index if col.startswith('Tank') and not col.endswith('_Volume')]
            
            for col in tank_cols:
                tank_num_str = col[4:].replace('_Status', '')
                if tank_num_str.isdigit():
                    tank_id = int(tank_num_str)
                    status = latest_row[col]
                    if pd.notna(status) and isinstance(status, str):
                        tank_status[tank_id] = status.strip().upper()
    
    # Method 2: Read from HORIZONTAL snapshot format
    if snapshot_df is not None and not snapshot_df.empty and '_Timestamp' in snapshot_df.columns:
        # Filter to correct timestamp
        matched_rows = snapshot_df[snapshot_df['_Timestamp'] <= timestamp]
        
        if matched_rows.empty:
            latest_snapshot = snapshot_df.iloc[0]
        else:
            latest_snapshot = matched_rows.iloc[-1]
        
        # In horizontal format: columns after volumes contain status values
        # We need to figure out where volumes end and statuses begin
        
        # Count numeric columns (volumes) first
        volume_end_idx = 1  # Start after timestamp column
        for i in range(1, len(snapshot_df.columns)):
            col_name = snapshot_df.columns[i]
            if col_name == '_Timestamp':
                continue
            value = latest_snapshot[col_name]
            
            if pd.notna(value):
                try:
                    float(str(value).replace(',', ''))
                    volume_end_idx = i + 1
                except:
                    break
        
        # Now read status values
        status_start_idx = volume_end_idx
        tank_id = 1
        
        for i in range(status_start_idx, len(snapshot_df.columns)):
            col_name = snapshot_df.columns[i]
            if col_name == '_Timestamp':
                continue
            
            value = latest_snapshot[col_name]
            
            if pd.notna(value):
                value_str = str(value).strip().upper()
                
                # Check if it's a valid status
                if value_str in ['FILLING', 'EMPTY', 'READY', 'FEEDING', 'SUSPENDED', 
                                'SETTLED', 'LAB', 'FILLED', 'SETTLING', 'MAINTENANCE', 'CLEANING']:
                    tank_status[tank_id] = value_str
                    tank_id += 1
    
    # Fill in any missing tanks up to num_tanks
    for i in range(1, num_tanks + 1):
        if i not in tank_status:
            tank_status[i] = 'READY'
    
    # Update num_tanks if we detected more
    actual_num_tanks = max(max(tank_status.keys()) if tank_status else 0, num_tanks)
    
    return tank_status, actual_num_tanks

def get_tank_volume(snapshot_df, timestamp, tank_id):
    """Get tank volume from snapshots - HANDLES HORIZONTAL FORMAT"""
    
    # If no snapshot data available, return 0
    if snapshot_df is None or snapshot_df.empty:
        return 0
    
    # HORIZONTAL FORMAT: First column is timestamp, subsequent columns 1 to N are tank volumes
    # Tank 1 volume is in column index 1, Tank 2 in column index 2, etc.
    
    if '_Timestamp' in snapshot_df.columns:
        # Filter to correct timestamp
        matched_rows = snapshot_df[snapshot_df['_Timestamp'] <= timestamp]
        
        if matched_rows.empty:
            latest_snapshot = snapshot_df.iloc[0]
        else:
            latest_snapshot = matched_rows.iloc[-1]
        
        # Tank ID maps to column index
        # Column 0 = Timestamp, Column 1 = Tank 1, Column 2 = Tank 2, etc.
        if tank_id < len(snapshot_df.columns):
            col_name = snapshot_df.columns[tank_id]
            
            if col_name == '_Timestamp':
                return 0
            
            value = latest_snapshot[col_name]
            
            # Clean and convert value
            if pd.isna(value):
                return 0
            
            vol_str = str(value).replace(',', '').replace(' ', '').strip()
            
            # Skip if it's a status string (not a number)
            status_keywords = ['FILLING', 'EMPTY', 'READY', 'FEEDING', 'SUSPENDED', 'SETTLED', 'LAB', 
                             'FILLED', 'SETTLING', 'MAINTENANCE', 'CLEANING']
            if vol_str.upper() in status_keywords:
                return 0
            
            try:
                volume = float(vol_str)
                return max(0, volume)  # Ensure non-negative
            except:
                return 0
    
    # Fallback: Try vertical format (columns named Tank1, Tank2, etc.)
    possible_cols = [f'Tank{tank_id}', f'Tank{tank_id}_Volume', f'Tank {tank_id}', f'tank{tank_id}']
    latest_snapshot = snapshot_df.iloc[-1]
    
    for col in possible_cols:
        if col in latest_snapshot.index:
            vol_str = str(latest_snapshot[col]).replace(',', '').replace(' ', '')
            try:
                volume = float(vol_str)
                return max(0, volume)
            except:
                continue
    
    return 0

def get_crude_mix(crude_mix_dict):
    """Get crude mix from HTML input data"""
    return crude_mix_dict if crude_mix_dict else None

def display_tank_grid(tank_status, snapshot_df, timestamp, num_tanks):
    """Display tank status grid with dynamic layout based on number of tanks"""
    st.subheader(f"🛢️ Tank Status Grid - {num_tanks} Tanks")
    st.caption("Move the time slider above to see tank status changes over time")
    
    # Calculate optimal grid layout
    if num_tanks <= 4:
        cols_per_row = num_tanks
    elif num_tanks <= 9:
        cols_per_row = 3
    elif num_tanks <= 16:
        cols_per_row = 4
    elif num_tanks <= 25:
        cols_per_row = 5
    else:
        cols_per_row = 6  # Max 6 columns for readability
    
    num_rows = math.ceil(num_tanks / cols_per_row)
    
    # Create dynamic grid
    tank_id = 1
    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            if tank_id <= num_tanks:
                with cols[col_idx]:
                    state = tank_status.get(tank_id, 'READY')
                    color = STATE_COLORS.get(state, '#6b7280')
                    volume = get_tank_volume(snapshot_df, timestamp, tank_id)
                    
                    # Adjust font sizes based on number of tanks for better fit
                    if num_tanks <= 16:
                        tank_font_size = 24
                        state_font_size = 16
                        volume_font_size = 14
                        padding = "20px"
                    elif num_tanks <= 25:
                        tank_font_size = 20
                        state_font_size = 14
                        volume_font_size = 12
                        padding = "15px"
                    else:
                        tank_font_size = 18
                        state_font_size = 12
                        volume_font_size = 11
                        padding = "10px"
                    
                    # Display tank card with better formatting
                    st.markdown(f"""
                    <div style='background-color: {color}; 
                                padding: {padding}; 
                                border-radius: 12px; 
                                text-align: center; 
                                color: white; 
                                margin: 3px;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                                min-height: 100px;'>
                        <div style='font-size: {tank_font_size}px; font-weight: bold; margin-bottom: 5px;'>Tank {tank_id}</div>
                        <div style='font-size: {state_font_size}px; margin: 5px 0; background: rgba(255,255,255,0.2); 
                                    padding: 3px; border-radius: 5px;'>{state}</div>
                        <div style='font-size: {volume_font_size}px; font-weight: bold;'>{volume:,.0f} bbl</div>
                    </div>
                    """, unsafe_allow_html=True)
                tank_id += 1

def display_crude_mix(crude_mix_dict):
    """Display crude mix table showing crude names and percentages"""
    crude_data = get_crude_mix(crude_mix_dict)
    
    if crude_data and len(crude_data) > 0:
        st.subheader("🛢️ Crude Mix Composition")
        
        # Create a dataframe for display
        mix_df = pd.DataFrame([
            {"Crude Name": crude, "Percentage (%)": value}
            for crude, value in crude_data.items()
        ])
        
        # Display in a nice table
        col1, col2 = st.columns([2, 3])
        with col1:
            st.dataframe(mix_df, use_container_width=True, width='stretch')
        
        with col2:
            # If percentages are numeric, show a pie chart
            try:
                numeric_values = [float(v) if isinstance(v, (int, float)) else float(str(v).replace('%', '').strip()) 
                                for v in crude_data.values()]
                
                if sum(numeric_values) > 0:
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[go.Pie(
                        labels=list(crude_data.keys()),
                        values=numeric_values,
                        hole=0.3
                    )])
                    fig.update_layout(
                        title='Crude Mix Distribution',
                        height=300,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except:
                # If can't create chart, just show the table
                pass

def main():
    st.title("🛢️ Crude Oil Tank Simulation Dashboard")
    st.markdown("### Dynamic Multi-Tank System Monitor")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Source")
        folder_path = st.text_input("Data Folder Path", 
                                   value=os.path.expanduser("~/Downloads"),
                                   help="Enter the path to folder containing simulation files")
        
        if st.button("🔄 Reload Data", type="primary", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        st.header("📊 Data Files Status")
        
        # Show which files are available
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            log_found = any(f.startswith('simulation_log_') for f in files)
            snapshot_found = any('snapshot' in f.lower() for f in files)
            
            if log_found:
                st.success("✅ Log file found")
            else:
                st.warning("⚠️ No log file")
                
            if snapshot_found:
                st.success("✅ Snapshot file found")
            else:
                st.warning("⚠️ No snapshot file")
    
    # Load data
    log_df, summary_df, cargo_df, snapshot_df, crude_mix, processing_rate_html = load_data(folder_path)
    
    # Processing rate configuration - after loading data
    with st.sidebar:
        st.markdown("---")
        st.header("⚙️ Configuration")
        
        # Processing rate input - read-only, from HTML
        if processing_rate_html:
            st.metric("Processing Rate (bbl/day)", f"{processing_rate_html:,.0f}", help="From simulation configuration")
            processing_rate_input = processing_rate_html
        else:
            st.warning("No processing rate found. Using default.")
            processing_rate_input = 50000.0
    
    if log_df is None and snapshot_df is None:
        st.error("❌ No data files found. Please check the folder path.")
        st.info(f"Looking in: {folder_path}")
        st.info("Expected files: simulation_log_*.csv or *snapshot*.csv/xlsx")
        return
    
    # Initialize session state for selected time
    if 'selected_time' not in st.session_state:
        if log_df is not None and not log_df.empty:
            st.session_state.selected_time = pd.Timestamp(log_df['Timestamp'].min()).to_pydatetime()
        elif snapshot_df is not None and '_Timestamp' in snapshot_df.columns:
            st.session_state.selected_time = pd.Timestamp(snapshot_df['_Timestamp'].min()).to_pydatetime()
        else:
            st.session_state.selected_time = datetime.now()
    
    # Get time range - Convert all to datetime.datetime for consistency
    min_time = st.session_state.selected_time
    max_time = st.session_state.selected_time
    
    if log_df is not None and not log_df.empty:
        min_time = pd.Timestamp(log_df['Timestamp'].min()).to_pydatetime()
        max_time = pd.Timestamp(log_df['Timestamp'].max()).to_pydatetime()
    elif snapshot_df is not None and '_Timestamp' in snapshot_df.columns:
        min_time = pd.Timestamp(snapshot_df['_Timestamp'].min()).to_pydatetime()
        max_time = pd.Timestamp(snapshot_df['_Timestamp'].max()).to_pydatetime()
    
    # Ensure min and max are different (add 1 hour if same)
    if min_time == max_time:
        max_time = min_time + timedelta(hours=1)
    
    # Time selector with dropdowns
    st.markdown("### ⏰ Select Time Point")
    
    # Get time range from data
    if log_df is not None and not log_df.empty:
        data_min_time = log_df['Timestamp'].min()
        data_max_time = log_df['Timestamp'].max()
    elif snapshot_df is not None and '_Timestamp' in snapshot_df.columns:
        data_min_time = snapshot_df['_Timestamp'].min()
        data_max_time = snapshot_df['_Timestamp'].max()
    else:
        data_min_time = min_time
        data_max_time = max_time
    
    # Generate all dates in range
    current_date = data_min_time.date()
    end_date = data_max_time.date()
    date_list = []
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    
    date_options = [d.strftime('%d/%m/%Y') for d in date_list]
    
    # Create two columns for date and time dropdowns
    col1, col2 = st.columns(2)
    
    with col1:
        # Date dropdown
        selected_date_str = st.selectbox(
            "📅 Select Date",
            options=date_options,
            index=0,
            key='date_selector',
            help="Select the date"
        )
        selected_date = datetime.strptime(selected_date_str, '%d/%m/%Y').date()
    
    with col2:
        # Generate all time options in 1-minute intervals for 24 hours
        time_options = []
        for hour in range(24):
            for minute in range(60):
                time_options.append(f"{hour:02d}:{minute:02d}")
        
        # Time dropdown
        selected_time_str = st.selectbox(
            "🕐 Select Time",
            options=time_options,
            index=0,
            key='time_selector',
            help="Select the time (1-minute intervals)"
        )
        
        # Combine date and time
        selected_time = datetime.combine(selected_date, datetime.strptime(selected_time_str, '%H:%M').time())
    
    # Convert to datetime if it's a pandas Timestamp
    if isinstance(selected_time, pd.Timestamp):
        selected_time = selected_time.to_pydatetime()
    
    st.session_state.selected_time = selected_time
    
    # Detect number of tanks from data
    num_tanks = detect_number_of_tanks(log_df, snapshot_df)
    
    # Get tank status at selected time - NOW PASSING SNAPSHOT_DF TOO
    tank_status, actual_num_tanks = get_tank_status(log_df, snapshot_df, selected_time, num_tanks)
    
    # Update num_tanks if more were detected
    num_tanks = actual_num_tanks
    
    # Display info about detected tanks
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📊 System has **{num_tanks} tanks**")
    with col2:
        st.info(f"🕐 Viewing: **{selected_time.strftime('%H:%M')}**")
    with col3:
        st.info(f"📅 Date: **{selected_time.strftime('%d/%m/%Y')}**")
    
    # Key metrics
    st.markdown("---")
    
    # Calculate real-time certified stock at selected time
    certified_stock_at_time = 0.0
    for tank_id in range(1, num_tanks + 1):
        if tank_status.get(tank_id) in ['READY', 'FEEDING']:
            volume = get_tank_volume(snapshot_df, selected_time, tank_id)
            certified_stock_at_time += volume
    
    # Convert to MMbbl
    certified_stock_mmbl = certified_stock_at_time / 1_000_000
    
    # Display certified stock for selected time prominently
    st.markdown("### 📊 Certified Stock at Selected Time")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Certified Stock (MMbbl)", f"{certified_stock_mmbl:.3f}")
    with col2:
        if processing_rate_input and processing_rate_input > 0:
            days_remaining = certified_stock_at_time / processing_rate_input
            st.metric("Days Remaining", f"{days_remaining:.2f}")
        else:
            st.metric("Days Remaining", "N/A")
    with col3:
        ready_feeding_count = sum(1 for s in tank_status.values() if s in ['READY', 'FEEDING'])
        st.metric("Ready + Feeding Tanks", f"{ready_feeding_count}")
    
    st.markdown("---")
    
    # Calculate metrics
    metrics = {}
    for state in STATE_COLORS.keys():
        count = sum(1 for s in tank_status.values() if s == state)
        if count > 0:
            metrics[state] = count
    
    # Display metrics dynamically based on what states are present
    if metrics:
        cols = st.columns(min(len(metrics), 6))  # Max 6 columns
        for idx, (state, count) in enumerate(metrics.items()):
            with cols[idx % len(cols)]:
                color = STATE_COLORS.get(state, '#6b7280')
                # Use emoji based on state
                emoji = {
                    'READY': '🟢',
                    'FEEDING': '🔵',
                    'FILLING': '🟠',
                    'EMPTY': '🔴',
                    'FILLED': '🟣',
                    'SETTLING': '🟡',
                    'LAB': '🔷',
                    'SUSPENDED': '⚫'
                }.get(state, '⚪')
                st.metric(f"{emoji} {state}", count)
    
    st.markdown("---")
    
    # Tank grid - now passing snapshot_df and dynamic num_tanks
    display_tank_grid(tank_status, snapshot_df, selected_time, num_tanks)
    
    st.markdown("---")
    
    # Charts in tabs
    tabs = ["🛢️ Crude Mix", "📋 Events Log"]
    if snapshot_df is not None and not snapshot_df.empty:
        tabs.append("📊 Certified Stock")
    if cargo_df is not None:
        tabs.append("🚢 Cargo Report")
    if summary_df is not None:
        tabs.append("📈 Daily Summary")
    
    tab_objects = st.tabs(tabs)
    
    tab_idx = 0
    
    # Crude Mix tab
    with tab_objects[tab_idx]:
        if crude_mix and len(crude_mix) > 0:
            st.subheader("🛢️ Crude Mix Target Configuration")
            
            # Create a dataframe for display
            mix_df = pd.DataFrame([
                {"Crude Name": crude, "Percentage (%)": value}
                for crude, value in crude_mix.items()
            ])
            
            # Display table
            st.dataframe(mix_df, width='stretch', hide_index=True)
            
            # Display pie chart
            try:
                numeric_values = list(crude_mix.values())
                if sum(numeric_values) > 0:
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[go.Pie(
                        labels=list(crude_mix.keys()),
                        values=numeric_values,
                        hole=0.3
                    )])
                    fig.update_layout(
                        title='Crude Mix Distribution',
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except:
                pass
        else:
            st.warning("No crude mix data found in simulation log READY_1 events.")
    
    tab_idx += 1
    
    # Events Log tab
    with tab_objects[tab_idx]:
        if log_df is not None and not log_df.empty:
            st.subheader("All Events")
            
            # Show ALL events (removed time filter - no longer filtering by selected_time)
            filtered_log = log_df.tail(500)  # Show last 500 events instead of filtering by time
            
            # Format timestamp to remove seconds
            if 'Timestamp' in filtered_log.columns:
                filtered_log = filtered_log.copy()
                filtered_log['Timestamp'] = filtered_log['Timestamp'].dt.strftime('%d/%m/%Y %H:%M')
            
            # Search functionality
            col1, col2 = st.columns([3, 1])
            with col1:
                search = st.text_input("🔍 Search events", "", placeholder="Type to filter events...")
            with col2:
                show_all = st.checkbox("Show all columns", value=False)
            
            if search:
                # Search in all text columns
                mask = pd.Series([False] * len(filtered_log))
                for col in filtered_log.columns:
                    if filtered_log[col].dtype == 'object':
                        mask |= filtered_log[col].str.contains(search, case=False, na=False)
                filtered_log = filtered_log[mask]
            
            # Determine which columns to display
            if show_all:
                display_cols = list(filtered_log.columns)
            else:
                # Show only the most relevant columns
                possible_cols = ['Timestamp', 'Level', 'Event', 'Tank', 'Cargo', 'Message', 'Status']
                display_cols = [col for col in possible_cols if col in filtered_log.columns]
            
            if not filtered_log.empty:
                st.markdown("""
                <style>
                    .dataframe {
                        font-size: 11px !important;
                    }
                    .dataframe td, .dataframe th {
                        padding: 2px 4px !important;
                        font-size: 11px !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                st.dataframe(
                    filtered_log[display_cols].sort_values('Timestamp', ascending=True).reset_index(drop=True),
                    width='stretch',
                    height=400
                )
            else:
                st.info("No events found matching your criteria")
        else:
            st.warning("No event log data available")
    
    tab_idx += 1
    
    # Inventory tab - NOW USING SNAPSHOTS
    if snapshot_df is not None and not snapshot_df.empty and tab_idx < len(tab_objects):
        with tab_objects[tab_idx]:
            st.subheader("Certified Stock Over Time")
            
            if '_Timestamp' in snapshot_df.columns:
                # Get date range from snapshots
                data_min_time = snapshot_df['_Timestamp'].min()
                data_max_time = snapshot_df['_Timestamp'].max()
                
                # Generate date list
                current_date = data_min_time.date()
                end_date = data_max_time.date()
                date_list = []
                while current_date <= end_date:
                    date_list.append(current_date)
                    current_date += timedelta(days=1)
                
                date_options = [d.strftime('%d/%m/%Y') for d in date_list]
                
                # Create two columns for date and time dropdowns (SAME AS SELECT TIME POINT)
                st.markdown("#### Select Date and Time to View Certified Stock")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Date dropdown
                    selected_inv_date_str = st.selectbox(
                        "📅 Select Date",
                        options=date_options,
                        index=len(date_options)-1,  # Default to last date
                        key='inv_date_selector',
                        help="Select the date"
                    )
                    selected_inv_date = datetime.strptime(selected_inv_date_str, '%d/%m/%Y').date()
                
                with col2:
                    # Generate all time options in 1-minute intervals for 24 hours
                    time_options = []
                    for hour in range(24):
                        for minute in range(60):
                            time_options.append(f"{hour:02d}:{minute:02d}")
                    
                    # Time dropdown
                    selected_inv_time_str = st.selectbox(
                        "🕐 Select Time",
                        options=time_options,
                        index=len(time_options)-1,  # Default to 23:59
                        key='inv_time_selector',
                        help="Select the time (1-minute intervals)"
                    )
                    
                    # Combine date and time
                    selected_inv_datetime = datetime.combine(selected_inv_date, datetime.strptime(selected_inv_time_str, '%H:%M').time())
                
                # Calculate certified stock for ALL snapshots (for the graph)
                timestamps = []
                certified_stocks = []
                
                for idx, row in snapshot_df.iterrows():
                    timestamp = row['_Timestamp']
                    certified_stock = 0.0
                    
                    # Sum READY and FEEDING tank volumes (SAME AS SELECT TIME POINT)
                    for tank_id in range(1, num_tanks + 1):
                        state_col = f'State{tank_id}'
                        tank_col = f'Tank{tank_id}'
                        
                        if state_col in row.index and tank_col in row.index:
                            state = str(row[state_col]).strip().upper()
                            
                            if state in ['READY', 'FEEDING']:
                                volume_str = str(row[tank_col]).replace(',', '').strip()
                                try:
                                    volume = float(volume_str)
                                    certified_stock += volume
                                except:
                                    pass
                    
                    timestamps.append(timestamp)
                    certified_stocks.append(certified_stock / 1_000_000)  # Convert to MMbbl
                
                # Create dataframe for plotting
                chart_df = pd.DataFrame({
                    'Timestamp': timestamps,
                    'Certified Stock (MMbbl)': certified_stocks
                })
                
                # Get certified stock at selected date/time
                matched_rows = chart_df[chart_df['Timestamp'] <= selected_inv_datetime]
                if not matched_rows.empty:
                    selected_row = matched_rows.iloc[-1]
                    selected_certified_stock = selected_row['Certified Stock (MMbbl)']
                    selected_timestamp = selected_row['Timestamp']
                else:
                    selected_certified_stock = 0.0
                    selected_timestamp = selected_inv_datetime
                
                # Create line chart
                fig = px.line(
                    chart_df,
                    x='Timestamp',
                    y='Certified Stock (MMbbl)',
                    title='Certified Stock Timeline',
                    labels={'Certified Stock (MMbbl)': 'Certified Stock (Million Barrels)', 'Timestamp': 'Date & Time'}
                )
                
                # Add marker for selected date/time
                fig.add_scatter(
                    x=[selected_timestamp],
                    y=[selected_certified_stock],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='circle'),
                    name='Selected Point',
                    showlegend=True
                )
                
                fig.update_xaxes(tickformat='%d/%m/%Y', tickangle=-45)
                fig.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show certified stock stats
                st.markdown("""
                <style>
                    div[data-testid="stMetric"],
                    div[data-testid="metric-container"],
                    .stMetric {
                        background-color: #000000 !important;
                        color: white !important;
                        padding: 20px !important;
                        border-radius: 10px !important;
                        border: 1px solid #333 !important;
                    }
                    div[data-testid="stMetric"] label,
                    div[data-testid="stMetric"] > div > div > div {
                        color: #ffffff !important;
                        background-color: transparent !important;
                    }
                    div[data-testid="stMetric"] [data-testid="stMetricValue"],
                    div[data-testid="stMetric"] div[data-testid="stMarkdownContainer"] {
                        color: #ffffff !important;
                        background-color: transparent !important;
                    }
                    div[data-testid="stMetric"] * {
                        background-color: transparent !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Certified Stock (Selected Date/Time)", f"{selected_certified_stock:.2f} MMbbl")
                with col2:
                    st.metric("Average (All Time)", f"{chart_df['Certified Stock (MMbbl)'].mean():.2f} MMbbl")
                with col3:
                    st.metric("Max (All Time)", f"{chart_df['Certified Stock (MMbbl)'].max():.2f} MMbbl")
                
                # Calculate days remaining
                st.markdown("---")
                
                if processing_rate_input and processing_rate_input > 0:
                    selected_certified_stock_bbl = selected_certified_stock * 1_000_000
                    days_remaining = selected_certified_stock_bbl / processing_rate_input
                    st.info(f"📊 **Certified stock will last for {days_remaining:.2f} days** (at processing rate of {processing_rate_input:,.0f} bbl/day)")
                else:
                    st.warning("⚠️ Please enter a valid processing rate in the sidebar")
            else:
                st.warning("No snapshot data available")
        tab_idx += 1
    
    # Cargo Report tab
    if cargo_df is not None and tab_idx < len(tab_objects):
        with tab_objects[tab_idx]:
            st.subheader("Cargo Schedule")
            
            # Format berth gap hours as decimals if the column exists
            cargo_display = cargo_df.copy()
            
            # Remove duplicate rows (keep first occurrence)
            cargo_display = cargo_display.drop_duplicates()
            
            # Remove Cargo Type column if it exists
            columns_to_remove = ['Cargo Type', 'CARGO TYPE', 'Cargo_Type', 'cargo_type']
            for col in columns_to_remove:
                if col in cargo_display.columns:
                    cargo_display = cargo_display.drop(columns=[col])
            
            if 'BERTH GAP(hrs)' in cargo_display.columns:
                cargo_display['BERTH GAP(hrs)'] = cargo_display['BERTH GAP(hrs)'].apply(
                    lambda x: f"{float(x):.2f}" if pd.notna(x) else x
                )
            
            st.dataframe(cargo_display,width='content', height=400)
        tab_idx += 1
    
    # Daily Summary tab
    if summary_df is not None and tab_idx < len(tab_objects):
        with tab_objects[tab_idx]:
            if 'Date' in summary_df.columns:
                st.subheader("Daily Processing Summary")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(summary_df,width='stretch', height=400)
                
                with col2:
                    if 'Processing (bbl)' in summary_df.columns:
                        # Convert to numeric, removing commas
                        processing_values = pd.to_numeric(summary_df['Processing (bbl)'].astype(str).str.replace(',', ''), errors='coerce')
                        st.metric("Total Processed", 
                                f"{processing_values.sum():,.0f} bbl")
                        st.metric("Daily Average", 
                                f"{processing_values.mean():,.0f} bbl")
                
                # Daily chart
                valid_summary_dates = summary_df['Date'].dropna()
                if not valid_summary_dates.empty and 'Processing (bbl)' in summary_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=summary_df.dropna(subset=['Date'])['Date'],
                        y=summary_df.dropna(subset=['Date'])['Processing (bbl)'],
                        name='Processing',
                        marker_color='#3b82f6'
                    ))
                    fig.update_layout(
                        title='Daily Processing Volume',
                        xaxis_title='Date',
                        yaxis_title='Processing (barrels)',
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()