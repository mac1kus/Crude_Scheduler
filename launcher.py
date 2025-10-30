import os
import subprocess
import sys

# Detect if running as a bundled exe
if getattr(sys, 'frozen', False):
    # When running as an exe, the dist folder is one level above the temp _MEI folder
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

app_path = os.path.join(base_dir, "streamlit_app.py")

print(f"Starting Streamlit using {app_path}")
try:
    subprocess.run(["python", "-m", "streamlit", "run", app_path], check=True)
except Exception as e:
    print(f"Error launching Streamlit: {e}")
    input("Press Enter to close...")
