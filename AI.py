import json
from datetime import datetime, UTC
from pathlib import Path
import os
from typing import Any
import io
import base64
def get_hackclub_api_key():
    # Fallback to a default value if not set in environment
    HACKCLUB_API_KEY = None
    return os.getenv('HACKCLUB_API_KEY') or HACKCLUB_API_KEY

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm

# Try importing optional backends
try:
    import openai  # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore
    OPENAI_AVAILABLE = False



st.set_page_config(page_title='Cortex Ai', layout='wide', page_icon='🤖')

HISTORY_FILE = Path(__file__).with_name('chat_history.json')
SETTINGS_FILE = Path(__file__).with_name('user_settings.json')

def load_saved_chats() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
            # Validate data structure
            if isinstance(data, list):
                return data
            else:
                return []
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        print(f"Debug: Error loading chat history: {e}")
        return []

def persist_saved_chats(chats: list[dict]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open('w', encoding='utf-8') as fh:
        json.dump(chats, fh, ensure_ascii=False, indent=2)

def load_user_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        with SETTINGS_FILE.open('r', encoding='utf-8') as fh:
            return json.load(fh)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError):
        return {}

def save_user_settings(settings: dict) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_FILE.open('w', encoding='utf-8') as fh:
        json.dump(settings, fh, ensure_ascii=False, indent=2)

def save_current_settings():
    """Save current session state settings to file"""
    current_settings = {
        'backend': st.session_state.backend,
        'model_name': st.session_state.model_name,
        'theme': getattr(st.session_state, 'theme', 'dark'),
        'profile_name': st.session_state.profile_name,
        'profile_email': st.session_state.profile_email,
        'profile_mobile': st.session_state.profile_mobile,
        'profile_address': st.session_state.profile_address,
    }
    save_user_settings(current_settings)

def lemniscate(t, a=1.0):
    # Bernoulli lemniscate parametric form
    denom = 1 + np.sin(t)**2
    x = a * np.cos(t) / denom
    y = a * np.sin(t) * np.cos(t) / denom
    return np.vstack((x, y)).T

def unit_normals(points):
    # compute tangent and normals for a polyline
    diffs = np.diff(points, axis=0)
    tangents = diffs / np.linalg.norm(diffs, axis=1)[:, None]
    # for last point, repeat last tangent
    tangents = np.vstack((tangents, tangents[-1]))
    normals = np.column_stack((-tangents[:,1], tangents[:,0]))
    # normalize normals
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    return normals

def build_ribbon(points, width=0.18, n_segments=200):
    normals = unit_normals(points)
    left = points + normals * (width/2)
    right = points - normals * (width/2)
    # build polygon strips per segment for gradient coloring
    polys = []
    tvals = np.linspace(0, 1, len(points), False)
    for i in range(len(points)-1):
        quad = np.array([ left[i], left[i+1], right[i+1], right[i] ])
        polys.append((quad, tvals[i]))
    return polys

def plot_rainbow_ribbon(ax, polys, cmap_name='hsv'):
    cmap = plt.colormaps.get_cmap(cmap_name)
    verts = [p[0] for p in polys]
    tvals = np.array([p[1] for p in polys])
    # color by t along the path
    colors = cmap(tvals)
    coll = PolyCollection(verts, facecolors=colors, edgecolors='none', linewidths=0)
    ax.add_collection(coll)

def generate_logo_base64():
    # parameter t
    t = np.linspace(0, 2*np.pi, 600, False)
    pts = lemniscate(t, a=1.2)

    # rotate and scale a bit to match aesthetic
    theta = np.deg2rad(-20)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    pts = pts.dot(R.T)

    # build ribbon polygons
    polys = build_ribbon(pts, width=0.45)

    # plotting
    fig, ax = plt.subplots(figsize=(2,2), dpi=150)
    plot_rainbow_ribbon(ax, polys, cmap_name='hsv')

    # add a subtle inner highlight (thin white stroke along center)
    center_x = pts[:,0]
    center_y = pts[:,1]
    ax.plot(center_x, center_y, color=(1,1,1,0.18), linewidth=6, solid_capstyle='round')

    # add a soft darker shadow underneath (offset and blurred look via alpha)
    shadow = pts + np.array([0.06, -0.06])
    ax.plot(shadow[:,0], shadow[:,1], color=(0,0,0,0.12), linewidth=18, solid_capstyle='round')

    ax.set_aspect('equal')
    ax.axis('off')
    # tight margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    return img_base64

# Generate logo once
LOGO_BASE64 = generate_logo_base64()
LOGO_URL = f"data:image/png;base64,{LOGO_BASE64}"

def get_theme_styles(theme='dark'):
    """Get theme-specific CSS styles"""
    if theme == 'transparent':
        return '''
        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .stApp {
            background: linear-gradient(120deg, rgba(34,193,195,0.35) 0%, rgba(253,187,45,0.25) 100%, rgba(131,58,180,0.25) 60%);
            background-size: 200% 200%;
            animation: gradientBG 12s ease infinite;
            color: #22223b !important;
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
        }
        .stSidebar {
            background: rgba(255,255,255,0.18) !important;
            border: 1.5px solid rgba(255,255,255,0.22) !important;
            box-shadow: 0 8px 32px 0 rgba(31,38,135,0.12) !important;
            backdrop-filter: blur(24px) !important;
            transition: background 0.5s, box-shadow 0.5s;
        }
        .stSidebar .stMarkdown {
            color: #22223b !important;
        }
        .stChatMessage {
            background: rgba(255,255,255,0.22) !important;
            color: #22223b !important;
            border-radius: 22px !important;
            border: 1.5px solid rgba(255,255,255,0.32) !important;
            box-shadow: 0 8px 32px 0 rgba(31,38,135,0.12) !important;
            backdrop-filter: blur(32px) !important;
            transition: background 0.5s, box-shadow 0.5s;
        }
        .stChatMessage:hover {
            background: rgba(255,255,255,0.32) !important;
            box-shadow: 0 12px 40px 0 rgba(31,38,135,0.18) !important;
        }
        .stTextInput input, .stChatInput textarea {
            background: rgba(255,255,255,0.22) !important;
            color: #22223b !important;
            border-color: rgba(255,255,255,0.32) !important;
            backdrop-filter: blur(18px) !important;
            border-radius: 12px !important;
            font-size: 1.08rem !important;
            transition: background 0.5s, box-shadow 0.5s;
        }
        .stTextInput input:focus, .stChatInput textarea:focus {
            background: rgba(255,255,255,0.32) !important;
            box-shadow: 0 0 0 2px #a78bfa44 !important;
        }
        .stSelectbox select {
            background: rgba(255,255,255,0.22) !important;
            color: #22223b !important;
            border-color: rgba(255,255,255,0.32) !important;
            backdrop-filter: blur(18px) !important;
            border-radius: 12px !important;
        }
        .stButton button {
            background: linear-gradient(90deg, #a78bfa 0%, #43e97b 100%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 12px 0 rgba(67,233,123,0.12) !important;
            transition: background 0.3s, box-shadow 0.3s;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #43e97b 0%, #a78bfa 100%) !important;
            box-shadow: 0 4px 24px 0 rgba(67,233,123,0.18) !important;
        }
        .stChatInput {
            background: rgba(255,255,255,0.22) !important;
            border-radius: 16px !important;
            border: 1.5px solid rgba(255,255,255,0.32) !important;
            box-shadow: 0 8px 32px 0 rgba(31,38,135,0.12) !important;
            backdrop-filter: blur(32px) !important;
            transition: background 0.5s, box-shadow 0.5s;
        }
        .stChatInputContainer {
            background: transparent !important;
        }
        '''
    else:  # default dark theme
        return '''
        .stApp {
            background-color: #0e1117 !important;
        }
        '''

_STYLE = '''
/* Main app background */
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
.main-container {
    color: #e2e8f0;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    background: transparent;
}
/* Chat panel */
.panel {
    border-radius: 16px;
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(148, 163, 184, 0.1);
    backdrop-filter: blur(10px);
}
/* Sidebar cards */
.sidebar-card {
    border-radius: 14px;
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
    padding: 1rem 1.25rem;
    margin-bottom: 0.85rem;
    border: 1px solid rgba(99, 102, 241, 0.15);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
/* Example/capability/limitation cards */
.example-card {
    border-radius: 14px;
    background: linear-gradient(145deg, rgba(51, 65, 85, 0.6), rgba(30, 41, 59, 0.7));
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(139, 92, 246, 0.2);
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
}
.example-card:hover {
    border-color: rgba(139, 92, 246, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.15);
}
/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e2f 0%, #151521 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.1);
}
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}
section[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
}
/* Text inputs */
.stTextInput input {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
}
/* Chat input */
.stChatInput {
    border-radius: 14px !important;
    background: transparent !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: none !important;
}
.stChatInput textarea {
    background: transparent !important;
    color: #e2e8f0 !important;
}
.stChatInput > div {
    background: transparent !important;
}
.stChatInput input {
    background: transparent !important;
}
/* Force chat input container transparent */
[data-testid="stChatInput"] {
    background: transparent !important;
}
[data-testid="stChatInput"] > div {
    background: transparent !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
}
.stChatInputContainer {
    background: transparent !important;
}
/* Chat bottom container */
[data-testid="stBottom"] {
    background: transparent !important;
}
[data-testid="stBottom"] > div {
    background: transparent !important;
}
.stBottom {
    background: transparent !important;
}
/* Main block container */
[data-testid="stMainBlockContainer"] {
    background: transparent !important;
}
.block-container {
    background: transparent !important;
}
/* Chat messages */
.stChatMessage {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.7)) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    padding: 1rem !important;
    margin-bottom: 0.75rem !important;
}
/* Hide default robot/user avatars and replace with custom logo */
.stChatMessage [data-testid="stChatMessageAvatarAssistant"],
.stChatMessage [data-testid="stChatMessageAvatarUser"] {
    display: none !important;
}
/* Custom avatar container styling */
.stChatMessage > div:first-child {
    position: relative;
}
/* Assistant message avatar */
[data-testid="stChatMessage"][aria-label*="assistant"] .stChatMessageAvatar,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div:first-child::before {
    content: "";
    display: inline-block;
    width: 32px;
    height: 32px;
    background-image: url("data:image/png;base64,''' + LOGO_BASE64 + '''");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    border-radius: 6px;
}
/* User message styling - show custom icon */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:first-child::before {
    content: "";
    display: inline-block;
    width: 32px;
    height: 32px;
    background-image: url("data:image/png;base64,''' + LOGO_BASE64 + '''");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    border-radius: 6px;
    filter: hue-rotate(180deg);
}
/* Success/info messages */
.stSuccess {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.1)) !important;
    border: 1px solid rgba(34, 197, 94, 0.3) !important;
    border-radius: 10px !important;
}
/* Headings */
h1, h2, h3, h4, h5 {
    background: linear-gradient(135deg, #f8fafc 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.5);
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #6366f1, #8b5cf6);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #8b5cf6, #a78bfa);
}
/* Streamlit header styling - keep visible for Deploy button */
header[data-testid="stHeader"] {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.98), rgba(15, 23, 42, 0.98)) !important;
    backdrop-filter: blur(15px) !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    z-index: 9998 !important;
}
/* Style the toolbar buttons */
.stToolbar {
    color: #e2e8f0 !important;
}
.stToolbar button {
    color: #e2e8f0 !important;
    background: rgba(99, 102, 241, 0.15) !important;
    border-radius: 6px !important;
}
.stToolbar button:hover {
    background: rgba(99, 102, 241, 0.25) !important;
}
/* Fixed top-left header overlay */
.top-header {
    position: fixed;
    top: 0;
    left: 60px;
    right: 0;
    z-index: 10000;
    background: transparent;
    backdrop-filter: none;
    border-bottom: none;
    padding: 1rem 0.5rem;
    height: 70px;
    display: flex;
    align-items: center;
    pointer-events: none;
    box-shadow: none;
}
.top-header > * {
    pointer-events: auto;
}
.top-header h1 {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #f8fafc 0%, #a78bfa 50%, #8b5cf6 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: 0.02em;
}
/* Logo styling */
.logo-container {
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.3s ease;
}
.logo-container:hover {
    transform: scale(1.1);
}
/* Adjust main content to account for fixed header */
.main-content {
    margin-top: 70px !important;
    padding-top: 2rem !important;
    background: transparent !important;
}
/* Adjust sidebar to account for header */
section[data-testid="stSidebar"] {
    margin-top: 70px !important;
    height: calc(100vh - 70px) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0 !important;
}
/* Style the sidebar collapse button */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarNavCollapseButton"] {
    color: #e2e8f0 !important;
    background: rgba(99, 102, 241, 0.15) !important;
    border-radius: 6px !important;
    z-index: 10000 !important;
}
button[data-testid="stSidebarCollapseButton"]:hover,
button[data-testid="stSidebarNavCollapseButton"]:hover {
    background: rgba(99, 102, 241, 0.25) !important;
}
'''
_SEND_MESSAGE_STYLE = f'''
.stChatInput textarea {{
    background-image: url("data:image/png;base64,{LOGO_BASE64}") !important;
    background-repeat: no-repeat !important;
    background-position: 16px center !important;
    background-size: 28px 28px !important;
    padding-left: 3.5rem !important;
    padding-right: 5.5rem !important;
}}.
stChatInput textarea::placeholder {{
    color: rgba(226, 232, 240, 0.85) !important;
}}
/* Drag and drop styling */
.chat-input-container {{
    position: relative !important;
}}
.drag-overlay {{
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    background: rgba(99, 102, 241, 0.1) !important;
    border: 2px dashed rgba(99, 102, 241, 0.5) !important;
    border-radius: 14px !important;
    display: none !important;
    align-items: center !important;
    justify-content: center !important;
    z-index: 10002 !important;
    pointer-events: none !important;
}}
.drag-overlay.drag-active {{
    display: flex !important;
}}
.drag-overlay-text {{
    color: #8b5cf6 !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
}}
.stChatInput {{
    transition: all 0.3s ease !important;
}}
.stChatInput.drag-over {{
    background: rgba(99, 102, 241, 0.05) !important;
    border-color: rgba(99, 102, 241, 0.6) !important;
    transform: scale(1.01) !important;
}}
'''

# Initialize session state first
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'saved_chats' not in st.session_state:
    st.session_state.saved_chats = load_saved_chats()
if 'chat_title_input' not in st.session_state:
    st.session_state.chat_title_input = 'New chat'
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'selected_chat_id' not in st.session_state:
    st.session_state.selected_chat_id = None
if 'rename_input' not in st.session_state:
    st.session_state.rename_input = ''

# Load saved user settings
saved_settings = load_user_settings()

if 'backend' not in st.session_state:
    st.session_state.backend = saved_settings.get('backend', 'hackclub')
if 'model_name' not in st.session_state:
    st.session_state.model_name = saved_settings.get('model_name', 'hackclub/model1')
if 'theme' not in st.session_state:
    st.session_state.theme = saved_settings.get('theme', 'dark')

if 'show_settings' not in st.session_state:
    st.session_state.show_settings = False
if 'settings_page' not in st.session_state:
    st.session_state.settings_page = 'application'
if 'profile_name' not in st.session_state:
    st.session_state.profile_name = saved_settings.get('profile_name', '')
if 'profile_mobile' not in st.session_state:
    st.session_state.profile_mobile = saved_settings.get('profile_mobile', '')
if 'profile_email' not in st.session_state:
    st.session_state.profile_email = saved_settings.get('profile_email', '')
if 'profile_address' not in st.session_state:
    st.session_state.profile_address = saved_settings.get('profile_address', '')

# Apply styles after session state is initialized
st.markdown(f"<style>{get_theme_styles(st.session_state.theme)}{_STYLE}{_SEND_MESSAGE_STYLE}</style>", unsafe_allow_html=True)

with st.sidebar:
    # Conversations section
    st.markdown("**Conversations**")
    
    # New conversation button
    if st.button('+ New conversation', use_container_width=True):
        # Save current conversation if it has messages
        if st.session_state.messages:
            entry = {
                'id': datetime.now(UTC).strftime('%Y%m%d%H%M%S%f'),
                'name': f'Chat {len(st.session_state.saved_chats) + 1}',
                'timestamp': datetime.now(UTC).isoformat(),
                'model': st.session_state.model_name,
                'backend': st.session_state.backend,
                'messages': [message.copy() for message in st.session_state.messages],
            }
            st.session_state.saved_chats.insert(0, entry)
            persist_saved_chats(st.session_state.saved_chats)
        
        # Clear current conversation
        st.session_state.messages = []
        st.session_state.selected_chat_id = None
        st.session_state.rename_input = ''
        st.rerun()
    
    # Show saved chats
    if st.session_state.saved_chats:
        for entry in st.session_state.saved_chats[:10]:
            chat_name = entry['name'][:30] + '...' if len(entry['name']) > 30 else entry['name']
            
            # Create columns for chat button and delete button
            col1, col2 = st.columns([4, 1])
            
            with col1:
                if st.button(chat_name, key=f"chat-{entry['id']}", use_container_width=True):
                    st.session_state.messages = [message.copy() for message in entry['messages']]
                    st.session_state.selected_chat_id = entry['id']
                    st.session_state.rename_input = entry['name']
                    st.rerun()
            
            with col2:
                if st.button('🗑️', key=f"delete-{entry['id']}", help=f"Delete '{chat_name}'"):
                    # Remove the chat from saved_chats
                    st.session_state.saved_chats = [chat for chat in st.session_state.saved_chats if chat['id'] != entry['id']]
                    persist_saved_chats(st.session_state.saved_chats)
                    
                    # Clear current messages if this was the selected chat
                    if st.session_state.selected_chat_id == entry['id']:
                        st.session_state.messages = []
                        st.session_state.selected_chat_id = None
                    
                    st.rerun()
    
    st.markdown("---")
    
    # Settings button
    st.markdown("---")
    if st.button('⚙️ Settings', use_container_width=True):
        st.session_state.show_settings = True
        st.rerun()

# Simplified Settings Modal
if st.session_state.show_settings:
    # Settings layout with sidebar and main content
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("## Settings")
        st.markdown("---")
        
        # Settings navigation menu
        if st.button("👤 Profile", use_container_width=True):
            st.session_state.settings_page = "profile"
        st.caption("User settings and preferences")
        
        if st.button("🔧 Application", use_container_width=True):
            st.session_state.settings_page = "application"
        st.caption("App preferences and settings")
        
        if st.button("🔒 Security", use_container_width=True):
            st.session_state.settings_page = "security"
        st.caption("Security and privacy settings")
        
        if st.button("🔔 Notifications", use_container_width=True):
            st.session_state.settings_page = "notifications"
        st.caption("Alert and update preferences")
    
    with col2:
        # Header with close button
        col2a, col2b = st.columns([3, 1])
        with col2b:
            if st.button("✕ Close", key="close_settings_top"):
                st.session_state.show_settings = False
                st.rerun()
        
        # Show content based on selected settings page
        if st.session_state.settings_page == "profile":
            with col2a:
                st.markdown("## Profile Settings")
            st.markdown("---")
            
            st.markdown("### Personal Information")
            
            # Profile input fields with change detection
            new_name = st.text_input('Full Name', value=st.session_state.profile_name, placeholder='Enter your full name', key="profile_name_input")
            new_mobile = st.text_input('Mobile Number', value=st.session_state.profile_mobile, placeholder='Enter your mobile number', key="profile_mobile_input")
            new_email = st.text_input('Email Address', value=st.session_state.profile_email, placeholder='Enter your email address', key="profile_email_input")
            new_address = st.text_area('Address', value=st.session_state.profile_address, placeholder='Enter your address', height=100, key="profile_address_input")
            
            # Check for changes and save if needed
            if (new_name != st.session_state.profile_name or 
                new_mobile != st.session_state.profile_mobile or 
                new_email != st.session_state.profile_email or 
                new_address != st.session_state.profile_address):
                
                st.session_state.profile_name = new_name
                st.session_state.profile_mobile = new_mobile
                st.session_state.profile_email = new_email
                st.session_state.profile_address = new_address
                save_current_settings()
                
            # Save button for manual save
            if st.button("💾 Save Profile", use_container_width=True):
                save_current_settings()
                st.success("✅ Profile saved successfully!")
        
        elif st.session_state.settings_page == "security":
            with col2a:
                st.markdown("## Security Settings")
            st.markdown("---")
            st.markdown("### Password & Security")
            st.text_input('Current Password', type='password', placeholder='Enter current password')
            st.text_input('New Password', type='password', placeholder='Enter new password')
            st.text_input('Confirm Password', type='password', placeholder='Confirm new password')
        
        elif st.session_state.settings_page == "notifications":
            with col2a:
                st.markdown("## Notification Settings")
            st.markdown("---")
            st.markdown("### Notification Preferences")
            st.checkbox('Email notifications', value=True)
            st.checkbox('Push notifications', value=True)
            st.checkbox('Chat sound alerts', value=True)
        
        else:  # application settings (default)
            with col2a:
                st.markdown("## Application Settings")
            st.markdown("---")
            
            # Theme selection
            st.markdown("### Theme Settings")
            theme_options = ['dark', 'transparent']
            current_theme_index = 0
            if st.session_state.theme in theme_options:
                current_theme_index = theme_options.index(st.session_state.theme)
            
            new_theme = st.selectbox('Select Theme', theme_options, 
                                   index=current_theme_index, 
                                   key="theme_selection")
            if new_theme != st.session_state.theme:
                st.session_state.theme = new_theme
                save_current_settings()
                st.rerun()
            
            st.markdown("---")
            
            # Backend configuration
            st.markdown("### AI Model Configuration")
            st.markdown("### Hack Club Models")
            hc_models = {
                'hackclub/model1': '🔧 Hack Club Model 1',
                'hackclub/model2': '🔨 Hack Club Model 2'
            }
            hc_keys = list(hc_models.keys())
            hc_labels = list(hc_models.values())
            current_index = 0
            if st.session_state.model_name in hc_keys:
                current_index = hc_keys.index(st.session_state.model_name)
            selected_label = st.radio(
                'Select Hack Club Model',
                hc_labels,
                index=current_index,
                key="hc_model_selection",
                horizontal=True
            )
            selected_index = hc_labels.index(selected_label)
            new_model = hc_keys[selected_index]
            if new_model != st.session_state.model_name:
                st.session_state.model_name = new_model
                save_current_settings()

            st.markdown("---")
            st.markdown("### Conversation Management")
            if st.button('🗑️ Clear all conversations', use_container_width=True):
                st.session_state.saved_chats = []
                persist_saved_chats([])
                st.session_state.messages = []
                st.session_state.selected_chat_id = None
                st.success('All conversations cleared!')
                st.rerun()
        
        st.markdown("---")
        if st.button('Save & Close', use_container_width=True, key="save_close"):
            st.session_state.show_settings = False
            st.rerun()

elif not st.session_state.show_settings:
    # Regular chat interface
    
    # Fixed top-left header with logo and title
    st.markdown(f"""
    <div class='top-header'>
        <div style='display: flex; align-items: center; gap: 0.75rem;'>
            <div class='logo-container'>
                <img src="data:image/png;base64,{LOGO_BASE64}" 
                     alt="Cortex AI Logo" 
                     style="width: 40px; height: 40px; border-radius: 8px;"/>
            </div>
            <h1>Cortex Ai</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        st.markdown("<div class='main-content'>", unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input('Send a message'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        # Build context with profile information
        profile_context = ""
        if (st.session_state.profile_name or st.session_state.profile_email or 
            st.session_state.profile_mobile or st.session_state.profile_address):
            profile_parts = []
            if st.session_state.profile_name:
                profile_parts.append(f"My name is {st.session_state.profile_name}")
            if st.session_state.profile_email:
                profile_parts.append(f"my email is {st.session_state.profile_email}")
            if st.session_state.profile_mobile:
                profile_parts.append(f"my phone number is {st.session_state.profile_mobile}")
            if st.session_state.profile_address:
                profile_parts.append(f"my address is {st.session_state.profile_address}")
            profile_context = f"User Profile: {', '.join(profile_parts)}. "
        # Prepare messages with profile context for AI
        messages_for_ai = st.session_state.messages.copy()
        if profile_context:
            # Add profile context as a system message at the beginning
            system_message = {
                'role': 'system', 
                'content': f"{profile_context}Please use this information when relevant to provide personalized responses."
            }
            messages_for_ai = [system_message] + messages_for_ai[-3:]  # System + last 3 messages
        else:
            messages_for_ai = messages_for_ai[-3:]  # Just last 3 messages
        full_response = ''
        try:
            from openrouter import OpenRouter
        except ImportError:
            OpenRouter = None
        if OpenRouter is None:
            full_response = "⚠️ openrouter library not installed. Run: pip install openrouter"
        else:
            api_key = get_hackclub_api_key()
            if not api_key:
                full_response = "⚠️ Hack Club API key not configured. Set the HACKCLUB_API_KEY environment variable."
            else:
                client = OpenRouter(
                    api_key=api_key,
                    server_url="https://ai.hackclub.com/proxy/v1"
                )
                formatted_messages = []
                for msg in messages_for_ai:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                try:
                    response = client.chat.send(
                        model=st.session_state.model_name,
                        messages=formatted_messages,
                        stream=False
                    )
                    if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message'):
                        full_response = response.choices[0].message.content
                    else:
                        full_response = f"❌ Hack Club API: Unexpected response: {response}"
                except Exception as e:
                    full_response = f"❌ Hack Club API error: {type(e).__name__}: {str(e)}"
        # Always append assistant response to messages
        st.session_state.messages.append({'role': 'assistant', 'content': full_response})
        st.rerun()

    # Display chat messages
    if st.session_state.messages:
        st.markdown("<div style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message['role'], avatar=LOGO_URL):
                st.markdown(message['content'])
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Examples, Capabilities, Limitations cards - only show when no messages
        columns = st.columns(3)
        examples = [
            'Explain quantum computing in simple terms',
            "Got any creative ideas for a 10 year old's birthday?",
            'How do I write a Javascript fetch request?',
        ]
        capabilities = [
            'Remembers what user said earlier in the conversation',
            'Allows user to provide follow-up corrections',
            'Trained to decline inappropriate requests',
        ]
        limitations = [
            'May occasionally generate incorrect information',
            'May occasionally produce harmful instructions',
            'Limited knowledge of world and events after 2021',
        ]

        icons = ['💡', '⚡', '⚠️']
        for col, heading, items, icon in zip(columns, ['Examples', 'Capabilities', 'Limitations'], [examples, capabilities, limitations], icons):
            with col:
                st.markdown(f"<h3 style='color:#e2e8f0; margin-bottom: 1rem;'>{icon} {heading}</h3>", unsafe_allow_html=True)
                for item in items:
                    st.markdown(f"<div class='example-card'><p style='margin:0; color:#c7d2fe; font-size: 0.95rem;'>{item}</p></div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
