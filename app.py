import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# High-End Engineering Suite UI Setup
st.set_page_config(page_title="EV Battery Thermal CAE Twin", layout="wide", initial_sidebar_state="expanded")

# --- CAD/CAE INDUSTRIAL GLOW THEME & CONTRAST FIXES ---
st.markdown("""
    <style>
    .main {background-color: #0b0f19; color: #e2e8f0;}

    /* Target Sidebar Elements Explicitly for Ultra-High Contrast */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }

    /* CRITICAL FIX: Force Dropdown/Selectbox Options Text to be Visible */
    div[data-baseweb="select"] * {
        color: #0f172a !important; /* Dark text inside the white input box */
    }
    ul[role="listbox"] * {
        color: #0f172a !important; /* Dark text inside the dropdown menu list */
    }

    /* Neumorphic Metric Panels */
    [data-testid="stMetricValue"] {
        color: #00f2fe !important;
        font-family: 'Courier New', monospace;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.3);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stMetric {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%) !important;
        padding: 22px !important;
        border-radius: 8px !important;
        border: 1px solid #38bdf8 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title & Structural Header
st.title("MULTI-PHYSICS CONJUGATE HEAT TRANSFER (CHT) DIGITAL TWIN")
st.markdown("**Computational Fluid Dynamics (CFD) Real-time Solver Environment | EV Battery Core Matrix**")
st.markdown("---")

# --- SIDEBAR SOLVER CONSTANTS ---
st.sidebar.markdown("BOUNDARY CONDITIONS")
design_option = st.sidebar.radio("CHASSIS CONFIGURATION:", ["Baseline Channel (Stagnant Wakes)", "Optimized Channel (Vortex Baffles)"])
coolant_type = st.sidebar.selectbox("WORKING FLUID:", ["Liquid Glycol-Water Mixture", "Air (Forced Convection)"])
flow_velocity = st.sidebar.slider("INLET VELOCITY (u_inf) [m/s]:", 0.5, 5.0, 2.0, step=0.5)
heat_load = st.sidebar.slider("HEAT FLUX PER ELEMENT (q) [W]:", 10, 50, 25)

# --- ADVANCED CFD MATH ENGINE ---
fluid_props = {
    "Air (Forced Convection)": {"rho": 1.2, "mu": 1.8e-5, "cp": 1005, "k": 0.026, "mult": 1.0},
    "Liquid Glycol-Water Mixture": {"rho": 1060, "mu": 2.5e-3, "cp": 3500, "k": 0.42, "mult": 0.12}
}
f_prop = fluid_props[coolant_type]

# Dynamic Physics Calculations
Re = (f_prop["rho"] * flow_velocity * 0.02) / f_prop["mu"]
Pr = (f_prop["cp"] * f_prop["mu"]) / f_prop["k"]

base_offsets = np.array([6.0, 14.0, 29.0, 42.0]) if design_option == "Baseline Channel (Stagnant Wakes)" else np.array([2.1, 3.5, 4.2, 4.8])
h_coeff = (0.023 * (Re**0.8) * (Pr**0.4) * (f_prop["k"] / 0.02)) * (1.4 if design_option == "Optimized Channel (Vortex Baffles)" else 1.0)

cell_temps = 298.15 + (heat_load / (h_coeff * 0.0012)) + base_offsets
max_temp = np.max(cell_temps)
delta_T = max_temp - np.min(cell_temps)
pressure_drop = 2.5 * (flow_velocity**2) * f_prop["rho"] * (1.8 if design_option == "Optimized Channel (Vortex Baffles)" else 1.0)

# --- CORE ENGINEERING KPIs ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="MAX JUNCTION TEMP (T_MAX)", value=f"{max_temp:.2f} K")
with col2:
    st.metric(label="CORE THERMAL DELTA (ΔT)", value=f"{delta_T:.2f} K")
with col3:
    st.metric(label="REYNOLDS NUMBER (Re)", value=f"{int(Re):,}")
with col4:
    st.metric(label="PRESSURE DROP (ΔP)", value=f"{int(pressure_drop)} Pa")

st.markdown("---")

# --- VISUALIZATION 1: 3D VECTOR STREAMLINES (CONES) ---
st.subheader("Real-time Fluid Dynamics Vector Streamlines")

x = np.linspace(0, 10, 35)
y = np.linspace(-3, 3, 22)
X, Y = np.meshgrid(x, y)

u = np.ones_like(X) * flow_velocity
v = np.zeros_like(Y)
cell_centers = [2.5, 5.0, 7.5, 10.0]

for xc in cell_centers:
    r_sq = (X - xc)**2 + Y**2
    r_sq[r_sq < 0.4] = 0.4
    if design_option == "Baseline Channel (Stagnant Wakes)":
        u -= 0.8 * flow_velocity * np.exp(-((X - xc - 0.6)**2 + Y**2)/0.5)
    else:
        u += 0.4 * flow_velocity * np.sin(X - xc) * np.exp(-r_sq/2)
        v += 0.5 * flow_velocity * Y * np.exp(-r_sq/2)

fig_vector = go.Figure(data=go.Cone(
    x=X.flatten(), y=Y.flatten(), z=np.zeros_like(X).flatten(),
    u=u.flatten(), v=v.flatten(), w=np.zeros_like(X).flatten(),
    colorscale="Viridis", sizemode="scaled", sizeref=0.35, showscale=True,
    colorbar=dict(title="Velocity (m/s)", titlefont=dict(color="#ffffff"), tickfont=dict(color="#ffffff"))
))
fig_vector.update_layout(template="plotly_dark", height=380, margin=dict(l=0, r=0, t=10, b=0),
                         xaxis=dict(title="Flow Channel Length"), yaxis=dict(title="Width"))
st.plotly_chart(fig_vector, use_container_width=True)

st.markdown("---")

# --- VISUALIZATION 2: FULL WIDTH 2D CONTOUR PLOT ---
st.subheader("Spatial Isothermal Contours (2D CHT Field Solver)")
grid_x, grid_y = np.meshgrid(np.linspace(0, 12, 100), np.linspace(-2.5, 2.5, 40))
thermal_field = np.ones_like(grid_x) * 298.15

for idx, xc in enumerate(cell_centers):
    dist_sq = (grid_x - xc)**2 + grid_y**2
    thermal_field += (cell_temps[idx] - 298.15) * np.exp(-dist_sq / (1.5 if design_option == "Baseline Channel (Stagnant Wakes)" else 0.6))

for xc in cell_centers:
    inside_cylinder = (grid_x - xc)**2 + grid_y**2 < 0.35
    thermal_field[inside_cylinder] = np.nan

fig_contour = go.Figure(data=go.Contour(
    z=thermal_field, x=np.linspace(0, 12, 100), y=np.linspace(-2.5, 2.5, 40),
    colorscale="Jet", line_width=0, connectgaps=False,
    colorbar=dict(title="Temperature (K)", titlefont=dict(color="#ffffff"), tickfont=dict(color="#ffffff"))
))
fig_contour.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10),
                         xaxis=dict(showgrid=False, title="Flow Direction Axis (mm)"), yaxis=dict(showgrid=False, title="Channel Width (mm)"))
st.plotly_chart(fig_contour, use_container_width=True)

st.markdown("---")

# --- VISUALIZATION 3: FULL WIDTH PERFORMANCE TRADEOFF PLOT ---
st.subheader("Heat Transfer Efficiency vs Pumping Penalty")
fig_tradeoff = make_subplots(specs=[[{"secondary_y": True}]])
v_arr = np.linspace(0.5, 5.0, 30)
Re_arr = (f_prop["rho"] * v_arr * 0.02) / f_prop["mu"]
Nu_arr = 0.023 * (Re_arr**0.8) * (Pr**0.4)
f_arr = 0.316 / (Re_arr**0.25)

fig_tradeoff.add_trace(go.Scatter(x=v_arr, y=Nu_arr, name="Nusselt Number (Nu)", line=dict(color="#10b981", width=3)), secondary_y=False)
fig_tradeoff.add_trace(go.Scatter(x=v_arr, y=f_arr, name="Friction Factor (f)", line=dict(color="#ef4444", width=2, dash="dash")), secondary_y=True)

fig_tradeoff.update_layout(template="plotly_dark", xaxis_title="Inlet Fluid Velocity (m/s)", height=380, margin=dict(l=10, r=10, t=10, b=10))
fig_tradeoff.update_yaxes(title_text="Convective Nusselt Scaling", secondary_y=False)
fig_tradeoff.update_yaxes(title_text="Darcy Friction Factor", secondary_y=True)
st.plotly_chart(fig_tradeoff, use_container_width=True)
