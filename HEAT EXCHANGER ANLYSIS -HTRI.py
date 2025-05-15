import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
from io import BytesIO
from datetime import datetime

# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Heat Exchanger FIV Analysis (TEMA/HTRI Standards)",
    page_icon="🔧",
    layout="wide"
)

# Constants (TEMA/HTRI Standards)
GRAVITY = 9.81  # m/s²
SPEED_OF_SOUND_WATER = 1481  # m/s
FEI_CONSTANTS = {
    "TEMA Standard": 3.0,
    "HTRI Conservative": 3.5,
    "HTRI Normal": 4.0,
    "HTRI Aggressive": 4.5,
    "Custom": None
}
LIFT_COEFFICIENT = 0.1000
ADDED_MASS_FACTOR = 1.9150
LOG_DECREMENT = 0.1000
STROUHAL_NUMBERS = {
    "Triangular": 0.33,
    "Square": 0.21,
    "Rotated Square": 0.24,
    "Rotated Triangular": 0.35
}

# Security
API_KEY = "M_A_K_1995"
user_key = st.sidebar.text_input("Enter API Key:", type="password")

if user_key != API_KEY:
    st.error("⚠️ Unauthorized access. Please enter a valid API Key.")
    st.stop()

# Now your app can start
st.title("Heat Exchanger FIV Analysis (TEMA/HTRI Standards)")
st.write("Comprehensive Flow-Induced Vibration Analysis Tool for Shell-and-Tube Heat Exchangers")

# Sidebar Inputs
with st.sidebar:
    st.header("📌 Design Parameters (TEMA CEM Type)")
    
    # Tube parameters
    tube_od = st.number_input("Tube OD (mm)", min_value=5.0, max_value=50.0, value=19.5, step=0.1)
    tube_thickness = st.number_input("Tube thickness (mm)", min_value=0.1, max_value=5.0, value=1.27, step=0.01)
    tube_id = tube_od - 2 * tube_thickness
    st.text_input("Tube ID (mm)", value=f"{tube_id:.2f}", disabled=True)
    tube_length = st.number_input("Tube length (mm)", min_value=1000.0, max_value=20000.0, value=3580.0, step=10.0)
    permissible_stress = st.number_input("Permissible stress (N/mm²)", min_value=10.0, max_value=500.0, value=54.1, step=1.0)
    modulus_elasticity = st.number_input("Modulus (N/mm²)", min_value=1e3, max_value=3e5, value=195000.0, step=1000.0)
    density_tube_material = st.number_input("Density (kg/m³)", min_value=5000.0, max_value=19000.0, value=8030.0, step=1.0)
    
    # Baffle parameters
    baffle_thickness = st.number_input("Baffle thickness (mm)", min_value=5.0, max_value=50.0, value=15.875, step=0.1)
    baffle_spacing_mid = st.number_input("Mid-span baffle spacing (mm)", min_value=100.0, max_value=2000.0, value=470.0, step=10.0)
    
    # Calculate number of baffles and spans
    number_of_baffles = max(1, int(tube_length / baffle_spacing_mid) - 1)
    total_unsupported_length = tube_length - (baffle_thickness * number_of_baffles)
    avg_span_length = total_unsupported_length / (number_of_baffles + 1)
    
    st.write(f"Calculated number of baffles: {number_of_baffles}")
    st.write(f"Average span length: {avg_span_length:.2f} mm")
    
    # Fluid parameters
    shell_side_fluid_density = st.number_input("Shell density (kg/m³)", min_value=1.0, max_value=2000.0, value=1000.0, step=1.0)
    tube_side_fluid_density = st.number_input("Tube density (kg/m³)", min_value=1.0, max_value=2000.0, value=1000.0, step=1.0)
    flow_velocity = st.number_input("Flow velocity (m/s)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    
    # Layout parameters
    tube_pitch = st.number_input("Tube pitch (mm)", min_value=10.0, max_value=50.0, value=23.8125, step=0.1)
    diametral_clearance = st.number_input("Diametral clearance (mm)", min_value=0.1, max_value=2.0, value=0.49276, step=0.01)
    tube_array_pattern = st.selectbox("Tube pattern", list(STROUHAL_NUMBERS.keys()))
    damping_ratio = st.number_input("Damping ratio", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    
    # FEI Constant selection
    fei_standard = st.selectbox("FEI Constant Standard", list(FEI_CONSTANTS.keys()))
    if fei_standard == "Custom":
        fluidelastic_instability_constant = st.number_input("Custom FEI constant", min_value=1.0, max_value=10.0, value=3.5, step=0.1)
    else:
        fluidelastic_instability_constant = FEI_CONSTANTS[fei_standard]
    
    # Other parameters
    added_mass_factor = st.number_input("Added mass factor", min_value=1.0, max_value=3.0, value=ADDED_MASS_FACTOR, step=0.01)
    lift_coefficient = st.number_input("Lift coefficient", min_value=0.01, max_value=1.0, value=LIFT_COEFFICIENT, step=0.01)
    log_decrement = st.number_input("Log decrement", min_value=0.001, max_value=0.5, value=LOG_DECREMENT, step=0.001)

def calculate_vibration_parameters(params):
    results = {}
    
    # Convert units to SI
    tube_od_m = params['tube_od'] / 1000  # mm to m
    tube_id_m = (params['tube_od'] - 2 * params['tube_thickness']) / 1000
    tube_length_m = params['tube_length'] / 1000
    tube_pitch_m = params['tube_pitch'] / 1000
    
    # Tube properties
    tube_cross_area = math.pi * (tube_od_m**2 - tube_id_m**2) / 4
    tube_mass_per_length = tube_cross_area * params['density_tube_material']
    tube_moment_inertia = math.pi * (tube_od_m**4 - tube_id_m**4) / 64
    
    # Calculate equivalent modulus
    E = params['modulus_elasticity'] * 1e6  # N/mm² to N/m²
    
    # Calculate spans
    number_of_baffles = max(1, int(params['tube_length'] / params['baffle_spacing_mid']) - 1)
    total_unsupported_length = params['tube_length'] - (params['baffle_thickness'] * number_of_baffles)
    avg_span_length = total_unsupported_length / (number_of_baffles + 1)
    results['Average Span Length'] = avg_span_length / 1000  # Convert to meters
    
    # 1. Natural Frequency Calculation
    fn = (3.516 / (2 * math.pi)) * math.sqrt((E * tube_moment_inertia) / 
          (tube_mass_per_length * (results['Average Span Length']**4)))
    results['Natural Frequency'] = fn
    
    # 2. Vortex Shedding
    strouhal = STROUHAL_NUMBERS[params['tube_array_pattern']]
    results['Strouhal Number'] = strouhal
    vortex_freq = strouhal * params['flow_velocity'] / tube_od_m
    results['Vortex Shedding Frequency'] = vortex_freq
    
    # 3. Turbulent Buffeting
    results['Turbulent Buffeting Force'] = 0.5 * params['shell_side_fluid_density'] * \
                                         (params['flow_velocity']**2) * tube_od_m * tube_length_m
    
    # 4. Fluid Elastic Instability
    mass_damping = (2 * math.pi * params['log_decrement'] * tube_mass_per_length) / \
                  (params['shell_side_fluid_density'] * tube_od_m**2)
    results['Fluid Elastic Instability Factor'] = params['fluidelastic_instability_constant'] * math.sqrt(mass_damping)
    critical_velocity = results['Fluid Elastic Instability Factor'] * fn * tube_od_m
    results['Critical Reduced Velocity'] = critical_velocity
    
    # 5. Acoustic Resonance
    results['Axial Resonance'] = SPEED_OF_SOUND_WATER / (2 * tube_length_m)
    results['Angular Resonance'] = SPEED_OF_SOUND_WATER / (2 * tube_pitch_m)
    
    # 6. Mid-span Deflection
    results['Max Displacement'] = (5 * tube_mass_per_length * GRAVITY * (results['Average Span Length']**4)) / \
                                 (384 * E * tube_moment_inertia) * 1000  # mm
    
    # 7. Wear Damage
    results['Wear Contact Events'] = int(1e6 * params['flow_velocity']**3 * (params['baffle_thickness']/1000))
    
    # 8. Fatigue Analysis
    dynamic_pressure = 0.5 * params['shell_side_fluid_density'] * params['flow_velocity']**2
    results['Fatigue Stress'] = dynamic_pressure * tube_od_m / (2 * params['tube_thickness']/1000) / 1e6  # MPa
    
    # 9. Noise Level
    results['Noise Level'] = 20 * math.log10(params['flow_velocity'] * 100)  # dB
    
    # 10. Pressure Drop
    results['Pressure Drop'] = 0.1 * (tube_length_m/(params['baffle_spacing_mid']/1000)) * \
                             params['shell_side_fluid_density'] * params['flow_velocity']**2 / 1e5  # bar
    
    # Additional calculations
    results['Gap Velocity Ratio'] = params['flow_velocity'] / critical_velocity
    results['Vortex Shedding Amplitude'] = 0.0582  # Standard value
    results['Vortex Shedding Ratio'] = results['Vortex Shedding Amplitude'] / (params['tube_pitch'] - params['tube_od'])
    
    return results

def check_acceptance_criteria(results, params):
    criteria = {}
    
    # 1. Vortex Shedding
    ratio = results['Vortex Shedding Frequency'] / results['Natural Frequency']
    criteria['Vortex Shedding'] = {
        'Status': ratio < 0.5 or ratio > 1.5,
        'Value': f"{ratio:.2f}",
        'Limit': "0.5-1.5"
    }
    
    # 2. Turbulent Buffeting
    criteria['Turbulent Buffeting'] = {
        'Status': results['Turbulent Buffeting Force'] < 1000,
        'Value': f"{results['Turbulent Buffeting Force']:.1f} N",
        'Limit': "<1000 N"
    }
    
    # 3. Fluid Elastic Instability
    velocity_ratio = params['flow_velocity'] / results['Critical Reduced Velocity']
    criteria['Fluid Elastic Instability'] = {
        'Status': velocity_ratio < 0.5,
        'Value': f"{velocity_ratio:.2f}",
        'Limit': "<0.5"
    }
    
    # 4. Acoustic Resonance
    axial_ratio = results['Vortex Shedding Frequency'] / results['Axial Resonance']
    angular_ratio = results['Vortex Shedding Frequency'] / results['Angular Resonance']
    criteria['Acoustic Resonance'] = {
        'Status': (axial_ratio < 0.8 or axial_ratio > 1.2) and (angular_ratio < 0.8 or angular_ratio > 1.2),
        'Value': f"Axial: {axial_ratio:.2f}, Angular: {angular_ratio:.2f}",
        'Limit': "0.8-1.2"
    }
    
    # 5. Mid-span Collision
    max_deflection = params['diametral_clearance'] / 2
    criteria['Mid-span Collision'] = {
        'Status': results['Max Displacement'] < max_deflection,
        'Value': f"{results['Max Displacement']:.2f} mm",
        'Limit': f"<{max_deflection:.2f} mm"
    }
    
    # 6. Wear Damage
    criteria['Wear Damage'] = {
        'Status': results['Wear Contact Events'] < 10000,
        'Value': f"{results['Wear Contact Events']}",
        'Limit': "<10000"
    }
    
    # 7. Fatigue Failure
    criteria['Fatigue Failure'] = {
        'Status': results['Fatigue Stress'] < 0.5*params['permissible_stress'],
        'Value': f"{results['Fatigue Stress']:.1f} MPa",
        'Limit': f"<{0.5*params['permissible_stress']:.1f} MPa"
    }
    
    # 8. Excessive Noise
    criteria['Excessive Noise'] = {
        'Status': results['Noise Level'] < 85,
        'Value': f"{results['Noise Level']:.1f} dB",
        'Limit': "<85 dB"
    }
    
    # 9. Pressure Drop
    criteria['Pressure Drop'] = {
        'Status': results['Pressure Drop'] < 1.0,
        'Value': f"{results['Pressure Drop']:.2f} bar",
        'Limit': "<1.0 bar"
    }
    
    # 10. Stress Corrosion
    criteria['Stress Corrosion'] = {
        'Status': results['Fatigue Stress'] < 0.3*params['permissible_stress'],
        'Value': f"{results['Fatigue Stress']:.1f} MPa",
        'Limit': f"<{0.3*params['permissible_stress']:.1f} MPa"
    }
    
    return criteria

def create_velocity_vibration_graph(results, params):
    """Create a graph showing vibration risk vs flow velocity"""
    velocities = np.linspace(0.1, 5.0, 50)  # Range of velocities from 0.1 to 5 m/s
    
    # Calculate vortex shedding frequencies across velocity range
    strouhal = STROUHAL_NUMBERS[params['tube_array_pattern']]
    vortex_freqs = [strouhal * v / (params['tube_od']/1000) for v in velocities]
    
    # Calculate FEI critical velocities
    mass_damping = (2 * math.pi * params['log_decrement'] * 
                   (math.pi * ((params['tube_od']/1000)**2 - 
                    ((params['tube_od']-2*params['tube_thickness'])/1000)**2) / 4 * 
                    params['density_tube_material'])) / \
                  (params['shell_side_fluid_density'] * (params['tube_od']/1000)**2)
    fei_factor = params['fluidelastic_instability_constant'] * math.sqrt(mass_damping)
    critical_velocity = fei_factor * results['Natural Frequency'] * (params['tube_od']/1000)
    critical_velocities = [critical_velocity] * len(velocities)
    
    # Calculate ratios
    vortex_ratios = [vf / results['Natural Frequency'] for vf in vortex_freqs]
    fei_ratios = [v / critical_velocity for v in velocities]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(velocities, vortex_ratios, 'b-', label='Vortex Shedding Ratio (f/fn)')
    ax.plot(velocities, fei_ratios, 'r--', label='FEI Ratio (V/Vc)')
    
    # Add critical regions
    ax.axhspan(0.8, 1.2, color='red', alpha=0.1, label='Vortex Shedding Danger Zone')
    ax.axhline(0.5, color='green', linestyle=':', label='FEI Threshold (0.5)')
    ax.axvline(params['flow_velocity'], color='black', linestyle='-', 
               label=f'Design Velocity ({params["flow_velocity"]} m/s)')
    
    ax.set_title('Vibration Risk vs Flow Velocity')
    ax.set_xlabel('Flow Velocity (m/s)')
    ax.set_ylabel('Vibration Risk Ratio')
    ax.grid(True)
    ax.legend()
    
    return fig

def create_vibration_graph(results, params):
    """Create vibration response graph"""
    time = np.linspace(0, 0.1, 1000)  # Short time period to show vibration
    displacement = results['Vortex Shedding Amplitude'] * np.sin(2 * np.pi * results['Natural Frequency'] * time)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, displacement, 'b-', linewidth=2)
    ax.set_title('Tube Vibration Response')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Displacement (mm)')
    ax.grid(True)
    
    return fig

def add_diagonal_watermark(fig):
    """Add diagonal watermark to a figure"""
    fig.text(0.5, 0.5, 'MAK-FIV Analysis',
             rotation=45, fontsize=40, color='gray',
             alpha=0.1, ha='center', va='center',
             bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none", alpha=0.1))

def create_pdf_report(params, results, criteria):
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.pyplot import figure, text, axis, savefig, close
    from matplotlib.table import Table
    import numpy as np
    
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Title Page
        fig = figure(figsize=(8.5, 11))
        text(0.5, 0.8, 'CEM Heat Exchanger FIV Analysis Report', 
             ha='center', va='center', fontsize=18, fontweight='bold')
        text(0.5, 0.75, 'TEMA/HTRI Standards Compliance', 
             ha='center', va='center', fontsize=14)
        text(0.5, 0.7, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
             ha='center', va='center', fontsize=12)
        
        # Add diagonal watermark
        add_diagonal_watermark(fig)
        
        axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        close()
        
        # Vibration Data Summary
        fig = figure(figsize=(8.5, 11))
        text(0.1, 0.95, 'Vibration Data Summary', fontsize=16, fontweight='bold')
        
        # Create a table with vibration data
        fig_table = figure(figsize=(8, 4))
        ax = fig_table.add_subplot(111)
        
        vibration_data = [
            ["Natural Frequency", f"{results['Natural Frequency']:.3f} Hz"],
            ["Average Span Length", f"{results['Average Span Length']:.3f} m"],
            ["Tube Pitch", f"{params['tube_pitch']} mm"],
            ["Tube Gap", f"{params['tube_pitch'] - params['tube_od']:.3f} mm"],
            ["FEI Constant", f"{params['fluidelastic_instability_constant']}"],
            ["Lift Coefficient", f"{params['lift_coefficient']}"],
            ["Added Mass Factor", f"{params['added_mass_factor']}"],
            ["Log Decrement", f"{params['log_decrement']}"]
        ]
        
        table = ax.table(cellText=vibration_data,
                        colLabels=["Parameter", "Value"],
                        loc='center',
                        cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.axis('off')
        ax.set_title('Vibration Data', fontweight='bold')
        
        # Add diagonal watermark to table figure
        add_diagonal_watermark(fig_table)
        
        pdf.savefig(fig_table, bbox_inches='tight')
        close(fig_table)
        
        # Add diagonal watermark to main figure
        add_diagonal_watermark(fig)
        
        axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        close()
        
        # Analysis Results
        fig = figure(figsize=(8.5, 11))
        text(0.1, 0.95, 'Analysis Results', fontsize=16, fontweight='bold')
        
        # Create a table with analysis results
        fig_table = figure(figsize=(8, 4))
        ax = fig_table.add_subplot(111)
        
        analysis_data = [
            ["Frequency", f"{results['Natural Frequency']:.3f} Hz"],
            ["Gap Velocity Ratio", f"{results['Gap Velocity Ratio']:.4f}"],
            ["Max Vortex Shedding Amplitude", f"{results['Vortex Shedding Amplitude']:.4f} mm"],
            ["Vortex Shedding Ratio", f"{results['Vortex Shedding Ratio']:.4f}"]
        ]
        
        table = ax.table(cellText=analysis_data,
                        colLabels=["Parameter", "Value"],
                        loc='center',
                        cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.axis('off')
        ax.set_title('Analysis Results', fontweight='bold')
        
        # Add diagonal watermark to table figure
        add_diagonal_watermark(fig_table)
        
        pdf.savefig(fig_table, bbox_inches='tight')
        close(fig_table)
        
        # Add diagonal watermark to main figure
        add_diagonal_watermark(fig)
        
        axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        close()
        
        # Visualizations
        fig = figure(figsize=(8.5, 11))
        text(0.1, 0.95, 'Visualizations', fontsize=16, fontweight='bold')
        axis('off')
        
        # Add diagonal watermark
        add_diagonal_watermark(fig)
        
        pdf.savefig(fig, bbox_inches='tight')
        close()
        
        # Add velocity-vibration graph
        fig_vel = create_velocity_vibration_graph(results, params)
        add_diagonal_watermark(fig_vel)
        pdf.savefig(fig_vel, bbox_inches='tight')
        close(fig_vel)
        
        # Add vibration graph
        fig_vib = create_vibration_graph(results, params)
        add_diagonal_watermark(fig_vib)
        pdf.savefig(fig_vib, bbox_inches='tight')
        close(fig_vib)
        
        # Detailed Acceptance Criteria
        fig = figure(figsize=(8.5, 11))
        text(0.1, 0.95, 'Detailed Acceptance Criteria', fontsize=16, fontweight='bold')
        axis('off')
        
        # Create a table with all criteria
        fig_table = figure(figsize=(8, 8))
        ax = fig_table.add_subplot(111)
        
        criteria_data = []
        for key in criteria:
            criteria_data.append([
                key,
                criteria[key]['Value'],
                criteria[key]['Limit'],
                "PASS" if criteria[key]['Status'] else "FAIL"
            ])
        
        table = ax.table(cellText=criteria_data,
                        colLabels=["Mechanism", "Value", "Limit", "Status"],
                        loc='center',
                        cellLoc='center')
        
        # Color cells based on status
        for i in range(1, len(criteria_data)+1):
            status_cell = table[i, 3]
            if "FAIL" in criteria_data[i-1][3]:
                status_cell.set_facecolor('lightcoral')
            else:
                status_cell.set_facecolor('lightgreen')
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax.axis('off')
        ax.set_title('Detailed Acceptance Criteria', fontweight='bold')
        
        # Add diagonal watermark to table figure
        add_diagonal_watermark(fig_table)
        
        pdf.savefig(fig_table, bbox_inches='tight')
        close(fig_table)
        
        # Add diagonal watermark to main figure
        add_diagonal_watermark(fig)
        
        pdf.savefig(fig, bbox_inches='tight')
        close()
    
    return buf

# Main App
st.title("CEM Heat Exchanger Flow-Induced Vibration Analysis")
st.subheader("TEMA/HTRI Standards Compliance")

# Prepare parameters
params = {
    'tube_od': tube_od,
    'tube_thickness': tube_thickness,
    'tube_length': tube_length,
    'density_tube_material': density_tube_material,
    'permissible_stress': permissible_stress,
    'modulus_elasticity': modulus_elasticity,
    'baffle_thickness': baffle_thickness,
    'baffle_spacing_mid': baffle_spacing_mid,
    'shell_side_fluid_density': shell_side_fluid_density,
    'tube_side_fluid_density': tube_side_fluid_density,
    'flow_velocity': flow_velocity,
    'tube_pitch': tube_pitch,
    'diametral_clearance': diametral_clearance,
    'tube_array_pattern': tube_array_pattern,
    'damping_ratio': damping_ratio,
    'added_mass_factor': added_mass_factor,
    'fluidelastic_instability_constant': fluidelastic_instability_constant,
    'lift_coefficient': lift_coefficient,
    'log_decrement': log_decrement
}

# Calculations
results = calculate_vibration_parameters(params)
criteria = check_acceptance_criteria(results, params)

# Display Results in Output Summary Format
st.header("OUTPUT SUMMARY")
st.subheader("FLOW INDUCED VIBRATION MECHANISMS")

# Vibration Data
st.markdown("### Vibration Data")
vib_col1, vib_col2 = st.columns(2)
with vib_col1:
    st.write(f"Average span length: {results['Average Span Length']:.3f} m")
    st.write(f"Tube pitch: {params['tube_pitch']} mm")
    st.write(f"FEI constant: {params['fluidelastic_instability_constant']}")
with vib_col2:
    st.write(f"Tube gap: {params['tube_pitch'] - params['tube_od']:.3f} mm")
    st.write(f"Lift coefficient: {params['lift_coefficient']}")
    st.write(f"Log decrement: {params['log_decrement']}")

# Analysis Results
st.markdown("### Analysis Results")
analysis_data = {
    "Parameter": ["Frequency (Hz)", "Gap Velocity Ratio", "Max Vortex Shedding Amplitude (mm)", "Vortex Shedding Ratio"],
    "Value": [f"{results['Natural Frequency']:.3f}", f"{results['Gap Velocity Ratio']:.4f}", 
              f"{results['Vortex Shedding Amplitude']:.4f}", f"{results['Vortex Shedding Ratio']:.4f}"]
}
st.table(pd.DataFrame(analysis_data))

# Detailed Mechanism Analysis
st.subheader("DETAILED MECHANISM ANALYSIS")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### VORTEX SHEDDING")
    st.write(f"I) Natural Frequency = {results['Natural Frequency']:.3f} Hz")
    st.write(f"II) Strouhal number = {results['Strouhal Number']:.2f}")
    st.write(f"III) Vortex shedding frequency = {results['Vortex Shedding Frequency']:.2f} Hz")
    status = "✅ ACCEPTABLE" if criteria['Vortex Shedding']['Status'] else "❌ NOT ACCEPTABLE"
    st.markdown(f"**STATUS OF VORTEX SHEDDING:** {status}")

    st.markdown("### TURBULENT BUFFETING")
    st.write(f"I) Turbulent Buffeting Force = {results['Turbulent Buffeting Force']:.1f} N")
    status = "✅ ACCEPTABLE" if criteria['Turbulent Buffeting']['Status'] else "❌ NOT ACCEPTABLE"
    st.markdown(f"**STATUS OF TURBULENT BUFFETING:** {status}")

with col2:
    st.markdown("### FLUID ELASTIC INSTABILITY")
    st.write(f"I) Fluid Elastic Instability Factor = {results['Fluid Elastic Instability Factor']:.4f}")
    st.write(f"II) Critical Reduced Velocity = {results['Critical Reduced Velocity']:.4f} m/s")
    status = "✅ ACCEPTABLE" if criteria['Fluid Elastic Instability']['Status'] else "❌ NOT ACCEPTABLE"
    st.markdown(f"**STATUS OF FLUID ELASTIC INSTABILITY:** {status}")

    st.markdown("### ACOUSTIC RESONANCE")
    st.write(f"Axial Resonance = {results['Axial Resonance']:.2f} Hz")
    st.write(f"Angular Resonance = {results['Angular Resonance']:.2f} Hz")
    status = "✅ ACCEPTABLE" if criteria['Acoustic Resonance']['Status'] else "❌ NOT ACCEPTABLE"
    st.markdown(f"**STATUS OF ACOUSTIC RESONANCE:** {status}")

# Damage Effects
st.subheader("POSSIBILITY DAMAGING EFFECT OF THE FIV ON HEAT EXCHANGER")
damage_cols = st.columns(3)
with damage_cols[0]:
    st.write(f"I) Max Displacement = {results['Max Displacement']:.4f} mm")
    st.write(f"IV) Noise Level = {results['Noise Level']:.1f} dB")
with damage_cols[1]:
    st.write(f"II) Mid-span Collision Risk = {'YES' if not criteria['Mid-span Collision']['Status'] else 'NO'}")
    st.write(f"V) Pressure Drop = {results['Pressure Drop']:.4f} bar")
with damage_cols[2]:
    st.write(f"III) Wear Contact Events = {results['Wear Contact Events']}")
    st.write(f"VI) Stress Corrosion Cracking Risk = {'HIGH' if not criteria['Stress Corrosion']['Status'] else 'LOW'}")

# Visualizations
st.header("Vibration Analysis Visualizations")
fig1 = create_velocity_vibration_graph(results, params)
fig2 = create_vibration_graph(results, params)

viz_col1, viz_col2 = st.columns(2)
with viz_col1:
    st.pyplot(fig1)
    st.caption("Figure 1: Vibration risk factors vs flow velocity. Shows critical regions for vortex shedding and FEI.")
with viz_col2:
    st.pyplot(fig2)
    st.caption("Figure 2: Time-domain vibration response at natural frequency.")

# PDF Report Generation
if st.button("📥 Generate Comprehensive PDF Report"):
    with st.spinner('Generating professional report...'):
        pdf_buffer = create_pdf_report(params, results, criteria)
        
    st.success('Report generated successfully!')
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="CEM_Heat_Exchanger_FIV_Analysis.pdf",
        mime="application/pdf"
    )