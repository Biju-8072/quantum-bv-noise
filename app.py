import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Suppress minor deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ---------------------------------------------------------
# Page Configuration & Dark Navy Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bernstein-Vazirani Noise Analysis",
    page_icon="⚛️",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Dark Navy Background and Base Text */
    .stApp {
        background-color: #0B132B;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Divider */
    hr {
        border-color: #1E293B !important;
        margin: 1.5rem 0 !important;
    }

    /* Input widgets and containers */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input {
        background-color: #162035 !important;
        color: #F8FAFC !important;
        border-color: #334155 !important;
    }

    /* Primary Run Simulation Button */
    div.stButton > button:first-child {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 2rem !important;
        cursor: pointer;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8 !important;
        border: none !important;
    }

    /* Streamlit DataFrame styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Quantum Simulation Logic
# ---------------------------------------------------------
def create_bv_circuit(hidden_string: str) -> QuantumCircuit:
    """
    Builds the Bernstein-Vazirani quantum circuit for a given binary hidden string.
    Uses n input qubits and 1 ancilla qubit (placed at index n).
    """
    n = len(hidden_string)
    qc = QuantumCircuit(n + 1, n)

    # Prepare ancilla qubit in state |->
    qc.x(n)
    qc.h(n)

    # Initialize input qubits to superposition |+>
    for i in range(n):
        qc.h(i)

    qc.barrier()

    # Oracle: CNOT gates from input qubits to ancilla for bits equal to '1'
    # In Qiskit's little-endian convention, reverse the string index
    for i, bit in enumerate(reversed(hidden_string)):
        if bit == "1":
            qc.cx(i, n)

    qc.barrier()

    # Apply final Hadamard transform to input qubits
    for i in range(n):
        qc.h(i)

    # Measure input qubits into classical register
    qc.measure(range(n), range(n))
    return qc


def build_depolarizing_noise_model(error_rate: float) -> NoiseModel:
    """
    Creates a depolarizing noise model applied to single-qubit (h, x)
    and two-qubit (cx) gates.
    """
    noise_model = NoiseModel()
    if error_rate > 0.0:
        error_1q = depolarizing_error(error_rate, 1)
        error_2q = depolarizing_error(error_rate, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ["h", "x"])
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
    return noise_model


def simulate_bv(hidden_string: str, noise_pct: float, shots: int) -> tuple[int, int, float]:
    """
    Executes the Bernstein-Vazirani circuit on Qiskit Aer simulator
    with specified depolarizing noise level and shot count.
    """
    circuit = create_bv_circuit(hidden_string)
    error_rate = noise_pct / 100.0

    if error_rate <= 0.0:
        simulator = AerSimulator()
    else:
        noise_model = build_depolarizing_noise_model(error_rate)
        simulator = AerSimulator(noise_model=noise_model)

    job = simulator.run(circuit, shots=shots)
    counts = job.result().get_counts()

    correct_shots = counts.get(hidden_string, 0)
    success_prob = round(correct_shots / shots, 4)

    return shots, correct_shots, success_prob


# ---------------------------------------------------------
# A. Project Title
# ---------------------------------------------------------
st.title("Noise Robustness Analysis of the Bernstein–Vazirani Quantum Algorithm: A Comparative Study of 3-bit, 6-bit, and 9-bit Hidden Strings")
st.markdown(
    "An experimental quantum simulation analyzing the impact of depolarizing gate noise on the success "
    "probability of the Bernstein–Vazirani algorithm across 3-bit, 6-bit, and 9-bit secret strings."
)

st.markdown("---")

# ---------------------------------------------------------
# B. Simulation Settings
# ---------------------------------------------------------
st.header("Simulation Settings")

col1, col2 = st.columns([1, 1])

with col1:
    length_option = st.selectbox(
        "Hidden-string length:",
        options=["Compare All", "3-bit", "6-bit", "9-bit"],
        index=0,
    )

    # Input strings based on selection
    if length_option == "3-bit":
        hidden_input_3 = st.text_input("Hidden String (3-bit):", value="101")
        target_strings = [("3-bit", hidden_input_3)]
    elif length_option == "6-bit":
        hidden_input_6 = st.text_input("Hidden String (6-bit):", value="110101")
        target_strings = [("6-bit", hidden_input_6)]
    elif length_option == "9-bit":
        hidden_input_9 = st.text_input("Hidden String (9-bit):", value="101101101")
        target_strings = [("9-bit", hidden_input_9)]
    else:
        # Compare All: display the default hidden strings
        st.markdown("**Default Hidden Strings:**")
        c3, c6, c9 = st.columns(3)
        with c3:
            hidden_input_3 = st.text_input("3-bit string:", value="101")
        with c6:
            hidden_input_6 = st.text_input("6-bit string:", value="110101")
        with c9:
            hidden_input_9 = st.text_input("9-bit string:", value="101101101")
        target_strings = [
            ("3-bit", hidden_input_3),
            ("6-bit", hidden_input_6),
            ("9-bit", hidden_input_9),
        ]

with col2:
    max_noise_level = st.slider(
        "Depolarizing noise level (%):",
        min_value=0.1,
        max_value=5.0,
        value=5.0,
        step=0.1,
        help="Simulates depolarizing gate error rates from 0.1% up to this selected value.",
    )

    shots_input = st.number_input(
        "Number of measurement shots:",
        min_value=128,
        max_value=10000,
        value=2048,
        step=256,
    )

run_simulation = st.button("Run Simulation")

# Determine noise points to test
if max_noise_level == 5.0:
    noise_points = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0]
elif max_noise_level <= 0.1:
    noise_points = [0.1]
else:
    raw_points = np.linspace(0.1, max_noise_level, 6)
    noise_points = sorted(list(set([round(float(p), 2) for p in raw_points])))

# ---------------------------------------------------------
# Validation and Simulation Execution
# ---------------------------------------------------------
expected_lengths = {"3-bit": 3, "6-bit": 6, "9-bit": 9}

if run_simulation:
    # Validate strings
    is_valid = True
    for label, s in target_strings:
        s_clean = s.strip()
        expected_len = expected_lengths[label]
        if len(s_clean) != expected_len:
            st.error(f"Error: {label} string '{s}' must have length {expected_len}. Currently {len(s_clean)}.")
            is_valid = False
            break
        if not set(s_clean).issubset({"0", "1"}):
            st.error(f"Error: {label} string '{s}' must contain only binary digits ('0' and '1').")
            is_valid = False
            break

    if is_valid:
        with st.spinner("Executing quantum circuits on Qiskit Aer simulator..."):
            records = []
            plot_series = {}

            for label, s in target_strings:
                s_clean = s.strip()
                probs = []
                for p in noise_points:
                    tot, corr, prob = simulate_bv(s_clean, p, int(shots_input))
                    records.append({
                        "Hidden-string length": label,
                        "Hidden string": s_clean,
                        "Noise level": f"{p:.1f}%",
                        "Total shots": tot,
                        "Correct shots": corr,
                        "Success probability": prob,
                    })
                    probs.append(prob)
                plot_series[label] = {
                    "string": s_clean,
                    "noise_points": noise_points,
                    "probs": probs,
                }

            # Save in session state
            st.session_state["results_df"] = pd.DataFrame(records)
            st.session_state["plot_series"] = plot_series
            st.session_state["length_option"] = length_option

st.markdown("---")

# ---------------------------------------------------------
# C. Main Graph
# ---------------------------------------------------------
st.header("Main Graph")

if "results_df" in st.session_state:
    plot_data = st.session_state["plot_series"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor("#0B132B")
    ax.set_facecolor("#162035")

    color_map = {
        "3-bit": ("#38BDF8", "o"),  # Sky Blue, Circle
        "6-bit": ("#34D399", "s"),  # Emerald Green, Square
        "9-bit": ("#FB923C", "^"),  # Amber Orange, Triangle
    }

    for label, data in plot_data.items():
        color, marker = color_map.get(label, ("#38BDF8", "o"))
        ax.plot(
            data["noise_points"],
            data["probs"],
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2.2,
            label=f"{label} ({data['string']})",
        )

    ax.set_title("Success Probability vs. Noise Level", fontsize=14, fontweight="bold", color="#FFFFFF", pad=12)
    ax.set_xlabel("Noise Level (%)", fontsize=11, fontweight="bold", color="#E2E8F0", labelpad=8)
    ax.set_ylabel("Success Probability", fontsize=11, fontweight="bold", color="#E2E8F0", labelpad=8)
    ax.set_ylim(-0.02, 1.05)

    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.grid(True, linestyle="--", alpha=0.3, color="#64748B")
    ax.legend(facecolor="#0B132B", edgecolor="#334155", labelcolor="#FFFFFF", fontsize=10, loc="lower left")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
else:
    st.info("Click 'Run Simulation' to generate the quantum noise robustness graph.")

st.markdown("---")

# ---------------------------------------------------------
# D. Results Table
# ---------------------------------------------------------
st.header("Results Table")

if "results_df" in st.session_state:
    st.dataframe(st.session_state["results_df"], use_container_width=True, hide_index=True)
else:
    st.info("Click 'Run Simulation' to populate the results table.")

st.markdown("---")

# ---------------------------------------------------------
# E. About This Project
# ---------------------------------------------------------
st.header("About This Project")

st.markdown(
    """
    ### 1. The Bernstein–Vazirani Algorithm
    The **Bernstein–Vazirani algorithm** solves a fundamental black-box problem: finding an unknown $n$-bit secret string $s$ 
    from a quantum oracle function:
    $$f(x) = s \\cdot x \\pmod 2$$
    - **Classical Complexity:** A classical algorithm requires $n$ independent evaluations (queries) to determine each bit of $s$.
    - **Quantum Advantage:** By utilizing quantum superposition and phase kickback, the Bernstein–Vazirani algorithm retrieves the complete secret string $s$ with 100% theoretical determinism in **a single query** ($O(1)$).

    ### 2. Depolarizing Noise Model
    In real-world quantum hardware (NISQ era), qubits undergo environmental decoherence and gate inaccuracies.
    - **Depolarizing Noise:** With probability $p$, a qubit state is replaced with a completely mixed state, effectively introducing bit-flip ($X$), phase-flip ($Z$), or combined bit-phase ($Y$) errors.
    - In this simulation, single-qubit depolarizing error $p$ is applied to single-qubit gates ($H, X$), and two-qubit depolarizing error is applied to Controlled-NOT ($CNOT$) gates.

    ### 3. Noise Sensitivity Across Bit Lengths
    As the hidden string length scales from 3-bit to 6-bit and 9-bit:
    - **Circuit Depth & Gate Count:** Longer strings require more input qubits ($n+1$ total qubits) and significantly more Hadamard and CNOT operations in the oracle.
    - **Error Accumulation:** Because quantum errors compound multiplicatively across consecutive gates, the overall circuit fidelity decays exponentially with gate count:
      $$P_{\\text{success}} \\approx (1 - p_{\\text{gate}})^{N_{\\text{gates}}}$$
    - **Algorithmic Outcome:** As evidenced in the graph and table, 9-bit strings experience a much steeper reduction in success probability compared to 3-bit strings under identical noise rates.
    """
)

