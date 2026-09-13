"""
Vehicle Route Optimization using Genetic Algorithm
Soft Computing Techniques — Mini Project (Streamlit Front-End)

Run locally:  streamlit run app.py
Deploy:       push this folder to GitHub, then deploy on streamlit.io/cloud
              pointing to this file as the main app.
"""

import math
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="VRP Optimizer — Genetic Algorithm", page_icon="🧬", layout="wide")

st.title("🧬 Vehicle Route Optimization using Genetic Algorithm")
st.caption("Soft Computing Techniques — Mini Project")

with st.expander("ℹ️ About this project", expanded=False):
    st.markdown(
        """
This app solves a **Vehicle Route Optimization** problem (a Traveling Salesman Problem)
using a **Genetic Algorithm**:

- **Chromosome** = a route (an ordering/permutation of locations)
- **Fitness** = `1 / total route distance` (shorter routes are fitter)
- **Selection** = Tournament Selection
- **Crossover** = Order Crossover (OX1)
- **Mutation** = Swap Mutation
- **Elitism** = best routes are carried over unchanged each generation

Enter your locations (or use the sample data), tune the GA parameters in the sidebar,
then click **Optimize Route**.
"""
    )

# ----------------------------------------------------------------------------
# Genetic Algorithm core (same logic as the Colab notebook)
# ----------------------------------------------------------------------------

def total_distance(route, coords):
    dist = 0.0
    n = len(route)
    for i in range(n):
        a = coords[route[i]]
        b = coords[route[(i + 1) % n]]
        dist += math.dist(a, b)
    return dist


def fitness(route, coords):
    d = total_distance(route, coords)
    return 1.0 / d if d > 0 else float("inf")


def create_route(num_cities):
    route = list(range(num_cities))
    random.shuffle(route)
    return route


def initial_population(pop_size, num_cities):
    return [create_route(num_cities) for _ in range(pop_size)]


def tournament_selection(population, fitnesses, k=5):
    contenders = random.sample(range(len(population)), k)
    best = max(contenders, key=lambda idx: fitnesses[idx])
    return population[best]


def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    fill_values = [g for g in parent2 if g not in child]
    pointer = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill_values[pointer]
            pointer += 1
    return child


def swap_mutation(route, mutation_rate):
    route = route[:]
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route


def genetic_algorithm(coords, pop_size, generations, elite_size, mutation_rate, tournament_k,
                       progress_callback=None):
    num_cities = len(coords)
    population = initial_population(pop_size, num_cities)
    best_distance_history = []
    best_route_overall = None
    best_distance_overall = float("inf")

    for gen in range(generations):
        fitnesses = [fitness(r, coords) for r in population]
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        population = [r for r, f in ranked]
        fitnesses = [f for r, f in ranked]

        current_best_route = population[0]
        current_best_distance = total_distance(current_best_route, coords)
        if current_best_distance < best_distance_overall:
            best_distance_overall = current_best_distance
            best_route_overall = current_best_route[:]
        best_distance_history.append(best_distance_overall)

        next_population = population[:elite_size]
        while len(next_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_k)
            parent2 = tournament_selection(population, fitnesses, tournament_k)
            child = ordered_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_rate)
            next_population.append(child)
        population = next_population

        if progress_callback is not None and (gen % 5 == 0 or gen == generations - 1):
            progress_callback((gen + 1) / generations)

    return best_route_overall, best_distance_overall, best_distance_history


# ----------------------------------------------------------------------------
# Sidebar — GA hyperparameters
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Genetic Algorithm Settings")
pop_size = st.sidebar.slider("Population size", 20, 500, 150, step=10)
generations = st.sidebar.slider("Generations", 50, 1000, 300, step=10)
elite_size = st.sidebar.slider("Elite size", 1, 50, 15, step=1)
mutation_rate = st.sidebar.slider("Mutation rate", 0.0, 0.5, 0.02, step=0.01)
tournament_k = st.sidebar.slider("Tournament size", 2, 15, 5, step=1)
seed = st.sidebar.number_input("Random seed (for reproducibility)", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Built for Soft Computing Techniques mini-project · Colab backend + Streamlit frontend")

# ----------------------------------------------------------------------------
# Location input
# ----------------------------------------------------------------------------
st.subheader("📍 Step 1 — Enter Your Locations")

default_data = pd.DataFrame(
    {
        "Name": ["A", "B", "C", "D", "E", "F"],
        "X": [0, 2, 5, 7, 9, 3],
        "Y": [0, 6, 2, 8, 1, 9],
    }
)

if "locations_df" not in st.session_state:
    st.session_state.locations_df = default_data.copy()

col1, col2 = st.columns([3, 1])
with col2:
    rand_n = st.number_input("Random count", min_value=3, max_value=25, value=8, key="rand_n", label_visibility="collapsed")
    if st.button("🎲 Generate random locations", use_container_width=True):
        rand_names = [chr(65 + i) if i < 26 else f"L{i}" for i in range(rand_n)]
        rng = np.random.default_rng()
        st.session_state.locations_df = pd.DataFrame(
            {
                "Name": rand_names,
                "X": rng.uniform(0, 10, rand_n).round(2),
                "Y": rng.uniform(0, 10, rand_n).round(2),
            }
        )
    if st.button("🔄 Reset to sample data", use_container_width=True):
        st.session_state.locations_df = default_data.copy()

st.caption("Add, edit, or delete rows below. Coordinates can be any consistent unit (km, grid units, etc.)")
edited_df = st.data_editor(
    st.session_state.locations_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Name": st.column_config.TextColumn("Location Name", required=True),
        "X": st.column_config.NumberColumn("X Coordinate", required=True),
        "Y": st.column_config.NumberColumn("Y Coordinate", required=True),
    },
    key="editor",
)
st.session_state.locations_df = edited_df

# Clean and validate
clean_df = edited_df.dropna()
clean_df = clean_df[clean_df["Name"].astype(str).str.strip() != ""]

st.markdown("---")
st.subheader("🚀 Step 2 — Optimize the Route")

run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    run_clicked = st.button("Optimize Route", type="primary", use_container_width=True)

if run_clicked:
    if len(clean_df) < 3:
        st.error("Please enter at least 3 valid locations (with Name, X, Y) to optimize a route.")
    else:
        random.seed(int(seed))
        np.random.seed(int(seed))

        names = clean_df["Name"].astype(str).tolist()
        coords = clean_df[["X", "Y"]].to_numpy(dtype=float)

        progress_bar = st.progress(0, text="Evolving population...")

        def update_progress(frac):
            progress_bar.progress(min(frac, 1.0), text=f"Evolving population... {int(frac * 100)}%")

        best_route, best_distance, history = genetic_algorithm(
            coords,
            pop_size=pop_size,
            generations=generations,
            elite_size=min(elite_size, pop_size - 1),
            mutation_rate=mutation_rate,
            tournament_k=min(tournament_k, pop_size),
            progress_callback=update_progress,
        )
        progress_bar.empty()

        st.session_state["result"] = {
            "names": names,
            "coords": coords,
            "best_route": best_route,
            "best_distance": best_distance,
            "history": history,
        }

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
if "result" in st.session_state:
    res = st.session_state["result"]
    names = res["names"]
    coords = res["coords"]
    best_route = res["best_route"]
    best_distance = res["best_distance"]
    history = res["history"]

    route_names = [names[i] for i in best_route] + [names[best_route[0]]]

    st.success(f"**Optimal route found!**  Total distance: **{best_distance:.3f}** units")
    st.markdown("### 🧭 Optimal Route Order")
    st.markdown(" → ".join(f"**{n}**" for n in route_names))

    tab1, tab2, tab3 = st.tabs(["🗺️ Route Map", "📈 Convergence Curve", "📋 Data Table"])

    with tab1:
        tour_coords = coords[best_route + [best_route[0]]]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=tour_coords[:, 0],
                y=tour_coords[:, 1],
                mode="lines+markers+text",
                line=dict(color="#E63946", width=3),
                marker=dict(size=16, color="#1D3557"),
                text=[names[i] for i in best_route] + [names[best_route[0]]],
                textposition="top center",
                textfont=dict(size=14, color="black"),
                name="Route",
            )
        )
        fig.update_layout(
            title=f"Optimized Vehicle Route (Total Distance: {best_distance:.2f})",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            height=550,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                y=history,
                mode="lines",
                line=dict(color="#2E86AB", width=2),
                name="Best Distance",
            )
        )
        fig2.update_layout(
            title="GA Convergence: Best Route Distance per Generation",
            xaxis_title="Generation",
            yaxis_title="Best Distance Found",
            height=450,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        dist_table = pd.DataFrame(
            {"Order": range(1, len(route_names) + 1), "Location": route_names}
        )
        st.dataframe(dist_table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download route as CSV",
            data=dist_table.to_csv(index=False),
            file_name="optimized_route.csv",
            mime="text/csv",
        )
else:
    st.info("Enter your locations above and click **Optimize Route** to see results.")
