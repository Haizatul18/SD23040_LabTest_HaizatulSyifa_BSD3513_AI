import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Callable, List


# ===================== GA PROBLEM DEFINITION =====================
@dataclass
class GAProblem:
    name: str
    dim: int
    fitness_fn: Callable[[np.ndarray], float]


def make_bit_peak_problem(dim: int, peak_ones: int, max_fitness: int) -> GAProblem:
    """
    Fitness peaks when number of ones = peak_ones
    Max fitness = max_fitness
    """
    def fitness(x: np.ndarray) -> float:
        ones = int(np.sum(x))
        return float(max_fitness - abs(ones - peak_ones))

    return GAProblem(
        name="Bit Pattern GA (Peak at 40 Ones)",
        dim=dim,
        fitness_fn=fitness,
    )


# ===================== GA OPERATORS =====================
def init_population(pop_size: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, dim), dtype=np.int8)


def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    candidates = rng.integers(0, len(fitness), size=k)
    return int(candidates[np.argmax(fitness[candidates])])


def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    if len(a) <= 1:
        return a.copy(), b.copy()
    point = rng.integers(1, len(a))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def bit_mutation(x: np.ndarray, mutation_rate: float, rng: np.random.Generator):
    y = x.copy()
    mask = rng.random(x.shape) < mutation_rate
    y[mask] = 1 - y[mask]
    return y


def evaluate_population(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop])


# ===================== GENETIC ALGORITHM =====================
def run_ga(problem: GAProblem):
    # ---- FIXED PARAMETERS (AS REQUIRED) ----
    POPULATION_SIZE = 300
    CHROMOSOME_LENGTH = 80
    GENERATIONS = 50
    CROSSOVER_RATE = 0.9
    MUTATION_RATE = 0.01
    TOURNAMENT_K = 3
    ELITISM = 2
    SEED = 42

    rng = np.random.default_rng(SEED)

    population = init_population(POPULATION_SIZE, CHROMOSOME_LENGTH, rng)
    fitness = evaluate_population(population, problem)

    best_history: List[float] = []
    avg_history: List[float] = []

    chart_placeholder = st.empty()

    for gen in range(GENERATIONS):
        # Record statistics
        best_history.append(np.max(fitness))
        avg_history.append(np.mean(fitness))

        # Live chart
        df = pd.DataFrame({
            "Best Fitness": best_history,
            "Average Fitness": avg_history
        })
        chart_placeholder.line_chart(df)

        # Elitism
        elite_idx = np.argsort(fitness)[-ELITISM:]
        elites = population[elite_idx]

        new_population = []

        while len(new_population) < POPULATION_SIZE - ELITISM:
            p1 = population[tournament_selection(fitness, TOURNAMENT_K, rng)]
            p2 = population[tournament_selection(fitness, TOURNAMENT_K, rng)]

            if rng.random() < CROSSOVER_RATE:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = bit_mutation(c1, MUTATION_RATE, rng)
            c2 = bit_mutation(c2, MUTATION_RATE, rng)

            new_population.append(c1)
            if len(new_population) < POPULATION_SIZE - ELITISM:
                new_population.append(c2)

        population = np.vstack([np.array(new_population), elites])
        fitness = evaluate_population(population, problem)

    # Final best solution
    best_index = np.argmax(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]

    return best_solution, best_fitness, df


# ===================== STREAMLIT WEB APP =====================
st.set_page_config(
    page_title="Genetic Algorithm – Bit Pattern",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Algorithm Web App – Bit Pattern Generation")

st.markdown("""
**Fixed Parameters (as required):**
- Population Size: **300**
- Chromosome Length: **80 bits**
- Generations: **50**
- Fitness Peak at: **40 ones**
- Maximum Fitness: **80**
""")

if st.button("Run Genetic Algorithm", type="primary"):
    problem = make_bit_peak_problem(
        dim=80,
        peak_ones=40,
        max_fitness=80
    )

    best_solution, best_fitness, history = run_ga(problem)

    st.subheader("📈 Fitness Over Generations")
    st.line_chart(history)

    st.subheader("🏆 Best Bit Pattern Found")
    bitstring = "".join(map(str, best_solution.tolist()))
    ones_count = int(np.sum(best_solution))

    st.code(bitstring, language="text")
    st.write(f"**Number of Ones:** {ones_count} / 80")
    st.write(f"**Best Fitness:** {best_fitness:.2f}")
    st.write("✅ Optimal solution occurs when number of ones = 40")
