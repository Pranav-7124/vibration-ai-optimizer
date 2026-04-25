"""
optimizer.py  Genetic Algorithm for Impact Damper Optimization
Replaces brute-force grid search with evolutionary optimization.

GA Parameters:
- Population: 120 individuals
- Generations: 60
- Tournament selection (k=3)
- Uniform crossover
- Gaussian mutation
"""

import joblib
import numpy as np

#  Load model & scaler 
model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

#  Parameter Bounds 
BOUNDS = {
    "mass_ratio": (0.005, 0.10),
    "clearance":  (0.10,  1.20),
    "location":   (0.25,  0.80),
}

POP_SIZE    = 120
N_GEN       = 60
CROSS_RATE  = 0.80
MUTATE_RATE = 0.15
TOURN_K     = 3


#  Helpers 

def _predict_amplitude(frequency, individual):
    """Predict amplitude for one individual [mr, cl, loc]."""
    mr, cl, loc = individual
    beta = frequency / 50.0            # freq_ratio
    # Approximate damping_ratio for the feature vector (avg structural)
    dr = 0.01 + 0.02 * mr             # rough estimate
    features = np.array([[frequency, mr, cl, loc, beta, dr]])
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)[0]


def _random_individual():
    return np.array([
        np.random.uniform(*BOUNDS["mass_ratio"]),
        np.random.uniform(*BOUNDS["clearance"]),
        np.random.uniform(*BOUNDS["location"]),
    ])


def _tournament_select(population, fitnesses):
    idxs = np.random.choice(len(population), TOURN_K, replace=False)
    best = idxs[np.argmin(fitnesses[idxs])]
    return population[best].copy()


def _crossover(p1, p2):
    """Uniform crossover."""
    mask = np.random.rand(len(p1)) < 0.5
    child1 = np.where(mask, p1, p2)
    child2 = np.where(mask, p2, p1)
    return child1, child2


def _mutate(individual):
    """Gaussian mutation with boundary clamping."""
    bounds_arr = np.array(list(BOUNDS.values()))
    ranges = bounds_arr[:, 1] - bounds_arr[:, 0]
    for i in range(len(individual)):
        if np.random.rand() < MUTATE_RATE:
            individual[i] += np.random.normal(0, 0.05 * ranges[i])
            individual[i] = np.clip(individual[i], bounds_arr[i, 0], bounds_arr[i, 1])
    return individual


def _clip_bounds(individual):
    bounds_arr = np.array(list(BOUNDS.values()))
    return np.clip(individual, bounds_arr[:, 0], bounds_arr[:, 1])


#  Main GA 

def find_best_config(frequency: float):
    """
    Run Genetic Algorithm to find optimal [mass_ratio, clearance, location]
    that minimizes vibration amplitude for the given frequency.

    Returns:
        dict with optimal parameters, predicted amplitude,
        and per-generation convergence history.
    """
    # Initialise population
    population = np.array([_random_individual() for _ in range(POP_SIZE)])
    fitnesses  = np.array([_predict_amplitude(frequency, ind) for ind in population])

    best_ever_idx = np.argmin(fitnesses)
    best_ever_ind = population[best_ever_idx].copy()
    best_ever_fit = fitnesses[best_ever_idx]

    convergence = []   # (generation, best_amplitude, mean_amplitude)

    for gen in range(N_GEN):
        new_population = []

        # Elitism  keep best 2
        elite_idxs = np.argsort(fitnesses)[:2]
        for idx in elite_idxs:
            new_population.append(population[idx].copy())

        # Fill rest with crossover + mutation
        while len(new_population) < POP_SIZE:
            p1 = _tournament_select(population, fitnesses)
            p2 = _tournament_select(population, fitnesses)

            if np.random.rand() < CROSS_RATE:
                c1, c2 = _crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = _mutate(_clip_bounds(c1))
            c2 = _mutate(_clip_bounds(c2))
            new_population.extend([c1, c2])

        population = np.array(new_population[:POP_SIZE])
        fitnesses  = np.array([_predict_amplitude(frequency, ind) for ind in population])

        gen_best_idx = np.argmin(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]
        gen_mean_fit = float(np.mean(fitnesses))

        if gen_best_fit < best_ever_fit:
            best_ever_fit = gen_best_fit
            best_ever_ind = population[gen_best_idx].copy()

        convergence.append({
            "generation": gen + 1,
            "best":       round(float(gen_best_fit), 6),
            "mean":       round(gen_mean_fit, 6),
        })

    mr, cl, loc = best_ever_ind
    return {
        "mass_ratio":   round(float(mr), 6),
        "clearance":    round(float(cl), 6),
        "location":     round(float(loc), 6),
        "amplitude":    round(float(best_ever_fit), 6),
        "convergence":  convergence,
    }