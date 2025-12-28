#!/usr/bin/env python3
"""
TSP + Deadlines (Caso 03)
------------------------
A truck starts at the depot (node 0) and must visit each customer exactly once.
Each customer i has a deadline (minutes). If arriving after deadline, the delivered
material is considered expired.

We assume:
- Speed: 1 distance unit == 1 minute (distance -> travel time)
- Service time: service_min at each customer (added on arrival)
- Objective: minimize total distance while satisfying deadlines,
  or minimize a penalized cost: distance_total + lambda * total_tardiness

This script implements a Genetic Algorithm (GA) using ONLY NumPy:
- Chromosome: permutation of customers [1..n]
- Fitness/cost: distance + lambda * tardiness
- Selection: tournament + elitism
- Crossover: Order Crossover (OX)
- Mutation: inversion (2-opt style) + insertion ("move")

It includes the sample dataset from the statement by default.

Usage:
    python tsp_deadlines_numpy.py

Notes:
- This is a heuristic (GA). For small instances it usually finds a good/feasible tour.
- If feasibility is hard, increase LAMBDA and/or generations/population size.
"""

import numpy as np


# -----------------------------
# Problem data (sample dataset)
# -----------------------------
def load_sample_dataset():
    """
    Returns:
        coords: (N,2) float array of x,y
        deadline: (N,) float array in minutes
        service: (N,) float array in minutes
        depot_index: int (0)
        customers: (n_customers,) int array [1..N-1]
    Dataset taken from the statement (small validation set).
    """
    # Columns: node_id, type, x, y, deadline_min, service_min
    # Node 0 is depot.
    data = np.array([
        [0, 0, 0, 0,  0, 0],
        [1, 1, 2, 7, 18, 2],
        [2, 1, 6, 4, 22, 2],
        [3, 1, 8, 9, 35, 3],
        [4, 1, 3, 1, 16, 2],
        [5, 1, 9, 2, 28, 2],
        [6, 1, 5, 8, 30, 2],
    ], dtype=float)

    coords = data[:, 2:4]
    deadline = data[:, 4]
    service = data[:, 5]
    depot_index = 0
    customers = np.arange(1, coords.shape[0], dtype=int)
    return coords, deadline, service, depot_index, customers


def compute_distance_matrix(coords):
    """
    Euclidean distance matrix using NumPy broadcasting.

    Args:
        coords: (N,2) array

    Returns:
        D: (N,N) array, D[i,j] = distance(i,j)
    """
    dif = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(dif * dif, axis=2))


# -----------------------------
# Route simulation & cost
# -----------------------------
def evaluate_route(perm, D, deadline, service, depot=0, return_to_depot=True):
    """
    Simulate visiting customers in the order given by perm.

    Time model:
        - start at t=0 at depot
        - travel time equals distance
        - upon arrival at customer i, record arrival time (before service)
        - then add service[i]
    Deadlines:
        tardiness_i = max(0, arrival_i - deadline_i)
    Distance:
        sum of travel distances along the path, including optional return to depot

    Args:
        perm: (n,) int array, permutation of customers (no depot)
        D: (N,N) distance matrix
        deadline: (N,) deadlines
        service: (N,) service time
        depot: depot node id
        return_to_depot: whether to close the tour (back to depot)

    Returns:
        total_dist: float
        total_tardiness: float
        arrival_times: (n,) float arrival times at each customer in perm order
    """
    t = 0.0
    total_dist = 0.0
    total_tard = 0.0

    arrival_times = np.empty(len(perm), dtype=float)

    prev = depot
    for k, node in enumerate(perm):
        travel = D[prev, node]
        total_dist += travel
        t += travel

        arrival_times[k] = t
        if node != depot:
            late = t - deadline[node]
            if late > 0:
                total_tard += late

        t += service[node]
        prev = node

    if return_to_depot:
        total_dist += D[prev, depot]
        # (Typically no deadline/service at depot)

    return total_dist, total_tard, arrival_times


def penalized_cost(perm, D, deadline, service, lam, depot=0, return_to_depot=True):
    """
    Penalized objective:
        cost = total_distance + lam * total_tardiness

    lam should be "large" to prioritize feasibility (meeting deadlines).
    """
    dist, tard, _ = evaluate_route(perm, D, deadline, service, depot, return_to_depot)
    return dist + lam * tard, dist, tard


# -----------------------------
# GA operators (NumPy only)
# -----------------------------
def init_population(pop_size, customers, rng):
    """
    Create initial population: random permutations of customers.
    Returns (pop_size, n_customers) int array.
    """
    n = len(customers)
    pop = np.empty((pop_size, n), dtype=int)
    for i in range(pop_size):
        pop[i] = rng.permutation(customers)
    return pop


def tournament_select(costs, k, rng):
    """
    Tournament selection (minimization):
      - pick k individuals at random
      - return index of the one with lowest cost
    """
    idx = rng.integers(0, len(costs), size=k)
    best = idx[np.argmin(costs[idx])]
    return best


def order_crossover_OX(p1, p2, rng):
    """
    Order Crossover (OX) for permutations.
    Steps:
      1) choose cut points a < b
      2) child[a:b] = p1[a:b]
      3) fill remaining positions in order from p2, skipping already present

    Returns:
        child (n,) int array
    """
    n = len(p1)
    a = rng.integers(0, n - 1)
    b = rng.integers(a + 1, n)

    child = np.full(n, -1, dtype=int)
    child[a:b] = p1[a:b]

    used = np.zeros(n, dtype=bool)  # used positions in p1? Not by value; we need set of values.
    # Map values to "used" via a value->index trick:
    # values are customer ids (not necessarily 0..n-1), so we create a boolean mask via membership check.
    # Since n is small/medium, we can do it with np.isin efficiently.
    # We'll just maintain a Python-free method:
    taken = child[a:b]

    fill = p2[~np.isin(p2, taken)]
    # Fill left part then right part
    pos = np.concatenate((np.arange(0, a), np.arange(b, n)))
    child[pos] = fill
    return child


def mutate_inversion(perm, rng):
    """
    Inversion mutation (2-opt style):
      - pick i < j and reverse perm[i:j]
    """
    n = len(perm)
    i = rng.integers(0, n - 1)
    j = rng.integers(i + 1, n)
    perm[i:j] = perm[i:j][::-1]


def mutate_insertion(perm, rng):
    """
    Insertion (move) mutation:
      - pick i != j
      - remove element at i and insert at j
    """
    n = len(perm)
    i = rng.integers(0, n)
    j = rng.integers(0, n - 1)
    if j >= i:
        j += 1  # ensure j != i
    val = perm[i]
    if i < j:
        perm[i:j] = perm[i + 1:j + 1]
        perm[j] = val
    else:
        perm[j + 1:i + 1] = perm[j:i]
        perm[j] = val


# -----------------------------
# Main GA loop
# -----------------------------
def solve_tsp_deadlines_ga(
    coords, deadline, service, depot, customers,
    pop_size=200,
    generations=1500,
    lam=200.0,
    elite_frac=0.05,
    tourn_k=3,
    cx_prob=0.9,
    mut_prob=0.3,
    inv_prob=0.7,
    return_to_depot=True,
    seed=42,
    verbose=True
):
    """
    Genetic Algorithm solver.

    Args:
        lam: penalty weight for tardiness
        elite_frac: fraction of best individuals copied directly each generation
        cx_prob: crossover probability per child
        mut_prob: mutation probability per child
        inv_prob: if mutating, probability to use inversion vs insertion
    Returns:
        best_perm, best_cost, best_dist, best_tard
    """
    rng = np.random.default_rng(seed)
    D = compute_distance_matrix(coords)

    pop = init_population(pop_size, customers, rng)

    elite_n = max(1, int(np.floor(elite_frac * pop_size)))

    best_perm = None
    best_cost = np.inf
    best_dist = np.inf
    best_tard = np.inf

    for gen in range(generations):
        # Evaluate population
        costs = np.empty(pop_size, dtype=float)
        dists = np.empty(pop_size, dtype=float)
        tards = np.empty(pop_size, dtype=float)

        for i in range(pop_size):
            c, d, t = penalized_cost(pop[i], D, deadline, service, lam, depot, return_to_depot)
            costs[i] = c
            dists[i] = d
            tards[i] = t

        # Track global best (prefer feasibility via lower tardiness, then cost)
        # A simple approach: compare by (tardiness, cost)
        idx_sorted = np.lexsort((costs, tards))  # primary: tards, secondary: costs
        cur_best = idx_sorted[0]

        if (tards[cur_best] < best_tard) or (tards[cur_best] == best_tard and costs[cur_best] < best_cost):
            best_perm = pop[cur_best].copy()
            best_cost = costs[cur_best]
            best_dist = dists[cur_best]
            best_tard = tards[cur_best]

        if verbose and (gen % max(1, generations // 10) == 0 or gen == generations - 1):
            print(f"Gen {gen:4d} | best tard={best_tard:8.3f} | best dist={best_dist:8.3f} | best cost={best_cost:10.3f}")

        # Elitism: copy top elite_n by (tardiness, cost)
        elites = pop[idx_sorted[:elite_n]].copy()

        # Create next generation
        new_pop = np.empty_like(pop)
        new_pop[:elite_n] = elites

        # Fill rest
        for i in range(elite_n, pop_size):
            # Select parents
            p1 = pop[tournament_select(costs, tourn_k, rng)]
            p2 = pop[tournament_select(costs, tourn_k, rng)]

            child = p1.copy()
            if rng.random() < cx_prob:
                child = order_crossover_OX(p1, p2, rng)

            if rng.random() < mut_prob:
                if rng.random() < inv_prob:
                    mutate_inversion(child, rng)
                else:
                    mutate_insertion(child, rng)

            new_pop[i] = child

        pop = new_pop

    return best_perm, best_cost, best_dist, best_tard, D


def pretty_print_solution(best_perm, D, deadline, service, depot=0, return_to_depot=True):
    """
    Print route with arrival times, deadlines, tardiness, total distance, total tardiness.
    """
    dist, tard, arrivals = evaluate_route(best_perm, D, deadline, service, depot, return_to_depot)

    route = np.concatenate(([depot], best_perm, [depot] if return_to_depot else []))
    print("\nBest route:")
    print("  " + " -> ".join(map(str, route.tolist())))

    print("\nPer-customer timing (arrival before service):")
    print("  node | arrival | deadline | tardiness | service")
    for k, node in enumerate(best_perm):
        arr = arrivals[k]
        dl = deadline[node]
        td = max(0.0, arr - dl)
        sv = service[node]
        print(f"  {node:>4d} | {arr:7.3f} | {dl:8.3f} | {td:9.3f} | {sv:7.3f}")

    print(f"\nTotals:")
    print(f"  total_distance = {dist:.3f}")
    print(f"  total_tardiness = {tard:.3f}")


def main():
    # Load dataset from the statement
    coords, deadline, service, depot, customers = load_sample_dataset()

    # GA hyperparameters (tune if needed)
    POP_SIZE = 250
    GENERATIONS = 2000
    LAMBDA = 250.0           # penalty weight (bigger => prioritize deadlines more)
    ELITE_FRAC = 0.06
    TOURN_K = 3
    CX_PROB = 0.9
    MUT_PROB = 0.35
    INV_PROB = 0.7
    RETURN_TO_DEPOT = True
    SEED = 7

    best_perm, best_cost, best_dist, best_tard, D = solve_tsp_deadlines_ga(
        coords=coords,
        deadline=deadline,
        service=service,
        depot=depot,
        customers=customers,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        lam=LAMBDA,
        elite_frac=ELITE_FRAC,
        tourn_k=TOURN_K,
        cx_prob=CX_PROB,
        mut_prob=MUT_PROB,
        inv_prob=INV_PROB,
        return_to_depot=RETURN_TO_DEPOT,
        seed=SEED,
        verbose=True
    )

    print("\n==============================")
    print("FINAL BEST (lexicographic by tardiness then cost)")
    print(f"best_perm = {best_perm.tolist()}")
    print(f"best_dist = {best_dist:.3f}")
    print(f"best_tard = {best_tard:.3f}")
    print(f"best_cost = {best_cost:.3f}   (cost = dist + lambda*tard)")
    print("==============================")

    pretty_print_solution(best_perm, D, deadline, service, depot, RETURN_TO_DEPOT)


if __name__ == "__main__":
    main()
