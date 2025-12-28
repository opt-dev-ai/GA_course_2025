#!/usr/bin/env python3
"""
Caso 05 - VRPTW (Vehicle Routing Problem with Time Windows / simple deadlines)
----------------------------------------------------------------------------
Several trucks depart from depot 0. Each customer must be served exactly once.

Constraints (from the statement):
- Capacity per truck: CAPACITY (default 8)
- Speed: 1 distance unit == 1 minute (distance -> travel time)
- Service time per customer: service_min column
- Deadline constraint: arrival_i <= deadline_i (lateness => material expires)

GA evaluation (as suggested):
- base_cost = total_distance
- capacity_penalty = beta * sum(excess_capacity_per_route)
- tardiness_penalty = lam * sum(max(0, arrival_i - deadline_i))
- fitness = base_cost + capacity_penalty + tardiness_penalty
Typically lam > beta to prioritize deadlines.

Representation:
- A chromosome is a 1D int array containing all customers [1..N-1] plus (K-1)
  separators (SEP = -1) to split into K routes.
- Example for K=3, customers 1..7:
    [2, 5, 1, -1, 4, 7, -1, 6, 3]
  means routes:
    depot -> 2 -> 5 -> 1 -> depot
    depot -> 4 -> 7 -> depot
    depot -> 6 -> 3 -> depot

This script uses ONLY NumPy (no random module, no itertools, etc.).
"""

import numpy as np

SEP = -1  # route separator token (not a customer id)


# -----------------------------
# Dataset (sample from statement)
# -----------------------------
def load_sample_dataset():
    """
    Returns:
        coords:   (N,2) float
        demand:   (N,) float
        deadline: (N,) float  minutes
        service:  (N,) float  minutes
        depot:    int (0)
        customers:(n,) int = [1..N-1]
    """
    # Columns: node_id, x, y, demand, deadline_min, service_min
    data = np.array([
        [0, 0, 0, 0,  0, 0],   # depot
        [1, 2, 7, 3, 20, 2],
        [2, 6, 4, 2, 22, 2],
        [3, 8, 9, 4, 40, 3],
        [4, 3, 1, 2, 18, 2],
        [5, 9, 2, 3, 30, 2],
        [6, 5, 8, 1, 32, 2],
        [7, 1, 4, 2, 16, 2],
    ], dtype=float)

    coords = data[:, 1:3]
    demand = data[:, 3]
    deadline = data[:, 4]
    service = data[:, 5]
    depot = 0
    customers = np.arange(1, coords.shape[0], dtype=int)
    return coords, demand, deadline, service, depot, customers


def distance_matrix(coords):
    """Euclidean distance matrix using broadcasting."""
    d = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(d * d, axis=2))


# -----------------------------
# Chromosome utilities
# -----------------------------
def normalize_chromosome(chrom, n_trucks):
    """
    Repair/normalize a chromosome so it has:
    - exactly (n_trucks-1) separators
    - no leading/trailing separators
    - no consecutive separators

    Works even if mutation/crossover produced degenerate forms.
    """
    chrom = chrom.copy()

    # Keep only valid tokens: SEP or positive customer ids
    # (Assumes customer ids are >0)
    chrom = chrom[(chrom == SEP) | (chrom > 0)]

    # Ensure correct number of separators
    need_sep = n_trucks - 1
    cur_sep = np.sum(chrom == SEP)

    if cur_sep > need_sep:
        # remove extras (from left to right)
        sep_idx = np.where(chrom == SEP)[0]
        remove = sep_idx[:(cur_sep - need_sep)]
        chrom = np.delete(chrom, remove)
    elif cur_sep < need_sep:
        # insert missing separators at random-ish positions (spread)
        # We'll insert at roughly evenly spaced indices.
        missing = need_sep - cur_sep
        if len(chrom) == 0:
            chrom = np.array([SEP] * need_sep, dtype=int)
        else:
            # Choose insertion positions excluding ends
            pos = np.linspace(1, len(chrom) - 1, missing, dtype=int)
            for p in pos:
                chrom = np.insert(chrom, p, SEP)

    # Remove leading/trailing SEP
    while len(chrom) > 0 and chrom[0] == SEP:
        chrom = chrom[1:]
    while len(chrom) > 0 and chrom[-1] == SEP:
        chrom = chrom[:-1]

    # Remove consecutive SEP by collapsing them
    if len(chrom) == 0:
        return chrom

    keep = np.ones(len(chrom), dtype=bool)
    for i in range(1, len(chrom)):
        if chrom[i] == SEP and chrom[i - 1] == SEP:
            keep[i] = False
    chrom = chrom[keep]

    # If we accidentally reduced separators below need_sep (rare), re-add
    cur_sep = np.sum(chrom == SEP)
    if cur_sep < need_sep:
        missing = need_sep - cur_sep
        pos = np.linspace(1, max(1, len(chrom) - 1), missing, dtype=int)
        for p in pos:
            chrom = np.insert(chrom, p, SEP)

        # Trim leading/trailing again
        while len(chrom) > 0 and chrom[0] == SEP:
            chrom = chrom[1:]
        while len(chrom) > 0 and chrom[-1] == SEP:
            chrom = chrom[:-1]

    return chrom


def split_routes(chrom):
    """
    Split chromosome into list of route arrays (customers only).
    Returns a list of 1D int arrays.
    """
    if len(chrom) == 0:
        return []

    sep_idx = np.where(chrom == SEP)[0]
    # boundaries: [-1, sep_idx..., len]
    bounds = np.concatenate((np.array([-1], dtype=int), sep_idx, np.array([len(chrom)], dtype=int)))
    routes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        r = chrom[a + 1:b]
        if len(r) > 0:
            routes.append(r)
        else:
            routes.append(np.array([], dtype=int))
    return routes


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(chrom, D, demand, deadline, service, depot, capacity, beta, lam):
    """
    Evaluate a chromosome:
      fitness = distance_total + beta * excess_capacity + lam * total_tardiness

    Route simulation (per statement hints):
    - For each route, start at depot with time t=0 and load=0
    - Travel time equals distance
    - arrival time is BEFORE service
    - then add service time
    """
    routes = split_routes(chrom)

    total_dist = 0.0
    total_tard = 0.0
    total_excess = 0.0

    for r in routes:
        # capacity
        load = np.sum(demand[r]) if len(r) else 0.0
        if load > capacity:
            total_excess += (load - capacity)

        # time simulation
        t = 0.0
        prev = depot
        for node in r:
            travel = D[prev, node]
            total_dist += travel
            t += travel

            late = t - deadline[node]
            if late > 0:
                total_tard += late

            t += service[node]
            prev = node

        # return to depot
        if len(r):
            total_dist += D[prev, depot]

    fitness = total_dist + beta * total_excess + lam * total_tard
    return fitness, total_dist, total_excess, total_tard


# -----------------------------
# GA operators (NumPy-only)
# -----------------------------
def init_population(pop_size, customers, n_trucks, rng):
    """
    Initialize population:
    - random customer permutation
    - random-ish separator positions
    """
    n = len(customers)
    need_sep = n_trucks - 1
    pop = np.empty((pop_size, n + need_sep), dtype=int)

    for i in range(pop_size):
        perm = rng.permutation(customers)
        # choose separator insertion positions among gaps (between customers)
        # gaps are positions 1..n-1 in the permutation array
        if need_sep > 0:
            gaps = rng.choice(np.arange(1, n, dtype=int), size=need_sep, replace=False)
            gaps.sort()
            chrom = perm.copy()
            # insert separators from left to right (offset increases)
            offset = 0
            for g in gaps:
                chrom = np.insert(chrom, g + offset, SEP)
                offset += 1
        else:
            chrom = perm
        pop[i] = normalize_chromosome(chrom, n_trucks)

    return pop


def tournament_select(costs, k, rng):
    """Minimization tournament selection."""
    idx = rng.integers(0, len(costs), size=k)
    return idx[np.argmin(costs[idx])]


def extract_customer_order(chrom):
    """Return the customers in chrom (remove separators)."""
    return chrom[chrom != SEP]


def get_split_pattern(chrom):
    """
    Return the route lengths implied by separators.
    Example: [2,5,1,SEP,4,7,SEP,6,3] -> lengths [3,2,2]
    """
    routes = split_routes(chrom)
    return np.array([len(r) for r in routes], dtype=int)


def apply_split_pattern(order, lengths, n_trucks):
    """
    Rebuild chromosome from pure customer order and route lengths.
    If lengths don't sum to len(order), we adjust by redistributing.
    """
    order = order.copy()
    n = len(order)

    lengths = lengths.copy()
    # Ensure correct number of routes
    if len(lengths) != n_trucks:
        # fallback: equal-ish split
        base = n // n_trucks
        rem = n - base * n_trucks
        lengths = np.array([base + (1 if i < rem else 0) for i in range(n_trucks)], dtype=int)

    # Fix sum
    s = int(np.sum(lengths))
    if s != n:
        # adjust by adding/removing to last routes
        diff = n - s
        for _ in range(abs(diff)):
            j = int(np.argmax(lengths)) if diff < 0 else int(np.argmin(lengths))
            if diff < 0 and lengths[j] > 0:
                lengths[j] -= 1
            elif diff > 0:
                lengths[j] += 1

    # Build with separators
    parts = []
    idx = 0
    for rlen in lengths:
        parts.append(order[idx:idx + rlen])
        idx += rlen

    # Interleave separators
    chrom = parts[0]
    for p in parts[1:]:
        chrom = np.concatenate((chrom, np.array([SEP], dtype=int), p))
    return chrom


def order_crossover_OX(p1_order, p2_order, rng):
    """
    OX crossover on customer-only permutations.
    Standard OX:
    - cut a<b
    - child[a:b] = p1[a:b]
    - fill remaining in order from p2 skipping existing
    """
    n = len(p1_order)
    a = rng.integers(0, n - 1)
    b = rng.integers(a + 1, n)

    child = np.full(n, -1, dtype=int)
    child[a:b] = p1_order[a:b]

    taken = child[a:b]
    fill = p2_order[~np.isin(p2_order, taken)]

    pos = np.concatenate((np.arange(0, a, dtype=int), np.arange(b, n, dtype=int)))
    child[pos] = fill
    return child


def mutate_swap_customers(chrom, rng):
    """Swap two customer positions (separators untouched)."""
    idx = np.where(chrom != SEP)[0]
    if len(idx) < 2:
        return
    a, b = rng.choice(idx, size=2, replace=False)
    chrom[a], chrom[b] = chrom[b], chrom[a]


def mutate_relocate_customer(chrom, rng):
    """
    Relocate: remove one customer token and insert it elsewhere (possibly across routes).
    """
    idx = np.where(chrom != SEP)[0]
    if len(idx) < 2:
        return
    src = int(rng.choice(idx))
    val = chrom[src]
    chrom2 = np.delete(chrom, src)

    # choose insertion position anywhere (including before/after SEP)
    ins = int(rng.integers(0, len(chrom2) + 1))
    chrom[:] = np.insert(chrom2, ins, val)


def mutate_inversion_within_route(chrom, rng):
    """
    2-opt style inside ONE route:
    - pick a route
    - invert a segment of customers inside that route
    """
    routes = split_routes(chrom)
    if len(routes) == 0:
        return

    # pick a route with >= 3 customers (otherwise inversion does nothing useful)
    lengths = np.array([len(r) for r in routes], dtype=int)
    eligible = np.where(lengths >= 3)[0]
    if len(eligible) == 0:
        return
    ridx = int(rng.choice(eligible))
    r = routes[ridx].copy()

    n = len(r)
    i = rng.integers(0, n - 1)
    j = rng.integers(i + 1, n)
    r[i:j] = r[i:j][::-1]
    routes[ridx] = r

    # rebuild chromosome preserving other routes
    # concatenate routes with separators
    out = routes[0]
    for rr in routes[1:]:
        out = np.concatenate((out, np.array([SEP], dtype=int), rr))
    chrom[:] = out


# -----------------------------
# Solver
# -----------------------------
def solve_vrptw_ga(
    coords, demand, deadline, service, depot, customers,
    n_trucks=3,
    capacity=8.0,
    pop_size=300,
    generations=2500,
    beta=50.0,
    lam=300.0,
    elite_frac=0.06,
    tourn_k=3,
    cx_prob=0.9,
    mut_prob=0.40,
    seed=7,
    verbose=True
):
    """
    GA for VRPTW with capacity and deadlines penalties.

    Elitism strategy:
    - Keep top elites by lexicographic (tardiness, excess, fitness) preference
      so feasibility is prioritized.
    - Also keep "best fully feasible" found.

    Returns:
        best_chrom, best_stats, D
        best_stats = dict(fitness, dist, excess, tard)
    """
    rng = np.random.default_rng(seed)
    D = distance_matrix(coords)

    pop = init_population(pop_size, customers, n_trucks, rng)
    chrom_len = pop.shape[1]
    elite_n = max(1, int(np.floor(elite_frac * pop_size)))

    best_feasible = None
    best_feasible_fit = np.inf
    best_any = None
    best_any_key = None  # (tard, excess, fitness)

    for gen in range(generations):
        fitness = np.empty(pop_size, dtype=float)
        dist = np.empty(pop_size, dtype=float)
        excess = np.empty(pop_size, dtype=float)
        tard = np.empty(pop_size, dtype=float)

        for i in range(pop_size):
            f, d, ex, td = evaluate(pop[i], D, demand, deadline, service, depot, capacity, beta, lam)
            fitness[i], dist[i], excess[i], tard[i] = f, d, ex, td

        # Key to sort: prioritize lower tardiness, then lower excess, then lower fitness
        keys = np.lexsort((fitness, excess, tard))
        best_i = keys[0]

        # Track best-any under feasibility-priority key
        cur_key = (tard[best_i], excess[best_i], fitness[best_i])
        if best_any_key is None or cur_key < best_any_key:
            best_any_key = cur_key
            best_any = pop[best_i].copy()

        # Track best fully feasible (tard=0 and excess=0)
        feas = np.where((tard == 0.0) & (excess == 0.0))[0]
        if len(feas) > 0:
            feas_best = feas[np.argmin(fitness[feas])]
            if fitness[feas_best] < best_feasible_fit:
                best_feasible_fit = fitness[feas_best]
                best_feasible = pop[feas_best].copy()

        if verbose and (gen % max(1, generations // 10) == 0 or gen == generations - 1):
            msg = f"Gen {gen:4d} | best(tard,excess,fit)=({best_any_key[0]:.3f},{best_any_key[1]:.3f},{best_any_key[2]:.3f})"
            if best_feasible is not None:
                msg += f" | best feasible fit={best_feasible_fit:.3f}"
            print(msg)

        # Elites
        elites = pop[keys[:elite_n]].copy()

        # Next generation
        new_pop = np.empty_like(pop)
        new_pop[:elite_n] = elites

        # Fill the rest
        for i in range(elite_n, pop_size):
            p1 = pop[tournament_select(fitness, tourn_k, rng)]
            p2 = pop[tournament_select(fitness, tourn_k, rng)]

            # Decompose into (order, split pattern)
            p1_order = extract_customer_order(p1)
            p2_order = extract_customer_order(p2)
            p1_split = get_split_pattern(p1)
            p2_split = get_split_pattern(p2)

            # Crossover: OX on order + inherit split from one parent
            if rng.random() < cx_prob:
                child_order = order_crossover_OX(p1_order, p2_order, rng)
                split = p1_split if rng.random() < 0.5 else p2_split
                child = apply_split_pattern(child_order, split, n_trucks)
            else:
                child = p1.copy()

            # Mutations
            if rng.random() < mut_prob:
                r = rng.random()
                if r < 0.40:
                    mutate_swap_customers(child, rng)
                elif r < 0.75:
                    mutate_relocate_customer(child, rng)
                else:
                    # 2-opt style within a route
                    mutate_inversion_within_route(child, rng)

            # Repair/normalize
            child = normalize_chromosome(child, n_trucks)
            # Ensure length matches (it should)
            if len(child) != chrom_len:
                # If mismatch, rebuild using current order and equal split
                order = extract_customer_order(child)
                child = apply_split_pattern(order, get_split_pattern(child), n_trucks)
                child = normalize_chromosome(child, n_trucks)

            new_pop[i] = child

        pop = new_pop

    # Return best feasible if available, else best_any
    best = best_feasible if best_feasible is not None else best_any
    f, d, ex, td = evaluate(best, D, demand, deadline, service, depot, capacity, beta, lam)
    stats = {"fitness": f, "dist": d, "excess": ex, "tard": td}
    return best, stats, D


def print_solution(chrom, stats, D, demand, deadline, service, depot, capacity):
    """Pretty print routes + timing per route."""
    routes = split_routes(chrom)

    print("\n==============================")
    print("BEST SOLUTION")
    print(f"chromosome: {chrom.tolist()}")
    print(f"fitness:  {stats['fitness']:.3f}")
    print(f"distance: {stats['dist']:.3f}")
    print(f"excess:   {stats['excess']:.3f}")
    print(f"tard:     {stats['tard']:.3f}")
    print("==============================\n")

    for r_id, r in enumerate(routes, start=1):
        load = np.sum(demand[r]) if len(r) else 0.0
        print(f"Route {r_id}: depot -> " + " -> ".join(map(str, r.tolist())) + " -> depot")
        print(f"  load = {load:.3f} / cap {capacity:.3f} | excess = {max(0.0, load-capacity):.3f}")

        t = 0.0
        prev = depot
        if len(r) == 0:
            print("  (empty route)\n")
            continue

        print("  node | arrival | deadline | tardiness | service")
        for node in r:
            t += D[prev, node]
            arr = t
            td = max(0.0, arr - deadline[node])
            print(f"  {int(node):>4d} | {arr:7.3f} | {deadline[node]:8.3f} | {td:9.3f} | {service[node]:7.3f}")
            t += service[node]
            prev = node

        t += D[prev, depot]
        print(f"  return_to_depot_time_added, route_end_time = {t:.3f}\n")


def main():
    coords, demand, deadline, service, depot, customers = load_sample_dataset()

    # --- Hyperparameters you can tune ---
    N_TRUCKS = 3          # "varios camiones" -> choose a number; change as needed
    CAPACITY = 8.0
    POP_SIZE = 350
    GENERATIONS = 3000
    BETA = 60.0           # capacity penalty weight
    LAMBDA = 350.0        # tardiness penalty weight (usually > beta)
    SEED = 7
    # -----------------------------------

    best, stats, D = solve_vrptw_ga(
        coords=coords,
        demand=demand,
        deadline=deadline,
        service=service,
        depot=depot,
        customers=customers,
        n_trucks=N_TRUCKS,
        capacity=CAPACITY,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        beta=BETA,
        lam=LAMBDA,
        seed=SEED,
        verbose=True
    )

    print_solution(best, stats, D, demand, deadline, service, depot, CAPACITY)


if __name__ == "__main__":
    main()
