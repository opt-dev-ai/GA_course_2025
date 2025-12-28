#!/usr/bin/env python3
"""
Caso 06 - VRPTW con coste fijo (NumPy-only GA)
---------------------------------------------
Objective (from statement):
    Minimize:  C_fixed * K  +  c * total_distance
Subject to:
    - Capacity per truck (8 units) (violations penalized)
    - Deadlines: every customer should be served before its deadline
      (lateness penalized strongly with large lambda)

Key idea:
    Chromosome = permutation of customers + separators.
    Number of NON-EMPTY routes defines K (trucks used).
    Empty routes are allowed, so K can vary even with a fixed number of separators.

Fitness:
    fitness = C_fixed*K + c*distance_total + beta*excess_capacity + lam*total_tardiness

Time model:
    - speed: 1 distance unit == 1 minute
    - start each route at time t=0 at depot
    - travel time = Euclidean distance
    - arrival time is BEFORE service time
    - then add service time

Only dependency: NumPy.
"""

import numpy as np

SEP = -1  # route separator token


# -----------------------------
# Dataset (sample from statement)
# -----------------------------
def load_sample_dataset():
    """
    Sample dataset in the PDF (small, for manual validation).

    Returns:
        coords:   (N,2) float
        demand:   (N,) float
        deadline: (N,) float (minutes)
        service:  (N,) float (minutes)
        depot:    int (0)
        customers:(n,) int array [1..N-1]
    """
    # columns: node_id, x, y, demand, deadline_min, service_min
    data = np.array([
        [ 0,  0,  0, 0,   0, 0],   # depot

        [ 1,  2,  7, 3,  20, 2],
        [ 2,  6,  4, 2,  22, 2],
        [ 3,  8,  9, 4,  40, 3],
        [ 4,  3,  1, 2,  18, 2],
        [ 5,  9,  2, 3,  30, 2],
        [ 6,  5,  8, 1,  32, 2],
        [ 7,  1,  4, 2,  16, 2],

        [ 8,  4, 12, 2,  28, 2],
        [ 9, 11,  6, 3,  34, 2],
        [10, 13,  3, 2,  36, 2],
        [11, 15,  9, 4,  55, 3],
        [12,  7, 14, 1,  42, 2],
        [13,  2, 15, 2,  38, 2],
        [14,  9, 13, 3,  50, 3],
        [15, 16,  2, 2,  45, 2],

        [16, 18,  6, 3,  60, 3],
        [17, 12, 12, 2,  58, 2],
        [18,  6, 18, 4,  65, 3],
        [19,  3, 10, 1,  26, 2],
        [20, 10, 16, 2,  70, 3],
        [21, 14, 14, 3,  75, 3],
        [22, 17, 11, 2,  72, 2],
        [23,  8,  1, 1,  24, 2],

        [24, 19,  3, 4,  80, 4],
        [25,  1, 18, 2,  62, 3],
    ], dtype=float)


    coords = data[:, 1:3]
    demand = data[:, 3]
    deadline = data[:, 4]
    service = data[:, 5]
    depot = 0
    customers = np.arange(1, coords.shape[0], dtype=int)
    return coords, demand, deadline, service, depot, customers


def distance_matrix(coords):
    """Euclidean distance matrix with broadcasting."""
    dif = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(dif * dif, axis=2))


# -----------------------------
# Chromosome helpers
# -----------------------------
def split_routes(chrom):
    """
    Split chromosome into routes (customers-only arrays).
    Empty routes are allowed (between consecutive separators).
    """
    if len(chrom) == 0:
        return []

    sep_idx = np.where(chrom == SEP)[0]
    bounds = np.concatenate((np.array([-1], dtype=int), sep_idx, np.array([len(chrom)], dtype=int)))

    routes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        routes.append(chrom[a + 1:b])  # may be empty
    return routes


def extract_customer_order(chrom):
    """Return customers in chromosome (remove separators)."""
    return chrom[chrom != SEP]


def normalize_chromosome(chrom, customers, max_trucks):
    """
    Repair chromosome to contain:
      - exactly all customers once
      - exactly (max_trucks-1) separators
    Any duplicates/missing customers are repaired.

    Strategy:
      1) take customer tokens in order, remove duplicates keeping first occurrence
      2) append missing customers (random order)
      3) rebuild separators count to max_trucks-1 (allowing empties)
    """
    need_sep = max_trucks - 1
    chrom = chrom.copy()

    # --- Keep tokens that are SEP or in customers ---
    # (customers are positive ints, SEP is -1)
    valid_cust = np.isin(chrom, customers)
    chrom = chrom[(chrom == SEP) | valid_cust]

    # --- Unique customers in first-seen order ---
    cust_seq = chrom[chrom != SEP]
    if len(cust_seq) > 0:
        _, first_idx = np.unique(cust_seq, return_index=True)
        cust_unique = cust_seq[np.sort(first_idx)]
    else:
        cust_unique = np.array([], dtype=int)

    # --- Append missing customers ---
    missing = customers[~np.isin(customers, cust_unique)]
    # randomize missing slightly to diversify (still NumPy-only)
    if len(missing) > 1:
        rng = np.random.default_rng()
        missing = rng.permutation(missing)
    cust_full = np.concatenate((cust_unique, missing))

    # --- Make sure we didn't lose customers (should be exact) ---
    if len(cust_full) != len(customers):
        # Fallback: force exact permutation
        cust_full = customers.copy()

    # --- Now rebuild with exactly need_sep separators ---
    # We place separators by selecting "gap indices" in [0..n] (between customers),
    # allowing multiple separators in the same gap => empty routes.
    n = len(cust_full)
    if need_sep <= 0:
        return cust_full

    rng = np.random.default_rng()
    gaps = rng.integers(0, n + 1, size=need_sep)  # allow repeats -> empties
    gaps.sort()

    out = cust_full.copy()
    offset = 0
    for g in gaps:
        out = np.insert(out, g + offset, SEP)
        offset += 1

    return out


def count_used_trucks(routes):
    """K = number of non-empty routes."""
    lengths = np.array([len(r) for r in routes], dtype=int)
    return int(np.sum(lengths > 0))


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(chrom, D, demand, deadline, service, depot, capacity,
             C_fixed, c_dist, beta, lam):
    """
    Compute:
      - K (used trucks)
      - distance_total
      - excess_capacity_total
      - tardiness_total
      - fitness = C_fixed*K + c_dist*distance + beta*excess + lam*tard

    Deadlines are penalized strongly (lam big).
    """
    routes = split_routes(chrom)
    K = count_used_trucks(routes)

    dist_total = 0.0
    excess_total = 0.0
    tard_total = 0.0

    for r in routes:
        if len(r) == 0:
            continue

        # capacity
        load = float(np.sum(demand[r]))
        if load > capacity:
            excess_total += (load - capacity)

        # time simulation
        t = 0.0
        prev = depot
        for node in r:
            dist_total += D[prev, node]
            t += D[prev, node]

            late = t - deadline[node]
            if late > 0:
                tard_total += late

            t += service[node]
            prev = node

        # return to depot
        dist_total += D[prev, depot]

    cost = C_fixed * K + c_dist * dist_total
    fitness = cost + beta * excess_total + lam * tard_total
    return fitness, cost, K, dist_total, excess_total, tard_total


# -----------------------------
# GA operators (NumPy-only)
# -----------------------------
def init_population(pop_size, customers, max_trucks, rng):
    """
    Build initial population with fixed separator count (max_trucks-1).
    K will vary by allowing empty routes naturally.
    """
    n = len(customers)
    need_sep = max_trucks - 1
    pop = np.empty((pop_size, n + need_sep), dtype=int)

    for i in range(pop_size):
        perm = rng.permutation(customers)

        # Random separator gaps (allow repeats -> empty routes)
        gaps = rng.integers(0, n + 1, size=need_sep)
        gaps.sort()

        chrom = perm.copy()
        offset = 0
        for g in gaps:
            chrom = np.insert(chrom, g + offset, SEP)
            offset += 1

        pop[i] = chrom

    return pop


def tournament_select(values, k, rng):
    """Minimization tournament selection."""
    idx = rng.integers(0, len(values), size=k)
    return idx[np.argmin(values[idx])]


def order_crossover_OX(p1, p2, rng):
    """
    Order crossover (OX) on pure permutations.
    Returns a permutation child.
    """
    n = len(p1)
    a = rng.integers(0, n - 1)
    b = rng.integers(a + 1, n)

    child = np.full(n, -1, dtype=int)
    child[a:b] = p1[a:b]

    taken = child[a:b]
    fill = p2[~np.isin(p2, taken)]
    pos = np.concatenate((np.arange(0, a, dtype=int), np.arange(b, n, dtype=int)))
    child[pos] = fill
    return child


def get_split_pattern(chrom):
    """
    Route lengths implied by separators (including empty routes).
    Example:
      [2, SEP, 3, 4, SEP, SEP, 5] -> lengths [1,2,0,1]
    """
    routes = split_routes(chrom)
    return np.array([len(r) for r in routes], dtype=int)


def apply_split_pattern(order, lengths, max_trucks):
    """
    Rebuild a chromosome from:
      - order: customer permutation
      - lengths: route sizes (len == max_trucks)
    If sum(lengths) differs from len(order), adjust lengths.
    """
    order = order.copy()
    n = len(order)

    lengths = lengths.copy()
    if len(lengths) != max_trucks:
        # fallback: equal-ish split with empties
        base = n // max_trucks
        rem = n - base * max_trucks
        lengths = np.array([base + (1 if i < rem else 0) for i in range(max_trucks)], dtype=int)

    # fix sum
    s = int(np.sum(lengths))
    if s != n:
        diff = n - s
        # adjust by adding/removing 1 at a time
        for _ in range(abs(diff)):
            if diff > 0:
                j = int(np.argmin(lengths))
                lengths[j] += 1
            else:
                j = int(np.argmax(lengths))
                if lengths[j] > 0:
                    lengths[j] -= 1

    # build
    parts = []
    idx = 0
    for rlen in lengths:
        parts.append(order[idx:idx + rlen])
        idx += rlen

    chrom = parts[0]
    for p in parts[1:]:
        chrom = np.concatenate((chrom, np.array([SEP], dtype=int), p))
    return chrom


def mutate_swap(chrom, rng):
    """Swap two customer tokens (separators unchanged)."""
    idx = np.where(chrom != SEP)[0]
    if len(idx) < 2:
        return
    a, b = rng.choice(idx, size=2, replace=False)
    chrom[a], chrom[b] = chrom[b], chrom[a]


def mutate_relocate(chrom, rng):
    """
    Relocate one customer token to another position (may change K by creating empties).
    """
    idx = np.where(chrom != SEP)[0]
    if len(idx) < 2:
        return
    src = int(rng.choice(idx))
    val = chrom[src]
    tmp = np.delete(chrom, src)
    ins = int(rng.integers(0, len(tmp) + 1))
    chrom[:] = np.insert(tmp, ins, val)


def mutate_inversion_within_route(chrom, rng):
    """2-opt style inversion inside one non-empty route."""
    routes = split_routes(chrom)
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

    # rebuild chromosome
    out = routes[0]
    for rr in routes[1:]:
        out = np.concatenate((out, np.array([SEP], dtype=int), rr))
    chrom[:] = out


def mutate_route_split_merge_style(chrom, rng):
    """
    'Route-level' operator hint (split/merge exploration) implemented without changing separator count:
    - Choose a customer and move it near a separator boundary (can create/empty a route),
      effectively changing K (# non-empty routes).
    """
    sep_idx = np.where(chrom == SEP)[0]
    cust_idx = np.where(chrom != SEP)[0]
    if len(sep_idx) == 0 or len(cust_idx) == 0:
        return

    cpos = int(rng.choice(cust_idx))
    val = chrom[cpos]
    tmp = np.delete(chrom, cpos)

    # choose a target near a separator boundary (+/- 1 position around sep)
    s = int(rng.choice(sep_idx))
    # after delete, sep indices might shift; clamp target
    target = int(np.clip(s + rng.integers(-1, 2), 0, len(tmp)))
    chrom[:] = np.insert(tmp, target, val)


# -----------------------------
# Solver
# -----------------------------
def solve_case06_ga(
    coords, demand, deadline, service, depot, customers,
    max_trucks=6,           # upper bound on available trucks (K can be <= max_trucks)
    capacity=8.0,
    C_fixed=25.0,
    c_dist=1.0,
    beta=50.0,
    lam=100.0,              # statement suggests big lambda, e.g. 100 per minute late
    pop_size=350,
    generations=3000,
    elite_frac=0.06,
    tourn_k=3,
    cx_prob=0.9,
    mut_prob=0.45,
    seed=7,
    verbose=True
):
    """
    Genetic Algorithm for Case 06.

    Sorting / elitism strategy (simple multiobjective):
      order individuals by lexicographic key:
        (tardiness_total, excess_capacity_total, total_cost)
      so the GA prioritizes being on-time, then capacity, then cost.

    Returns:
        best_chrom, best_stats, D
        best_stats = dict(fitness, cost, K, dist, excess, tard)
    """
    rng = np.random.default_rng(seed)
    D = distance_matrix(coords)

    pop = init_population(pop_size, customers, max_trucks, rng)
    elite_n = max(1, int(np.floor(elite_frac * pop_size)))

    best = None
    best_key = None

    for gen in range(generations):
        fitness = np.empty(pop_size, dtype=float)
        cost = np.empty(pop_size, dtype=float)
        Karr = np.empty(pop_size, dtype=int)
        dist = np.empty(pop_size, dtype=float)
        excess = np.empty(pop_size, dtype=float)
        tard = np.empty(pop_size, dtype=float)

        for i in range(pop_size):
            f, cst, k, d, ex, td = evaluate(
                pop[i], D, demand, deadline, service, depot, capacity,
                C_fixed, c_dist, beta, lam
            )
            fitness[i], cost[i], Karr[i], dist[i], excess[i], tard[i] = f, cst, k, d, ex, td

        # Lexicographic sort by (tard, excess, cost)
        order = np.lexsort((cost, excess, tard))
        best_i = order[0]
        cur_key = (tard[best_i], excess[best_i], cost[best_i])

        if best_key is None or cur_key < best_key:
            best_key = cur_key
            best = pop[best_i].copy()

        if verbose and (gen % max(1, generations // 10) == 0 or gen == generations - 1):
            print(
                f"Gen {gen:4d} | best(tard,excess,cost)=({best_key[0]:.3f},{best_key[1]:.3f},{best_key[2]:.3f})"
            )

        # Elitism
        elites = pop[order[:elite_n]].copy()

        # Next generation
        new_pop = np.empty_like(pop)
        new_pop[:elite_n] = elites

        for i in range(elite_n, pop_size):
            p1 = pop[tournament_select(fitness, tourn_k, rng)]
            p2 = pop[tournament_select(fitness, tourn_k, rng)]

            # Decompose: customer order + split pattern
            p1_order = extract_customer_order(p1)
            p2_order = extract_customer_order(p2)
            p1_split = get_split_pattern(p1)  # length max_trucks
            p2_split = get_split_pattern(p2)

            if rng.random() < cx_prob:
                child_order = order_crossover_OX(p1_order, p2_order, rng)
                split = p1_split if rng.random() < 0.5 else p2_split
                child = apply_split_pattern(child_order, split, max_trucks)
            else:
                child = p1.copy()

            # Mutations
            if rng.random() < mut_prob:
                r = rng.random()
                if r < 0.30:
                    mutate_swap(child, rng)
                elif r < 0.60:
                    mutate_relocate(child, rng)
                elif r < 0.85:
                    mutate_inversion_within_route(child, rng)
                else:
                    mutate_route_split_merge_style(child, rng)

            # Repair
            child = normalize_chromosome(child, customers, max_trucks)
            new_pop[i] = child

        pop = new_pop

    # Final stats for best
    f, cst, k, d, ex, td = evaluate(
        best, D, demand, deadline, service, depot, capacity,
        C_fixed, c_dist, beta, lam
    )
    stats = {"fitness": f, "cost": cst, "K": int(k), "dist": d, "excess": ex, "tard": td}
    return best, stats, D


def print_solution(chrom, stats, D, demand, deadline, service, depot, capacity):
    """Print solution routes, K, cost, and per-route timing."""
    routes = split_routes(chrom)
    used = [r for r in routes if len(r) > 0]

    print("\n==============================")
    print("BEST SOLUTION (Case 06)")
    print(f"chromosome: {chrom.tolist()}")
    print(f"K (used trucks): {stats['K']}")
    print(f"fixed+distance cost: {stats['cost']:.3f}")
    print(f"distance total: {stats['dist']:.3f}")
    print(f"excess capacity: {stats['excess']:.3f}")
    print(f"tardiness total: {stats['tard']:.3f}")
    print(f"fitness (with penalties): {stats['fitness']:.3f}")
    print("==============================\n")

    for r_id, r in enumerate(used, start=1):
        load = float(np.sum(demand[r]))
        print(f"Route {r_id}: depot -> " + " -> ".join(map(str, r.tolist())) + " -> depot")
        print(f"  load = {load:.3f} / cap {capacity:.3f} | excess = {max(0.0, load-capacity):.3f}")

        t = 0.0
        prev = depot
        print("  node | arrival | deadline | tardiness | service")
        for node in r:
            t += D[prev, node]
            arr = t
            td = max(0.0, arr - deadline[node])
            print(f"  {int(node):>4d} | {arr:7.3f} | {deadline[node]:8.3f} | {td:9.3f} | {service[node]:7.3f}")
            t += service[node]
            prev = node

        t += D[prev, depot]
        print(f"  route_end_time (after return) = {t:.3f}\n")


def main():
    coords, demand, deadline, service, depot, customers = load_sample_dataset()

    # Parameters from statement
    CAPACITY = 8.0
    C_FIXED = 25.0
    C_DIST = 1.0
    LAMBDA = 100.0  # suggested "lambda grande (p.ej., 100 por minuto)" :contentReference[oaicite:2]{index=2}
    BETA = 50.0     # tunable (capacity penalty)

    # GA hyperparameters (tune if needed)
    MAX_TRUCKS = 6      # upper bound; K will be <= MAX_TRUCKS
    POP_SIZE = 400
    GENERATIONS = 3500
    SEED = 7

    best, stats, D = solve_case06_ga(
        coords=coords,
        demand=demand,
        deadline=deadline,
        service=service,
        depot=depot,
        customers=customers,
        max_trucks=MAX_TRUCKS,
        capacity=CAPACITY,
        C_fixed=C_FIXED,
        c_dist=C_DIST,
        beta=BETA,
        lam=LAMBDA,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        seed=SEED,
        verbose=True
    )

    print_solution(best, stats, D, demand, deadline, service, depot, CAPACITY)


if __name__ == "__main__":
    main()
