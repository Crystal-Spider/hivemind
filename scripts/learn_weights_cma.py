# scripts/learn_weights_cma.py
from __future__ import annotations
import argparse, json, math, os, random, sys, time, multiprocessing as mp
from dataclasses import dataclass
from typing import List

THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from core.board import Board
from core.enums import GameState
from core.game import Move

try:
    from ai.brain import AlphaBetaPruner
except Exception:
    AlphaBetaPruner = None

Vec = List[float]
FEATURE_ORDER = ["contact","liberties","reach","mobility","pinned","beetle","material","supply"]

# ---------- CMA-ES (diagonal) ----------
@dataclass
class CMAState:
    mean: Vec
    sigma: float
    diag: Vec
    pc: Vec
    ps: Vec
    c1: float
    cmu: float
    cs: float
    ds: float
    cc: float
    mueff: float
    weights: Vec
    popsize: int

def init_cma(dim: int, popsize: int, init_mean: Vec | None, init_sigma: float) -> CMAState:
    lam = popsize
    mu = lam // 2
    raw_w = [math.log(mu + 0.5) - math.log(i+1) for i in range(mu)]
    w_sum = sum(raw_w)
    w = [wi / w_sum for wi in raw_w]
    mueff = 1.0 / sum(wi*wi for wi in w)
    cs = (mueff + 2) / (dim + mueff + 5)
    ds = 1 + cs + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
    cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
    c1 = 2 / ((dim + 1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
    mean = list(init_mean) if init_mean is not None else [0.0]*dim
    return CMAState(mean, init_sigma, [1.0]*dim, [0.0]*dim, [0.0]*dim, c1, cmu, cs, ds, cc, mueff, w, lam)

def ask(state: CMAState, rng: random.Random):
    samples = []
    for _ in range(state.popsize):
        z = [rng.gauss(0.0, 1.0) for _ in state.mean]
        y = [dz * d for dz, d in zip(z, state.diag)]
        x = [m + state.sigma * yi for m, yi in zip(state.mean, y)]
        samples.append((x, z))
    return samples

def tell(state: CMAState, samples, fitnesses):
    paired = list(zip(samples, fitnesses))
    paired.sort(key=lambda t: t[1])
    mu = len(state.weights)
    old_mean = state.mean[:]
    x_mu = [0.0]*len(state.mean)
    z_mu = [0.0]*len(state.mean)
    for k, w in enumerate(state.weights):
        (xk, zk) = paired[k][0]
        for i in range(len(x_mu)):
            x_mu[i] += w * xk[i]
            z_mu[i] += w * zk[i]
    state.mean = x_mu

    c = state.cs
    one_minus_c = 1 - c
    sqrt_term = math.sqrt(c * (2 - c) * state.mueff)
    for i in range(len(state.ps)):
        state.ps[i] = one_minus_c * state.ps[i] + sqrt_term * z_mu[i]

    chiN = math.sqrt(len(state.mean)) * (1 - 1/(4*len(state.mean)) + 1/(21*(len(state.mean))**2))
    norm_ps = math.sqrt(sum(v*v for v in state.ps))
    state.sigma *= math.exp((c / state.ds) * (norm_ps / chiN - 1))

    c1, cmu = state.c1, state.cmu
    for i in range(len(state.diag)):
        state.diag[i] = math.sqrt(max(1e-12, (1 - c1 - cmu) * (state.diag[i]**2)))
    for i in range(len(state.diag)):
        rank1 = c1 * ((state.mean[i] - old_mean[i]) / max(1e-12, state.sigma))**2
        rankmu = cmu * sum(state.weights[k] * (paired[k][0][1][i]**2) for k in range(mu))
        state.diag[i] = math.sqrt(max(1e-12, state.diag[i]**2 + rank1 + rankmu))
    for i in range(len(state.diag)):
        state.diag[i] = min(max(state.diag[i], 1e-3), 1e3)
    state.sigma = min(max(state.sigma, 1e-3), 5.0)

# ---------- Self-play using AlphaBetaPruner ----------
def _make_pruner(w: Vec):
    if AlphaBetaPruner is None:
        raise RuntimeError("AlphaBetaPruner not importable. Ensure ai/brain.py is on sys.path.")
    pr = AlphaBetaPruner()
    if hasattr(pr, "set_linear_weights"):
        pr.set_linear_weights(w)
    else:
        # last resort
        if hasattr(pr, "_weights"):
            pr._weights = list(w)
    return pr

def _call_find_best_move(pr, board, max_branch, depth, time_sec):
    # Try common signatures in order. Fallbacks are no-ops if missing.
    if hasattr(pr, "find_best_move"):
        try:
            return pr.find_best_move(board, max_branch, depth, time_sec)
        except TypeError:
            try:
                return pr.find_best_move(board, depth=depth, time_limit=time_sec, max_branching_factor=max_branch)
            except TypeError:
                return pr.find_best_move(board)
    if hasattr(pr, "_find_best_move"):
        try:
            return pr._find_best_move(board, max_branch, depth, time_sec)
        except TypeError:
            try:
                return pr._find_best_move(board, depth=depth, time_limit=time_sec, max_branching_factor=max_branch)
            except TypeError:
                return pr._find_best_move(board)
    return None

def play_game(wA: Vec, wB: Vec, max_plies: int, max_branch: int, depth: int, time_ms: int, epsilon: float, seed: int) -> float:
    rng = random.Random(seed)
    b = Board()
    prA = _make_pruner(wA)
    prB = _make_pruner(wB)
    ply = 0
    while not b.gameover and ply < max_plies:
        is_white = getattr(b, 'current_player_is_white', None)
        if is_white is None:
            is_white = (getattr(b, 'current_player_color', None) and b.current_player_color.name == 'WHITE')
        pr = prA if is_white else prB
        if epsilon > 0 and rng.random() < epsilon:
            moves = list(b.calculate_valid_moves())
            if not moves:
                break
            m = rng.choice(moves)
        else:
            m = _call_find_best_move(pr, b, max_branch, depth, time_ms / 1000.0)
        if not m:
            break
        # Use correct Board API depending on move type
        if isinstance(m, Move):
            b.play_parsed(m)
        else:
            b.play(m)
        ply += 1

    if b.state is GameState.DRAW:
        return 0.5
    if b.current_player_has_won:
        is_white = getattr(b, 'current_player_is_white', None)
        if is_white is None:
            is_white = (getattr(b, 'current_player_color', None) and b.current_player_color.name == 'WHITE')
        winner_is_white = is_white
    elif b.current_opponent_has_won:
        is_white = getattr(b, 'current_player_is_white', None)
        if is_white is None:
            is_white = (getattr(b, 'current_player_color', None) and b.current_player_color.name == 'WHITE')
        winner_is_white = not is_white
    else:
        return 0.5
    return 1.0 if winner_is_white else 0.0

# ---------- Fitness against a small pool ----------
def fitness(w: Vec, games_per_opponent: int, max_plies: int, max_branch: int, depth: int, time_ms: int, epsilon: float, seed: int) -> float:
    rng = random.Random(seed)
    opponents = [
        [10,10,10,2,4,6,1,1],
        [0,0,0,0,0,0,0,0],
        [-10,-10,-10,-2,-4,-6,-1,-1],
    ]
    res = []
    for opp in opponents:
        for _ in range(games_per_opponent):
            s1 = rng.randrange(1<<30)
            rW = play_game(w, opp, max_plies, max_branch, depth, time_ms, epsilon, s1)
            s2 = rng.randrange(1<<30)
            rB = play_game(opp, w, max_plies, max_branch, depth, time_ms, epsilon, s2)
            res.append(0.5 * (rW + (1 - rB)))
    winrate = sum(res) / max(1, len(res))
    l2 = 1e-4 * sum(v*v for v in w)
    return (1.0 - winrate) + l2

# ---------- Parallel wrapper ----------
def _worker_eval(args):
    w, gp_o, max_plies, max_branch, depth, time_ms, eps, seed = args
    try:
        return fitness(w, gp_o, max_plies, max_branch, depth, time_ms, eps, seed)
    except Exception as e:
        import traceback, sys
        print("\n[EVAL ERROR]", str(e), file=sys.stderr)
        traceback.print_exc()
        print("weights:", [round(v, 3) for v in w], file=sys.stderr)
        return 1e9

# ---------- Run CMA ----------
def run_cma(init_mean: Vec, init_sigma: float, pop: int, iters: int,
            games_per_opp: int, max_plies: int, max_branch: int, depth: int, time_ms: int,
            epsilon: float, seed: int, parallel: int):
    rng = random.Random(seed)
    state = init_cma(len(init_mean), pop, init_mean, init_sigma)
    best_w = list(init_mean)
    best_f = _worker_eval((init_mean, games_per_opp, max_plies, max_branch, depth, time_ms, epsilon, rng.randrange(1<<30)))

    pool = mp.Pool(processes=parallel) if parallel > 1 else None
    try:
        for it in range(1, iters+1):
            samples = ask(state, rng)
            seeds = [rng.randrange(1<<30) for _ in samples]
            tasks = [(x, games_per_opp, max_plies, max_branch, depth, time_ms, epsilon, s) for (x,_), s in zip(samples, seeds)]

            t_iter = time.time()
            total = len(tasks)
            fits: List[float] = [0.0]*total
            done = 0
            best_iter = float("inf")

            def _update():
                elapsed = time.time() - t_iter
                rate = (done / elapsed) if elapsed > 0 else 0.0
                remaining = ((total - done) / rate) if rate > 0 else 0.0
                bar_len = 28
                filled = int(bar_len * (done / total))
                bar = "[" + "#"*filled + "-"*(bar_len - filled) + "]"
                print(f"\n[iter {it}/{iters}] {bar} {done}/{total} evals • {elapsed:.1f}s elapsed • ETA {remaining:.1f}s • best_f {min(best_iter, best_f):.4f}", end="", file=sys.stderr, flush=True)

            if pool:
                for idx, fit in enumerate(pool.imap_unordered(_worker_eval, tasks, chunksize=1)):
                    fits[done] = fit
                    done += 1
                    if fit < best_iter:
                        best_iter = fit
                    if done == 1 or done % 2 == 0 or done == total:
                        _update()
            else:
                for i, t in enumerate(tasks):
                    fit = _worker_eval(t)
                    fits[done] = fit
                    done += 1
                    if fit < best_iter:
                        best_iter = fit
                    if done == 1 or done % 2 == 0 or done == total:
                        _update()

            # newline after progress bar
            print("", file=sys.stderr)

            tell(state, samples, fits)
            k = min(range(len(fits)), key=lambda i: fits[i])
            if fits[k] < best_f:
                best_f = fits[k]
                best_w = samples[k][0]

            iter_time = time.time() - t_iter
            eta_total = (iters - it) * iter_time
            print(f"[iter {it}] {iter_time:.1f}s • overall best_f={best_f:.4f} • sigma={state.sigma:.4f} • total ETA {eta_total:.1f}s", file=sys.stderr)
    finally:
        if pool:
            pool.close(); pool.join()
    return best_w, best_f

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="CMA-ES training for complex eval weights using AlphaBetaPruner.")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--init-sigma", type=float, default=2.0)
    ap.add_argument("--games-per-opp", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=220)
    ap.add_argument("--max-branch", type=int, default=30)
    ap.add_argument("--ab-depth", type=int, default=None)
    ap.add_argument("--ab-time-ms", type=int, default=5000)
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=max(1, mp.cpu_count()-1))
    ap.add_argument("--init-heuristic", action="store_true")
    ap.add_argument("--out", type=str, default="learned_weights_cma.json")
    args = ap.parse_args()

    if AlphaBetaPruner is None:
        raise SystemExit("AlphaBetaPruner not found. Ensure 'ai/brain.py' is importable from src root.")

    init = [10,10,10,2,4,6,1,1] if args.init_heuristic else [0.0]*8

    t0 = time.time()
    w, f = run_cma(
        init_mean=init, init_sigma=args.init_sigma, pop=args.pop, iters=args.iters,
        games_per_opp=args.games_per_opp, max_plies=args.max_plies, max_branch=args.max_branch,
        depth=args.ab_depth, time_ms=args.ab_time_ms, epsilon=args.epsilon,
        seed=args.seed, parallel=args.parallel
    )
    dt = time.time() - t0

    payload = {"weights": w, "order": FEATURE_ORDER, "meta": {
        "pop": args.pop, "iters": args.iters, "init_sigma": args.init_sigma,
        "games_per_opp": args.games_per_opp, "max_plies": args.max_plies,
        "max_branch": args.max_branch, "ab_depth": args.ab_depth, "ab_time_ms": args.ab_time_ms,
        "epsilon": args.epsilon, "seed": args.seed, "seconds": round(dt, 2), "fitness": f
    }}
    with open(args.out, "w", encoding="utf-8") as fjson:
        json.dump(payload, fjson, indent=2)
    print(f"Wrote {args.out}")
    print("weights =", [round(v, 4) for v in w])

if __name__ == "__main__":
    main()
