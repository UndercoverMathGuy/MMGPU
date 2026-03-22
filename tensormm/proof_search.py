"""Forward-chaining brute-force proof search for Metamath.

Algorithm:
  1. Seed a pool with axiom conclusions, floating hyp expressions,
     and the target theorem's essential hypotheses.
  2. At each depth level, try every available assertion against every
     combination of pool entries matching its floating hypothesis types.
  3. For each valid substitution (all essential hyps present in pool),
     derive the conclusion and add it to the pool.
  4. Optionally GPU-verify all new derivations at each depth level
     by shipping (pattern, σ, derived) triples to verify_flat.
  5. Stop when the target expression appears in the pool (or timeout/exhaustion).
  6. Reconstruct the proof by tracing provenance back through the pool.
"""
from __future__ import annotations

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from tensormm.cpu_verifier import CPUVerifier, VerificationResult, apply_substitution
from tensormm.parser import Assertion, FloatingHyp, ParsedDatabase


# ══════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════


Expr = tuple[str, ...]  # immutable expression for hashing


@dataclass(frozen=True)
class Provenance:
    """How a pool expression was derived."""

    theorem_label: str  # which assertion produced it
    substitution: dict[str, tuple[str, ...]]  # variable -> tokens (immutable)
    parent_exprs: tuple[Expr, ...]  # pool expressions used as hyp matches
    depth: int


@dataclass
class SearchResult:
    """Outcome of a proof search."""

    success: bool
    target: list[str]
    proof_labels: list[str] | None = None  # linear Metamath proof if found
    verification: VerificationResult | None = None
    depth_reached: int = 0
    pool_size: int = 0
    candidates_tried: int = 0
    candidates_valid: int = 0
    gpu_verified: int = 0
    gpu_rejected: int = 0
    elapsed: float = 0.0
    level_stats: list[dict] = field(default_factory=list)


class ExpressionPool:
    """Deduplicated pool of known expressions, indexed by typecode (first token).

    Each expression is stored once.  The first to derive it wins provenance.
    """

    def __init__(self) -> None:
        self._exprs: dict[Expr, Provenance | None] = {}  # expr -> provenance
        self._by_typecode: dict[str, list[Expr]] = defaultdict(list)

    def __contains__(self, expr: Expr) -> bool:
        return expr in self._exprs

    def __len__(self) -> int:
        return len(self._exprs)

    def add(self, expr: Expr, prov: Provenance | None = None) -> bool:
        """Add an expression.  Returns True if it was new."""
        if expr in self._exprs:
            return False
        self._exprs[expr] = prov
        if expr:
            self._by_typecode[expr[0]].append(expr)
        return True

    def get_by_typecode(self, tc: str) -> list[Expr]:
        return self._by_typecode.get(tc, [])

    def get_provenance(self, expr: Expr) -> Provenance | None:
        return self._exprs.get(expr)

    def all_exprs(self) -> list[Expr]:
        return list(self._exprs.keys())


# ══════════════════════════════════════════════════════════════════════
#  Theorem selection helpers
# ══════════════════════════════════════════════════════════════════════


def select_axioms_and_defs(db: ParsedDatabase) -> list[str]:
    """Return labels of all axioms (no theorems)."""
    return [lbl for lbl, a in db.assertions.items() if a.type == "axiom"]


def select_by_frequency(
    db: ParsedDatabase, top_n: int = 200
) -> list[str]:
    """Return the top-N most-referenced labels across all proofs."""
    counts: dict[str, int] = defaultdict(int)
    for a in db.assertions.values():
        if a.compressed_proof is not None:
            for lbl in a.compressed_proof.labels:
                counts[lbl] += 1
        elif a.proof is not None:
            for lbl in a.proof:
                counts[lbl] += 1
    ranked = sorted(counts, key=counts.__getitem__, reverse=True)
    # Only include assertions (not floating/essential hyps)
    return [lbl for lbl in ranked[:top_n] if lbl in db.assertions]


# ══════════════════════════════════════════════════════════════════════
#  GPU batch verification of derivations
# ══════════════════════════════════════════════════════════════════════


def _get_gpu_verifier():
    """Lazily create and return a GPU verifier (Metal or CUDA).

    Returns None if no GPU backend is available or dependencies are missing.
    """
    if torch.cuda.is_available():
        try:
            from tensormm.tensor_verifier import TensorVerifier
            return TensorVerifier(device=torch.device("cuda"))
        except Exception:
            pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            from tensormm.metal_verifier import MetalVerifier
            return MetalVerifier()
        except Exception:
            pass
    return None


def gpu_verify_derivations(
    derivations: list[tuple[Assertion, dict[str, tuple[str, ...]], Expr]],
    db: ParsedDatabase,
    verifier: object,
) -> list[bool]:
    """Batch-verify derivations on GPU.

    Each derivation is (assertion, substitution, derived_conclusion).
    We pack as verify_flat triples:
      pattern = assertion.expression
      σ       = the substitution
      target  = derived_conclusion

    Returns a list of bools: True if GPU confirms the derivation.
    """
    N = len(derivations)
    if N == 0:
        return []

    # Build symbol table: assign integer IDs to all symbols
    sym2id: dict[str, int] = {"": 0}  # 0 = pad
    next_id = 1

    def _sid(sym: str) -> int:
        nonlocal next_id
        if sym not in sym2id:
            sym2id[sym] = next_id
            next_id += 1
        return sym2id[sym]

    # Pre-scan all symbols to build vocabulary
    for assertion, subst, derived in derivations:
        for tok in assertion.expression:
            _sid(tok)
        for var, rep in subst.items():
            _sid(var)
            for tok in rep:
                _sid(tok)
        for tok in derived:
            _sid(tok)

    V = next_id

    # Compute dimensions
    P_max = max(len(a.expression) for a, _, _ in derivations)
    T_max = max(len(d) for _, _, d in derivations)
    S_max = 1
    for _, subst, _ in derivations:
        for rep in subst.values():
            S_max = max(S_max, len(rep))

    # Build numpy arrays
    patterns = np.zeros((N, P_max), dtype=np.int32)
    pat_lengths = np.zeros(N, dtype=np.int32)
    targets = np.zeros((N, T_max), dtype=np.int32)
    tgt_lengths = np.zeros(N, dtype=np.int32)
    sub_tables = np.zeros((N, V, S_max), dtype=np.int32)
    sub_lengths = np.ones((N, V), dtype=np.int32)

    # Identity: sub_tables[n, v, 0] = v
    ident = np.arange(V, dtype=np.int32)
    sub_tables[:, :, 0] = ident[np.newaxis, :]

    for i, (assertion, subst, derived) in enumerate(derivations):
        # Pattern
        pat = assertion.expression
        pat_lengths[i] = len(pat)
        for j, tok in enumerate(pat):
            patterns[i, j] = sym2id[tok]

        # Target
        tgt_lengths[i] = len(derived)
        for j, tok in enumerate(derived):
            targets[i, j] = sym2id[tok]

        # Substitution table
        for var, rep in subst.items():
            vid = sym2id[var]
            sub_lengths[i, vid] = len(rep)
            for s, tok in enumerate(rep):
                sub_tables[i, vid, s] = sym2id[tok]

    # Ship to GPU
    result = verifier.verify_flat(
        torch.from_numpy(patterns),
        torch.from_numpy(pat_lengths),
        torch.from_numpy(sub_tables),
        torch.from_numpy(sub_lengths),
        torch.from_numpy(targets),
        torch.from_numpy(tgt_lengths),
    )

    return result.tolist()


# ══════════════════════════════════════════════════════════════════════
#  Candidate generation
# ══════════════════════════════════════════════════════════════════════


def _get_mandatory_fhyps(
    assertion: Assertion, db: ParsedDatabase
) -> list[FloatingHyp]:
    """Return the ordered mandatory floating hypotheses for an assertion."""
    return [db.floating_hyps[lbl] for lbl in assertion.floating_hyps]


def _generate_candidates(
    assertion: Assertion,
    db: ParsedDatabase,
    pool: ExpressionPool,
    max_variables: int,
) -> list[tuple[dict[str, tuple[str, ...]], list[Expr]]]:
    """Generate valid (substitution, parent_exprs) pairs for one assertion.

    For each combination of pool entries matching the floating hyp types,
    build the substitution, check all essential hyps against the pool,
    and yield those that pass.
    """
    fhyps = _get_mandatory_fhyps(assertion, db)
    if len(fhyps) > max_variables:
        return []

    # Collect candidate expressions per floating hyp slot
    slot_candidates: list[list[Expr]] = []
    for fh in fhyps:
        cands = pool.get_by_typecode(fh.type_code)
        if not cands:
            return []  # no matching expressions → no derivations possible
        slot_candidates.append(cands)

    ehyp_exprs = [
        db.essential_hyps[lbl].expression for lbl in assertion.essential_hyps
    ]

    results: list[tuple[dict[str, tuple[str, ...]], list[Expr]]] = []

    for combo in itertools.product(*slot_candidates):
        # Build substitution: variable -> expression[1:] (strip typecode)
        subst: dict[str, tuple[str, ...]] = {}
        for fh, expr in zip(fhyps, combo):
            subst[fh.variable] = expr[1:]  # strip typecode

        # Check essential hypotheses
        all_match = True
        for ehyp_expr in ehyp_exprs:
            substituted = _apply_subst_tuple(ehyp_expr, subst)
            if substituted not in pool:
                all_match = False
                break

        if all_match:
            results.append((subst, list(combo)))

    return results


def _apply_subst_tuple(
    expression: Sequence[str], subst: dict[str, tuple[str, ...]]
) -> Expr:
    """Apply substitution and return an immutable Expr tuple."""
    result: list[str] = []
    for tok in expression:
        if tok in subst:
            result.extend(subst[tok])
        else:
            result.append(tok)
    return tuple(result)


# ══════════════════════════════════════════════════════════════════════
#  Proof reconstruction
# ══════════════════════════════════════════════════════════════════════


def _reconstruct_proof_stack(
    target: Expr,
    pool: ExpressionPool,
    db: ParsedDatabase,
) -> list[str]:
    """Reconstruct an uncompressed Metamath proof using stack-machine semantics.

    This builds the proof as a sequence of labels that, when executed by the
    Metamath stack machine, leaves exactly the target expression on the stack.

    The key insight: for each derived expression, we need to push its
    floating-hyp values (via $f labels for each token), then push essential
    hyp proofs (recursively), then push the assertion label.
    """
    proof_labels: list[str] = []

    def _emit(expr: Expr) -> None:
        """Emit proof labels that leave `expr` on the proof stack."""
        prov = pool.get_provenance(expr)

        if prov is None:
            # Base case: seeded expression.  Need to figure out what it is.
            expr_list = list(expr)
            if len(expr_list) == 2 and expr_list[1] in db.variables:
                for lbl, fh in db.floating_hyps.items():
                    if fh.type_code == expr_list[0] and fh.variable == expr_list[1]:
                        proof_labels.append(lbl)
                        return

            for lbl, eh in db.essential_hyps.items():
                if tuple(eh.expression) == expr:
                    proof_labels.append(lbl)
                    return

            for lbl, a in db.assertions.items():
                if (tuple(a.expression) == expr
                        and not a.floating_hyps
                        and not a.essential_hyps):
                    proof_labels.append(lbl)
                    return

            raise ValueError(
                f"Cannot find label for seeded expression: {expr}"
            )

        # Derived expression: push hyps then assertion label
        assertion = db.assertions[prov.theorem_label]
        fhyps = _get_mandatory_fhyps(assertion, db)

        for fh in fhyps:
            fh_value = prov.substitution[fh.variable]
            fh_expr = (fh.type_code,) + fh_value
            _emit(fh_expr)

        ehyp_exprs = [
            db.essential_hyps[lbl].expression for lbl in assertion.essential_hyps
        ]
        for ehyp_expr in ehyp_exprs:
            subst_ehyp = _apply_subst_tuple(ehyp_expr, prov.substitution)
            _emit(subst_ehyp)

        proof_labels.append(prov.theorem_label)

    _emit(target)
    return proof_labels


# ══════════════════════════════════════════════════════════════════════
#  Main search
# ══════════════════════════════════════════════════════════════════════


def forward_search(
    db: ParsedDatabase,
    target_label: str,
    *,
    max_depth: int = 10,
    timeout: float = 60.0,
    max_variables: int = 4,
    max_expr_len: int = 50,
    max_pool_size: int = 500_000,
    available_labels: list[str] | None = None,
    use_gpu: bool = False,
) -> SearchResult:
    """Run forward-chaining brute-force proof search.

    Parameters
    ----------
    db : ParsedDatabase
        The Metamath database.
    target_label : str
        Label of the theorem whose conclusion we want to derive.
    max_depth : int
        Maximum search depth (number of forward-chaining levels).
    timeout : float
        Maximum wall-clock seconds.
    max_variables : int
        Skip assertions with more than this many mandatory floating hyps.
    max_expr_len : int
        Prune derived expressions longer than this.
    max_pool_size : int
        Stop if pool exceeds this size.
    available_labels : list[str] | None
        Which assertion labels to use for candidate generation.
        If None, uses all axioms in the database.
    use_gpu : bool
        If True, GPU-verify every new derivation at each depth level.
    """
    t0 = time.perf_counter()

    if target_label not in db.assertions:
        return SearchResult(
            success=False, target=[], elapsed=time.perf_counter() - t0
        )

    target_assertion = db.assertions[target_label]
    target_expr: Expr = tuple(target_assertion.expression)

    # Select available assertions
    if available_labels is None:
        available_labels = select_axioms_and_defs(db)
    available_assertions = [
        db.assertions[lbl]
        for lbl in available_labels
        if lbl in db.assertions
    ]

    # GPU verifier (lazy init)
    gpu_verifier = None
    if use_gpu:
        gpu_verifier = _get_gpu_verifier()
        if gpu_verifier is None:
            print("  [warn] No GPU available, falling back to CPU-only search",
                  flush=True)
            use_gpu = False

    # ── 1. Initialize the pool ──────────────────────────────────────
    pool = ExpressionPool()

    # Add all axiom conclusions with no hypotheses
    for lbl, a in db.assertions.items():
        if a.type == "axiom" and not a.floating_hyps and not a.essential_hyps:
            pool.add(tuple(a.expression))

    # Add all floating hypothesis expressions
    for lbl, fh in db.floating_hyps.items():
        pool.add((fh.type_code, fh.variable))

    # Add the target theorem's essential hypotheses
    for ehyp_lbl in target_assertion.essential_hyps:
        eh = db.essential_hyps[ehyp_lbl]
        pool.add(tuple(eh.expression))

    # Early check
    if target_expr in pool:
        return SearchResult(
            success=True,
            target=list(target_expr),
            proof_labels=[],
            depth_reached=0,
            pool_size=len(pool),
            elapsed=time.perf_counter() - t0,
        )

    total_tried = 0
    total_valid = 0
    total_gpu_verified = 0
    total_gpu_rejected = 0
    level_stats: list[dict] = []

    # ── 2. Depth loop ───────────────────────────────────────────────
    for depth in range(1, max_depth + 1):
        if time.perf_counter() - t0 > timeout:
            break

        level_tried = 0
        level_valid = 0
        # Each entry: (derived_expr, provenance, assertion_obj, subst)
        new_derivations: list[tuple[Expr, Provenance, Assertion, dict]] = []

        for assertion in available_assertions:
            if time.perf_counter() - t0 > timeout:
                break

            candidates = _generate_candidates(
                assertion, db, pool, max_variables
            )

            for subst, parent_exprs_list in candidates:
                level_tried += 1
                derived = _apply_subst_tuple(
                    assertion.expression, subst
                )

                if len(derived) > max_expr_len:
                    continue
                if derived in pool:
                    level_valid += 1
                    continue

                level_valid += 1
                prov = Provenance(
                    theorem_label=assertion.label,
                    substitution=subst,
                    parent_exprs=tuple(
                        tuple(e) for e in parent_exprs_list
                    ),
                    depth=depth,
                )
                new_derivations.append((derived, prov, assertion, subst))

        # ── GPU verification of new derivations ─────────────────────
        gpu_ok_count = 0
        gpu_rej_count = 0

        if use_gpu and new_derivations:
            triples = [
                (a, s, d)
                for d, _prov, a, s in new_derivations
            ]
            t_gpu0 = time.perf_counter()
            results = gpu_verify_derivations(triples, db, gpu_verifier)
            t_gpu1 = time.perf_counter()

            # Filter: only keep GPU-confirmed derivations
            verified_derivations = []
            for (derived, prov, assertion, subst), ok in zip(
                new_derivations, results
            ):
                if ok:
                    verified_derivations.append((derived, prov))
                    gpu_ok_count += 1
                else:
                    gpu_rej_count += 1

            new_to_add = verified_derivations
        else:
            new_to_add = [(d, p) for d, p, _a, _s in new_derivations]
            t_gpu0 = t_gpu1 = 0.0

        # Add new expressions to pool
        actually_new = 0
        for expr, prov in new_to_add:
            if pool.add(expr, prov):
                actually_new += 1

        total_tried += level_tried
        total_valid += level_valid
        total_gpu_verified += gpu_ok_count
        total_gpu_rejected += gpu_rej_count

        gpu_tag = ""
        if use_gpu:
            gpu_ms = (t_gpu1 - t_gpu0) * 1000
            gpu_tag = (
                f" gpu={gpu_ok_count:,}✓/{gpu_rej_count}✗ {gpu_ms:.1f}ms"
            )

        stats = {
            "depth": depth,
            "candidates_tried": level_tried,
            "candidates_valid": level_valid,
            "new_expressions": actually_new,
            "pool_size": len(pool),
            "gpu_verified": gpu_ok_count,
            "gpu_rejected": gpu_rej_count,
            "elapsed": time.perf_counter() - t0,
        }
        level_stats.append(stats)
        print(
            f"  [depth {depth}] tried={level_tried:,} valid={level_valid:,} "
            f"new={actually_new:,} pool={len(pool):,}{gpu_tag} "
            f"t={stats['elapsed']:.2f}s",
            flush=True,
        )

        # Check target
        if target_expr in pool:
            proof_labels = _reconstruct_proof_stack(target_expr, pool, db)

            # Verify with CPU verifier (ground truth)
            verification = _verify_reconstructed(
                target_label, proof_labels, db
            )

            return SearchResult(
                success=verification.success if verification else False,
                target=list(target_expr),
                proof_labels=proof_labels,
                verification=verification,
                depth_reached=depth,
                pool_size=len(pool),
                candidates_tried=total_tried,
                candidates_valid=total_valid,
                gpu_verified=total_gpu_verified,
                gpu_rejected=total_gpu_rejected,
                elapsed=time.perf_counter() - t0,
                level_stats=level_stats,
            )

        # Pool size check
        if len(pool) >= max_pool_size:
            print(
                f"  [depth {depth}] Pool cap reached ({max_pool_size:,}) — stopping.",
                flush=True,
            )
            break

        # Growth check
        if actually_new == 0:
            print(
                f"  [depth {depth}] No new expressions — search space exhausted.",
                flush=True,
            )
            break

    return SearchResult(
        success=False,
        target=list(target_expr),
        depth_reached=depth if level_stats else 0,
        pool_size=len(pool),
        candidates_tried=total_tried,
        candidates_valid=total_valid,
        gpu_verified=total_gpu_verified,
        gpu_rejected=total_gpu_rejected,
        elapsed=time.perf_counter() - t0,
        level_stats=level_stats,
    )


# ══════════════════════════════════════════════════════════════════════
#  Verification of reconstructed proofs
# ══════════════════════════════════════════════════════════════════════


def _verify_reconstructed(
    target_label: str,
    proof_labels: list[str],
    db: ParsedDatabase,
) -> VerificationResult:
    """Temporarily patch the target assertion's proof to our reconstructed one,
    then verify it with the CPU verifier.
    """
    assertion = db.assertions[target_label]
    # Save original
    orig_proof = assertion.proof
    orig_compressed = assertion.compressed_proof

    try:
        # Patch in our proof
        assertion.proof = proof_labels
        assertion.compressed_proof = None

        verifier = CPUVerifier(db)
        result = verifier.verify_proof(target_label)
        return result
    finally:
        # Restore original
        assertion.proof = orig_proof
        assertion.compressed_proof = orig_compressed
