"""CPU reference verifier: stack-machine Metamath proof verification."""

from __future__ import annotations

import itertools
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from tensormm.parser import ParsedDatabase

# Cap workers to avoid diminishing returns on many-core systems.
_MAX_CPU_WORKERS = 32

# Worker process global — set by _init_cpu_worker or inherited via fork.
_CPU_WORKER_PARSED: ParsedDatabase | None = None


def _init_cpu_worker(parsed: ParsedDatabase) -> None:
    """ProcessPoolExecutor initializer for non-fork platforms."""
    global _CPU_WORKER_PARSED
    _CPU_WORKER_PARSED = parsed


@dataclass
class VerificationResult:
    """Result of verifying a single theorem's proof."""

    success: bool
    label: str
    error_message: str | None = None
    steps_verified: int = 0


Stmt = list[str]


def apply_substitution(expression: Stmt, substitution: dict[str, Stmt]) -> Stmt:
    """Apply a variable substitution to a statement.

    For each token: if it's a variable in the substitution dict, replace it
    with the substitution's token list. Otherwise keep the token as-is.
    """
    result: list[str] = []
    for tok in expression:
        if tok in substitution:
            result.extend(substitution[tok])
        else:
            result.append(tok)
    return result


def _verify_chunk(
    labels: list[str],
) -> dict[str, "VerificationResult"]:
    """Worker function for ProcessPoolExecutor.

    Uses process-global _CPU_WORKER_PARSED (set by initializer or inherited
    via fork). Creates a CPUVerifier in the worker and verifies the labels.
    """
    assert _CPU_WORKER_PARSED is not None
    verifier = CPUVerifier(_CPU_WORKER_PARSED)
    return {lbl: verifier.verify_proof(lbl) for lbl in labels}


class CPUVerifier:
    """Reference CPU implementation of the Metamath proof verification algorithm.

    Operates on ParsedDatabase directly using string tokens (no tensorization).
    Supports both uncompressed and compressed proofs.
    """

    def __init__(self, parsed: ParsedDatabase) -> None:
        self.db = parsed
        # Build a combined label lookup: label -> ("$f"|"$e"|"$a"|"$p", data)
        self._label_info: dict[str, tuple[str, object]] = {}

        for lbl, fh in parsed.floating_hyps.items():
            self._label_info[lbl] = ("$f", [fh.type_code, fh.variable])

        for lbl, eh in parsed.essential_hyps.items():
            self._label_info[lbl] = ("$e", eh.expression)

        for lbl, assertion in parsed.assertions.items():
            stmt_type = "$a" if assertion.type == "axiom" else "$p"
            self._label_info[lbl] = (stmt_type, assertion)

    def _treat_step(
        self,
        label: str,
        stack: list[Stmt],
        active_dvs: set[tuple[str, str]] | None = None,
    ) -> None:
        """Process a single proof step by label, modifying the stack in place.

        active_dvs: all $d pairs active in the scope of the theorem being
        proved.  Needed for the full Metamath $d check: when an invoked
        assertion has $d x y, every pair of variables (x0, y0) drawn from
        subst(x) and subst(y) must satisfy x0 != y0 AND $d x0 y0 must be
        active.
        """
        if label not in self._label_info:
            raise ValueError(f"Unknown label in proof: {label}")

        stmt_type, data = self._label_info[label]

        if stmt_type == "$f":
            # Floating hypothesis: push [typecode, variable]
            stack.append(list(data))
        elif stmt_type == "$e":
            # Essential hypothesis: push expression
            stack.append(list(data))
        elif stmt_type in ("$a", "$p"):
            # Assertion: pop hypotheses, compute substitution, push conclusion
            assertion = data
            f_hyp_labels = assertion.floating_hyps
            e_hyp_labels = assertion.essential_hyps

            npop = len(f_hyp_labels) + len(e_hyp_labels)
            sp = len(stack) - npop
            if sp < 0:
                raise ValueError(
                    f"Stack underflow: {label} requires {npop} hypotheses, "
                    f"but stack has {len(stack)} entries"
                )

            # Build substitution from floating hypotheses
            subst: dict[str, Stmt] = {}
            for flbl in f_hyp_labels:
                fh = self.db.floating_hyps[flbl]
                entry = stack[sp]
                if entry[0] != fh.type_code:
                    raise ValueError(
                        f"Type mismatch for {label}: expected '{fh.type_code}', "
                        f"got '{entry[0]}' in stack entry {entry}"
                    )
                subst[fh.variable] = entry[1:]
                sp += 1

            # Check essential hypotheses
            for elbl in e_hyp_labels:
                eh = self.db.essential_hyps[elbl]
                entry = stack[sp]
                expected = apply_substitution(eh.expression, subst)
                if entry != expected:
                    raise ValueError(
                        f"Essential hypothesis mismatch for {label}: "
                        f"stack has {entry}, expected {expected}"
                    )
                sp += 1

            # Check disjoint variable conditions
            for x, y in assertion.disjoint_vars:
                x_vars = {t for t in subst.get(x, []) if t in self.db.variables}
                y_vars = {t for t in subst.get(y, []) if t in self.db.variables}
                for x0, y0 in itertools.product(x_vars, y_vars):
                    if x0 == y0:
                        raise ValueError(
                            f"Disjoint variable violation in {label}: "
                            f"{x0} appears in substitutions for both {x} and {y}"
                        )
                    if active_dvs is not None:
                        pair = (min(x0, y0), max(x0, y0))
                        if pair not in active_dvs:
                            raise ValueError(
                                f"Disjoint variable violation in {label}: "
                                f"${label} requires $d {x} {y}, substitution maps to "
                                f"{x0} and {y0} but no active $d {x0} {y0}"
                            )

            # Pop consumed entries and push the substituted conclusion
            del stack[len(stack) - npop :]
            stack.append(apply_substitution(assertion.expression, subst))

    def _treat_saved_stmt(self, stmt: Stmt, stack: list[Stmt]) -> None:
        """Push a previously saved statement onto the stack (compressed proof Z-reuse)."""
        stack.append(list(stmt))

    def verify_proof(self, label: str) -> VerificationResult:
        """Verify the proof of a single theorem."""
        if label not in self.db.assertions:
            return VerificationResult(
                success=False,
                label=label,
                error_message=f"Label '{label}' not found in assertions",
            )

        assertion = self.db.assertions[label]
        if assertion.type != "theorem":
            return VerificationResult(
                success=False,
                label=label,
                error_message=f"Label '{label}' is an axiom, not a theorem",
            )

        stack: list[Stmt] = []
        steps = 0
        active_dvs = assertion.all_disjoint_vars or None

        try:
            if assertion.compressed_proof is not None:
                # Compressed proof processing
                cp = assertion.compressed_proof
                plabels = cp.labels
                label_end = len(plabels)
                saved_stmts: list[Stmt] = []

                for proof_int in cp.proof_ints:
                    if proof_int == -1:
                        # Z: save current stack top
                        if not stack:
                            raise ValueError("Z save on empty stack")
                        saved_stmts.append(list(stack[-1]))
                    elif proof_int < label_end:
                        # Reference a label
                        self._treat_step(plabels[proof_int], stack, active_dvs)
                        steps += 1
                    else:
                        # Reference a saved statement
                        saved_idx = proof_int - label_end
                        if saved_idx >= len(saved_stmts):
                            raise ValueError(
                                f"Saved stmt index {saved_idx} out of range "
                                f"(only {len(saved_stmts)} saved)"
                            )
                        self._treat_saved_stmt(saved_stmts[saved_idx], stack)
                        steps += 1

            elif assertion.proof is not None:
                # Uncompressed proof: each token is a label
                for step_label in assertion.proof:
                    self._treat_step(step_label, stack, active_dvs)
                    steps += 1
            else:
                return VerificationResult(
                    success=False, label=label, error_message="Theorem has no proof"
                )

            # Final check: stack must have exactly one entry matching conclusion
            if len(stack) != 1:
                return VerificationResult(
                    success=False,
                    label=label,
                    error_message=f"Stack has {len(stack)} entries at end of proof, expected 1",
                    steps_verified=steps,
                )

            if stack[0] != assertion.expression:
                return VerificationResult(
                    success=False,
                    label=label,
                    error_message=(
                        f"Final stack entry {stack[0]} does not match "
                        f"conclusion {assertion.expression}"
                    ),
                    steps_verified=steps,
                )

            return VerificationResult(success=True, label=label, steps_verified=steps)

        except ValueError as e:
            return VerificationResult(
                success=False,
                label=label,
                error_message=str(e),
                steps_verified=steps,
            )

    def verify_all(
        self, max_workers: int | None = None
    ) -> dict[str, VerificationResult]:
        """Verify all theorems in the database using a ProcessPoolExecutor.

        Each theorem's proof is verified independently in a worker process,
        bypassing the GIL for true CPU parallelism.

        Args:
            max_workers: Number of worker processes. Defaults to os.cpu_count().
        """
        theorem_labels = [
            label
            for label, assertion in self.db.assertions.items()
            if assertion.type == "theorem"
        ]

        if not theorem_labels:
            return {}

        global _CPU_WORKER_PARSED
        workers = min(max_workers or os.cpu_count() or 1, _MAX_CPU_WORKERS)
        results: dict[str, VerificationResult] = {}

        chunk_size = max(1, (len(theorem_labels) + workers - 1) // workers)
        chunks = [
            theorem_labels[i : i + chunk_size]
            for i in range(0, len(theorem_labels), chunk_size)
        ]

        # On Linux: fork inherits parent memory — zero pickle cost.
        # On macOS/Windows: use initializer (pickles once per worker).
        if sys.platform == "linux":
            _CPU_WORKER_PARSED = self.db
            ctx = multiprocessing.get_context("fork")
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        else:
            pool = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_cpu_worker,
                initargs=(self.db,),
            )

        with pool as executor:
            futures = {
                executor.submit(_verify_chunk, chunk): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                chunk_results = future.result()
                results.update(chunk_results)

        return results
