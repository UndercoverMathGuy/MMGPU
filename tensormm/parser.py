"""Parse .mm files into structured data."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FloatingHyp:
    """A floating hypothesis: variable type declaration like 'wph $f wff ph $.'"""
    label: str
    type_code: str       # e.g., "wff", "class", "setvar"
    variable: str        # the variable being typed


@dataclass
class EssentialHyp:
    """An essential hypothesis: logical hypothesis like 'min $e |- P $.'"""
    label: str
    expression: list[str]  # the hypothesis expression as symbol strings


@dataclass
class CompressedProof:
    """Structured representation of a compressed proof."""
    labels: list[str]        # the label list (f_hyps + e_hyps + explicit labels)
    proof_ints: list[int]    # decoded integer references (-1 = Z/save)


@dataclass
class Assertion:
    """An axiom or theorem."""
    label: str
    type: str                  # "axiom" or "theorem"
    expression: list[str]      # the conclusion expression as symbol strings
    floating_hyps: list[str]   # ordered list of mandatory floating hyp labels
    essential_hyps: list[str]  # ordered list of mandatory essential hyp labels
    proof: list[str] | None    # proof steps as labels (None for axioms), uncompressed
    compressed_proof: CompressedProof | None  # structured compressed proof data
    disjoint_vars: list[tuple[str, str]]


@dataclass
class ParsedDatabase:
    """Raw parsed data from a .mm file."""
    constants: set[str] = field(default_factory=set)
    variables: set[str] = field(default_factory=set)
    floating_hyps: dict[str, FloatingHyp] = field(default_factory=dict)
    essential_hyps: dict[str, EssentialHyp] = field(default_factory=dict)
    assertions: dict[str, Assertion] = field(default_factory=dict)


@dataclass
class _Frame:
    """Scope frame tracking declarations within a ${ ... $} block."""
    variables: set[str] = field(default_factory=set)
    floating_hyps: list[FloatingHyp] = field(default_factory=list)
    f_labels: dict[str, str] = field(default_factory=dict)  # variable -> label
    essential_hyps: list[EssentialHyp] = field(default_factory=list)
    disjoint_vars: set[tuple[str, str]] = field(default_factory=set)


def _decompress_proof(proof_tokens: list[str]) -> CompressedProof:
    """Decompress a compressed proof from its token representation.

    Format after $=:
        ( label1 label2 ... ) ENCODEDSTRING
    where ENCODEDSTRING uses:
        A-T (0-19): terminal digits in base-20 encoding
        U-Y: non-terminal digits (5 * cur_int + ord(ch) - 84)
        Z: save current stack top to heap (-1 in proof_ints)
    """
    # Find the label block between ( and )
    assert proof_tokens[0] == "(", f"Compressed proof must start with '(', got '{proof_tokens[0]}'"
    try:
        paren_end = proof_tokens.index(")")
    except ValueError:
        raise ValueError("Compressed proof missing closing ')'")

    labels = proof_tokens[1:paren_end]

    # Everything after ) is the encoded string, joined together
    encoded = "".join(proof_tokens[paren_end + 1:])

    # Decode the encoded string into integer references
    proof_ints: list[int] = []
    cur_int = 0
    for ch in encoded:
        if ch == "Z":
            proof_ints.append(-1)
        elif "A" <= ch <= "T":
            proof_ints.append(20 * cur_int + ord(ch) - 65)
            cur_int = 0
        elif "U" <= ch <= "Y":
            cur_int = 5 * cur_int + ord(ch) - 84
        else:
            raise ValueError(f"Invalid character '{ch}' in compressed proof")

    return CompressedProof(labels=labels, proof_ints=proof_ints)


class _FrameStack:
    """Stack of scope frames, managing Metamath's ${ ... $} scoping."""

    def __init__(self) -> None:
        self.frames: list[_Frame] = []

    def push(self) -> None:
        self.frames.append(_Frame())

    def pop(self) -> _Frame:
        return self.frames.pop()

    def add_variable(self, var: str) -> None:
        self.frames[-1].variables.add(var)

    def add_floating_hyp(self, fhyp: FloatingHyp) -> None:
        self.frames[-1].floating_hyps.append(fhyp)
        self.frames[-1].f_labels[fhyp.variable] = fhyp.label

    def add_essential_hyp(self, ehyp: EssentialHyp) -> None:
        self.frames[-1].essential_hyps.append(ehyp)

    def add_disjoint(self, vars_list: list[str]) -> None:
        frame = self.frames[-1]
        for i, v1 in enumerate(vars_list):
            for v2 in vars_list[i + 1:]:
                pair = (min(v1, v2), max(v1, v2))
                frame.disjoint_vars.add(pair)

    def is_variable(self, tok: str) -> bool:
        return any(tok in f.variables for f in self.frames)

    def lookup_float_label(self, var: str) -> str | None:
        """Return the label of the active $f statement typing this variable."""
        for frame in self.frames:
            if var in frame.f_labels:
                return frame.f_labels[var]
        return None

    def make_assertion(
        self,
        stmt: list[str],
        constants: set[str],
    ) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """Build the mandatory hypothesis lists and disjoint var conditions for an assertion.

        Returns (floating_hyp_labels, essential_hyp_labels, disjoint_var_pairs).

        CRITICAL: Mandatory floating hyps are ONLY those typing variables that
        appear in the essential hypotheses or conclusion. NOT all in-scope floats.
        """
        # Collect all essential hyps in scope (order matters)
        e_hyps = [eh for frame in self.frames for eh in frame.essential_hyps]
        e_labels = [eh.label for eh in e_hyps]

        # Collect variables appearing in essential hyps + conclusion
        mand_vars: set[str] = set()
        for eh in e_hyps:
            for tok in eh.expression:
                if self.is_variable(tok):
                    mand_vars.add(tok)
        for tok in stmt:
            if self.is_variable(tok):
                mand_vars.add(tok)

        # Collect mandatory floating hyps — only those typing mandatory variables
        # Preserve order: iterate frames bottom-to-top, then hyps in order within each frame
        f_labels: list[str] = []
        seen_vars: set[str] = set()
        for frame in self.frames:
            for fhyp in frame.floating_hyps:
                if fhyp.variable in mand_vars and fhyp.variable not in seen_vars:
                    f_labels.append(fhyp.label)
                    seen_vars.add(fhyp.variable)

        # Collect disjoint var conditions involving mandatory variables
        dvs: list[tuple[str, str]] = []
        for frame in self.frames:
            for x, y in frame.disjoint_vars:
                if x in mand_vars and y in mand_vars:
                    if (x, y) not in dvs:
                        dvs.append((x, y))

        return f_labels, e_labels, dvs


def parse_mm_file(filepath: str) -> ParsedDatabase:
    """Parse a .mm file and return structured data.

    Reads the entire file, splits into whitespace-delimited tokens,
    and processes them sequentially with a scope stack.
    """
    with open(filepath, "r", encoding="ascii", errors="replace") as f:
        raw = f.read()

    tokens = raw.split()
    db = ParsedDatabase()
    fs = _FrameStack()
    fs.push()  # global scope

    i = 0
    label: str | None = None

    while i < len(tokens):
        tok = tokens[i]

        # Skip comments: $( ... $)
        if tok == "$(":
            i += 1
            while i < len(tokens) and tokens[i] != "$)":
                i += 1
            i += 1  # skip $)
            continue

        # Constant declaration: $c ... $.
        if tok == "$c":
            i += 1
            while i < len(tokens) and tokens[i] != "$.":
                db.constants.add(tokens[i])
                i += 1
            i += 1  # skip $.
            continue

        # Variable declaration: $v ... $.
        if tok == "$v":
            i += 1
            while i < len(tokens) and tokens[i] != "$.":
                var = tokens[i]
                db.variables.add(var)
                fs.add_variable(var)
                i += 1
            i += 1  # skip $.
            continue

        # Disjoint variable: $d ... $.
        if tok == "$d":
            i += 1
            vars_list: list[str] = []
            while i < len(tokens) and tokens[i] != "$.":
                vars_list.append(tokens[i])
                i += 1
            i += 1  # skip $.
            fs.add_disjoint(vars_list)
            continue

        # Open scope
        if tok == "${":
            fs.push()
            i += 1
            continue

        # Close scope
        if tok == "$}":
            fs.pop()
            i += 1
            continue

        # Floating hypothesis: label $f type variable $.
        if tok == "$f":
            if label is None:
                raise ValueError("$f statement without a label")
            i += 1
            stmt_tokens: list[str] = []
            while i < len(tokens) and tokens[i] != "$.":
                stmt_tokens.append(tokens[i])
                i += 1
            i += 1  # skip $.
            if len(stmt_tokens) != 2:
                raise ValueError(f"$f must have exactly 2 tokens, got {stmt_tokens}")
            fhyp = FloatingHyp(label=label, type_code=stmt_tokens[0], variable=stmt_tokens[1])
            db.floating_hyps[label] = fhyp
            fs.add_floating_hyp(fhyp)
            label = None
            continue

        # Essential hypothesis: label $e expression $.
        if tok == "$e":
            if label is None:
                raise ValueError("$e statement without a label")
            i += 1
            expr: list[str] = []
            while i < len(tokens) and tokens[i] != "$.":
                if tokens[i] == "$(":
                    i += 1
                    while i < len(tokens) and tokens[i] != "$)":
                        i += 1
                    i += 1  # skip $)
                    continue
                expr.append(tokens[i])
                i += 1
            i += 1  # skip $.
            ehyp = EssentialHyp(label=label, expression=expr)
            db.essential_hyps[label] = ehyp
            fs.add_essential_hyp(ehyp)
            label = None
            continue

        # Axiom: label $a expression $.
        if tok == "$a":
            if label is None:
                raise ValueError("$a statement without a label")
            i += 1
            expr = []
            while i < len(tokens) and tokens[i] != "$.":
                if tokens[i] == "$(":
                    i += 1
                    while i < len(tokens) and tokens[i] != "$)":
                        i += 1
                    i += 1  # skip $)
                    continue
                expr.append(tokens[i])
                i += 1
            i += 1  # skip $.
            f_labels, e_labels, dvs = fs.make_assertion(expr, db.constants)
            assertion = Assertion(
                label=label,
                type="axiom",
                expression=expr,
                floating_hyps=f_labels,
                essential_hyps=e_labels,
                proof=None,
                compressed_proof=None,
                disjoint_vars=dvs,
            )
            db.assertions[label] = assertion
            label = None
            continue

        # Theorem with proof: label $p expression $= proof $.
        if tok == "$p":
            if label is None:
                raise ValueError("$p statement without a label")
            i += 1
            expr = []
            while i < len(tokens) and tokens[i] != "$=":
                if tokens[i] == "$(":
                    i += 1
                    while i < len(tokens) and tokens[i] != "$)":
                        i += 1
                    i += 1  # skip $)
                    continue
                expr.append(tokens[i])
                i += 1
            i += 1  # skip $=

            # Read proof tokens (skip embedded comments)
            proof_tokens: list[str] = []
            while i < len(tokens) and tokens[i] != "$.":
                if tokens[i] == "$(":
                    i += 1
                    while i < len(tokens) and tokens[i] != "$)":
                        i += 1
                    i += 1  # skip $)
                    continue
                proof_tokens.append(tokens[i])
                i += 1
            i += 1  # skip $.

            f_labels, e_labels, dvs = fs.make_assertion(expr, db.constants)

            # Detect compressed vs uncompressed proof
            compressed_proof: CompressedProof | None = None
            uncompressed_proof: list[str] | None = None

            if proof_tokens and proof_tokens[0] == "(":
                # Compressed proof — build full label list:
                # f_hyp_labels + e_hyp_labels + explicit labels from ( ... )
                cp = _decompress_proof(proof_tokens)
                full_labels = f_labels + e_labels + cp.labels
                compressed_proof = CompressedProof(
                    labels=full_labels,
                    proof_ints=cp.proof_ints,
                )
            else:
                uncompressed_proof = proof_tokens

            assertion = Assertion(
                label=label,
                type="theorem",
                expression=expr,
                floating_hyps=f_labels,
                essential_hyps=e_labels,
                proof=uncompressed_proof,
                compressed_proof=compressed_proof,
                disjoint_vars=dvs,
            )
            db.assertions[label] = assertion
            label = None
            continue

        # If token starts with $, it's an unknown keyword
        if tok.startswith("$"):
            raise ValueError(f"Unknown keyword: {tok}")

        # Otherwise it's a label for the next statement
        label = tok
        i += 1

    return db
