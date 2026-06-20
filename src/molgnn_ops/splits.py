import math
import random
import re
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover - exercised only in installations without RDKit
    Chem = None
    MurckoScaffold = None

_SPLIT_NAMES = ("train", "val", "test")
_SMILES_TOKEN_PATTERN = re.compile(
    r"\[[^\]]+\]|Br|Cl|[A-Z][a-z]?|[bcnops]|\*|%\d{2}|\d|[()=#$@+\-\\/.:]"
)
_ATOM_PATTERN = re.compile(r"Br|Cl|[A-Z][a-z]?|[bcnops]|\*")


def _validate_split_inputs(
    n: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> None:
    if n <= 0:
        raise ValueError("n must be greater than 0")

    fractions = (train_frac, val_frac, test_frac)
    if any(fraction < 0 or fraction > 1 for fraction in fractions):
        raise ValueError("Split fractions must each be between 0 and 1")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Split fractions must sum to 1.0")


def _split_sizes(
    n: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> dict[str, int]:
    fractions = (train_frac, val_frac, test_frac)
    exact_sizes = [n * fraction for fraction in fractions]
    positive_splits = sum(fraction > 0 for fraction in fractions)
    if n >= positive_splits:
        sizes = [int(fraction > 0) for fraction in fractions]
    else:
        sizes = [0] * len(fractions)

    while sum(sizes) < n:
        index = max(
            range(len(sizes)),
            key=lambda candidate: exact_sizes[candidate] - sizes[candidate],
        )
        sizes[index] += 1
    return dict(zip(_SPLIT_NAMES, sizes, strict=True))


def random_split_indices(
    n: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[int]]:
    """Return a deterministic random partition of indices from zero to ``n - 1``."""
    _validate_split_inputs(n, train_frac, val_frac, test_frac)
    sizes = _split_sizes(n, train_frac, val_frac, test_frac)

    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    train_end = sizes["train"]
    val_end = train_end + sizes["val"]
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def _fallback_scaffold_key(smiles: str) -> str:
    """Create a topology-like token key that is deterministic but not chemical."""
    tokens = _SMILES_TOKEN_PATTERN.findall(smiles)
    if not tokens or "".join(tokens) != smiles:
        return f"fallback_shape:{smiles}"

    ring_labels: dict[str, int] = {}
    shape_tokens: list[str] = []
    for token in tokens:
        if token.startswith("["):
            shape_tokens.append("a" if len(token) > 1 and token[1].islower() else "A")
        elif _ATOM_PATTERN.fullmatch(token):
            shape_tokens.append("a" if token.islower() else "A")
        elif token.isdigit() or token.startswith("%"):
            ring_label = token.removeprefix("%")
            ring_number = ring_labels.setdefault(ring_label, len(ring_labels))
            shape_tokens.append(f"R{ring_number}")
        elif token not in {"-", "@", "/", "\\"}:
            shape_tokens.append(token)

    return f"fallback_shape:{'|'.join(shape_tokens)}"


def scaffold_key_from_smiles(smiles: str) -> str:
    """Return a Bemis-Murcko key, or a documented non-chemical fallback key."""
    normalized_smiles = "".join(smiles.split())
    if not normalized_smiles:
        raise ValueError("SMILES must not be empty")

    if Chem is not None and MurckoScaffold is not None:
        molecule = Chem.MolFromSmiles(normalized_smiles)
        if molecule is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=molecule,
                includeChirality=False,
            )
            return f"bemis_murcko:{scaffold or '<acyclic>'}"

    return _fallback_scaffold_key(normalized_smiles)


def scaffold_split_indices(
    smiles: list[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[int]]:
    """Split indices while keeping every scaffold group in exactly one partition."""
    n = len(smiles)
    _validate_split_inputs(n, train_frac, val_frac, test_frac)

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(smiles):
        grouped_indices[scaffold_key_from_smiles(value)].append(index)

    groups = list(grouped_indices.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)

    fractions = dict(
        zip(_SPLIT_NAMES, (train_frac, val_frac, test_frac), strict=True)
    )
    target_sizes = {
        name: n * fraction for name, fraction in fractions.items()
    }
    split_indices = {name: [] for name in _SPLIT_NAMES}

    for group in groups:
        eligible_splits = [name for name in _SPLIT_NAMES if fractions[name] > 0]
        destination = max(
            eligible_splits,
            key=lambda name: target_sizes[name] - len(split_indices[name]),
        )
        split_indices[destination].extend(group)

    return split_indices
