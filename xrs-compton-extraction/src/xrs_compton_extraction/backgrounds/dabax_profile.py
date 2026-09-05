"""Read an explicitly supplied DABAX ComptonProfiles.dat without xraylib.

Shell_N labels are retained verbatim. They are table-column identifiers, not
automatically assigned spectroscopic shell names. Occupations come from UOCCUP.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .core_profile import BIGGS_REFERENCE_DOI, ElementIdentity


class DabaxProfileSource:
    """Per-electron partial profiles with linear interpolation in |pz|.

    No network access, implicit path, external dependency or extrapolation.
    The table itself is not copied into package distributions.
    """

    def __init__(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        self._tables = {}
        self._symbols = {}
        current = None
        date = "unspecified"

        def finish():
            if current is None:
                return
            z, symbol = current["z"], current["symbol"]
            if z in self._tables or symbol in self._symbols:
                raise ValueError("duplicate DABAX element")
            try:
                data = np.asarray(current["rows"], dtype=float)
                labels = current["labels"]
                occupation = np.asarray(current["occupation"], dtype=float)
                binding = np.asarray(current["binding"], dtype=float)
                columns = current["columns"]
            except (KeyError, ValueError) as exc:
                raise ValueError(f"incomplete DABAX block: {symbol}") from exc
            if (data.ndim != 2 or data.shape[0] < 2 or data.shape[1] != columns
                    or len(labels) != columns or labels[:2] != ["pz", "total"]
                    or len(set(labels)) != columns or columns < 3
                    or len(occupation) != columns-2 or len(binding) != columns-2):
                raise ValueError(f"inconsistent DABAX columns: {symbol}")
            if any(not re.fullmatch(r"Shell_[1-9][0-9]*", s) for s in labels[2:]):
                raise ValueError(f"unsupported DABAX shell label: {symbol}")
            if (not np.isfinite(data).all() or np.any(data < 0)
                    or data[0, 0] != 0 or np.any(np.diff(data[:, 0]) <= 0)
                    or not np.isfinite(occupation).all() or np.any(occupation <= 0)
                    or not np.isfinite(binding).all() or np.any(binding < 0)):
                raise ValueError(f"invalid DABAX numeric data: {symbol}")
            if not np.isclose(occupation.sum(), z, rtol=0, atol=1e-6):
                raise ValueError(f"DABAX occupations do not sum to Z: {symbol}")
            for array in (data, occupation, binding):
                array.setflags(write=False)
            self._tables[z] = (symbol, tuple(labels[2:]), occupation, binding, data)
            self._symbols[symbol] = z

        for line in raw.decode("utf-8-sig").splitlines():
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == "#D":
                date = " ".join(parts[1:])
            elif key == "#S":
                finish()
                if len(parts) != 3:
                    raise ValueError("invalid DABAX element header")
                current = {"z": int(parts[1]), "symbol": parts[2], "rows": []}
            elif current is not None:
                if key == "#N":
                    current["columns"] = int(parts[1])
                elif key == "#L":
                    current["labels"] = parts[1:]
                elif key == "#UOCCUP":
                    current["occupation"] = [float(v) for v in parts[1:]]
                elif key == "#UBIND":
                    current["binding"] = [float(v) for v in parts[1:]]
                elif not key.startswith("#"):
                    current["rows"].append([float(v) for v in parts])
        finish()
        if not self._tables:
            raise ValueError("no DABAX element blocks")
        self._provenance = MappingProxyType({
            "provider": "local_dabax", "source_file": str(path),
            "source_sha256": hashlib.sha256(raw).hexdigest(), "table_date": date,
            "reference_doi": BIGGS_REFERENCE_DOI, "occupancy_source": "#UOCCUP",
            "binding_energy_source": "#UBIND (eV)",
            "shell_convention": "native Shell_N column labels; no inferred spectroscopic mapping",
            "partial_normalization": "per electron; occupancy applied once by core builder",
            "interpolation": "linear in absolute pz; no extrapolation",
        })

    @property
    def provenance(self):
        return self._provenance

    def resolve_element(self, element):
        z = self._symbols.get(element) if isinstance(element, str) else element
        if isinstance(z, bool) or z not in self._tables:
            raise ValueError(f"element absent from DABAX table: {element}")
        return ElementIdentity(z, self._tables[z][0])

    def _table(self, element):
        return self._tables[self.resolve_element(element).atomic_number]

    def available_shells(self, element):
        return self._table(element)[1]

    def shell_label(self, shell):
        if not isinstance(shell, str) or not re.fullmatch(r"Shell_[1-9][0-9]*", shell):
            raise ValueError("use the explicit DABAX Shell_N label, not xraylib shell names")
        return shell

    def _index(self, element, shell):
        try:
            return self.available_shells(element).index(self.shell_label(shell))
        except ValueError as exc:
            raise ValueError(f"unknown DABAX shell {element}:{shell}") from exc

    def electron_occupancy(self, element, shell):
        return float(self._table(element)[2][self._index(element, shell)])

    def binding_energy_ev(self, element, shell):
        return float(self._table(element)[3][self._index(element, shell)])

    def momentum_support_au(self, element):
        """Tabulated positive momentum interval; no implicit infinite tails."""
        grid = self._table(element)[4]
        return float(grid[0, 0]), float(grid[-1, 0])

    def _evaluate(self, element, pz, column):
        grid = self._table(element)[4]
        query = np.abs(np.asarray(pz, dtype=float))
        if query.ndim != 1 or not np.isfinite(query).all():
            raise ValueError("pz must be a finite one-dimensional array")
        if np.any(query > grid[-1, 0]):
            raise ValueError("DABAX profile extrapolation is not allowed")
        return np.interp(query, grid[:, 0], grid[:, column])

    def total_profile(self, element, pz_au):
        return self._evaluate(element, pz_au, 1)

    def partial_profile(self, element, shell, pz_au):
        return self._evaluate(element, pz_au, 2+self._index(element, shell))
