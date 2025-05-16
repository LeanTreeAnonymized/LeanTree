from dataclasses import dataclass
from typing import Self, Callable, Any

import numpy as np

from search.base import Proof


@dataclass
class ProofSearchResult:
    theorem: str
    proof: Proof | None
    error: Any | None
    runtime: float | None  # In seconds.
    timed_out: bool
    metadata: dict | None

    @classmethod
    def from_proof_found(cls, theorem: str, proof: Proof, runtime: float, metadata: dict | None = None) -> Self:
        return ProofSearchResult(theorem, proof, None, runtime, False, metadata)

    @classmethod
    def from_proof_not_found(cls, theorem: str, runtime: float, metadata: dict | None = None) -> Self:
        return ProofSearchResult(theorem, None, None, runtime, False, metadata)

    @classmethod
    def from_error(cls, theorem: str, error: any, metadata: dict | None = None) -> Self:
        return ProofSearchResult(theorem, None, error, None, False, metadata)

    @classmethod
    def from_timed_out(cls, theorem: str, runtime: int, metadata: dict | None = None) -> Self:
        return ProofSearchResult(theorem, None, None, runtime, True, metadata)

    def serialize(self) -> dict:
        return {
            "theorem": self.theorem,
            "proof": self.proof.serialize() if self.proof else None,
            "error": str(self.error),
            "runtime": self.runtime,
            "timed_out": self.timed_out,
            "metadata": self.metadata,
        }

    @classmethod
    def deserialize(cls, data: dict) -> Self:
        return ProofSearchResult(
            data["theorem"],
            Proof.deserialize(data["proof"]) if data.get("proof") is not None else None,
            data["error"],
            data["runtime"],
            data["timed_out"],
            data["metadata"],
        )


@dataclass
class EvalStats:
    search_results: list[ProofSearchResult]

    @classmethod
    def empty(cls) -> Self:
        return EvalStats([])

    def serialize(self) -> dict:
        def get_portion(selector: Callable[[ProofSearchResult], bool]) -> float:
            return len([res for res in self.search_results if selector(res)]) / len(self.search_results)

        result = {
            "proven": get_portion(lambda res: res.proof is not None),
            "no_proof": get_portion(lambda res: res.proof is None and res.error is None and not res.timed_out),
            "failed": get_portion(lambda res: res.error is not None),
            "timed_out": get_portion(lambda res: res.timed_out)
        }
        assert np.isclose(sum(result.values()), 1)
        return result

    def pretty_print(self) -> str:
        return "\n".join(
            f"{name}: {value:%} ({int(value * len(self.search_results))} / {len(self.search_results)})"
            for name, value in self.serialize().items()
        )
