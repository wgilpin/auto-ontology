from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Cluster:
    """
    Describes a single cluster of entities.
    """
    freqs: dict
    freqs_unknown: dict
    class_freqs: np.ndarray
    frac: float
    n: int
    label: str
    entity_id: int
    clus_no: int
    name: str
    classes: Optional[list[dict]] = None

ClusterList = dict[str, Cluster]
