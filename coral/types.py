import enum
from typing import Any, Literal

CigarString = str
Strand = Literal["+", "-"]

AmpliconInterval = list[Any]  # tuple[str, int, int, int]
CnsInterval = Any  # tuple[str, int, int]
Edge = tuple[int, int]
PathConstraint = list[list[Any]]
