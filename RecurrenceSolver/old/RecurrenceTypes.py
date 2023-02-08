from enum import Enum, auto

class RecurrenceTypes(Enum):
    COARSE_GRAINED = auto()
    EXACT = auto()
    TAU = auto()