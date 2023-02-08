from enum import Enum, auto

class BaseCaseTypes(Enum):
    SVPInDimensionKOnly = auto()
    SVPOnly = auto()
    DSP = auto()
    HKZ = auto()
    SVPAndLLL = auto()