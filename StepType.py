from enum import Enum, auto

class StepType(Enum):
    RECURSIVE = auto()
    DUALITY = auto()
    LLL = auto()
    HKZ = auto()
    SVP = auto()
    DSP = auto()
    TRIVIAL = auto()
    STUCK = auto()
    SORTALLL = auto()

    def is_base(self):
        return self in [StepType.LLL,
                        StepType.SORTALLL,
                        StepType.HKZ,
                        StepType.SVP,
                        StepType.DSP,
                        StepType.STUCK,
                        StepType.TRIVIAL]

    def is_duality(self):
        return self == StepType.DUALITY

    def short_name(self):
        mapping = {
            StepType.RECURSIVE: "R",
            StepType.DUALITY: "DUAL",
            StepType.LLL: "L",
            StepType.SVP: "S",
            StepType.DSP: "DSP",
            StepType.TRIVIAL: "T",
            StepType.STUCK: "NSOLN",
            StepType.SORTALLL: "SL",
        }
        return mapping[self]

