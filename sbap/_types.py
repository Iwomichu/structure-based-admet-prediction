from typing import NewType

InteractionType = NewType("InteractionType", str)
DockingScore = NewType("DockingScore", float)
Receptor = NewType("Receptor", str)
ReceptorInteractionCombination = NewType("ReceptorInteractionCombination", tuple[Receptor, InteractionType])
UnsanitizedInteractionFingerprint = NewType(
    "UnsanitizedInteractionFingerprint",
    list[tuple[ReceptorInteractionCombination, bool]],
)
InteractionFingerprint = NewType("InteractionFingerprint", list[int])
