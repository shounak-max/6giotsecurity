"""Threat models and poisoning attack implementations."""

from fl_ids.attacks.base import BaseAttack
from fl_ids.attacks.byzantine import SignFlipAttack, LabelFlipAttack, RandomNoiseAttack
from fl_ids.attacks.backdoor import AdditiveBackdoorAttack
from fl_ids.attacks.adaptive import AdaptiveAttacker
from fl_ids.attacks.mimicry import AtypicalMimicryAttack

__all__ = [
    "BaseAttack",
    "SignFlipAttack",
    "LabelFlipAttack",
    "RandomNoiseAttack",
    "AdditiveBackdoorAttack",
    "AdaptiveAttacker",
    "AtypicalMimicryAttack",
]
