"""Robust aggregators and defense modules."""

from fl_ids.defense.base import BaseAggregator
from fl_ids.defense.fedavg import FedAvg
from fl_ids.defense.coordinate_filters import TrimmedMean, CoordinateMedian
from fl_ids.defense.distance_filters import MultiKrum, FoolsGold
from fl_ids.defense.trust_filters import FLTrust, RepGuard
from fl_ids.defense.compression import SketchCompressor
from fl_ids.defense.novel_defense import ClassAwareTrustDefense

__all__ = [
    "BaseAggregator",
    "FedAvg",
    "TrimmedMean",
    "CoordinateMedian",
    "MultiKrum",
    "FoolsGold",
    "FLTrust",
    "RepGuard",
    "SketchCompressor",
    "ClassAwareTrustDefense",
]
