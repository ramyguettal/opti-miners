"""
model/drone.py
==============
Drone dataclass — represents a single delivery drone.
"""
from dataclasses import dataclass


@dataclass(slots=True)
class Drone:
    """A drone in the delivery fleet.

    Attributes:
        id:               Unique drone identifier (0-based).
        max_payload_kg:   Maximum weight the drone can carry [kg].
        battery_wh:       Maximum battery capacity [Wh].
        speed_ms:         Cruise speed [m/s].
    """
    id: int
    max_payload_kg: float
    battery_wh: float
    speed_ms: float
