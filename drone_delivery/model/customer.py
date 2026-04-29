"""
model/customer.py
=================
Customer dataclass — represents one delivery destination.
"""
from dataclasses import dataclass


@dataclass(slots=True)
class Customer:
    """A customer node requiring a drone delivery.

    Attributes:
        id:     Unique customer identifier (1-based).
        x:      X-coordinate on the grid [metres].
        y:      Y-coordinate on the grid [metres].
        demand: Package weight to deliver [kg].
    """
    id: int
    x: float
    y: float
    demand: float
