"""Slot registry. One renderer module can back many per-person slots;
per-slot title/ttl override the module defaults.

Sanitized for publication: the real registry also carries weight, workout
and kids-allowance slots; person keys are the household members' names.
"""
from . import calories_today

REGISTRY = {
    "calories_alice": {"module": calories_today, "title": "Калории · Alice"},
    "calories_bob": {"module": calories_today, "title": "Калории · Bob"},
}
