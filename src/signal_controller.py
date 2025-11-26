from typing import Dict, Tuple
import time

class AdaptiveSignalController:
    """
    Two-phase controller: lanes keys 'A' and 'B' (e.g., NS and EW).
    Green time per phase adapts to measured density with clipping.
    """
    def __init__(self,
                 min_green: int = 10,
                 max_green: int = 60,
                 base_green: int = 15,
                 k_per_vehicle: float = 2.0):
        self.min_green = min_green
        self.max_green = max_green
        self.base_green = base_green
        self.k = k_per_vehicle

        self.active_lane = "A"
        self.phase_end_ts = 0.0
        self.current_green = base_green

    def _compute_green(self, density: int) -> int:
        green = int(self.base_green + self.k * density)
        green = max(self.min_green, min(self.max_green, green))
        return green

    def start_phase(self, densities: Dict[str, int], now: float = None) -> Tuple[str, int]:
        now = now or time.time()
        next_lane = max(densities.keys(), key=lambda k: densities[k]) if densities else self.active_lane
        self.active_lane = next_lane
        self.current_green = self._compute_green(densities.get(next_lane, 0))
        self.phase_end_ts = now + self.current_green
        return self.active_lane, self.current_green

    def maybe_advance(self, densities: Dict[str, int], now: float = None) -> Tuple[str, int, int]:
        """
        Returns (active_lane, remaining_sec, current_green).
        Advances phase if time elapsed, picking next lane (other of A/B) but prefers the one with higher density.
        """
        now = now or time.time()
        if now >= self.phase_end_ts:
            lanes = list(densities.keys()) if densities else ["A", "B"]
            if "A" not in lanes:
                lanes.append("A")
            if "B" not in lanes:
                lanes.append("B")
            other = "B" if self.active_lane == "A" else "A"
            next_lane = max([self.active_lane, other], key=lambda k: densities.get(k, 0))
            self.active_lane = next_lane
            self.current_green = self._compute_green(densities.get(next_lane, 0))
            self.phase_end_ts = now + self.current_green
        remaining = max(0, int(self.phase_end_ts - now))
        return self.active_lane, remaining, self.current_green
