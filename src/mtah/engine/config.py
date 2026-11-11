"""Engine configuration: explicit, deterministic, hand-set physical priors.

None of these values were fitted, tuned or selected against any match data or
ground truth (public or private). They are football-physics starting points
chosen from first principles and documented in docs/POSSESSION_ENGINE.md.
Any future calibrated value must come from properly licensed external data
about other games, and its provenance must be recorded next to the field.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

ENGINE_VERSION = "possession-rules-v0.2"


@dataclass(frozen=True)
class EngineConfig:
    # --- spatial control evidence -------------------------------------------
    # A player whose tracked centre is within this distance of the ball centre
    # can be the controlling player (stride length plus tracking jitter).
    control_radius_m: float = 1.5
    # Nearest home and nearest away both inside the radius and closer to each
    # other's distance than this margin -> contested, not control.
    contest_margin_m: float = 0.5
    # Faster than this the ball is treated as travelling (pass, shot,
    # clearance), not under a player's control. Sprint dribbling stays below it.
    max_control_ball_speed_mps: float = 12.0
    # Lag (frames) for deriving ball speed from positions when the source
    # speed is missing. A multi-frame lag damps per-frame position jitter.
    speed_lag_frames: int = 5

    # --- temporal possession reasoning ---------------------------------------
    # Team-level control evidence frames needed before a run counts as
    # controlled possession (0.4 s at 25 fps). Shorter runs are touches.
    confirm_frames: int = 10
    # A team-level run survives up to this many frames without supporting
    # evidence (ball missing, in flight between teammates, loose).
    run_gap_frames: int = 50

    # --- dead-ball inference -------------------------------------------------
    # Ball centre beyond a touch/goal line by more than this is out of play.
    out_margin_m: float = 1.0
    # After a stoppage, play restarts at the first frame the ball is observed
    # inside the pitch moving at least this fast.
    restart_speed_mps: float = 1.0
    # Dead runs shorter than this are dropped as tracking noise.
    min_dead_frames: int = 25
    # Ball observed slower than this, with nobody within control radius, for
    # at least stationary_frames -> placed ball awaiting a restart.
    stationary_speed_mps: float = 0.3
    stationary_frames: int = 75
    # Smart Tagging codes treated as restart evidence (normalized upper case).
    set_piece_codes: tuple = ("SET PIECES",)
    # Search window (frames, either side of a set-piece start) for the restart.
    set_piece_window_frames: int = 50
    # Maximum frames a set piece may extend a dead stretch backwards through
    # unobserved or stationary ball frames.
    set_piece_max_lead_frames: int = 250
    # Use a set piece's team tag as the restart possessor only when tracking
    # gives no confirmed control in that live stretch.
    use_set_piece_team_prior: bool = True

    def __post_init__(self):
        floats = ("control_radius_m", "contest_margin_m", "max_control_ball_speed_mps", "out_margin_m",
                  "restart_speed_mps", "stationary_speed_mps")
        for name in floats:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name}: expected finite non-negative number")
        ints = ("speed_lag_frames", "confirm_frames", "run_gap_frames", "min_dead_frames", "stationary_frames",
                "set_piece_window_frames", "set_piece_max_lead_frames")
        for name in ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}: expected positive integer")
        if not isinstance(self.use_set_piece_team_prior, bool):
            raise ValueError("use_set_piece_team_prior: expected bool")
        if isinstance(self.set_piece_codes, str):
            # tuple("SET PIECES") would silently become single characters.
            raise ValueError("set_piece_codes must be a sequence of codes, not one string")
        codes = tuple(self.set_piece_codes)
        if not all(isinstance(c, str) and c == c.strip().upper() and c for c in codes):
            raise ValueError("set_piece_codes must be normalized upper-case strings")
        object.__setattr__(self, "set_piece_codes", codes)

    def digest(self) -> str:
        payload = {"engine": ENGINE_VERSION, **asdict(self)}
        payload["set_piece_codes"] = list(payload["set_piece_codes"])
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
