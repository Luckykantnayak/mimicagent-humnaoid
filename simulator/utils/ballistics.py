"""Ballistic flight windows: the unpowered base path between take-off and landing.

Pure math shared by the controller's flight mechanism (and its tests). In a declared
flight window the base follows z = z0 + v0*t - g*t^2/2 with constant horizontal
velocity, so the window's duration fixes the launch velocity. A window too long for a
plausible launch cannot be made ballistic; it is reported, never stretched into a hover.
"""
import numpy as np

GRAVITY = 9.81


def plan_window(p0, landing, dt, launch_cap, lookahead_frames):
    """Solve one flight window.

    p0       : (3,) achieved root at the last supported frame (take-off).
    landing  : (frames_after_takeoff, landing_root(3,)) of the first supported frame
               found by look-ahead, or None if none was found within lookahead_frames.
    Returns {"ok", "k", ...}: ok windows carry the parabola (p0, vz, vxy, T); rejected
    windows carry the launch-speed excess ("violation", m/s) and "tmax", the longest
    flight gravity allows between these heights at the launch cap.
    """
    p0 = np.asarray(p0, float)
    if landing is None:
        T, z1, xy1 = (lookahead_frames + 1) * dt, p0[2], p0[:2]
    else:
        T = landing[0] * dt
        z1, xy1 = float(landing[1][2]), np.asarray(landing[1][:2], float)
    vz = (z1 - p0[2] + 0.5 * GRAVITY * T * T) / T
    if landing is None or vz > launch_cap:
        drop = p0[2] - z1
        tmax = (launch_cap + np.sqrt(max(0.0, launch_cap ** 2 + 2.0 * GRAVITY * drop))) / GRAVITY
        return {"ok": False, "k": 0, "violation": float(vz - launch_cap), "tmax": float(tmax),
                "T": float(T), "z0": float(p0[2]), "z1": float(z1), "vz": float(vz)}
    return {"ok": True, "k": 0, "p0": p0.copy(), "vz": float(vz),
            "vxy": (xy1 - p0[:2]) / T, "T": float(T)}


def position(window, k, dt):
    """Root position k frames after take-off on an ok window."""
    tau = min(k * dt, window["T"])
    p0 = window["p0"]
    return np.array([p0[0] + window["vxy"][0] * tau, p0[1] + window["vxy"][1] * tau,
                     p0[2] + window["vz"] * tau - 0.5 * GRAVITY * tau * tau])
