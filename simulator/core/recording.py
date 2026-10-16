"""Shared capture timing for evaluation trajectories and their video previews."""
WARMUP_FRAMES = 50


def recording_window(sim_freq, skill_freq, cycles):
    """Capture an endpoint-exclusive clip after IK settles at phase zero."""
    start = WARMUP_FRAMES + int(round(0.3 * sim_freq))
    return start, start + int(round(cycles / skill_freq * sim_freq))


def prepare_recording_step(runner, start_frame):
    if runner.frame == WARMUP_FRAMES:
        runner.motion_gen.t = 0.0
        runner.reset_bridge()
    # BaseMotionGenerator reads t before incrementing. Include start_frame so
    # the first captured state is phase 0 and the last never wraps to phase 0.
    if WARMUP_FRAMES <= runner.frame <= start_frame:
        runner.motion_gen.t = 0.0
