"""Stationary interface example; no benchmark skill implementations."""
from motion.base import BaseMotionGenerator


class StandingMotionGenerator(BaseMotionGenerator):
    def __init__(self, init_ee_pos):
        super().__init__(init_ee_pos, freq=0.25)

    def contacts(self, phase):
        return ["left_foot", "right_foot"]

    def compute_ee_position_body_frame(self, ee_name, phase):
        if ee_name in ("left_foot", "right_foot"):
            return self.init_ee_pos[ee_name].copy()
        return None
