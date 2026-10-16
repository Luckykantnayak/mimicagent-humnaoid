# Physical summary — g1
(generated from g1.xml; do not edit by hand)

## Kinematics

- Base body `pelvis` nominal standing height: 0.78 m
- Total height (tallest point at rest): 1.21 m

- End-effector reach (how far each EE extends from its limb root; keep commanded targets within this):
    - left_foot: reach ≈ 0.70 m; nominal position (rel. base) ≈ [-0.03, +0.12, -0.75] m
    - right_foot: reach ≈ 0.70 m; nominal position (rel. base) ≈ [-0.03, -0.12, -0.75] m
    - left_hand: reach ≈ 0.50 m; nominal position (rel. base) ≈ [+0.26, +0.19, -0.02] m
    - right_hand: reach ≈ 0.50 m; nominal position (rel. base) ≈ [+0.26, -0.19, -0.02] m

- Joint position limits (radians; degrees in parentheses):
    - left_hip_pitch_joint: [-2.531, +2.880] rad ([-145°, +165°])
    - left_hip_roll_joint: [-0.524, +2.967] rad ([-30°, +170°])
    - left_hip_yaw_joint: [-2.758, +2.758] rad ([-158°, +158°])
    - left_knee_joint: [-0.087, +2.880] rad ([-5°, +165°])
    - left_ankle_pitch_joint: [-0.873, +0.524] rad ([-50°, +30°])
    - left_ankle_roll_joint: [-0.262, +0.262] rad ([-15°, +15°])
    - right_hip_pitch_joint: [-2.531, +2.880] rad ([-145°, +165°])
    - right_hip_roll_joint: [-2.967, +0.524] rad ([-170°, +30°])
    - right_hip_yaw_joint: [-2.758, +2.758] rad ([-158°, +158°])
    - right_knee_joint: [-0.087, +2.880] rad ([-5°, +165°])
    - right_ankle_pitch_joint: [-0.873, +0.524] rad ([-50°, +30°])
    - right_ankle_roll_joint: [-0.262, +0.262] rad ([-15°, +15°])
    - waist_yaw_joint: [-2.618, +2.618] rad ([-150°, +150°])
    - waist_roll_joint: [-0.520, +0.520] rad ([-30°, +30°])
    - waist_pitch_joint: [-0.520, +0.520] rad ([-30°, +30°])
    - left_shoulder_pitch_joint: [-3.089, +2.670] rad ([-177°, +153°])
    - left_shoulder_roll_joint: [-1.588, +2.252] rad ([-91°, +129°])
    - left_shoulder_yaw_joint: [-2.618, +2.618] rad ([-150°, +150°])
    - left_elbow_joint: [-1.047, +2.094] rad ([-60°, +120°])
    - left_wrist_roll_joint: [-1.972, +1.972] rad ([-113°, +113°])
    - left_wrist_pitch_joint: [-1.614, +1.614] rad ([-92°, +92°])
    - left_wrist_yaw_joint: [-1.614, +1.614] rad ([-92°, +92°])
    - right_shoulder_pitch_joint: [-3.089, +2.670] rad ([-177°, +153°])
    - right_shoulder_roll_joint: [-2.252, +1.588] rad ([-129°, +91°])
    - right_shoulder_yaw_joint: [-2.618, +2.618] rad ([-150°, +150°])
    - right_elbow_joint: [-1.047, +2.094] rad ([-60°, +120°])
    - right_wrist_roll_joint: [-1.972, +1.972] rad ([-113°, +113°])
    - right_wrist_pitch_joint: [-1.614, +1.614] rad ([-92°, +92°])
    - right_wrist_yaw_joint: [-1.614, +1.614] rad ([-92°, +92°])

- Limb segment lengths:
    - pelvis: 79.3 cm
    - left_hip_pitch_link: 12.1 cm
    - left_hip_roll_link: 6.0 cm
    - left_hip_yaw_link: 12.7 cm
    - left_knee_link: 19.4 cm
    - left_ankle_pitch_link: 30.0 cm
    - left_ankle_roll_link: 1.8 cm
    - right_hip_pitch_link: 12.1 cm
    - right_hip_roll_link: 6.0 cm
    - right_hip_yaw_link: 12.7 cm
    - right_knee_link: 19.4 cm
    - right_ankle_pitch_link: 30.0 cm
    - right_ankle_roll_link: 1.8 cm
    - waist_roll_link: 3.5 cm
    - torso_link: 1.9 cm
    - left_shoulder_pitch_link: 25.8 cm
    - left_shoulder_roll_link: 4.0 cm
    - left_shoulder_yaw_link: 10.3 cm
    - left_elbow_link: 8.2 cm
    - left_wrist_roll_link: 10.1 cm
    - left_wrist_pitch_link: 3.8 cm
    - left_wrist_yaw_link: 4.6 cm
    - left_hand_contact: 9.0 cm
    - right_shoulder_pitch_link: 25.8 cm
    - right_shoulder_roll_link: 4.0 cm
    - right_shoulder_yaw_link: 10.3 cm
    - right_elbow_link: 8.2 cm
    - right_wrist_roll_link: 10.1 cm
    - right_wrist_pitch_link: 3.8 cm
    - right_wrist_yaw_link: 4.6 cm
    - right_hand_contact: 9.0 cm
    - torso_contact: 13.8 cm
    - head_contact: 37.5 cm
    - pelvis_contact: 7.9 cm

## Dynamics (feasible velocities & accelerations)

Estimates grounded in standard mechanics (the sim itself applies no forces — these bound what is *physically plausible*, so keep commanded velocities within them). Sources cited at the end.

- Total mass: 35.1 kg
- Per actuated joint — torque limit (source: actuatorfrcrange), effective inertia at rest, max acceleration |q̈| ≲ τ/I, and peak velocity |q̇| ≲ √(A·τ/I) for a full-range oscillation:
    - left_hip_pitch_joint: τ ≤ 88 N·m, I ≈ 0.903 kg·m² → |q̈| ≲ 95 rad/s², |q̇| ≲ 16.1 rad/s
    - left_hip_roll_joint: τ ≤ 88 N·m, I ≈ 0.733 kg·m² → |q̈| ≲ 120 rad/s², |q̇| ≲ 14.5 rad/s
    - left_hip_yaw_joint: τ ≤ 88 N·m, I ≈ 0.074 kg·m² → |q̈| ≲ 1191 rad/s², |q̇| ≲ 57.3 rad/s
    - left_knee_joint: τ ≤ 139 N·m, I ≈ 0.113 kg·m² → |q̈| ≲ 1220 rad/s², |q̇| ≲ 42.5 rad/s
    - left_ankle_pitch_joint: τ ≤ 50 N·m, I ≈ 0.003 kg·m² → |q̈| ≲ 17948 rad/s², |q̇| ≲ 111.9 rad/s
    - left_ankle_roll_joint: τ ≤ 50 N·m, I ≈ 0.000 kg·m² → |q̈| ≲ 129157 rad/s², |q̇| ≲ 183.9 rad/s
    - right_hip_pitch_joint: τ ≤ 88 N·m, I ≈ 0.903 kg·m² → |q̈| ≲ 95 rad/s², |q̇| ≲ 16.1 rad/s
    - right_hip_roll_joint: τ ≤ 88 N·m, I ≈ 0.733 kg·m² → |q̈| ≲ 120 rad/s², |q̇| ≲ 14.5 rad/s
    - right_hip_yaw_joint: τ ≤ 88 N·m, I ≈ 0.074 kg·m² → |q̈| ≲ 1192 rad/s², |q̇| ≲ 57.3 rad/s
    - right_knee_joint: τ ≤ 139 N·m, I ≈ 0.113 kg·m² → |q̈| ≲ 1220 rad/s², |q̇| ≲ 42.5 rad/s
    - right_ankle_pitch_joint: τ ≤ 50 N·m, I ≈ 0.003 kg·m² → |q̈| ≲ 17948 rad/s², |q̇| ≲ 111.9 rad/s
    - right_ankle_roll_joint: τ ≤ 50 N·m, I ≈ 0.000 kg·m² → |q̈| ≲ 129157 rad/s², |q̇| ≲ 183.9 rad/s
    - waist_yaw_joint: τ ≤ 88 N·m, I ≈ 0.290 kg·m² → |q̈| ≲ 304 rad/s², |q̇| ≲ 28.2 rad/s
    - waist_roll_joint: τ ≤ 50 N·m, I ≈ 0.853 kg·m² → |q̈| ≲ 59 rad/s², |q̇| ≲ 5.5 rad/s
    - waist_pitch_joint: τ ≤ 50 N·m, I ≈ 0.622 kg·m² → |q̈| ≲ 73 rad/s², |q̇| ≲ 6.2 rad/s
    - left_shoulder_pitch_joint: τ ≤ 25 N·m, I ≈ 0.164 kg·m² → |q̈| ≲ 141 rad/s², |q̇| ≲ 20.1 rad/s
    - left_shoulder_roll_joint: τ ≤ 25 N·m, I ≈ 0.119 kg·m² → |q̈| ≲ 203 rad/s², |q̇| ≲ 19.8 rad/s
    - left_shoulder_yaw_joint: τ ≤ 25 N·m, I ≈ 0.033 kg·m² → |q̈| ≲ 761 rad/s², |q̇| ≲ 44.6 rad/s
    - left_elbow_joint: τ ≤ 25 N·m, I ≈ 0.034 kg·m² → |q̈| ≲ 679 rad/s², |q̇| ≲ 32.6 rad/s
    - left_wrist_roll_joint: τ ≤ 25 N·m, I ≈ 0.000 kg·m² → |q̈| ≲ 68228 rad/s², |q̇| ≲ 366.8 rad/s
    - left_wrist_pitch_joint: τ ≤ 5 N·m, I ≈ 0.005 kg·m² → |q̈| ≲ 966 rad/s², |q̇| ≲ 39.5 rad/s
    - left_wrist_yaw_joint: τ ≤ 5 N·m, I ≈ 0.002 kg·m² → |q̈| ≲ 2710 rad/s², |q̇| ≲ 66.1 rad/s
    - right_shoulder_pitch_joint: τ ≤ 25 N·m, I ≈ 0.164 kg·m² → |q̈| ≲ 141 rad/s², |q̇| ≲ 20.1 rad/s
    - right_shoulder_roll_joint: τ ≤ 25 N·m, I ≈ 0.119 kg·m² → |q̈| ≲ 203 rad/s², |q̇| ≲ 19.8 rad/s
    - right_shoulder_yaw_joint: τ ≤ 25 N·m, I ≈ 0.033 kg·m² → |q̈| ≲ 761 rad/s², |q̇| ≲ 44.6 rad/s
    - right_elbow_joint: τ ≤ 25 N·m, I ≈ 0.034 kg·m² → |q̈| ≲ 679 rad/s², |q̇| ≲ 32.6 rad/s
    - right_wrist_roll_joint: τ ≤ 25 N·m, I ≈ 0.000 kg·m² → |q̈| ≲ 68228 rad/s², |q̇| ≲ 366.8 rad/s
    - right_wrist_pitch_joint: τ ≤ 5 N·m, I ≈ 0.005 kg·m² → |q̈| ≲ 966 rad/s², |q̇| ≲ 39.5 rad/s
    - right_wrist_yaw_joint: τ ≤ 5 N·m, I ≈ 0.002 kg·m² → |q̈| ≲ 2710 rad/s², |q̇| ≲ 66.1 rad/s

- Torso forward speed (Froude number Fr = v²/gL, L = leg length 0.78 m):
    - walk→run transition (Fr≈0.5): ≈ 1.96 m/s
    - walking ceiling (Fr=1, foot leaves ground above this): ≈ 2.77 m/s
    - dynamic running gait (Fr≈4): up to ≈ 5.55 m/s

- Torso vertical jump & flight (leg push F≈τ/ℓ, then projectile h=v_z²/2g, t_air=2v_z/g — approximate):
    - net upward accel ≈ 17 m/s², take-off speed ≈ 2.79 m/s
    - apex height ≈ 0.40 m, flight time ≈ 0.57 s
    - one full aerial rotation needs ω ≈ 2π/t_air ≈ 11.0 rad/s (632°/s)

- Torso angular rates about roll(x)/pitch(y)/yaw(z) — whole-body rotational inertia (exact), grounded α=τ/I with a planted foot, and the aerial regime where I·ω is conserved:
    - roll: I ≈ 4.05 kg·m²; grounded |α| ≲ 27 rad/s², |ω| ≲ 6.5 rad/s over a 90° swing
    - pitch: I ≈ 3.75 kg·m²; grounded |α| ≲ 6 rad/s², |ω| ≲ 3.2 rad/s over a 90° swing
    - yaw: I ≈ 0.52 kg·m²; grounded |α| ≲ 215 rad/s², |ω| ≲ 18.4 rad/s over a 90° swing
    - aerial: a flip/twist needs ω ≥ 2π/t_air (see flight time above); the legs impart angular momentum L = I·ω at take-off. A ~9 kg quadruped (MIT Mini Cheetah) lands 360° backflips, so this regime is real.

- Sources: Froude number & v=√(gL) — R. McN. Alexander / en.wikipedia.org/wiki/Froude_number; aerial rotation & backflip feasibility — MIT Mini Cheetah, Katz et al., ICRA 2019; Unitree Go2 top speed 2.5–5 m/s — unitree.com/go2. Torque source is the MJCF `actuatorfrcrange`; jump/rate force terms are statics estimates.
