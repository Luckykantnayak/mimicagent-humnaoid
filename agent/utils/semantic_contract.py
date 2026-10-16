"""Validated task criteria, independent of the mutable execution plan."""
import hashlib
import json
import math

PARAMS = {
    'net_progress': {'axis', 'min', 'max'},
    'peak_rise': {'axis', 'min', 'max', 'phase_window'},
    'swing_lift': {'ee', 'min', 'max'},
    'phase_offset': {'ee_a', 'ee_b', 'expected', 'tol', 'period_s'},
    'contact_alternation': {'ee_a', 'ee_b', 'min_pairs'},
    'base_height_band': {'min', 'max'},
    'object_coincident': {'object', 'ee', 'max_dist'},
    'end_state': {'mode', 'base_tol', 'joint_tol', 'orientation_tol', 'min_advance'},
    'stance_no_drift': {'ee', 'max_drift'},
    'contact_over_surface': {'ee', 'x_range', 'y_range', 'surface_z', 'tol', 'min_dwell_s', 'phase_window'},
    'signal_window': {'signal', 'window', 'stat', 'min', 'max', 'min_duration_s'},
    'event_order': {'events', 'max_gap_s'},
}

# Signals measurable per frame (see test_motion_quality._signal). <body> is any
# registered contact name (end-effectors AND contact bodies such as knees/pelvis).
SIGNALS = ('root_x', 'root_y', 'root_z', 'upright', 'rot_roll', 'rot_pitch', 'rot_yaw', 'airborne')
BODY_SIGNALS = ('height', 'x', 'y', 'clearance', 'contact', 'touching')
WINDOWS = ('all', 'start', 'end', 'airborne', 'grounded')
STATS = ('min', 'max', 'mean', 'first', 'last')


def validate_signal(name, spec=None):
    """Raise ValueError unless `name` is a measurable signal for this robot."""
    if not isinstance(name, str) or not name:
        raise ValueError('signal must be a non-empty string')
    if name in SIGNALS:
        return
    kind, _, arg = name.partition(':')
    roster = set()
    if spec is not None:
        roster = set(getattr(spec, 'ee_names', []) or []) | {b['name'] for b in getattr(spec, 'contact_bodies', [])}
    def body(n):
        if roster and n not in roster:
            raise ValueError(f'signal {name!r}: unknown body {n!r}; bodies are {sorted(roster)}')
    if kind in BODY_SIGNALS and arg:
        body(arg); return
    if kind == 'above' and arg.count(':') == 1:
        a, b = arg.split(':'); body(a); body(b); return
    if kind == 'contacts' and arg:
        for n in arg.split(','):
            body(n.strip())
        return
    if kind == 'joint' and arg:
        joints = set(getattr(spec, 'joint_names', []) or []) if spec is not None else set()
        if joints and arg not in joints:
            raise ValueError(f'signal {name!r}: unknown joint {arg!r}')
        return
    if kind in ('object_x', 'object_y', 'object_z') and arg:
        return
    raise ValueError(f'unknown signal {name!r}; use one of {list(SIGNALS)} or <kind>:<name> with kind in '
                     f'{list(BODY_SIGNALS)} / above:<a>:<b> / contacts:<a>,<b> / joint:<joint> / object_z:<object>')


def _validate_window(name, window, spec):
    if isinstance(window, str):
        if window not in WINDOWS:
            raise ValueError(f'{name}: window must be one of {list(WINDOWS)}, [a, b] or a condition')
    elif isinstance(window, (list, tuple)):
        if (len(window) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in window)
                or not 0 <= window[0] < window[1] <= 1):
            raise ValueError(f'{name}: window [a, b] must satisfy 0 <= a < b <= 1')
    elif isinstance(window, dict):
        validate_signal(window.get('signal'), spec)
        if not {'min', 'max'} & window.keys():
            raise ValueError(f'{name}: a condition window needs min and/or max')
    else:
        raise ValueError(f'{name}: invalid window {window!r}')


def _validate_events(name, events, spec):
    if not isinstance(events, list) or not 2 <= len(events) <= 12:
        raise ValueError(f'{name}: events must be a list of 2-12 conditions')
    for ev in events:
        if not isinstance(ev, dict):
            raise ValueError(f'{name}: each event must be an object')
        validate_signal(ev.get('signal'), spec)
        if not {'min', 'max'} & ev.keys():
            raise ValueError(f'{name}: each event needs min and/or max')
        for key, val in ev.items():
            if key == 'signal':
                continue
            if key not in {'min', 'max', 'min_duration_s'} or isinstance(val, bool) \
                    or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f'{name}: event field {key!r} must be one of min/max/min_duration_s (numbers)')
        if ev.get('min', -math.inf) > ev.get('max', math.inf):
            raise ValueError(f'{name}: event bounds reversed')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_invariants(invariants, spec=None):
    if not isinstance(invariants, list):
        raise ValueError('invariants must be a list')
    names = set()
    roster = set(getattr(spec, 'contact_names', []) or []) | set(getattr(spec, 'ee_names', []) or [])
    roster |= {b['name'] for b in getattr(spec, 'contact_bodies', [])}
    for inv in invariants:
        if not isinstance(inv, dict):
            raise ValueError('each invariant must be an object')
        kind, name, p = inv.get('kind'), inv.get('name'), inv.get('params')
        if kind not in PARAMS or not isinstance(p, dict):
            raise ValueError(f'unknown kind or malformed params: {kind}')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('invariant names must be nonempty and unique')
        names.add(name)
        unknown = set(p) - PARAMS[kind]
        if unknown:
            raise ValueError(f'{name}: unsupported parameters {sorted(unknown)}')
        for key, val in p.items():
            if key in {'ee', 'ee_a', 'ee_b'}:
                if not isinstance(val, str) or (roster and val not in roster):
                    raise ValueError(f'{name}: unknown contact {val}')
            elif key in {'axis', 'mode', 'object'}:
                if not isinstance(val, str):
                    raise ValueError(f'{name}: {key} must be a string')
            elif key == 'signal':
                validate_signal(val, spec)
            elif key == 'stat':
                if val not in STATS:
                    raise ValueError(f'{name}: stat must be one of {list(STATS)}')
            elif key == 'window':
                _validate_window(name, val, spec)
            elif key == 'events':
                _validate_events(name, val, spec)
            elif key in {'phase_window', 'x_range', 'y_range'}:
                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    raise ValueError(f'{name}: {key} must contain two bounds')
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in val) or val[0] >= val[1]:
                    raise ValueError(f'{name}: invalid {key}')
                if key == 'phase_window' and not 0 <= val[0] < val[1] <= 1:
                    raise ValueError(f'{name}: phase_window must lie in [0,1]')
            elif not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                raise ValueError(f'{name}: {key} must be a finite number')
            elif key in {'tol', 'max_dist', 'max_drift', 'base_tol', 'joint_tol', 'orientation_tol', 'min_dwell_s',
                         'period_s', 'min_duration_s', 'max_gap_s'} and val <= 0:
                raise ValueError(f'{name}: {key} must be positive')
        required = {'swing_lift': {'ee', 'min'}, 'phase_offset': {'ee_a', 'ee_b', 'expected'},
                    'base_height_band': {'min', 'max'}, 'object_coincident': {'object', 'ee', 'max_dist'},
                    'contact_alternation': {'ee_a', 'ee_b'},
                    'stance_no_drift': {'ee', 'max_drift'}, 'contact_over_surface': {'ee'},
                    'end_state': {'mode'}, 'signal_window': {'signal'},
                    'event_order': {'events'}}.get(kind, set())
        if not required <= p.keys():
            raise ValueError(f'{name}: missing {sorted(required - p.keys())}')
        if kind in {'net_progress', 'peak_rise', 'signal_window'} and not {'min', 'max'} & p.keys():
            raise ValueError(f'{name}: requires an explicit bound')
        if p.get('min', -math.inf) > p.get('max', math.inf):
            raise ValueError(f'{name}: reversed bounds')
        axes = {'x', 'y', 'z', 'horizontal', 'roll', 'pitch', 'yaw'} if kind == 'net_progress' else {'x', 'y', 'z'}
        if 'axis' in p and p['axis'] not in axes:
            raise ValueError(f'{name}: unsupported axis')
        if kind in {'phase_offset', 'contact_alternation'} and p['ee_a'] == p['ee_b']:
            raise ValueError(f'{name}: alternation needs two different contacts')
        if kind == 'contact_alternation' and (not isinstance(p.get('min_pairs', 1), int) or p.get('min_pairs', 1) < 1):
            raise ValueError(f'{name}: min_pairs must be a positive integer')
        if kind == 'end_state' and p['mode'] not in {'returns', 'advances'}:
            raise ValueError(f'{name}: unsupported end_state mode')
        if kind == 'contact_over_surface' and not {'x_range', 'y_range'} & p.keys():
            raise ValueError(f'{name}: surface footprint is required')
    return invariants


def validate_contract(contract, spec=None):
    if not isinstance(contract, dict):
        raise ValueError('goal contract must be an object')
    checks = validate_invariants(contract.get('invariants'), spec)
    requirements = contract.get('requirements')
    if not isinstance(requirements, list) or not requirements:
        raise ValueError('requirements must explicitly cover the original request')
    ids = [r.get('id') for r in requirements if isinstance(r, dict)]
    if len(ids) != len(requirements) or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('requirement IDs must be unique strings')
    coverage = set()
    for inv in checks:
        refs = inv.get('requirement_ids')
        if not isinstance(refs, list) or not refs or not set(refs) <= set(ids):
            raise ValueError('each goal check must reference known requirement_ids')
        coverage.update(refs)
    for req in requirements:
        if not isinstance(req.get('text'), str) or not req['text']:
            raise ValueError('each requirement needs its original meaning as text')
        if req['id'] not in coverage and not req.get('unsupported_reason'):
            raise ValueError(f"uncovered requirement: {req['id']}")
    sliders = contract.get('sliding_contacts', [])
    if not isinstance(sliders, list) or not all(isinstance(x, str) for x in sliders):
        raise ValueError('sliding_contacts must be a list of contact names')
    if spec is not None:
        roster = set(spec.ee_names) | set(spec.contact_body_names)
        if not set(sliders) <= roster:
            raise ValueError('sliding_contacts must name real contacts')
        environment = spec.get_environment(contract.get('environment'))
        objects = {o['name'] for o in environment.get('objects', [])}
        for inv in checks:
            if inv['kind'] == 'object_coincident' and inv['params']['object'] not in objects:
                raise ValueError('object_coincident must name an object present in the goal scene')
    return contract


def goal_prompt(state):
    contract = state.get('goal_contract')
    if not contract:
        return ''
    return ('\n\nAUTHORITATIVE USER REQUEST AND FIXED ACCEPTANCE CRITERIA:\n'
            + json.dumps(contract, indent=2)
            + '\nSatisfy every requirement. You may change the execution plan, but cannot '
              'delete, relax, replace, or reinterpret these goal checks. Plan checks are supplementary.\n')


def goal_invariants(state):
    contract = state.get('goal_contract')
    if not contract:
        return []
    validate_contract(contract)
    if digest(contract) != state.get('goal_contract_id'):
        raise ValueError('immutable goal contract changed during the run')
    checks = [dict(inv, name='goal/' + inv['name']) for inv in contract['invariants']]
    for req in contract['requirements']:
        if req.get('unsupported_reason'):
            checks.append({'name': 'goal/unsupported/' + req['id'], 'kind': 'unsupported',
                           'params': {}, 'why': req['unsupported_reason']})
    return checks
