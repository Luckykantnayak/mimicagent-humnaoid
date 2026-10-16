"""Compile and independently review the user's goal once, before planning."""
import json
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from utils.semantic_contract import digest, validate_contract, goal_invariants
from utils.robot_spec import load_robot_spec
from utils.tools import extract_json_from_message
from agents.base_agent import merge_usage

REVIEW_INSTRUCTIONS = (
    '\nYou are now reviewing a proposed contract, not generating one. Compare EVERY explicit '
    'requirement in the original request to the criteria. Reject omitted requirements, invented '
    'restrictions, wrong units/axes/signs, and proxies that allow a different behavior to pass. '
    'Requirements that no primitive can measure must stay explicitly unsupported (they block a '
    'completed outcome); do not reject a contract because a partial proxy could be measured. '
    'Reasonable numeric tolerances for qualitative goals are permitted; wording preferences, '
    'redundant checks, and alternative reasonable tolerances are not rejection grounds. Return '
    '{"accepted": true|false, "issues": [strings]}. Set accepted=true when the contract is usable, '
    'even if you list minor notes in issues.')


def make_goal_contract_node(agent, data, cfg):
    spec = load_robot_spec(cfg)

    def node(state):
        raw = state['skill_desc'].get('user_request')
        if not raw:
            raise ValueError('Original user_request is required; formatted skill text is not authoritative.')
        if state.get('goal_contract'):
            goal_invariants(state)
            if state['goal_contract'].get('user_request') != raw:
                raise ValueError('Cannot resume a goal contract for a different user request')
            return {'outcome': 'in_progress'}
        folder = os.path.join(cfg['paths']['skill_output_dir'], cfg['robot']['name'], state['skill_name'])
        os.makedirs(folder, exist_ok=True)
        usage = dict(state.get('token_usage') or {})
        error = ''
        previous_contract = None
        fallback = None          # last schema-valid contract (used if review never accepts)
        attempts = cfg.get('goal_contract', {}).get('max_attempts', 4)

        def add_usage(response):
            usage.update(merge_usage(usage, response))

        def publish(contract, verdict, accepted):
            contract = dict(contract, user_request=raw)
            contract_id = digest(contract)
            with open(os.path.join(folder, state['skill_name'] + '_goal.json'), 'w') as f:
                json.dump({'contract_id': contract_id, 'contract': contract, 'review': verdict,
                           'review_accepted': accepted}, f, indent=2)
            with open(os.path.join(folder, state['skill_name'] + '_report.json'), 'w') as f:
                json.dump({'outcome': 'in_progress', 'goal_contract_id': contract_id}, f, indent=2)
            return {'goal_contract': contract, 'goal_contract_id': contract_id,
                    'goal_review_accepted': accepted, 'token_usage': usage,
                    'outcome': 'in_progress', 'messages': [AIMessage(content=json.dumps(contract))]}

        for attempt in range(attempts):
            try:
                response = agent.generate([
                    SystemMessage(content=data['system']),
                    HumanMessage(content=data['human'].replace('{USER_SKILL}', raw)
                                 + ('\nPrevious contract to repair:\n' + json.dumps(previous_contract)
                                    if previous_contract is not None else '')
                                 + error),
                ])
                add_usage(response)
                with open(os.path.join(folder, f'goal_attempt_{attempt + 1:04d}.json'), 'w') as f:
                    json.dump({'content': response.content, 'response_metadata': response.response_metadata},
                              f, indent=2, default=str)
                contract = extract_json_from_message(response.content)
                previous_contract = contract
                validate_contract(contract, spec)
                envs = {'flat'} | {e['name'] for e in spec.environments}
                if contract.get('environment') not in envs:
                    raise ValueError(f'environment must be one of {sorted(envs)}')
                fallback = contract
                review = agent.generate([
                    SystemMessage(content=data['system'] + REVIEW_INSTRUCTIONS),
                    HumanMessage(content=json.dumps({'original_request': raw, 'proposed_contract': contract})),
                ])
                add_usage(review)
                with open(os.path.join(folder, f'goal_review_{attempt + 1:04d}.json'), 'w') as f:
                    json.dump({'content': review.content, 'response_metadata': review.response_metadata},
                              f, indent=2, default=str)
                verdict = extract_json_from_message(review.content)
                if verdict.get('accepted') is not True:
                    raise ValueError('Goal coverage review: ' + json.dumps(verdict))
                return publish(contract, verdict, True)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                print(f'[goal_contract] attempt {attempt + 1}/{attempts}: {exc}', flush=True)
                error = '\nCorrect this contract validation error without weakening the request: ' + str(exc)
        if fallback is not None:
            # Generation continues against the best schema-valid contract; the outcome can
            # then be at most best_effort, because no reviewer accepted its coverage.
            print('[goal_contract] ⚠️ no contract passed review; continuing with the last valid one '
                  '(outcome capped at best_effort)', flush=True)
            return publish(fallback, {'accepted': False, 'issues': [error.strip()]}, False)
        # No usable contract at all: still generate (the motion is the deliverable), but
        # without fixed acceptance criteria the outcome can only be 'unverified'.
        print('[goal_contract] ⚠️ could not compile any valid goal contract; continuing without one '
              '(outcome will be unverified)', flush=True)
        with open(os.path.join(folder, state['skill_name'] + '_report.json'), 'w') as f:
            json.dump({'outcome': 'in_progress', 'goal_contract_id': None,
                       'goal_contract_error': error.strip()}, f, indent=2)
        return {'token_usage': usage, 'outcome': 'in_progress', 'goal_review_accepted': False}
    return node
