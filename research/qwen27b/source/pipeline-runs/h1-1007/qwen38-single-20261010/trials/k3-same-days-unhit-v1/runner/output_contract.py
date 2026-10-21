"""Explicit coverage and scoped claims. Validation does not certify semantics."""
import json
import math

SUBJECTS = ('self', 'other_retail', 'main_force', 'other', 'text', 'unknown')
TIMES = ('completed', 'planned', 'current', 'past', 'conditional', 'general', 'unknown')
STATES = ('expressed', 'explicitly_negated', 'ambiguous')


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def object_schema(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def response_schema(schema):
    string = {'type': 'string'}
    explanation = {'type': 'string', 'minLength': 1}
    nullable_score = {'type': ['number', 'null'], 'minimum': 0, 'maximum': 1}
    details = {'quote': string, 'subject': {'type': 'string', 'enum': list(SUBJECTS)},
               'target': string, 'time': {'type': 'string', 'enum': list(TIMES)},
               'time_text': string, 'intensity': nullable_score, 'support': nullable_score}
    claim = object_schema({'state': {'type': 'string', 'enum': list(STATES)},
                           **details, 'reason': explanation})
    absent = object_schema({'state': {'type': 'string', 'enum': ['not_expressed']}})
    present = object_schema({'claims': {'type': 'array', 'items': claim, 'minItems': 1}})
    # Named, required slots: no omission-as-negative or cross-list duplicate IDs.
    checks = object_schema({x['id']: {'anyOf': [absent, present]}
                            for x in schema['core_candidates']})
    observation = object_schema({'kind': {'type':'string','enum':['emotion','stance','action','style','other']}, 'meaning': string, 'state': {'type':'string','enum':list(STATES)}, **details, 'reason': explanation})
    return object_schema({'id': string, 'observations': {'type': 'array', 'items': observation}, 'checks': checks})


def keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        missing = sorted(set(expected) - set(value)) if isinstance(value, dict) else list(expected)
        extra = sorted(set(value) - set(expected)) if isinstance(value, dict) else []
        raise ValueError(f'{name}: incomplete or unexpected fields; missing={missing}, extra={extra}')


def score(value):
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                             and math.isfinite(value) and 0 <= value <= 1)


def details(item, text, name):
    if not isinstance(item['quote'], str) or not item['quote'] or item['quote'] not in text:
        raise ValueError(f'{name}: evidence is not an exact nonempty text span')
    if item['subject'] not in SUBJECTS or item['time'] not in TIMES:
        raise ValueError(f'{name}: invalid subject or time')
    if not isinstance(item['target'], str) or not item['target'].strip():
        raise ValueError(f'{name}: missing target')
    if not isinstance(item['time_text'], str) or (item['time_text'] and item['time_text'] not in text):
        raise ValueError(f'{name}: time_text must be empty or an exact text span')
    if not score(item['intensity']) or not score(item['support']):
        raise ValueError(f'{name}: scores must be finite 0..1 numbers or null')


def normalize(content, row, schema):
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Empty/non-text assistant content')
    obj = strict_json(content)
    keys(obj, ('id', 'checks', 'observations'), 'response')
    if obj['id'] != row['id']:
        raise ValueError('Response ID does not exactly match request ID')
    order = [x['id'] for x in schema['core_candidates']]
    keys(obj['checks'], order, 'checks')
    judgments, issues = {}, []
    fields = ('state', 'quote', 'subject', 'target', 'time', 'time_text', 'intensity', 'support', 'reason')
    for meaning in order:
        check = obj['checks'][meaning]
        if check == {'state': 'not_expressed'}:
            judgments[meaning] = {'decision': 'not_supported', 'state': 'not_expressed',
                                 'explicit_coverage': True, 'quote': '', 'claims': [],
                                 'intensity': None, 'support': None}
            continue
        keys(check, ('claims',), meaning)
        claims = check['claims']
        if not isinstance(claims, list) or not claims:
            raise ValueError(f'{meaning}: claims cannot be empty; report not_expressed explicitly')
        seen, scopes = {}, {}
        for index, claim in enumerate(claims):
            name = f'{meaning}[{index}]'
            keys(claim, fields, name)
            if claim['state'] not in STATES:
                raise ValueError(f'{name}: invalid state')
            details(claim, row['text'], name)
            if not isinstance(claim['reason'], str) or not claim['reason'].strip():
                raise ValueError(f'{name}: every expressed/negated/ambiguous claim requires a nonempty reason')
            if claim['state'] != 'explicitly_negated' and claim['support'] is None:
                raise ValueError(f'{name}: expressed/ambiguous claims require a support estimate')
            if claim['state'] == 'explicitly_negated' and claim['intensity'] is not None:
                raise ValueError(f'{name}: denial is not a positive emotion-intensity measurement')
            scope = tuple(claim[k] for k in ('subject', 'target', 'time', 'time_text'))
            exact = (*scope, claim['quote'])
            if exact in seen:
                raise ValueError(f'{meaning}: duplicate/conflicting state for the same scoped evidence')
            seen[exact] = claim['state']
            scopes.setdefault(scope, set()).add(claim['state'])
        for scope, states in scopes.items():
            if {'expressed', 'explicitly_negated'}.issubset(states):
                # Different spans may concern different events. Never silently choose one.
                issues.append({'meaning': meaning, 'type': 'potential_scope_conflict',
                               'scope': list(scope), 'action': 'review_original_text'})
        positives = [x for x in claims if x['state'] == 'expressed']
        uncertain = [x for x in claims if x['state'] == 'ambiguous']
        decision = 'supported' if positives else 'abstain' if uncertain else 'not_supported'
        representative = (positives or uncertain or claims)[0]
        if len(claims) == 1:
            judgment = {**representative, 'decision': decision, 'claims': claims}
        else:
            # Compatibility view only: existential label comparison against the frozen baseline.
            # Downstream scores MUST use scoped claims, never this projection.
            judgment = {'decision': decision, 'state': 'multiple_contexts',
                        'quote': representative['quote'], 'claims': claims,
                        'projection_only': True, 'intensity': None, 'support': None}
        judgments[meaning] = {**judgment, 'explicit_coverage': True}
    observations = obj['observations']
    if not isinstance(observations, list):
        raise ValueError('observations must be a list')
    for index, item in enumerate(observations):
        name = f'observations[{index}]'
        keys(item, ('kind', 'meaning', 'state', 'quote', 'subject', 'target', 'time', 'time_text', 'intensity', 'support', 'reason'), name)
        if item['kind'] not in ('emotion','stance','action','style','other') or item['state'] not in STATES:
            raise ValueError(f'{name}: invalid observation kind or state')
        if item['state'] == 'explicitly_negated' and item['intensity'] is not None:
            raise ValueError(f'{name}: negated observation intensity must be null')
        if not isinstance(item['meaning'], str) or not item['meaning'].strip():
            raise ValueError(f'{name}: empty meaning')
        if not isinstance(item['reason'], str) or not item['reason'].strip():
            raise ValueError(f'{name}: every additional observation requires a nonempty reason')
        details(item, row['text'], name)
    return {'id': row['id'], 'status': 'ok', 'judgments': judgments,
            'checks': obj['checks'], 'observations': observations,
            'review_issues': issues, 'needs_scope_review': bool(issues),
            'scores_are_calibrated_probabilities': False,
            'quality_notice': 'Structure validated; semantic correctness and score quality not certified.'}
