from wspbench.report import build_report, clustered_interval
import hashlib
import json

CONFIG = {'id': 'test-model'}
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()


def matrix():
    return [{
        'id': f'{skill}-{task}-{attempt}', 'task_id': f'{skill}-{task}',
        'parent_id': f'{skill}-{task}', 'skill': skill, 'family': skill + '_family',
        'split': 'official', 'phase': 'primary', 'attempt': attempt,
        'model_id': 'test-model', 'provider': 'test', 'success': task == 0,
        'status': 'completed', 'audit': {'status': 'approved'}, 'cost_usd': .1,
        'suite_digest': 'suite', 'config_digest': 'config', 'model_config_digest': CONFIG_HASH,
        'worker_image_digest': 'image', '_integrity_valid': True, 'model_config': CONFIG,
    } for skill in ('read', 'repair', 'write') for task in range(2) for attempt in range(3)]


def freeze():
    return {'suite_digest': 'suite', 'config_digest': 'config', 'worker_image_digest': 'image',
            'models': [CONFIG], 'expected_matrix': [{'model_id': r['model_id'], 'task_id': r['task_id'], 'attempt': r['attempt']} for r in matrix()]}


def report(records):
    return build_report(records, expected_tasks=6, frozen=freeze())


def test_complete_matrix_averages_all_attempts():
    value = report(matrix())
    assert value['models'][0]['skills']['read']['score'] == .5
    assert value['models'][0]['skills']['read']['n_tasks'] == 2


def test_partial_or_pending_cannot_be_official():
    records = matrix()
    assert not report(records[:-1])['models']
    records[0]['audit']['status'] = 'pending'
    assert not report(records)['models']


def test_rejected_success_is_zero_and_infrastructure_blocks_publication():
    records = matrix()
    records[0]['audit']['status'] = 'rejected'
    score = report(records)['models'][0]['skills']['read']['score']
    assert abs(score - 1 / 3) < 1e-10
    records[0]['status'] = 'infrastructure_error'
    assert not report(records)['models']


def test_attempt_duplication_cannot_create_complete_matrix():
    records = matrix()
    records[1]['attempt'] = records[0]['attempt']
    assert not report(records)['models']


def test_bootstrap_uses_parent_not_episode_count():
    assert clustered_interval({'a': [0.] * 100, 'b': [1.]})[0] == .5


def test_missing_freeze_or_mixed_provenance_is_not_verified():
    assert not build_report(matrix(), expected_tasks=6)['models']
    records = matrix()
    records[0]['suite_digest'] = 'different-suite'
    assert not report(records)['models']


def test_trace_tampering_prevents_verification():
    records = matrix()
    records[0]['_integrity_valid'] = False
    assert not report(records)['models']


def test_family_continuous_metrics_preserve_units_and_missingness():
    records = matrix()
    records[0]['metrics'] = {'max_absolute_error': .002}
    value = report(records)['models'][0]['skills']['read']
    assert 'metrics' not in value
    metric = value['families']['read_family']['metrics']['max_absolute_error']
    assert metric == {'mean': .002, 'n_tasks': 1, 'n_observations': 1}


def test_legacy_sanity_tasks_never_become_headline_results():
    records = matrix()
    for record in records:
        if record['skill'] == 'read':
            record['family'] = 'read_visual'
    value = report(records)
    assert not value['models']
    assert 'sanity suite excluded from headline scores' in value['pending'][0]['reasons']
