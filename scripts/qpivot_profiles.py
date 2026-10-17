"""Shared, auditable bounded-dictionary profiles for Global QPivot performance."""
import json
from pathlib import Path

DEFAULT_CONFIG=Path(__file__).with_name('qpivot_performance_profiles.json')


def load_profiles(path=DEFAULT_CONFIG):
    config=json.loads(Path(path).read_text())
    for dataset,profile in config['datasets'].items():
        if profile.get('scope')!='global' or not 3<=profile['pmax']<=512:
            raise ValueError(f'{dataset}: expected Global PCA with Pmax in [3,512]')
        if not 1<=profile['fixed_nprobe']<=profile['nlist']:
            raise ValueError(f'{dataset}: invalid fixed_nprobe')
        if not profile['nprobes'] or any(not 1<=n<=profile['nlist'] for n in profile['nprobes']):
            raise ValueError(f'{dataset}: invalid nprobes')
        if not profile['budget_grid'] or max(profile['budget_grid'])!=profile['pmax'] or min(profile['budget_grid'])<3:
            raise ValueError(f'{dataset}: budget grid must end at its persisted Pmax')
    if not set(config['default_datasets'])<=set(config['datasets']):
        raise ValueError('unknown default dataset')
    return config


CONFIG=load_profiles()
PROFILES=CONFIG['datasets']
