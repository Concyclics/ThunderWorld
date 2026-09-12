"""Explicit disjoint experiment cohorts; historical evaluation is unchanged."""
import hashlib
import json
from pathlib import Path


class Cohort:
    def __init__(self, value):
        self.value = value
        if value.get('schema_version') != 1:
            raise ValueError('Unsupported cohort schema')
        self.seeds = value['seeds']
        expected = {'calibration', 'trajectory', 'validation', 'final'}
        if set(self.seeds) != expected:
            raise ValueError('Declare calibration, trajectory, validation, final cohorts')
        seen = set(value.get('historical_excluded_seeds', []))
        for role in ('calibration', 'trajectory', 'validation', 'final'):
            seeds = self.seeds[role]
            if not seeds or any(type(s) is not int or s < 0 for s in seeds):
                raise ValueError('Seeds must be nonempty nonnegative integer lists')
            if len(set(seeds)) != len(seeds) or seen.intersection(seeds):
                raise ValueError('Cohorts overlap or repeat seeds')
            seen.update(seeds)
        if value.get('steps') != 35 or value.get('phases') != [0, 7, 18, 28, 35]:
            raise ValueError('This protocol requires 35 steps and frozen four phases')

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.value, sort_keys=True).encode()).hexdigest()

    def validate(self, role, seeds, *, complete=False):
        expected = self.seeds[role]
        if len(set(seeds)) != len(seeds) or not set(seeds).issubset(expected):
            raise ValueError(f'Seeds outside declared {role} cohort')
        if complete and set(seeds) != set(expected):
            raise ValueError(f'Incomplete {role} cohort')
