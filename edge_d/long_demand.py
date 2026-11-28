"""Optional EXP086 macro demand process; physical time remains in World.t.

Only current intensities/identities enter the existing observation interface.
Residence age and future transitions are audit metadata, never policy inputs.
"""
import numpy as np

from ._engine.dynamic import DynamicWorld
from .world import stable_seed


class LongDemandWorld(DynamicWorld):
    def __init__(self, original, config, seed):
        # Retain the exact network, population, groups, workload and route tables.
        self.__dict__.update(original.__dict__)
        self.__dict__.pop('advance', None)  # remove the old profile-bound method
        self.__dict__.pop('comparison_rate_reference', None)
        self.long_config = dict(config)
        self.center_rng = np.random.default_rng(stable_seed(seed, 'centers'))
        self.noise_rng = np.random.default_rng(stable_seed(seed, 'residuals'))
        initial = np.random.default_rng(stable_seed(seed, 'initial'))
        self.daily_amplitude = float(initial.uniform(*config['daily_amplitude']))
        self.macro_time = float(initial.uniform(0, 86400))
        self.prehistory_slots = int(initial.integers(config['prehistory_slots'][0], config['prehistory_slots'][1]+1))
        self.macro_index = 0
        self.regime, self.seasonal = True, False
        self.regime_id = int(self.center_rng.choice(3, p=self.params['regime_initial']))
        self.hot_region = int(self.center_rng.integers(self.c.regions))
        self.hot_service = int(self.center_rng.integers(len(self.c.services)))
        dimensions = {'global': None, 'region': self.c.regions, 'service': len(self.c.services)}
        self.residuals = {k: self.noise_rng.normal(0, config['residual_stationary_sd'][k], size=size)
                          for k, size in dimensions.items()}
        self.remaining = self._duration()
        self.residence_slots = self.remaining
        self.transitions = []
        self._reveal_long()
        for _ in range(self.prehistory_slots):
            self.advance()
        self.t = 0.  # demand prehistory does NOT advance or warm a physical queue
        self.start_macro_time = self.macro_time
        self.start_macro_index = self.macro_index

    @property
    def micro_time(self):
        return self.t

    def _duration(self):
        p = self.long_config
        hours = self.center_rng.lognormal(np.log(p['residence_median_hours']), p['residence_log_sd'])
        return max(1, int(np.floor(hours*3600/p['macro_seconds']+.5)))

    def _reveal_long(self):
        self.log_global = float(np.log(self.params['regime_levels'][self.regime_id])+self.residuals['global'])
        self.log_region = self.residuals['region'].copy()
        self.log_service = self.residuals['service'].copy()
        super().reveal()
        # Daily multiplier has mean exactly one over each complete 12-slot day.
        self.daily_factor = 1+self.daily_amplitude*np.sin(2*np.pi*self.macro_time/86400)
        self.scale *= self.daily_factor
        # Preserve the physical meaning of log-global context: observed intensity.
        self.log_global = float(np.log(self.scale))

    def advance(self):
        p = self.long_config
        self.macro_index += 1
        self.macro_time += p['macro_seconds']
        self.remaining -= 1
        if self.remaining == 0:
            component = str(self.center_rng.choice(p['change_components'], p=p['change_probabilities']))
            if component in ('region','joint'):
                self.hot_region = int(self.center_rng.choice([i for i in range(self.c.regions) if i != self.hot_region]))
            if component in ('service','joint'):
                self.hot_service = int(self.center_rng.choice([i for i in range(len(self.c.services)) if i != self.hot_service]))
            if component in ('load','joint'):
                self.regime_id = int(self.center_rng.choice([i for i in range(3) if i != self.regime_id]))
            self.transitions.append(dict(index=self.macro_index, component=component,
                                         completed_residence=self.residence_slots))
            self.remaining = self.residence_slots = self._duration()
        for key, value in self.residuals.items():
            rho = p['residual_ar'][key]
            sd = p['residual_stationary_sd'][key]
            self.residuals[key] = rho*value+self.noise_rng.normal(0, sd*np.sqrt(1-rho*rho), size=np.shape(value) or None)
        self._reveal_long()
        if hasattr(self, 'comparison_rate_reference'):
            self.comparison_rate_reference.advance()
            reference = self.comparison_rate_reference.rates(0).sum()
            self.scale *= reference/self.rates(0).sum()

    def audit_state(self):
        return dict(macro_time=self.macro_time, micro_time=self.micro_time,
                    macro_index=self.macro_index, prehistory_slots=self.prehistory_slots,
                    day_factor=float(self.daily_factor), daily_amplitude=self.daily_amplitude,
                    regime=self.regime_id, hot_region=self.hot_region, hot_service=self.hot_service,
                    remaining_for_audit_only=self.remaining, residence_slots=self.residence_slots)


def attach(env, config, seed):
    """Replace demand only, after profile adaptation; match to NEW reference flow."""
    old = env.w
    reference = getattr(old, 'comparison_rate_reference', None)
    env.w = LongDemandWorld(old, config, seed)
    if reference is not None:
        env.w.comparison_rate_reference = LongDemandWorld(reference, config, seed)
        env.w.scale *= env.w.comparison_rate_reference.rates(0).sum()/env.w.rates(0).sum()
    return env
