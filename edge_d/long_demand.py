"""Opt-in macro demand with the original user attachment and Poisson sampler."""

import numpy as np

from ._engine.dynamic import DynamicWorld
from .world import stable_seed


class LongDemandWorld(DynamicWorld):
    @classmethod
    def enable(cls,w,config,seed,first=None):
        # Reuse the world already constructed with the registered workload/user model.
        rate = float(w.rates(w.t).sum())
        w.__class__ = cls
        w.__dict__.pop('advance',None)  # Disable the adapter's legacy per-period matching.
        w.long_config = config
        w.context_dim = 3+w.c.regions+len(w.c.services)
        w.macro_gs = 0
        w.phase = np.asarray(w.c.region_phases)
        w.state_rng,w.ar_rng,w.pulse_rng = [np.random.default_rng(stable_seed('EXP065-demand',seed,k))
                                        for k in ('state','ar','pulse')]
        w.day_amplitude = w.state_rng.uniform(*config['amplitude'],w.c.regions)
        w.phase_offset = w.state_rng.uniform(0.,2*np.pi)
        w.amplitude = w.shift = 0.  # Daily phase is sampled by macro slot, never physical seconds.
        w.region_mass = np.bincount(w.regions,weights=w.user_counts,minlength=w.c.regions)/w.user_counts.sum()
        w.baseline_scale = rate/(w.user_counts.sum()*w.c.user_rate)
        w.center = np.zeros(1+w.c.regions+len(w.c.services))
        w.noise = w.center.copy()
        w.pulse_end = 0
        w.pulse = None
        w.start_region = int(w.state_rng.integers(w.c.regions))
        w.start_service = int(w.state_rng.integers(len(w.c.services)))
        w.new_segment(first)
        w.reveal_long()
        return w

    def new_segment(self,kind=None):
        q,rng = self.long_config,self.state_rng
        self.mode = kind if kind is not None else str(rng.choice(q['modes'],p=q['probabilities']))
        self.regime_id = q['modes'].index(self.mode)
        lo,hi,median = q['durations'][self.regime_id]
        while True:
            length = int(np.rint(rng.lognormal(np.log(median),q['duration_log_sd'])))
            if lo<=length<=hi:
                break
        self.segment_start,self.segment_end = self.macro_gs,self.macro_gs+length
        self.previous_center = self.center.copy()
        self.target_center = self.center.copy()
        split = 1+self.c.regions
        if self.mode in ('regional','compound'):
            self.start_region = int(rng.choice([i for i in range(self.c.regions) if i!=self.start_region]))
            self.target_center[1:split] = 0.
            self.target_center[1+self.start_region] = np.log(rng.uniform(*q['region_hot_gain']))
        if self.mode in ('service','compound'):
            self.start_service = int(rng.choice([i for i in range(len(self.c.services)) if i!=self.start_service]))
            self.target_center[split:] = 0.
            self.target_center[split+self.start_service] = np.log(rng.uniform(*q['service_hot_gain']))
        if self.mode in ('intensity','compound'):
            self.target_center[0] = np.log(rng.uniform(*q['global_center']))
        self.transition = int(rng.integers(q['transition_length'][0],q['transition_length'][1]+1)) if rng.random()<.5 else 1

    def reveal_long(self):
        fraction = min(1.,(self.macro_gs-self.segment_start+1)/self.transition)
        self.center = self.previous_center+fraction*(self.target_center-self.previous_center)
        split = 1+self.c.regions
        signal = self.center+self.noise
        self.scale = self.baseline_scale*float(np.exp(signal[0]))
        region = np.exp(signal[1:split])
        # Separate compositional migration from total rates before adding daily cycles and bursts.
        self.daily_factor = 1+self.day_amplitude*np.sin(2*np.pi*self.macro_gs/self.long_config['slots_per_day']+self.phase_offset-self.phase)
        self.region_scale = region/(self.region_mass@region)*self.daily_factor
        marks = np.broadcast_to(np.exp(signal[split:]),(self.c.n,len(self.c.services))).copy()
        self.hot_region,self.hot_service = int(np.argmax(region)),int(np.argmax(signal[split:]))
        if self.macro_gs>=self.pulse_end:
            self.pulse = None
            rng,q = self.pulse_rng,self.long_config
            if rng.random()<q['pulse_probability']:
                length = int(rng.integers(q['pulse_duration'][0],q['pulse_duration'][1]+1))
                self.pulse = (int(rng.integers(self.c.regions)),int(rng.integers(len(self.c.services))),
                              float(rng.uniform(*q['pulse_gain'])))
                self.pulse_end = self.macro_gs+length
        if self.pulse is not None:
            r,s,gain = self.pulse
            self.region_scale[r] *= gain
            marks[self.regions==r,s] *= gain
        self.set_marks(marks)

    def advance(self):
        self.macro_gs += 1
        self.t += self.c.period_seconds  # episode() retains control of actual event seconds.
        if self.macro_gs==self.segment_end:
            self.new_segment()
        q = self.long_config
        self.noise = np.repeat(q['ar'],[1,self.c.regions,len(self.c.services)])*self.noise + self.ar_rng.normal(
            0.,np.repeat(q['innovations'],[1,self.c.regions,len(self.c.services)]))
        self.reveal_long()

    def context(self):
        phase = 2*np.pi*self.macro_gs/self.long_config['slots_per_day']
        rates = self.rates(self.t)
        total = rates.sum()
        regions = np.bincount(self.regions,weights=rates,minlength=self.c.regions)/total
        services = (rates@self.mark)/total
        return np.r_[np.sin(phase),np.cos(phase),np.log(total),regions,services].astype(np.float32)
