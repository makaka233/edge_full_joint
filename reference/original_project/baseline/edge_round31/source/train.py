"""Discrete SAC + DDQN rollout; optional independent FINISH experience, never action selection by lookahead."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,hashlib,copy
from pathlib import Path
import numpy as np,torch
from edge_rl.config import Config
from edge_rl.agents import SchedulingAgent
from edge_rl.replay import SReplay
from environment import Env
from sac import SAC,Replay,Transition,macro_smdp_path,migrate_macro_smdp
from hierarchy import HierarchicalSAC
from warm_hierarchy import TransferredMacroSAC
from checkpoint import save_checkpoint,load_checkpoint,read_core
from rehearsal import Rehearsal,snapshot

def write(p,x):
    with open(p,'a',encoding='utf8') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def digest(net):return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in net.parameters())).hexdigest()
def config(path,seed=None):
    x=json.loads(Path(path).read_text());c=Config.load(Path(path).with_name('world.json'))
    if seed is not None:x['seed']=seed
    c.cpu_allocator=x.get('cpu_allocator','kkt')
    if c.cpu_allocator not in ('kkt','ps','finish_pf','fcfs','srpt'):raise ValueError('Unknown CPU allocator')
    c.device=x['device'];c.s_next_context=x['s_next_context'];c.torch_threads=x['torch_threads'];
    if x.get('s_competition_context',False):c.s_competition_context=True
    if x.get('s_structured_context',False):c.s_structured_context=True
    if x.get('s_reservation_context',False):c.s_reservation_context=True
    if x.get('s_resource_credit',False):c.s_resource_credit=True
    if x.get('s_system_credit',False):
        c.s_system_credit=True;c.s_system_credit_scale=float(x.get('s_system_credit_scale',20.));c.gamma_s=1.
    if x.get('s_objective_aligned',False):
        if not getattr(c,'s_system_credit',False) or x.get('cost_model',{}).get('latency_objective','rate_weighted') not in ('period_mean','rate_weighted'):raise ValueError('aligned S requires system credit and a recognized latency objective')
        c.s_objective_aligned=True;c.s_objective_reward_scale=float(x['reward_scale'])
        if x.get('s_local_auxiliary',False):c.s_local_auxiliary=True
        if x.get('s_center_local_advantage',False):
            if not x.get('s_local_auxiliary',False):raise ValueError('centered local advantage requires the local auxiliary')
            c.s_center_local_advantage=True
    if x.get("s_local_period_mean",False):
        if x.get('s_system_credit',False) or x.get('s_resource_credit',False) or x['cost_model']['latency_objective']!='period_mean':raise ValueError('local period mean requires ordinary per-request cost and period-mean D objective')
        c.s_local_period_mean=True;c.gamma_s=1.;c.s_local_reward_scale=float(x['reward_scale'])
    if x.get("s_externality_pairs",False):c.s_externality_pairs=True
    if x.get("s_pair_policy",False):
        assert x.get("s_externality_pairs",False)
        c.s_pair_policy=True; c.s_pair_temperature=float(x.get("s_pair_temperature",.15)); c.s_pair_kl=float(x.get("s_pair_kl",.15))
    if x.get("s_policy_evaluated_local",False):
        assert x.get("s_externality_pairs",False) and x["cost_model"]["latency_objective"]=="period_mean"
        c.s_policy_evaluated_local=True
    if x.get("s_iid_pg",False):
        assert x.get("s_pair_policy",False)
        c.s_iid_pg=True;c.s_iid_epsilon_prior=bool(x.get("s_iid_epsilon_prior",False))
    if x.get("s_mc_pg",False):
        assert x.get("s_system_credit") and x.get("s_local_auxiliary")
        c.s_mc_pg=True;c.s_mc_no_update=bool(x.get("s_mc_no_update",False));c.s_mc_kl=float(x.get("s_mc_kl",.01));c.s_mc_lr=float(x.get("s_mc_lr",.001))
        if x.get("s_mc_natural",False):c.s_mc_natural=True;c.s_mc_trust_kl=float(x.get("s_mc_trust_kl",.005));c.s_mc_fisher_damping=float(x.get("s_mc_fisher_damping",.01))
    if x.get('d_macro_smdp',False):
        assert not x.get('hierarchy') and not x.get('d_binary_advantage') and not x.get('d_potential_scale',0.) and not x.get('d_counterfactual_replay',False)
        assert not x.get('d_fresh_weight',1.) or x.get('d_fresh_phase_balanced',False)
    if x.get('d_stage_blocks',False):
        assert not x.get('hierarchy') and not x.get('d_binary_advantage') and not x.get('d_binary_projection',0.) and not x.get('d_separate_finish',False)
    if x.get('d_macro_forward_kl',False):assert x.get('d_macro_smdp',False) and not x.get('d_binary_projection',0.)
    if x.get('d_decompose_return',False):
        assert x.get('d_afterstate_features') and x.get('d_shared_cpu_features') and not any(x.get(k) for k in ('hierarchy','d_macro_smdp','d_potential_scale','d_binary_advantage','d_risk_weight','d_exact_cpu_effect','d_finish_probe','d_counterfactual_replay'))
    if x.get('d_build_pair_pg',False):assert not any(x.get(k) for k in ('d_decompose_return','d_layout_pair','d_binary_advantage','d_binary_projection','d_macro_smdp','d_potential_scale','d_finish_probe','hierarchy','d_gate_explore')) and x.get('demand_streams',1)==1
    if x.get('d_layout_pair',False):assert x.get('d_decompose_return',False) and x.get('demand_streams',1)==1
    streams=int(x.get('demand_streams',1))
    if streams<1:raise ValueError('demand_streams must be >=1')
    if streams>1 and (not x.get('continuing') or any(x.get(k) for k in ('hierarchy','d_binary_advantage','d_counterfactual_replay','d_finish_probe','s_rehearsal','s_externality_pairs','s_mc_pg'))):raise ValueError('multi-stream training supports continuing core SAC/DDQN only')
    return c,x
def environment(c,x):
    e=Env(c,x['population_seed'],x['episode_steps'],x['windows'],x['window_seconds'],x['warm_seconds'],x['tail_seconds'],x['failure_penalty_seconds'],x.get('demand_model'),x.get('cost_model'),x.get('continuing',False));e.risk_threshold=x.get('d_risk_threshold',1.)
    if x.get('d_stage_blocks',False):
        e.stage_blocks=True;e.A+=e.K;e.KEEP=e.A-3;e.BUILD=e.A-2;e.FINISH=e.A-1
    return e
def agents(c,x,e):
    e.w.binary_advantage=x.get('d_binary_advantage',False);e.w.binary_pair_warmup=x.get('d_binary_pair_warmup',16);e.w.binary_pair_lr=x.get('d_binary_pair_lr',.001);e.w.binary_projection=x.get('d_binary_projection',0.);e.w.binary_projection_warmup=x.get('d_binary_projection_warmup',0);e.w.binary_actor_damping=x.get('d_binary_actor_damping',0.);e.w.binary_actor_scope=x.get('d_binary_actor_scope','both');e.w.separate_finish=x.get('d_separate_finish',False);e.w.risk_weight=x.get('d_risk_weight',0.);e.w.risk_lr=x.get('d_risk_lr',x['d_lr']);e.w.shared_cpu_features=x.get('d_shared_cpu_features',False);e.w.exact_cpu_effect=x.get('d_exact_cpu_effect',False);e.w.factorized_stage_policy=x.get('d_factorized_stage_policy',False);e.w.finish_features=x.get('d_finish_features',False);e.w.actor_lr=x.get('d_actor_lr',x['d_lr']);e.w.auto_alpha=x.get('d_auto_alpha',False);e.w.alpha_target_kl=x.get('d_alpha_target_kl',.3);e.w.alpha_lr=x.get('d_alpha_lr',.01);e.w.shared_cpu_potential=x.get('d_shared_cpu_potential',0.)
    e.w.stage_blocks=bool(x.get("d_stage_blocks",False));e.w.retain_context=bool(x.get("d_retain_context",False));e.w.chain_context=bool(x.get("d_chain_context",False));e.w.count_context=bool(x.get("d_count_context",False));e.w.cost_potential=(dict(e.cost,scale=x["reward_scale"]) if x.get("d_cost_potential",False) else None)
    e.w.build_pair_pg=bool(x.get("d_build_pair_pg",False));e.w.decompose_return=bool(x.get("d_decompose_return",False));e.w.critic_mse=bool(x.get("d_critic_mse",False));e.w.macro_forward_kl=bool(x.get("d_macro_forward_kl",False));e.w.fresh_weight=float(x.get("d_fresh_weight",1.));e.w.construction_alpha=x.get("d_construction_alpha",x["d_alpha"]);e.w.pair_horizon=x.get("d_pair_horizon",1)
    cls=(TransferredMacroSAC if x['hierarchy'].get('transfer') else HierarchicalSAC) if x.get('hierarchy') else SAC
    if x.get('d_independent_macro',False):
        from round18_algorithms import MacroSAC
        cls=MacroSAC
    d=cls(e.w,x['seed']+11,x['d_hidden'],x['d_alpha'],x['d_lr'],x['d_tau'],x['device'],x.get('d_potential_scale',0.),x.get('d_afterstate_features',False),**({'macro':x['hierarchy']} if x.get('hierarchy') else {}));torch.manual_seed(x['seed']+29)
    if x.get('s_mc_pg',False):
        from round21_algorithms import MCScheduler
        return d,MCScheduler(c)
    if x.get('s_externality_pairs',False):
        from round18_algorithms import ExternalityScheduler
        if x.get("s_iid_pg",False):
            from round20_algorithms import IIDPairScheduler
            return d,IIDPairScheduler(c)
        if x.get("s_pair_policy",False):
            from round19_algorithms import PairPolicyScheduler
            return d,PairPolicyScheduler(c)
        return d,ExternalityScheduler(c)
    return d,SchedulingAgent(c)
def rng_for(seed,*ids):return np.random.default_rng(np.random.SeedSequence([seed,*ids]))
def fresh_indices(path,n,rng,balanced=False):
    # Same semantic groups as the ordinary SAC replay: macro, coverage, add, finish.
    if not balanced:return rng.integers(len(path),size=n)
    groups=[[],[],[],[]]
    for i,t in enumerate(path):
        k=0 if t.s.phase==0 else 1 if t.s.phase==1 else 3 if t.a==len(t.mask)-1 else 2
        groups[k].append(i)
    groups=[g for g in groups if g]
    return np.array([rng.choice(groups[int(rng.integers(len(groups)))]) for _ in range(n)],dtype=np.int64)

def reset(e,seed,ep):
    r=rng_for(seed,10,ep);return e.reset(int(r.integers(0,2**31-1)),int(r.integers(12)))
def rollout(e,d,s,seed,gs,epsilon,deterministic=False,collect=True,reward_scale=.1,gate_epsilon=0.,finish_probe=False,risk_threshold=1.,risk_mode="event",diagnostic=None,pair_callback=None,pair_sampling="geometric",aux_callback=None):
    e.current_s_version=int(getattr(s,"version",-1));rng=rng_for(seed,20,gs);path=[];mat=None;gate_p=0.;gate_behavior=0.;probe=None;probe_info=None
    while mat is None:
        st=e.state();mask=e.legal().copy();a,lr,p=d.act(st,mask,rng,deterministic,gate_epsilon if st.phase==0 else 0.);mat=e.apply(a)
        if st.phase==0 and mask[e.KEEP]:
            gate_p=p if a==e.KEEP else 1-p;ge=(d.me if isinstance(d,TransferredMacroSAC) else (1. if d.mcount<d.mwarm else d.me)) if isinstance(d,HierarchicalSAC) and gate_epsilon else gate_epsilon;gate_behavior=(1-ge)*gate_p+ge/2
        if mat is None:path.append(Transition(st,mask,a,0.,e.state(),e.legal().copy(),1.,False,False,lr))
        else:
            # Choose an off-policy observation state independently of its outcome.
            # Clone before commit; the live action is already chosen and cannot change.
            probe_env=None
            if finish_probe and a==e.FINISH:
                eligible=[t for t in path if t.s.phase==2 and t.mask[e.FINISH]]
                if eligible:
                    prng=rng_for(seed,71,gs)
                    # Stratify toward early FINISH decisions, not toward good/bad outcomes.
                    cutoff=max(1,(len(eligible)+1)//2)
                    pool=eligible[:cutoff] if prng.random()<.75 else eligible
                    pick=pool[int(prng.integers(len(pool)))];probe_env=copy.deepcopy(e)
                    probe_env.draft=pick.s.draft.copy();probe_env.phase=2;probe_env._mask=None
            if diagnostic is not None:
                from spike_audit import inspect_terminal
                diagnostic(inspect_terminal(e,d,s,seed,gs,epsilon,st,mask,a,mat,p,lr,rng_for))
            chosen_pair=None;pair_env=None
            if pair_callback is not None:
                from binary_advantage import choose_pair
                chosen_pair=choose_pair(e,path,st,mask,a,lr,seed,gs,rng_for,pair_sampling)
                if chosen_pair is not None:pair_env=copy.deepcopy(e)
            ns,info,ts,term,trunc=e.commit(mat,s.scheduler(rng_for(seed,30,gs),epsilon),int(rng_for(seed,90,gs).integers(0,2**31-1)),gs,collect)
            path.append(Transition(st,mask,a,info['reward']*reward_scale,ns,e.legal().copy(),e.c.gamma_d,term,trunc,lr,float(info['tail_fraction'] if risk_mode=='tail' else info['failed']>0 or info['penalized_delay']>risk_threshold)))
            if chosen_pair is not None:
                from binary_advantage import paired_experience
                result=paired_experience(pair_env,d,s,path,info,chosen_pair,seed,gs,epsilon,reward_scale,rng_for,return_path=aux_callback is not None)
                pair,rec=result[:2]
                if aux_callback is not None:aux_callback(result[2])
                rec["pair_sampling"]=pair_sampling;pair_callback(pair,rec)
            if probe_env is not None:
                pm=probe_env.legal().copy();assert np.array_equal(pm,pick.mask)
                px=probe_env.state();assert np.array_equal(px.demand,pick.s.demand)
                pmat=probe_env.apply(probe_env.FINISH)
                pn,pi,_,pt,pc=probe_env.commit(pmat,s.scheduler(rng_for(seed,30,gs),epsilon),int(rng_for(seed,90,gs).integers(0,2**31-1)),gs,False)
                assert pi['trace_sha256']==info['trace_sha256'] and np.array_equal(pn.demand,ns.demand)
                # Off-policy one-step transition: no fresh on-policy MC target.
                probe=Transition(px,pm,probe_env.FINISH,pi['reward']*reward_scale,pn,probe_env.legal().copy(),e.c.gamma_d,pt,pc,0.)
                probe_info={k:pi[k] for k in ('replicas','reward','penalized_delay','failed','n','trace_sha256')}
                probe_info['matrix_hex']=np.packbits(pmat.ravel()).tobytes().hex()
                probe_info['next_demand_hash']=hashlib.sha256(pn.demand.tobytes()).hexdigest()
    for tr in path:
        tr.partner_version=e.current_s_version;tr.collection_step=gs
    if probe is not None:
        probe.partner_version=e.current_s_version;probe.collection_step=gs
    info.update(gs=gs,gate_policy_keep_probability=gate_p,gate_behavior_keep_probability=gate_behavior,actions=len(path),decision='KEEP' if path[0].a==e.KEEP else 'REDEPLOY',terminated=term,truncated=trunc,epsilon=epsilon,matrix_hex=np.packbits(mat.ravel()).tobytes().hex(),demand_hash=hashlib.sha256(path[0].s.demand.tobytes()).hexdigest(),demand=path[0].s.demand.tolist(),next_demand_hash=hashlib.sha256(ns.demand.tobytes()).hexdigest(),s_transitions=len(ts))
    if getattr(e,'stage_blocks',False):
        info['block_actions']=sum(e.K*e.N<=t.a<e.KEEP for t in path)
        info['block_added_replicas']=sum(int(t.ns.draft.sum()-t.s.draft.sum()) for t in path if e.K*e.N<=t.a<e.KEEP)
    if finish_probe:info['finish_probe']=probe_info
    return path,info,ts,probe

def evaluate(c,x,d,s,seed,steps=24,deterministic=False):
    e=environment(c,x);reset(e,seed,0);rows=[]
    for i in range(1,steps+1):
        if not getattr(e,'continuing',False) and e.period==e.episode_steps:reset(e,seed,e.episode+1)
        _,info,_,_=rollout(e,d,s,seed,i,0.,deterministic,False);rows.append(info)
    return rows

def summary(rows):
    n=sum(r['n'] for r in rows);ncomp=sum(r['completed'] for r in rows)
    return dict(steps=len(rows),reward=float(np.mean([r['reward'] for r in rows])),penalized_e2e=sum(r['penalized_delay']*r['n'] for r in rows)/max(1,n),completed_e2e=sum((r['mean_completed'] or 0)*r['completed'] for r in rows)/max(1,ncomp),n=n,failed=sum(r['failed'] for r in rows),replicas=float(np.mean([r['replicas'] for r in rows])),keep=sum(r['decision']=='KEEP' for r in rows),changes=float(np.mean([r['changed'] for r in rows])),period_mean_e2e=float(np.mean([r['penalized_delay'] for r in rows])),delay_cost=float(np.mean([r['delay_cost'] for r in rows])),holding_cost=float(np.mean([r['holding_cost'] for r in rows])),switch_cost=float(np.mean([r['switch_cost'] for r in rows])),rate_weighted_reward=float(np.mean([-(r.get('rate_weighted_delay_cost',r['delay_cost'])+r['holding_cost']+r['switch_cost']+r.get('migration_cost',0.)) for r in rows])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/experiment_regime_slow_actor.json');p.add_argument('--steps',type=int,default=128);p.add_argument('--out',required=True);p.add_argument('--seed',type=int);p.add_argument('--mode',choices=['learn','freeze','s_only','d_only'],default='learn');p.add_argument('--resume');p.add_argument('--init-s',help='Load mature S only; fresh D, environment and replay');p.add_argument('--resume-allow-gate-explore',action='store_true',help='Only allow an explicit change of macro behavior exploration on a compatible checkpoint');p.add_argument('--resume-allow-smdp',action='store_true',help='Explicitly migrate a compatible completed-path replay to macro SMDP semantics');p.add_argument('--no-checkpoint',action='store_true');p.add_argument('--policy-out',help='Optional evaluation-only local export (not a resume checkpoint)');p.add_argument('--save-every',type=int,default=32);p.add_argument('--eval-every',type=int,default=0);p.add_argument('--eval-steps',type=int,default=12);p.add_argument('--diagnose-gs',nargs='*',type=int,default=[]);a=p.parse_args();c,x=config(a.config,a.seed);torch.set_num_threads(x['torch_threads']);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    e=environment(c,x);d,s=agents(c,x,e);dr=Replay(x['d_replay_periods']);aux_dr=Replay(x['d_replay_periods']) if x.get('d_counterfactual_replay',False) else None;sr=SReplay(c.s_replay_transitions,c.replay_max_age);local_sr=SReplay(c.s_replay_transitions,c.replay_max_age) if getattr(c,"s_local_auxiliary",False) else None;gs0=0;best=-float('inf');seed=x['seed'];rgr=rng_for(seed,40);reset(e,seed,0)
    stream_count=int(x.get('demand_streams',1));stream_seeds=[seed+1000000*i for i in range(stream_count)];envs=[e]
    for stream_seed in stream_seeds[1:]:
        other=environment(c,x);reset(other,stream_seed,0);envs.append(other)
    if a.resume:
        ck=load_checkpoint(a.resume);
        if a.resume_allow_gate_explore or a.resume_allow_smdp:
            old=dict(ck['config']);new=dict(x)
            if a.resume_allow_gate_explore:old.pop('d_gate_explore',None);new.pop('d_gate_explore',None);assert 0<=x.get('d_gate_explore',0.)<=1
            if a.resume_allow_smdp:old.pop('d_macro_smdp',None);new.pop('d_macro_smdp',None);assert x.get('d_macro_smdp',False)
            assert old==new and ck['physical']==vars(c)
        else:assert ck['config']==x and ck['physical']==vars(c)
        d.load_state_dict(ck['d']);s.load_state_dict(ck['s']);e=ck['env'];dr=ck['dr'];sr=ck['sr'];local_sr=ck.get('local_sr');aux_dr=ck.get('aux_dr',aux_dr);gs0=ck['gs'];best=ck['best'];rgr.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
        envs=ck.get('envs',[e]);assert len(envs)==stream_count
        if x.get('d_macro_smdp',False) and not ck['config'].get('d_macro_smdp',False):
            assert a.resume_allow_smdp and not x.get('d_potential_scale',0.) and x.get('d_fresh_weight',1.)==0
            print('MIGRATED MACRO OPTION PATHS',migrate_macro_smdp(dr,d),flush=True)
        dr.compact()
        if aux_dr is not None:aux_dr.compact()
        # Release deserialization scratch memory on Linux; optional elsewhere.
        del ck
        import gc;gc.collect()
        if os.name=='posix':
            try:
                import ctypes;ctypes.CDLL('libc.so.6').malloc_trim(0)
            except (OSError,AttributeError):pass
        for name in ['train.jsonl','validation.jsonl','rehearsal.jsonl','binary_pairs.jsonl','spike_audit.jsonl','external_pairs.jsonl','layout_pairs.jsonl','build_pairs.jsonl','macro_pairs.jsonl']:
            f=out/name
            if f.exists():f.write_text(''.join(l+'\n' for l in f.read_text().splitlines() if json.loads(l)['gs']<=gs0))
    elif (out/'train.jsonl').exists():raise ValueError('Refusing to overwrite existing run')
    if a.init_s:
        assert not a.resume and a.mode=='d_only'
        ss0=read_core(a.init_s);s.load_state_dict(ss0['s']);del ss0
    if x.get('d_macro_pair_replay',False):assert x.get('d_macro_smdp',False) and a.mode=='d_only' and stream_count==1
    if x.get('s_rehearsal') and not hasattr(sr,'rehearsal'):sr.rehearsal=Rehearsal(x['s_rehearsal'],c.replay_max_age)
    (out/'effective_config.json').write_text(json.dumps(dict(experiment=x,physical=vars(c),mode=a.mode,init_s=a.init_s),indent=2));initial=dict(d=digest(d.actor),s=digest(s.online));start=time.monotonic()
    def save(gs):
        payload=dict(config=x,physical=vars(c),d=d.state_dict(),s=s.state_dict(),env=e,dr=dr,sr=sr,local_sr=local_sr,aux_dr=aux_dr,gs=gs,best=best,rng=rgr.bit_generator.state,torch_rng=torch.get_rng_state());
        if stream_count>1:payload['envs']=envs
        save_checkpoint(payload,out/'latest.pt')
    if a.eval_every and gs0==0 and x.get('initial_validation',False):
        rows=evaluate(c,x,d,s,seed+100000,a.eval_steps);v=summary(rows)|dict(gs=0,validation_seed=seed+100000,trace_hashes=[r['trace_sha256'] for r in rows])
        write(out/'validation.jsonl',v);print('VALID INITIAL',v,flush=True)
    for gs in range(gs0+1,a.steps+1):
        stream=(gs-1)%stream_count;e=envs[stream];local_gs=(gs-1)//stream_count+1;rollout_seed=stream_seeds[stream]
        if not getattr(e,'continuing',False) and e.period==e.episode_steps:reset(e,seed,e.episode+1)
        eps=c.epsilon_end+(c.epsilon_start-c.epsilon_end)*max(0,1-(gs-1+x.get("s_exploration_offset",0))/c.epsilon_decay_steps)
        if x.get('s_frozen_greedy',False) and a.mode=='d_only':eps=0.
        macro_pre=copy.deepcopy(e) if x.get('d_macro_pair_replay',False) else None
        macro_alt=None
        rehears=getattr(sr,'rehearsal',None) if a.mode in ('learn','s_only') else None
        revealed=snapshot(e) if rehears else None
        external_pre=copy.deepcopy(e) if x.get("s_externality_pairs",False) and a.mode in ("learn","s_only") else None
        layout_pre=copy.deepcopy(e) if x.get('d_layout_pair',False) and a.mode in ('learn','d_only') else None
        paired_layout=None;precomputed_fy=None
        build_pre=copy.deepcopy(e) if x.get('d_build_pair_pg',False) and a.mode in ('learn','d_only') else None
        pairs=[];aux_paths=[]
        def record_pair(pair,rec):pairs.append(pair);write(out/'binary_pairs.jsonl',rec)
        t=time.monotonic();path,info,ts,probe=rollout(e,d,s,rollout_seed,local_gs,eps,False,a.mode in ('learn','s_only'),x['reward_scale'],(1. if x.get('hierarchy') else x.get('d_gate_explore',0.)) if a.mode in ('learn','d_only') else 0.,x.get('d_finish_probe',False) and a.mode in ('learn','d_only'),x.get('d_risk_threshold',1.),x.get('d_risk_mode','event'),(lambda r:write(out/'spike_audit.jsonl',r)) if gs in a.diagnose_gs else None,record_pair if x.get('d_binary_advantage',False) and a.mode in ('learn','d_only') else None,x.get('d_binary_pair_sampling','geometric'),aux_paths.append if aux_dr is not None and a.mode in ('learn','d_only') else None);ds={};ss={}
        if macro_pre is not None:
            from macro_pair_replay import alternative
            macro_alt,macro_log=alternative(macro_pre,e,d,s,path,info,rollout_seed,local_gs,eps,x['reward_scale'],rng_for)
            if macro_log is not None:write(out/'macro_pairs.jsonl',macro_log);info['macro_pair']=macro_log
        if build_pre is not None:
            from build_pair_pg import collect_build_pair,update_build_pg
            build_pair,build_log=collect_build_pair(build_pre,e,path,info,d,s,rollout_seed,local_gs,eps,x['reward_scale'],rng_for)
            if build_log is not None:
                write(out/'build_pairs.jsonl',build_log);info['build_pair']=build_log
                # Fresh, pre-critic/pre-actor update; both paths from same policy.
                precomputed_fy=d.fresh_targets(path)
                ds.update(update_build_pg(d,build_pair))
        if layout_pre is not None:
            from layout_pair import collect_layout_pair
            paired_layout,layout_log=collect_layout_pair(layout_pre,e,d,s,info,rollout_seed,local_gs,eps,x['reward_scale'],rng_for)
            if layout_log is not None:write(out/'layout_pairs.jsonl',layout_log);info['layout_pair']=layout_log
        if stream_count>1:
            info.update(gs=gs,stream_id=stream,stream_step=local_gs,stream_seed=rollout_seed)
            for tr in path:tr.collection_step=gs
            for tr in ts:tr.collection_step=gs
        if external_pre is not None:
            from round18_algorithms import external_samples
            if x.get('s_iid_pg',False) and x.get('s_iid_no_update',False):
                ex,exlogs,eh=[],[],info['trace_sha256']
            elif x.get('s_iid_pg',False):
                from round20_algorithms import iid_samples
                ex,exlogs,eh=iid_samples(external_pre,e.old,s,seed,gs,int(rng_for(seed,90,gs).integers(0,2**31-1)),rng_for(seed,1971,gs),x.get('s_pairs_per_window',6),eps)
                # One fresh on-policy update BEFORE the reference DDQN changes.
                ss.update(s.update_iid(ex,eps));assert eh==info['trace_sha256']
            else:
                ex,exlogs,eh=external_samples(external_pre,e.old,s,seed,gs,int(rng_for(seed,90,gs).integers(0,2**31-1)),rng_for(seed,1971,gs),x.get('s_pairs_per_window',2));assert eh==info['trace_sha256'];s.add_external(ex,gs)
            for rec in exlogs:write(out/'external_pairs.jsonl',rec)
        if x.get('s_mc_pg',False) and a.mode in ('learn','s_only'):
            ss.update(s.update_mc(ts,eps))
        if a.mode in ('learn','d_only'):
            if x.get('hierarchy'):
                ds=d.macro_update(path,rgr);path=path[1:]
            if path:
                fy=precomputed_fy if precomputed_fy is not None else d.fresh_targets(path);rfy=d.risk_fresh_targets(path) if getattr(d,'risk_weight',0.) else None;dr.add(macro_smdp_path(path,d) if x.get('d_macro_smdp',False) else path)
                fresh_path=path
                if macro_alt is not None:
                    afy=d.fresh_targets(macro_alt);dr.add(macro_smdp_path(macro_alt,d))
                    # Both paths have on-policy construction conditional on their macro action.
                    # Pool both paths; the existing phase-balanced fresh sampler is unchanged.
                    fresh_path=path+macro_alt;fy=fy+afy
                if probe is not None:dr.add_probe(probe)
                if aux_dr is not None:
                    for ap in aux_paths:aux_dr.add(ap)
                for _ in range(x['d_updates_per_period'] if gs>x.get('d_learning_starts',0) else 0):
                    inds=rgr.integers(len(fresh_path),size=x['fresh_batch']);
                    if x.get('d_fresh_phase_balanced',False):inds=fresh_indices(fresh_path,x['fresh_batch'],rng_for(seed,231,gs,d.updates),True)
                    na=int(x['d_batch']*x.get('d_aux_fraction',.25)) if aux_dr is not None and len(aux_dr) else 0
                    items=(dr.sample_factorial(x['d_batch']-na,rgr,x['d_replay_phase_mass'],x['d_replay_macro_balance']) if 'd_replay_phase_mass' in x else dr.sample_global_phase(x['d_batch']-na,rgr) if x.get('d_replay_global_phase_balance',False) else dr.sample(x['d_batch']-na,rgr,x.get('d_finish_probe_fraction',0.)))+(aux_dr.sample(na,rgr) if na else [])
                    ds.update(d.update(items,[fresh_path[i] for i in inds],[fy[i] for i in inds],**({'rfy':[rfy[i] for i in inds]} if rfy is not None else {}),**({'period_r':path[-1].r,'layout_pair':paired_layout} if x.get('d_decompose_return',False) else {})))
                if aux_dr is not None:ds.update(d_aux_transitions=len(aux_dr),d_aux_added=sum(len(ap) for ap in aux_paths))
        if x.get('d_binary_advantage',False) and a.mode in ('learn','d_only'):ds.update(d.learn_binary_pairs(pairs,rng_for(seed,195,gs),x.get('d_binary_pair_updates',4)))
        if a.mode in ('learn','s_only'):
            sr.add(ts);sr.prune(gs)
            if local_sr is not None:
                if x.get("s_policy_evaluated_local",False):
                    # Constant reference multiplier preserves the exact period-mean objective.
                    factor=e.cost["reference_rate"]*e.windows*e.window/max(1,info["n"])
                    for tr in e.local_transitions:tr.reward*=factor
                local_sr.add(e.local_transitions);local_sr.prune(gs)
            if rehears:
                aux=rehears.interact(e,s,gs,seed,rng_for)
                if aux is not None:write(out/'rehearsal.jsonl',aux)
                rehears.add(revealed,info)
            lam=x.get('s_system_trace_lambda',0.)
            ft=s.system_trace_targets(ts,lam) if lam and ts and external_pre is None else None
            for ui in range(x['s_updates_per_period']):
                if len(sr)>=32:
                    if local_sr is not None and len(local_sr)>=32:ss.update(s.update_local(local_sr.sample(x['s_batch'],gs,rgr)))
                    if external_pre is not None or x.get("s_mc_pg",False):continue
                    batch=sr.sample(x['s_batch'],gs,rgr)
                    if ft is not None:
                        nf=x['s_batch']//2;ix=rng_for(seed,111,gs,ui).integers(len(ts),size=nf)
                        ss.update(s.update(batch[:-nf],[ts[i] for i in ix],ft[ix]))
                    else:ss.update(s.update(rehears.blend(batch,gs,rng_for(seed,713,gs,ui)) if rehears else batch))
            if external_pre is not None and not x.get("s_iid_pg",False):ss.update(s.update_external(rng_for(seed,1972,gs),x["s_updates_per_period"]))
            s.version+=1
        info.update(s_collection_version=e.current_s_version,s_policy_version=s.version,hazard=int(info['failed']>0 or info['penalized_delay']>x.get('d_risk_threshold',1.)),risk_label=float(path[-1].hazard) if path else float(info['tail_fraction'] if x.get('d_risk_mode')=='tail' else info['failed']>0 or info['penalized_delay']>x.get('d_risk_threshold',1.)),d_updates=d.updates,s_updates=s.updates,wall_s=time.monotonic()-t,**ds,**ss);write(out/'train.jsonl',info)
        print(f"{a.mode} GS{gs:04} E{info['episode']} P{info['period']:02} R={info['reward']:.4f} E2E={info['penalized_delay']:.4f} n={info['n']} fail={info['failed']} rep={info['replicas']} {info['decision']} actions={info['actions']} wall={info['wall_s']:.2f}",flush=True)
        if a.eval_every and gs%a.eval_every==0:
            rows=evaluate(c,x,d,s,seed+100000,a.eval_steps);v=summary(rows)|dict(gs=gs,validation_seed=seed+100000,trace_hashes=[r['trace_sha256'] for r in rows]);write(out/'validation.jsonl',v);print('VALID',v,flush=True)
            if v['reward']>best and not a.no_checkpoint:
                best=v['reward'];payload=dict(d=d.actor.state_dict(),s=s.online.state_dict(),gs=gs,**({'macro':{'q1':d.mq1.state_dict(),'q2':d.mq2.state_dict(),**({'actor':d.ma_actor.state_dict()} if hasattr(d,'ma_actor') else {})}} if x.get('hierarchy') else {}))
                if getattr(s,'local_auxiliary',False):payload.update(s_local=s.local_online.state_dict(),config=x)
                if getattr(d,'binary_projection',0.):payload.update(config=x,d_q1=d.q1.state_dict(),d_q2=d.q2.state_dict(),d_updates=d.updates)
                if getattr(d,'binary_advantage',False):payload.update(d_binary_adv=d.binary_adv.state_dict(),binary_pair_count=d.binary_pair_count)
                if hasattr(d,'macro_adv'):payload.update(d_macro_adv=d.macro_adv.state_dict())
                torch.save(payload,out/'best.pt')
        if not a.no_checkpoint and a.mode!='freeze' and gs%a.save_every==0 and gs<a.steps:save(gs)
    if not a.no_checkpoint and a.mode!='freeze':save(a.steps)
    if a.policy_out:torch.save(dict(config=x,gs=a.steps,d=d.state_dict(),s=s.state_dict()),a.policy_out)
    (out/'completed.json').write_text(json.dumps(dict(gs=a.steps,mode=a.mode,seconds=time.monotonic()-start,initial_hash=initial,final_hash=dict(d=digest(d.actor),s=digest(s.online)),d_updates=d.updates,s_updates=s.updates),indent=2));print('DONE',a.steps,flush=True)
if __name__=='__main__':main()
