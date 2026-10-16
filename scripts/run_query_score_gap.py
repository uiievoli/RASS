#!/usr/bin/env python3
"""Compare cheap direction scores against fixed-budget global Greedy references."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from run_query_energy_native import ROOT, read, write, sha

METHODS=['prefix','query_energy','moment_mean','z_score','greedy',
         'moment_mean_listweighted','z_score_eps1e8','z_score_eps1e4']
OUT=ROOT/'logs/query_score_gap_20261016'


def cases():
    captures=ROOT/'logs/fixed_p_list_rank_20261013'
    old=json.loads((captures/'protocol.json').read_text())
    db=next(c for c in old['cases'] if c['dataset']=='dbpedia1536m_holdout' and c['scope']=='global')
    sift=json.loads((ROOT/'logs/sift1m_query_moment_native_20261016/protocol.json').read_text())
    result=[dict(name='dbpedia1536_np160',dataset='dbpedia1536m_holdout',d=1536,m=256,np=160,nq=20,
        budgets=[32,65,96,129],capture_p=128,index=db['index'],
        dictionary=str(ROOT/'logs/query_subset_oracle_20261016/dictionary.bin'),
        reuse_greedy=str(ROOT/'logs/query_subset_oracle_20261016/orders.csv'),
        visits=str(captures/'dbpedia1536m_holdout/global/visits.csv'),
        query=str(captures/'views/dbpedia1536m_holdout/origin/dbpedia1536m_holdout_query.fvecs'))]
    for npb in [34,100]:
        result.append(dict(name=f'sift1m_np{npb}',dataset='sift1m',d=128,m=128,np=npb,nq=32,
            budgets=[8,16,32],capture_p=40,index=sift['index']['path'],dictionary=str(ROOT/'logs/sift1m_query_moment_native_20261016/dictionary.bin'),
            reuse_greedy='-',visits=str(captures/'sift1m/per_list/visits.csv'),query=sift['query']))
    for c in result:
        ids=json.loads((captures/f'views/{c["dataset"]}/query_ids.json').read_text())['query_ids']
        c['source_query_ids']=ids[:c['nq']]
    return result


def validate(directory,c):
    rows=read(directory/'queries.csv');lists=read(directory/'lists.csv');orders=read(directory/'orders.csv')
    key=lambda r:(int(r['query_id']),int(r['directions']),r['method'])
    by={key(r):r for r in rows};plan={key(r):list(map(int,r['order'].split())) for r in orders}
    assert len(by)==len(rows)==len(orders)==c['nq']*len(c['budgets'])*len(METHODS)
    assert len(lists)==len(rows)*c['np']
    sums=defaultdict(lambda:np.zeros(5,dtype=np.int64));fields=['candidates','triangle_pruned','mp_checked','mp_pruned','overall_pruned']
    list_by={}
    for r in lists:
        sums[key(r)]+=np.array([int(r[f]) for f in fields]);list_by[key(r)+(int(r['list_id']),)]=r
        if int(r['probe_rank'])==0:assert int(r['overall_pruned'])==0
    for k,r in by.items():
        np.testing.assert_array_equal(sums[k],[int(r[f]) for f in fields])
        assert int(r['candidates'])==int(r['mp_checked'])+int(r['triangle_pruned'])
        assert int(r['overall_pruned'])==int(r['triangle_pruned'])+int(r['mp_pruned'])
        assert len(plan[k])==k[1] and len(set(plan[k]))==k[1] and all(0<=i<c['m'] for i in plan[k])
    for q in range(c['nq']):
        for method in METHODS:
            for a,b in zip(c['budgets'],c['budgets'][1:]):assert plan[q,a,method]==plan[q,b,method][:a]
    safety=read(directory/'safety.csv')
    assert len(safety)==c['nq'] and all(int(r['false_prunes'])==0 and int(r['final_plan_checks'])==len(METHODS)*len(c['budgets']) for r in safety)
    # Independently rebuild mixture statistics and rank all score families.
    moments={}
    for r in read(directory/'list_moments.csv'):
        lid=int(r['list_id']);dim=int(r['direction'])
        if lid not in moments:moments[lid]=(int(r['candidates']),np.zeros(c['m']),np.zeros(c['m']))
        moments[lid][1][dim]=float(r['mean']);moments[lid][2][dim]=float(r['variance'])
    scores=read(directory/'query_scores.csv');assert len(scores)==c['nq']*c['m']
    for q in range(c['nq']):
        rr=sorted((r for r in scores if int(r['query_id'])==q),key=lambda r:int(r['direction']))
        z=np.array([float(r['query_projection']) for r in rr]);mu=np.array([float(r['pooled_mean']) for r in rr]);var=np.array([float(r['pooled_variance']) for r in rr])
        visits=[r for r in lists if int(r['query_id'])==q and int(r['directions'])==c['budgets'][0] and r['method']=='prefix']
        n=sum(int(r['candidates']) for r in visits)
        expected_mu=sum(moments[int(r['list_id'])][0]*moments[int(r['list_id'])][1] for r in visits)/n
        expected_var=sum(moments[int(r['list_id'])][0]*(moments[int(r['list_id'])][2]+(moments[int(r['list_id'])][1]-expected_mu)**2) for r in visits)/n
        np.testing.assert_allclose(mu,expected_mu,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(var,expected_var,atol=1e-12,rtol=1e-12)
        assert np.isfinite(var).all() and (var>=0).all()
        expected={'query_energy':z*z,'moment_mean':(z-mu)**2,
            'moment_mean_listweighted':sum(moments[int(r['list_id'])][0]*(z-moments[int(r['list_id'])][1])**2 for r in visits)/n}
        for method,ratio in [('z_score',1e-6),('z_score_eps1e8',1e-8),('z_score_eps1e4',1e-4)]:
            expected[method]=(z-mu)**2/(var+ratio*max(var.mean(),1e-30))
        for method,values in expected.items():
            np.testing.assert_allclose(values,[float(r[method]) for r in rr],rtol=1e-11,atol=1e-12)
            order=np.argsort(-values,kind='stable').tolist()
            for p in c['budgets']:assert order[:p]==plan[q,p,method],(q,p,method)
    # Recompute projections, moments, exact distances, bounds and pruning from raw vectors.
    bank=np.fromfile(directory/'dictionary.bin',dtype='<f8').reshape(c['m']+1,c['d']);center,basis=bank[0],bank[1:]
    qraw=np.fromfile(c['query'],dtype='<i4').reshape(-1,c['d']+1)[:,1:].copy().view('<f4').astype(np.float64)
    checks=0
    for r in read(directory/'validation_samples.csv'):
        q=int(r['query_id']);lid=int(r['list_id']);n=int(r['candidates']);tau=float(r['entry_tau2'])
        raw=np.fromfile(directory/f'validation_samples/{q}_{lid}.f32',dtype='<f4').reshape(n+1,c['d']).astype(np.float64)
        centroid,x=raw[0],raw[1:];xc=x-center;qc=qraw[q]-center;z=basis@xc.T;a=basis@qc
        xn=np.sum(xc*xc,axis=1);qn=qc@qc;exact=np.sum((x-qraw[q])**2,axis=1)
        local_q=np.sum((qraw[q]-centroid)**2);local_x=np.sum((x-centroid)**2,axis=1)
        tri=(np.sqrt(local_q)-np.sqrt(local_x))**2>tau
        np.testing.assert_allclose(z.mean(1),moments[lid][1],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(z.var(1),moments[lid][2],rtol=1e-10,atol=1e-10)
        for p in c['budgets']:
            for method in METHODS:
                dims=plan[q,p,method];bound=np.sum((a[dims,None]-z[dims])**2,axis=0)+(np.sqrt(max(0,qn-np.sum(a[dims]**2)))-np.sqrt(np.maximum(0,xn-np.sum(z[dims]**2,axis=0))))**2
                assert not np.any((bound>tau)&(exact<=tau))
                assert np.max(bound-exact)<=1e-8*max(1,float(exact.max()))
                actual=list_by[q,p,method,lid]
                assert int(actual['triangle_pruned'])==int(tri.sum())
                assert int(actual['mp_pruned'])==int(((bound>tau)&~tri).sum()),(q,lid,p,method)
                checks+=1
    if c['reuse_greedy']!='-':
        old={(int(r['query_id']),int(r['directions']),r['method']):r for r in read(ROOT/'logs/query_subset_oracle_20261016/queries.csv') if r['method'] in ['prefix','query_energy','greedy']}
        for k,r in by.items():
            if k in old:assert all(r[f]==old[k][f] for f in fields),(k,r,old[k])
        assert sha(directory/'dictionary.bin')==sha(Path(c['dictionary']))
    return dict(query_plans=len(rows),list_plans=len(lists),independent_raw_vector_plan_checks=checks,false_prunes=0)


def summarize(out,config):
    result=[];paired=[];validation={}
    rng=np.random.default_rng(20261016)
    for c in config:
        directory=out/c['name'];validation[c['name']]=validate(directory,c)
        rows=read(directory/'queries.csv');by={(int(r['query_id']),int(r['directions']),r['method']):r for r in rows}
        samples=rng.integers(c['nq'],size=(4000,c['nq']))
        for p in c['budgets']:
            n=np.array([int(by[q,p,'prefix']['candidates']) for q in range(c['nq'])])
            counts={method:np.array([int(by[q,p,method]['overall_pruned']) for q in range(c['nq'])]) for method in METHODS}
            prefix,greedy=counts['prefix'],counts['greedy'];headroom=greedy-prefix
            remain_g=n-greedy
            for method in METHODS:
                count=counts[method];remaining=n-count;gain=count-prefix
                bs_gain=100*gain[samples].sum(1)/n[samples].sum(1)
                ratio=float(remaining.sum()/remain_g.sum()) if remain_g.sum()>0 else None
                bs_ratio=remaining[samples].sum(1)/np.maximum(remain_g[samples].sum(1),1)
                numerator=(greedy-count).sum();denom=headroom.sum()
                gap=float(numerator/denom) if denom>0 else None
                bs_denom=headroom[samples].sum(1);valid=bs_denom>0
                bs_gap=(greedy-count)[samples].sum(1)[valid]/bs_denom[valid]
                gap_lo,gap_hi=np.quantile(bs_gap,[.025,.975]) if valid.any() else [None,None]
                gain_lo,gain_hi=np.quantile(bs_gain,[.025,.975]);ratio_lo,ratio_hi=np.quantile(bs_ratio,[.025,.975])
                result.append(dict(case=c['name'],dataset=c['dataset'],scope='global_query',nprobe=c['np'],dictionary_m=c['m'],directions=p,native_P=p+1,method=method,
                    queries=c['nq'],candidates=int(n.sum()),pruned=int(count.sum()),remaining=int(remaining.sum()),pruning_pct=float(100*count.sum()/n.sum()),
                    gain_vs_prefix_pp=float(100*gain.sum()/n.sum()),gain_ci95_low_pp=float(gain_lo),gain_ci95_high_pp=float(gain_hi),
                    candidates_ratio_vs_greedy=ratio,ratio_ci95_low=float(ratio_lo),ratio_ci95_high=float(ratio_hi),
                    greedy_gap=gap,recovered_headroom_pct=100*(1-gap) if gap is not None else None,
                    gap_ci95_low=None if gap_lo is None else float(gap_lo),gap_ci95_high=None if gap_hi is None else float(gap_hi),
                    bootstrap_positive_headroom_fraction=float(valid.mean()),queries_positive_greedy_headroom=int((headroom>0).sum()),
                    queries_better_than_greedy=int((count>greedy).sum()),queries_worse_than_prefix=int((count<prefix).sum())))
                for q in range(c['nq']):
                    paired.append(dict(case=c['name'],query_id=q,directions=p,method=method,candidates=int(n[q]),remaining=int(remaining[q]),
                        greedy_remaining=int(remain_g[q]),ratio_vs_greedy=float(remaining[q]/remain_g[q]) if remain_g[q]>0 else None,
                        greedy_headroom=int(headroom[q]),greedy_gap=float((greedy[q]-count[q])/headroom[q]) if headroom[q]>0 else None,
                        gain_vs_prefix_pp=float(100*gain[q]/n[q])))
    write(out/'summary.csv',result);write(out/'paired_metrics.csv',paired)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    report=['# 低成本方向选择：QE / MM / Z-score 与 Greedy gap','',
        '## 口径与配置','',
        '- 全部主实验为Global PCA，每查询一套方向子集，跨访问list共用；不是per-list PCA。',
        '- DBpedia-1536：M256、nprobe160、20查询、P=32/65/96/129；SIFT1M：M128、nprobe34/100、32查询、P=8/16/32。',
        '- nlist1000、top-k1；固定原生list入口阈值，整个访问list纳入候选分母，首个heap未建立的list保留。',
        '- 使用与前轮完全相同的冻结float64正交字典；SIFT阈值来自原per-list PCA40查询，但所有本轮方法使用Global基。',
        '- μ为访问list按完整候选数加权的投影总体均值；σ²包含list内方差和list间均值差异。',
        '- QE=z²；MM=(z−μ)²；Z=(z−μ)²/(σ²+ε)，ε=1e−6×mean_i(σ_i²)，并设1e−30的方差尺度下限。',
        '- 补充保存ε比例1e−8/1e−4及旧list加权MM对照；未按测试结果挑选ε。',
        '- 组合搜索和投影开销不计，本轮不测在线latency，不训练模型。离线Greedy使用32线程不代表线上查询线程。',
        '- DBpedia复用旧Greedy路径并重新计算剪枝，逐查询结果与旧oracle一致；SIFT从完整128方向重新运行Greedy。','',
        '## 指标','',
        '`F(S) = 同候选、同阈值下的总剪枝率（Triangle + MP）`。',
        '`CandidateRatio(S) = remaining(S) / remaining(Greedy)`，越接近1越接近Greedy；小于1表示超过该Greedy。',
        '`Gap(S) = [F(Greedy)−F(S)] / [F(Greedy)−F(Prefix)]`；0表示恢复全部Greedy增益，1表示与前缀相同，超过1表示比前缀差。',
        '`Recovered = 1−Gap`。总体指标由候选计数先汇总再求比值，不是逐查询比值平均。',
        'Greedy并非全局最优，允许负gap/超过100%恢复率，不做截断。Greedy不优于前缀时gap记NA。',
        '区间为4000次按查询配对bootstrap；仅在Greedy增益为正的重采样上定义gap，CSV保存有效比例。','',
        '## 主结果','',
        '| 数据集/nprobe | P | 方法 | 剪枝率% | 剩余候选数 | /Greedy剩余 | Greedy gap | 恢复空间% |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in result:
        if r['method'] not in METHODS[:5]:continue
        gap='NA' if r['greedy_gap'] is None else f"{r['greedy_gap']:.3f}"
        recovered='NA' if r['recovered_headroom_pct'] is None else f"{r['recovered_headroom_pct']:.1f}"
        report.append(f"| {r['case']} | {r['directions']} | {r['method']} | {r['pruning_pct']:.3f} | {r['remaining']:,} | {r['candidates_ratio_vs_greedy']:.3f} | {gap} | {recovered} |")
    report+=['','## 限制与验证','',
        '不能将本轮冻结阈值的剪枝率混作前轮真实heap演进下的原生剪枝率；候选比值也不是查询加速比。',
        '仅比较低成本评分能否接近已找到的Greedy组合，不证明其接近全局最优或能够部署加速。',
        '每个最终方案检查原始向量精确距离，零误剪；独立NumPy从原始float32向量重算采样list投影、统计、下界和计数。',
        'summary.csv包含所有平滑敏感性、旧MM对照、区间和退化计数；paired_metrics.csv包含逐查询比值和gap；validation.json保存验证数量。','']
    (out/'report.md').write_text('\n'.join(report))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(config),2,figsize=(11,3.8*len(config)),squeeze=False)
    for row,c in enumerate(config):
        for method in ['query_energy','moment_mean','z_score']:
            rr=[r for r in result if r['case']==c['name'] and r['method']==method]
            axes[row,0].plot(c['budgets'],[r['candidates_ratio_vs_greedy'] for r in rr],'o-',label=method)
            axes[row,1].plot(c['budgets'],[r['greedy_gap'] for r in rr],'o-',label=method)
        axes[row,0].axhline(1,color='gray',linestyle=':');axes[row,1].axhline(0,color='gray',linestyle=':')
        axes[row,0].set(title=c['name'],xlabel='Directions P',ylabel='Remaining candidates / Greedy')
        axes[row,1].set(title=c['name'],xlabel='Directions P',ylabel='Greedy gap (lower is better)')
        for ax in axes[row]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/f'comparison.{ext}',dpi=180)
    print('\n'.join(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--threads',type=int,default=32)
    parser.add_argument('--summarize-only',action='store_true');args=parser.parse_args()
    if args.summarize_only:
        config=json.loads((args.out/'protocol.json').read_text())['cases'];summarize(args.out,config);return
    if args.threads<1:parser.error('threads must be positive')
    if (args.out/'protocol.json').exists():parser.error('choose a new output directory')
    args.out.mkdir(parents=True,exist_ok=True);config=cases();binary=ROOT/'build-predictor-perf/bin/query_score_gap'
    for c in config:
        c['hashes']={key:sha(Path(c[key])) for key in ['dictionary','query','visits']}
        if c['reuse_greedy']!='-':c['hashes']['reuse_greedy']=sha(Path(c['reuse_greedy']))
        c['index_stat']=dict(bytes=Path(c['index']).stat().st_size,mtime_ns=Path(c['index']).stat().st_mtime_ns)
    protocol=dict(cases=config,threads=args.threads,seed=20261016,epsilon_relative=1e-6,epsilon_sensitivity=[1e-8,1e-4],
        scope='Global PCA, one score-ranked subset per query across all probed lists',
        moments='candidate-count weighted mixture of full visited lists, population variance including between-list variance',
        binary_sha256=sha(binary),source_sha256=sha(ROOT/'src/query_score_gap.cpp'),runner_sha256=sha(Path(__file__)))
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',OMP_PLACES='cores',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for c in config:
        directory=args.out/c['name'];directory.mkdir(exist_ok=True)
        command=list(map(str,[binary,c['index'],c['query'],c['visits'],directory,c['nq'],c['m'],0,20261016,','.join(map(str,c['budgets'])),c['np'],c['dictionary'],c['reuse_greedy'],c['capture_p']]))
        (directory/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        print(f'Start {c["name"]}',flush=True);start=time.monotonic()
        with (directory/'run.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        (directory/'runtime.json').write_text(json.dumps(dict(offline_seconds=time.monotonic()-start),indent=2)+'\n')
    summarize(args.out,config);(args.out/'complete').touch()


if __name__=='__main__':main()
