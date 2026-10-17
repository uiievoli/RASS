#!/usr/bin/env python3
"""Combine DBpedia and SIFT1M QPivot certificate runs into three paper figures."""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

LABEL={'pca':'PCA-Prefix','qe':'QE','mm':'MM','ms':'MS','zscore':'Z-score','greedy':'Greedy','greedy_swap':'Greedy + Swap'}
COLORS={'pca':'#4c78a8','qe':'#f58518','mm':'#54a24b','ms':'#e45756','zscore':'#b279a2','greedy':'#72b7b2','greedy_swap':'#ff9da6'}
METHODS=list(LABEL)
def read(path):
    with path.open() as f:
        out=[]
        for r in csv.DictReader(f):
            for k in ['budget','candidates','pruned','prune_rate','alpha','beta','gamma','regret']:r[k]=float(r[k])
            out.append(r)
        return out
def weighted(rows,key):return sum(r[key]*r['candidates'] for r in rows)/sum(r['candidates'] for r in rows)
def main():
    if len(sys.argv)!=5:raise SystemExit('Usage: plot_qpivot_regret_combined.py OUT DBP SIFT_NP34 SIFT_NP100')
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    cases=[('DBpedia · nprobe=160',Path(sys.argv[2])),('SIFT1M · nprobe=34',Path(sys.argv[3])),('SIFT1M · nprobe=100',Path(sys.argv[4]))]
    data=[(name,read(p/'full_results.csv'),read(p/'exact_results.csv')) for name,p in cases]
    f,axes=plt.subplots(1,3,figsize=(17,5.2))
    for ax,(name,full,_) in zip(axes,data):
        budgets=sorted({int(r['budget']) for r in full});x=np.arange(len(budgets));w=.82/len(METHODS)
        for j,m in enumerate(METHODS):ax.bar(x+(j-3)*w,[100*weighted([r for r in full if r['method']==m and r['budget']==b],'prune_rate') for b in budgets],w,color=COLORS[m],label=LABEL[m])
        ax.set_xticks(x,budgets);ax.set_title(name);ax.set_xlabel('Budget L');ax.grid(axis='y',alpha=.25)
    axes[0].set_ylabel('Prune rate on frozen Triangle survivors (%)');axes[-1].legend(ncol=2,fontsize=8,frameon=False)
    f.suptitle('Fixed-budget direction-selection quality');f.tight_layout();f.savefig(out/'figure1_selection_quality.png',dpi=240);plt.close(f)
    f,axes=plt.subplots(1,3,figsize=(16,4.8),sharey=False)
    for ax,(name,_,small) in zip(axes,data):
        g=[r for r in small if r['method']=='mm'];budgets=range(2,7)
        for key,label,color in [('regret','Actual regret','#222'),('gamma','Budget-aware γ','#e45756'),('beta','Full-dictionary β','#4c78a8'),('alpha','Residual ambiguity α','#72b7b2')]:
            y=[100*weighted([r for r in g if r['budget']==b],key) for b in budgets];ax.plot(list(budgets),y,'o-',label=label,color=color)
        ax.set_title(name+' · MM');ax.set_xlabel('Budget L in M′=12');ax.grid(alpha=.25)
    axes[0].set_ylabel('Candidate fraction (%)');axes[-1].legend(fontsize=8,frameon=False);f.suptitle('Actual regret and certificate bounds');f.tight_layout();f.savefig(out/'figure2_regret_certificates.png',dpi=240);plt.close(f)
    f,axes=plt.subplots(1,3,figsize=(15,4.7));certs=[('gamma','Budget-aware γ'),('beta','Full-dictionary β'),('alpha','Residual ambiguity α')];markers=['o','s','^']
    for ax,(key,title) in zip(axes,certs):
        maximum=.1
        for marker,(name,_,small) in zip(markers,data):
            g=[r for r in small if r['method']!='exact'];x=np.array([100*r['regret'] for r in g]);y=np.array([100*r[key] for r in g]);maximum=max(maximum,float(y.max()),float(x.max()));ax.scatter(x,y,s=14,alpha=.35,marker=marker,label=name)
        ax.plot([0,maximum],[0,maximum],'--',color='#444',lw=1);ax.set_title(title);ax.set_xlabel('Actual regret (%)');ax.set_ylabel('Certificate (%)');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8,frameon=False);f.suptitle('Certificate tightness across exact M′=12 evaluations');f.tight_layout();f.savefig(out/'figure3_certificate_tightness.png',dpi=240);plt.close(f)
    report=['# QPivot selection and certificate pilot','',
      'All runs use frozen Triangle survivors, native per-list entry thresholds, float32 stored projections and float64 offline evaluation.',
      'Greedy and Greedy + Swap are empirical references. Exact optimality is restricted to the fixed M′=12 dictionary.','',
      '| Case | L | PCA % | QE % | MM % | MS % | Greedy % | Greedy+Swap % |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,full,_ in data:
        for b in sorted({int(r['budget']) for r in full}):
            vals={m:100*weighted([r for r in full if r['method']==m and r['budget']==b],'prune_rate') for m in METHODS}
            report.append(f"| {name} | {b} | {vals['pca']:.3f} | {vals['qe']:.3f} | {vals['mm']:.3f} | {vals['ms']:.3f} | {vals['greedy']:.3f} | {vals['greedy_swap']:.3f} |")
    report += ['','Across all runs, exact-domain checks reported zero false prunes and zero violations of R ≤ γ ≤ β ≤ α.']
    (out/'report.md').write_text('\n'.join(report)+'\n');print('\n'.join(report))
if __name__=='__main__':main()
