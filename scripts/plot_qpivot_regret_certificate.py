#!/usr/bin/env python3
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

METHODS=['pca','qe','mm','ms','zscore','greedy','greedy_swap']
LABEL={'pca':'PCA-Prefix','qe':'QE','mm':'MM','ms':'MS','zscore':'Z-score','greedy':'Greedy','greedy_swap':'Greedy + Swap','exact':'Exact (M′=12)'}
COLORS={'pca':'#4c78a8','qe':'#f58518','mm':'#54a24b','ms':'#e45756','zscore':'#b279a2','greedy':'#72b7b2','greedy_swap':'#ff9da6','exact':'#333333'}
def read(p):
    with p.open() as f:return [{**r,**{k:float(r[k]) for k in ['budget','candidates','pruned','prune_rate','alpha','beta','gamma','regret','exact_pruned']}} for r in csv.DictReader(f)]
def weighted(rows,key):return sum(r[key]*r['candidates'] for r in rows)/sum(r['candidates'] for r in rows)
def quant(rows,key,q):return float(np.quantile([r[key] for r in rows],q))
def main():
    root=Path(sys.argv[1]);fig=root/'figures';fig.mkdir(exist_ok=True);full=read(root/'full_results.csv');small=read(root/'exact_results.csv')
    budgets=sorted({int(r['budget']) for r in full});methods=[m for m in METHODS if any(r['method']==m for r in full)]
    x=np.arange(len(budgets));width=.82/max(1,len(methods));plt.figure(figsize=(10,5.8))
    for j,m in enumerate(methods):plt.bar(x+(j-(len(methods)-1)/2)*width,[100*weighted([r for r in full if r['method']==m and r['budget']==b],'prune_rate') for b in budgets],width,label=LABEL[m],color=COLORS[m])
    plt.xticks(x,budgets);plt.xlabel('Direction budget L');plt.ylabel('Prune rate on frozen Triangle survivors (%)');plt.title('QPivot fixed-budget selection quality');plt.grid(axis='y',alpha=.25);plt.legend(ncol=4,frameon=False);plt.tight_layout();plt.savefig(fig/'figure1_selection_quality.png',dpi=220);plt.close()
    chosen=['qe','greedy','greedy_swap'];f,axes=plt.subplots(1,3,figsize=(15,4.6),sharey=True)
    for ax,m in zip(axes,chosen):
        group=[r for r in small if r['method']==m];bs=sorted({int(r['budget']) for r in group})
        for key,label,color in [('regret','Actual regret','#222222'),('gamma','Budget-aware γ','#e45756'),('beta','Full-dictionary β','#4c78a8'),('alpha','Residual ambiguity α','#72b7b2')]:
            means=[100*weighted([r for r in group if r['budget']==b],key) for b in bs];lo=[100*quant([r for r in group if r['budget']==b],key,.1) for b in bs];hi=[100*quant([r for r in group if r['budget']==b],key,.9) for b in bs]
            ax.plot(bs,means,'o-',label=label,color=color);ax.fill_between(bs,lo,hi,color=color,alpha=.12)
        ax.set_title(LABEL[m]);ax.set_xlabel('Budget L in fixed M′=12');ax.grid(alpha=.25)
    axes[0].set_ylabel('Candidate fraction (%)');axes[-1].legend(frameon=False,fontsize=9);f.suptitle('Actual regret and certificates (mean; P10–P90 band)');f.tight_layout();f.savefig(fig/'figure2_regret_certificates.png',dpi=220);plt.close(f)
    f,axes=plt.subplots(1,3,figsize=(14.5,4.5));certs=[('gamma','Budget-aware γ'),('beta','Full-dictionary β'),('alpha','Residual ambiguity α')]
    points=[r for r in small if r['method']!='exact']
    for ax,(key,title) in zip(axes,certs):
        for m in METHODS: 
            g=[r for r in points if r['method']==m]
            if g:ax.scatter([100*r['regret'] for r in g],[100*r[key] for r in g],s=13,alpha=.45,label=LABEL[m],color=COLORS[m])
        lim=max([100*r[key] for r in points]+[100*r['regret'] for r in points]+[.1]);ax.plot([0,lim],[0,lim],'--',color='#555',lw=1);ax.set_title(title);ax.set_xlabel('Actual regret (%)');ax.set_ylabel('Certificate (%)');ax.grid(alpha=.2)
    axes[-1].legend(frameon=False,fontsize=8,ncol=2);f.suptitle('Certificate tightness inside the exact M′=12 experiment');f.tight_layout();f.savefig(fig/'figure3_certificate_tightness.png',dpi=220);plt.close(f)
    rows=[]
    for b in budgets:
        base=weighted([r for r in full if r['method']=='pca' and r['budget']==b],'prune_rate');greedy=weighted([r for r in full if r['method']=='greedy' and r['budget']==b],'prune_rate')
        for m in methods:
            g=[r for r in full if r['method']==m and r['budget']==b];rate=weighted(g,'prune_rate');den=greedy-base;captured=100*(rate-base)/den if abs(den)>1e-15 else float('nan')
            rows.append(dict(budget=b,method=m,prune_rate_pct=100*rate,exact_survivors=int(sum(r['candidates']-r['pruned'] for r in g)),greedy_improvement_captured_pct=captured,
                remaining_pct=100*(1-rate),alpha_pct=100*weighted(g,'alpha'),beta_pct=100*weighted(g,'beta'),gamma_pct=100*weighted(g,'gamma'),selection_cost={'pca':'very low','qe':'low','mm':'low','ms':'low','zscore':'low','greedy':'high','greedy_swap':'very high'}[m]))
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    violations=sum((r['regret']>r['gamma']+1e-12) or (r['gamma']>r['beta']+1e-12) or (r['beta']>r['alpha']+1e-12) for r in small)
    report=['# QPivot pivot selection and regret certificates','',f'Exact-domain certificate inequality violations: **{violations}**.','',
      '| L | Method | Prune % | Exact survivors | Greedy improvement captured % | α % | β % | γ % | Selection cost |','|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:report.append(f"| {r['budget']} | {LABEL[r['method']]} | {r['prune_rate_pct']:.3f} | {r['exact_survivors']:,} | {r['greedy_improvement_captured_pct']:.2f} | {r['alpha_pct']:.3f} | {r['beta_pct']:.3f} | {r['gamma_pct']:.3f} | {r['selection_cost']} |")
    report += ['','Greedy and Greedy + Swap are empirical references. Exact optimality applies only to the fixed M′=12 subdictionary.','All rates are candidate-count weighted over frozen Triangle survivors and native list-entry thresholds.']
    (root/'report.md').write_text('\n'.join(report)+'\n');print('\n'.join(report))
if __name__=='__main__':main()
