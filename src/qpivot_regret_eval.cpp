#include "Index.h"
#include <Eigen/Dense>
#include <omp.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>

using namespace tribase;
using MatrixD = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using MatrixF = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

struct Visit { size_t q,lid,rank,begin,end; double tau2; };
struct ListCache {
    MatrixF z;
    Eigen::VectorXd mean,variance;
    std::vector<double> global_norm2,local_norm2;
};
struct Candidates {
    size_t n=0,m=0;
    double qnorm2=0;
    Eigen::VectorXd qz;
    std::vector<float> z; // direction-major float32 storage
    std::vector<double> xnorm2,tau2,exact2;
};
struct Metrics { size_t pruned=0,alpha=0,beta=0,gamma=0,false_prunes=0; double max_excess=-1e300; };

static std::vector<std::string> split(const std::string& line,char delimiter=',') {
    std::vector<std::string> r; std::stringstream ss(line); std::string s;
    while(std::getline(ss,s,delimiter))r.push_back(s); return r;
}
static std::vector<size_t> parseSizes(const std::string& value) {
    std::vector<size_t> r; for(const auto& s:split(value,' '))if(!s.empty())r.push_back(std::stoull(s)); return r;
}
static std::string join(const std::vector<size_t>& values) {
    std::ostringstream out; for(size_t i=0;i<values.size();++i){if(i)out<<' ';out<<values[i];} return out.str();
}
static std::vector<size_t> ranked(size_t m,const std::function<double(size_t)>& score) {
    std::vector<size_t> order(m); std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return score(a)>score(b);}); return order;
}
static std::pair<size_t,double> countPlan(const Candidates& c,const std::vector<size_t>& plan) {
    double qspent=0; for(size_t dim:plan)qspent+=c.qz[dim]*c.qz[dim];
    const double qr=std::sqrt(std::max(0.,c.qnorm2-qspent)); size_t count=0; double strength=0;
#pragma omp parallel for reduction(+:count,strength) schedule(static)
    for(size_t i=0;i<c.n;++i) {
        double score=0,xspent=0;
        for(size_t dim:plan){const double x=c.z[dim*c.n+i],delta=c.qz[dim]-x;score+=delta*delta;xspent+=x*x;}
        const double gap=qr-std::sqrt(std::max(0.,c.xnorm2[i]-xspent)); const double lb2=score+gap*gap;
        count+=lb2>c.tau2[i]; strength+=lb2;
    }
    return {count,strength};
}
static std::vector<std::vector<size_t>> greedyPath(const Candidates& c,size_t max_l,const std::vector<size_t>& pool) {
    std::vector<std::vector<size_t>> history; std::vector<size_t> selected; std::vector<uint8_t> used(c.m);
    std::vector<double> score(c.n),xspent(c.n); double qspent=0;
    for(size_t step=0;step<max_l;++step) {
        std::vector<size_t> counts(c.m); std::vector<double> strength(c.m,-1e300);
#pragma omp parallel for schedule(dynamic,1)
        for(size_t p=0;p<pool.size();++p) { const size_t dim=pool[p]; if(used[dim])continue;
            const double qnext=qspent+c.qz[dim]*c.qz[dim]; const double qr=std::sqrt(std::max(0.,c.qnorm2-qnext));
            size_t count=0;double smooth=0; const float* row=c.z.data()+dim*c.n;
#pragma omp simd reduction(+:count,smooth)
            for(size_t i=0;i<c.n;++i){const double x=row[i],delta=c.qz[dim]-x;
                const double gap=qr-std::sqrt(std::max(0.,c.xnorm2[i]-xspent[i]-x*x));
                const double lb2=score[i]+delta*delta+gap*gap;count+=lb2>c.tau2[i];smooth+=lb2;}
            counts[dim]=count;strength[dim]=smooth;
        }
        size_t best=c.m;for(size_t dim:pool)if(!used[dim]&&(best==c.m||counts[dim]>counts[best]||
            (counts[dim]==counts[best]&&(strength[dim]>strength[best]||
             (strength[dim]==strength[best]&&dim<best)))))best=dim;
        if(best==c.m)throw std::runtime_error("greedy pool exhausted");
        used[best]=1;selected.push_back(best);qspent+=c.qz[best]*c.qz[best];const float* row=c.z.data()+best*c.n;
#pragma omp parallel for schedule(static)
        for(size_t i=0;i<c.n;++i){const double x=row[i],delta=c.qz[best]-x;score[i]+=delta*delta;xspent[i]+=x*x;}
        history.push_back(selected);
    }
    return history;
}
static std::vector<size_t> improveSwaps(const Candidates& c,std::vector<size_t> plan,const std::vector<size_t>& pool,size_t rounds) {
    auto current=countPlan(c,plan); std::vector<uint8_t> selected(c.m);for(size_t d:plan)selected[d]=1;
    for(size_t round=0;round<rounds;++round) {
        struct Swap {size_t out=0,in=0,count=0;double strength=-1e300;};
        std::vector<std::pair<size_t,size_t>> pairs;
        for(size_t a=0;a<plan.size();++a)for(size_t d:pool)if(!selected[d])pairs.push_back({a,d});
        std::vector<Swap> result(pairs.size());
#pragma omp parallel for schedule(dynamic,1)
        for(size_t p=0;p<pairs.size();++p) {
            auto [a,in]=pairs[p];std::vector<size_t> proposal=plan;proposal[a]=in;auto value=countPlan(c,proposal);
            result[p]={a,in,value.first,value.second};
        }
        Swap best{0,0,current.first,current.second};
        for(const auto& s:result)if(s.count>best.count||(s.count==best.count&&s.strength>best.strength))best=s;
        if(best.count<current.first||(best.count==current.first&&best.strength<=current.second))break;
        selected[plan[best.out]]=0;plan[best.out]=best.in;selected[best.in]=1;current={best.count,best.strength};
    }
    std::sort(plan.begin(),plan.end());return plan;
}
static Metrics certificates(const Candidates& c,const std::vector<size_t>& plan,const std::vector<size_t>& dictionary) {
    std::vector<uint8_t> chosen(c.m);for(size_t d:plan)chosen[d]=1;
    double qspent=0,qfull=0;for(size_t d:plan)qspent+=c.qz[d]*c.qz[d];for(size_t d:dictionary)qfull+=c.qz[d]*c.qz[d];
    Metrics out;
#pragma omp parallel
    {
        Metrics local;std::vector<double> products;products.reserve(dictionary.size());
#pragma omp for schedule(static)
        for(size_t i=0;i<c.n;++i) {
            double as=0,xspent=0,afull=0,xfull=0;products.clear();
            for(size_t d:dictionary){const double x=c.z[d*c.n+i],delta=c.qz[d]-x;afull+=delta*delta;xfull+=x*x;products.push_back(c.qz[d]*x);
                if(chosen[d]){as+=delta*delta;xspent+=x*x;}}
            const double rqs=std::sqrt(std::max(0.,c.qnorm2-qspent)),rxs=std::sqrt(std::max(0.,c.xnorm2[i]-xspent));
            const double rqm=std::sqrt(std::max(0.,c.qnorm2-qfull)),rxm=std::sqrt(std::max(0.,c.xnorm2[i]-xfull));
            const double bs=as+(rqs-rxs)*(rqs-rxs),us=as+(rqs+rxs)*(rqs+rxs),bm=afull+(rqm-rxm)*(rqm-rxm);
            std::nth_element(products.begin(),products.begin()+plan.size(),products.end());
            double low=0;for(size_t k=0;k<plan.size();++k)low+=products[k];
            const double w=std::max(0.,c.qnorm2+c.xnorm2[i]-2*low-2*rqm*rxm);const double h=std::min(bm,w);
            const double tau=c.tau2[i];local.pruned+=bs>tau;local.alpha+=(bs<=tau&&tau<us);local.beta+=(bs<=tau&&tau<bm);local.gamma+=(bs<=tau&&tau<h);
            local.max_excess=std::max(local.max_excess,bs-c.exact2[i]);local.false_prunes+=(bs>tau&&c.exact2[i]<=tau);
        }
#pragma omp critical
        {out.pruned+=local.pruned;out.alpha+=local.alpha;out.beta+=local.beta;out.gamma+=local.gamma;
         out.false_prunes+=local.false_prunes;out.max_excess=std::max(out.max_excess,local.max_excess);}
    }
    return out;
}

int main(int argc,char** argv) {
 try {
    if(argc!=14)throw std::invalid_argument("Usage: qpivot_regret_eval INDEX QUERIES VISITS DICTIONARY OUTPUT NQ M NPROBE CAPTURE_P BUDGETS SWAP_ROUNDS SEED EXACT_M");
    const size_t nq=std::stoull(argv[6]),m=std::stoull(argv[7]),nprobe=std::stoull(argv[8]),capture_p=std::stoull(argv[9]);
    const auto budgets=parseSizes(argv[10]);const size_t swap_rounds=std::stoull(argv[11]);const uint64_t seed=std::stoull(argv[12]);const size_t exact_m=std::stoull(argv[13]);
    if(!nq||!m||m>511||!nprobe||budgets.empty()||budgets.back()>m||exact_m!=12)throw std::invalid_argument("invalid configuration");
    std::filesystem::path out=argv[5];std::filesystem::create_directories(out);
    Index index;index.load_index(argv[1]);if(index.metric!=MetricType::METRIC_L2)throw std::invalid_argument("requires L2 index");
    auto [queries,available,d]=loadXvecs(argv[2]);if(size_t(d)!=index.d||nq>size_t(available)||m>index.d)throw std::invalid_argument("query mismatch");
    std::ifstream input(argv[3]);std::string line;if(!std::getline(input,line))throw std::runtime_error("missing visits");
    auto header=split(line);std::map<std::string,size_t> col;for(size_t i=0;i<header.size();++i)col[header[i]]=i;
    std::vector<std::vector<Visit>> visits(nq);std::vector<uint8_t> needed(index.nlist);
    while(std::getline(input,line)){auto row=split(line);auto get=[&](const char* k)->const std::string&{return row.at(col.at(k));};
        if(std::stoull(get("nprobe"))!=nprobe||std::stoull(get("pivot_count"))!=capture_p)continue;size_t q=std::stoull(get("query_id"));if(q>=nq)continue;
        Visit v{q,std::stoull(get("list_id")),std::stoull(get("probe_rank")),std::stoull(get("scan_begin")),std::stoull(get("scan_end")),
            std::bit_cast<float>(uint32_t(std::stoul(get("entry_tau2_bits"))))};
        if(v.lid>=index.nlist||v.end>index.lists[v.lid].list_size||v.begin>v.end)throw std::runtime_error("visit mismatch");visits[q].push_back(v);needed[v.lid]=1;
    }
    for(auto& vv:visits){std::sort(vv.begin(),vv.end(),[](auto&a,auto&b){return a.rank<b.rank;});if(vv.size()!=nprobe)throw std::runtime_error("missing visits");}
    std::ifstream dict(argv[4],std::ios::binary);Eigen::VectorXd center(index.d);MatrixD basis(m,index.d);
    dict.read(reinterpret_cast<char*>(center.data()),index.d*sizeof(double));dict.read(reinterpret_cast<char*>(basis.data()),m*index.d*sizeof(double));
    if(!dict||!center.allFinite()||!basis.allFinite())throw std::runtime_error("bad dictionary");
    const MatrixF basis_f=basis.cast<float>();
    std::vector<ListCache> cache(index.nlist);std::cout<<"Projecting float32 candidate signatures for visited lists"<<std::endl;
#pragma omp parallel for schedule(dynamic,1)
    for(size_t lid=0;lid<index.nlist;++lid)if(needed[lid]){auto& list=index.lists[lid];auto& c=cache[lid];const size_t n=list.list_size;
        MatrixF centered(n,index.d);c.global_norm2.resize(n);c.local_norm2.resize(n);
        for(size_t i=0;i<n;++i){double gn=0,ln=0;const float* x=list.get_candidate_codes()+i*index.d;for(size_t a=0;a<index.d;++a){
            const double gd=double(x[a])-center[a],ld=double(x[a])-index.centroid_codes[lid*index.d+a];centered(i,a)=float(gd);gn+=gd*gd;ln+=ld*ld;}c.global_norm2[i]=gn;c.local_norm2[i]=ln;}
        c.z.noalias()=basis_f*centered.transpose();if(n){c.mean=c.z.cast<double>().rowwise().mean();c.variance=(c.z.cast<double>().colwise()-c.mean).array().square().rowwise().mean();}
        else{c.mean=Eigen::VectorXd::Zero(m);c.variance=Eigen::VectorXd::Zero(m);}
    }
    std::vector<size_t> full_pool(m);std::iota(full_pool.begin(),full_pool.end(),0);
    // Fixed before evaluation: four PCA-head directions, four directions with
    // the largest aggregate QE on held-out calibration queries, then four
    // approximately uniform uncovered directions. Evaluation queries [0,nq)
    // never participate in constructing M'.
    std::vector<size_t> small{0,1,2,3};Eigen::VectorXd calibration=Eigen::VectorXd::Zero(m);
    const size_t calibration_end=std::min<size_t>(available,nq+12);
    for(size_t qi=nq;qi<calibration_end;++qi){Eigen::VectorXd q(index.d);for(size_t a=0;a<index.d;++a)q[a]=double(queries[qi*index.d+a])-center[a];calibration+=(basis*q).array().square().matrix();}
    if(calibration_end>nq){auto order=ranked(m,[&](size_t x){return calibration[x];});for(size_t x:order)if(std::find(small.begin(),small.end(),x)==small.end()){small.push_back(x);if(small.size()==8)break;}}
    for(size_t k=0;small.size()<exact_m&&k<4;++k){const size_t target=4+(m-5)*k/3;for(size_t delta=0;delta<m;++delta){const size_t x=std::min(m-1,target+delta);if(std::find(small.begin(),small.end(),x)==small.end()){small.push_back(x);break;}}}
    std::mt19937_64 rng(seed);std::vector<size_t> shuffled;for(size_t x=4;x<m;++x)if(std::find(small.begin(),small.end(),x)==small.end())shuffled.push_back(x);std::shuffle(shuffled.begin(),shuffled.end(),rng);
    for(size_t x:shuffled){if(small.size()==exact_m)break;small.push_back(x);}std::sort(small.begin(),small.end());if(small.size()!=exact_m)throw std::runtime_error("small dictionary failure");
    std::ofstream full(out/"full_results.csv"),exact(out/"exact_results.csv"),orders(out/"orders.csv"),safety(out/"safety.csv");
    const char* mh="query_id,budget,method,candidates,pruned,prune_rate,alpha,beta,gamma,regret,exact_pruned,subset\n";full<<mh;exact<<mh;
    orders<<"query_id,dictionary,budget,method,subset\n";safety<<"query_id,false_prunes,max_lb_excess,certificate_violations\n";
    full<<std::setprecision(17);exact<<std::setprecision(17);size_t total_false=0,total_violations=0;double overall_excess=-1e300;
    for(size_t qi=0;qi<nq;++qi){const auto& vv=visits[qi];Eigen::VectorXd qvec(index.d);for(size_t a=0;a<index.d;++a)qvec[a]=double(queries[qi*index.d+a])-center[a];
        Candidates c;c.m=m;c.qnorm2=qvec.squaredNorm();c.qz=basis*qvec;size_t capacity=0;for(const auto&v:vv)capacity+=v.end-v.begin;
        c.z.reserve(m*capacity);c.xnorm2.reserve(capacity);c.tau2.reserve(capacity);c.exact2.reserve(capacity);
        Eigen::VectorXd mm=Eigen::VectorXd::Zero(m),ms=Eigen::VectorXd::Zero(m),zs=Eigen::VectorXd::Zero(m);double total_weight=0,variance_scale=0;
        for(const auto&v:vv){const double w=v.end-v.begin;total_weight+=w;variance_scale+=w*cache[v.lid].variance.mean();}
        variance_scale=std::max(variance_scale/std::max(1.,total_weight),1e-30);const double eps=1e-6*variance_scale;
        for(const auto&v:vv){const double w=(v.end-v.begin)/std::max(1.,total_weight);const auto& lc=cache[v.lid];mm+=w*(c.qz-lc.mean).array().square().matrix();ms+=w*((c.qz-lc.mean).array().square()+lc.variance.array()).matrix();zs+=w*((c.qz-lc.mean).array().square()/(lc.variance.array()+eps)).matrix();}
        for(const auto&v:vv){const auto&lc=cache[v.lid];const auto&list=index.lists[v.lid];double local_q=0;for(size_t a=0;a<index.d;++a){double x=double(queries[qi*index.d+a])-index.centroid_codes[v.lid*index.d+a];local_q+=x*x;}
            for(size_t i=v.begin;i<v.end;++i){const double tri=std::pow(std::sqrt(local_q)-std::sqrt(lc.local_norm2[i]),2);if(tri>v.tau2)continue;
                c.xnorm2.push_back(lc.global_norm2[i]);c.tau2.push_back(v.tau2);double dist=0;const float* x=list.get_candidate_codes()+i*index.d;for(size_t a=0;a<index.d;++a){double delta=double(queries[qi*index.d+a])-x[a];dist+=delta*delta;}c.exact2.push_back(dist);
                for(size_t dim=0;dim<m;++dim)c.z.push_back(lc.z(dim,i));
            }}
        c.n=c.xnorm2.size(); // transpose candidate-major append into direction-major
        std::vector<float> soa(m*c.n);for(size_t i=0;i<c.n;++i)for(size_t dim=0;dim<m;++dim)soa[dim*c.n+i]=c.z[i*m+dim];c.z.swap(soa);
        auto qe=ranked(m,[&](size_t x){return c.qz[x]*c.qz[x];});auto mm_order=ranked(m,[&](size_t x){return mm[x];});auto ms_order=ranked(m,[&](size_t x){return ms[x];});auto z_order=ranked(m,[&](size_t x){return zs[x];});
        auto greedy=greedyPath(c,budgets.back(),full_pool);std::map<std::string,std::vector<size_t>> base{{"qe",qe},{"mm",mm_order},{"ms",ms_order},{"zscore",z_order}};
        for(size_t L:budgets){std::vector<size_t> prefix(L);std::iota(prefix.begin(),prefix.end(),0);std::map<std::string,std::vector<size_t>> plans{{"pca",prefix},{"greedy",greedy[L-1]}};
            for(auto&[name,o]:base)plans[name]=std::vector<size_t>(o.begin(),o.begin()+L);plans["greedy_swap"]=improveSwaps(c,greedy[L-1],full_pool,swap_rounds);
            for(auto&[name,plan]:plans){auto x=certificates(c,plan,full_pool);full<<qi<<','<<L<<','<<name<<','<<c.n<<','<<x.pruned<<','<<double(x.pruned)/c.n<<','<<double(x.alpha)/c.n<<','<<double(x.beta)/c.n<<','<<double(x.gamma)/c.n<<",nan,0,\""<<join(plan)<<"\"\n";
                orders<<qi<<",full,"<<L<<','<<name<<",\""<<join(plan)<<"\"\n";total_false+=x.false_prunes;overall_excess=std::max(overall_excess,x.max_excess);}
        }
        auto small_greedy=greedyPath(c,6,small);std::map<std::string,std::vector<size_t>> small_orders;
        for(auto&[name,o]:base){small_orders[name]=small;std::stable_sort(small_orders[name].begin(),small_orders[name].end(),[&](size_t a,size_t b){const auto& source=name=="qe"?qe:name=="mm"?mm_order:name=="ms"?ms_order:z_order;auto pa=std::find(source.begin(),source.end(),a),pb=std::find(source.begin(),source.end(),b);return pa<pb;});}
        for(size_t L=2;L<=6;++L){std::vector<size_t> best;size_t best_count=0;double best_strength=-1;
            std::vector<size_t> mask(L);std::iota(mask.begin(),mask.end(),0);while(true){std::vector<size_t> plan;for(size_t x:mask)plan.push_back(small[x]);auto value=countPlan(c,plan);if(value.first>best_count||(value.first==best_count&&value.second>best_strength)){best=plan;best_count=value.first;best_strength=value.second;}
                int p=int(L)-1;while(p>=0&&mask[p]==small.size()-L+p)--p;if(p<0)break;++mask[p];for(size_t k=p+1;k<L;++k)mask[k]=mask[k-1]+1;}
            std::vector<size_t> prefix(small.begin(),small.begin()+L);std::map<std::string,std::vector<size_t>> plans{{"pca",prefix},{"greedy",small_greedy[L-1]},{"exact",best}};
            for(auto&[name,o]:small_orders)plans[name]=std::vector<size_t>(o.begin(),o.begin()+L);plans["greedy_swap"]=improveSwaps(c,small_greedy[L-1],small,swap_rounds);
            for(auto&[name,plan]:plans){auto x=certificates(c,plan,small);const double regret=double(best_count-x.pruned)/c.n;const size_t violations=(regret>x.gamma/double(c.n)+1e-12)||(x.gamma>x.beta)||(x.beta>x.alpha);total_violations+=violations;
                exact<<qi<<','<<L<<','<<name<<','<<c.n<<','<<x.pruned<<','<<double(x.pruned)/c.n<<','<<double(x.alpha)/c.n<<','<<double(x.beta)/c.n<<','<<double(x.gamma)/c.n<<','<<regret<<','<<best_count<<",\""<<join(plan)<<"\"\n";
                orders<<qi<<",small,"<<L<<','<<name<<",\""<<join(plan)<<"\"\n";total_false+=x.false_prunes;overall_excess=std::max(overall_excess,x.max_excess);}
        }
        safety<<qi<<','<<total_false<<','<<overall_excess<<','<<total_violations<<'\n';full.flush();exact.flush();orders.flush();safety.flush();
        std::cout<<"query="<<qi<<" triangle_survivors="<<c.n<<" false_prunes_total="<<total_false<<" certificate_violations_total="<<total_violations<<std::endl;
    }
    std::ofstream meta(out/"native_summary.txt");meta<<"small_dictionary="<<join(small)<<"\nfalse_prunes="<<total_false<<"\nmax_lb_excess="<<overall_excess<<"\ncertificate_violations="<<total_violations<<'\n';
    if(total_violations)throw std::runtime_error("certificate inequality violation");if(total_false)throw std::runtime_error("false pruning detected");
 } catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}return 0;
}
