#include "Index.h"
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

using namespace tribase;
using Clock=std::chrono::steady_clock;

// Offline comparison only. Candidate sets and list-entry thresholds come from
// one native Triangle run; all selectors then consume identical stored rows.
struct Domain {
    size_t n=0,m=0;
    double qnorm=0;
    std::vector<double> qz,norm,tau,exact;
    std::vector<float> z;
};
struct State { std::vector<double> score,energy; double qenergy=0; };

static void extend(const Domain& c,State& state,size_t dim) {
    state.qenergy+=c.qz[dim]*c.qz[dim];
    const float* row=c.z.data()+dim*c.n;
#pragma omp parallel for simd schedule(static)
    for(size_t i=0;i<c.n;++i) {
        const double x=row[i],delta=c.qz[dim]-x;
        state.score[i]+=delta*delta;state.energy[i]+=x*x;
    }
}
static std::pair<size_t,double> evaluate(const Domain& c,const State& state,size_t dim) {
    const double qr=std::sqrt(std::max(0.,c.qnorm-state.qenergy-c.qz[dim]*c.qz[dim]));
    const float* row=c.z.data()+dim*c.n;
    size_t count=0;double strength=0;
#pragma omp simd reduction(+:count,strength)
    for(size_t i=0;i<c.n;++i) {
        const double x=row[i],delta=c.qz[dim]-x;
        const double gap=qr-std::sqrt(std::max(0.,c.norm[i]-state.energy[i]-x*x));
        const double lb=state.score[i]+delta*delta+gap*gap;
        count+=lb>c.tau[i];strength+=lb;
    }
    return {count,strength};
}
static void emit(const Domain& c,const State& state,size_t qi,size_t total,size_t triangle,
                 const std::string& method,const std::vector<size_t>& order,double seconds,
                 std::ofstream& output,std::ofstream& orders) {
    const double qr=std::sqrt(std::max(0.,c.qnorm-state.qenergy));
    size_t count=0,false_prunes=0;
#pragma omp parallel for reduction(+:count,false_prunes)
    for(size_t i=0;i<c.n;++i) {
        const double gap=qr-std::sqrt(std::max(0.,c.norm[i]-state.energy[i]));
        const double bound=state.score[i]+gap*gap;
        count+=bound>c.tau[i];
        // A scaled roundoff margin distinguishes material unsafe pruning from
        // float32 projection/norm rounding at a nearly exact threshold.
        false_prunes+=bound>c.tau[i]&&c.exact[i]<=c.tau[i]&&
            bound-c.exact[i]>2e-5*std::max(1.,c.exact[i]);
    }
    output<<qi<<','<<method<<','<<order.size()<<','<<order.size()+1<<','<<total<<','
          <<triangle<<','<<c.n<<','<<count<<','<<c.n-count<<','
          <<(c.n?double(count)/c.n:0.)<<','<<(total?double(triangle+count)/total:0.)
          <<','<<seconds<<','<<false_prunes<<'\n';
    orders<<qi<<','<<method<<','<<order.size()<<",\"";
    for(size_t i=0;i<order.size();++i){if(i)orders<<' ';orders<<order[i];}
    orders<<"\"\n";
    if(false_prunes)throw std::runtime_error("material unsafe lower bound detected");
}

int main(int argc,char** argv) {
 try {
#ifndef TRIBASE_ENABLE_STATS
    throw std::invalid_argument("qpivot_order_eval requires ENABLE_STATS=ON to freeze native list-entry thresholds");
#endif
    if(argc!=8)throw std::invalid_argument("Usage: qpivot_order_eval INDEX QUERIES OUTPUT NQ NPROBE BUDGETS_CSV NORMALIZE_0_OR_1");
    const size_t nq=std::stoull(argv[4]),probes=std::stoull(argv[5]);
    std::vector<size_t> budgets;std::stringstream stream(argv[6]);std::string cell;
    while(std::getline(stream,cell,','))budgets.push_back(std::stoull(cell));
    std::sort(budgets.begin(),budgets.end());budgets.erase(std::unique(budgets.begin(),budgets.end()),budgets.end());
    Index index;index.load_index(argv[1]);
    if(index.metric!=METRIC_L2||!(index.added_opt_level&OPT_TRIANGLE)||
       index.multipivot_scope!=MultiPivotScope::GLOBAL||index.multipivot_method!="pca"||
       index.global_pivots.count<2||!index.global_pivots.has_U(index.global_pivots.count))
        throw std::invalid_argument("requires rich Global PCA Triangle index");
    const size_t m=index.global_pivots.count-1;
    if(!nq||!probes||probes>index.nlist||budgets.empty()||!budgets.front()||budgets.back()>m)
        throw std::invalid_argument("invalid nq/nprobe/budgets");
    auto [queries,available,d]=loadXvecs(argv[2]);
    if(size_t(d)!=index.d||nq>size_t(available))throw std::invalid_argument("query mismatch");
    if(std::stoi(argv[7]))normalizeL2InPlace(queries.get(),nq,index.d);
    index.build_projection_moments();
    index.nprobe=probes;index.opt_level=OPT_TRIANGLE;index.set_multipivot_search(MultiPivotMode::NONE,0);
    index.collect_list_stats=true;
    std::vector<float> distances(nq);std::vector<idx_t> labels(nq);
    auto stats=index.search(nq,queries.get(),1,distances.data(),labels.data());
    if(stats.list_visits.size()!=nq*probes)throw std::runtime_error("incomplete native visits");
    std::sort(stats.list_visits.begin(),stats.list_visits.end(),[](const auto&a,const auto&b){
        return std::pair(a.query_id,a.probe_rank)<std::pair(b.query_id,b.probe_rank);});
    const std::filesystem::path out(argv[3]);std::filesystem::create_directories(out);
    listVisitsToCsv((out/"visits.csv").string(),stats.list_visits,false,"frozen_triangle",index.nlist,probes,
                   OPT_TRIANGLE,MultiPivotMode::NONE,MultiPivotScope::GLOBAL,"pca",m+1,index.multipivot_seed,1.);
    std::ofstream results(out/"results.csv"),orders(out/"orders.csv");
    results<<std::setprecision(17)<<"query_id,method,directions,pivots,total_candidates,triangle_pruned,triangle_survivors,mp_pruned,exact_survivors,conditional_prune_rate,overall_prune_rate,selection_seconds,false_prunes\n";
    orders<<"query_id,method,directions,order\n";
    for(size_t qi=0;qi<nq;++qi) {
        Domain c;c.m=m;c.qz.resize(m);const float* q=queries.get()+qi*index.d;
        std::vector<double> sq(m+1);std::vector<float> z(m),centered(index.d);float residue=0;
        if(!computePcaQuerySignature(q,index.d,index.global_pivots,m+1,sq.data(),z.data(),&residue,centered.data()))
            throw std::runtime_error("query signature failed");
        c.qnorm=sq[0];std::copy(z.begin(),z.end(),c.qz.begin());
        size_t total=0,triangle=0;std::vector<double> mm(m);
        for(size_t rank=0;rank<probes;++rank) {
            const auto& visit=stats.list_visits[qi*probes+rank];
            if(visit.query_id!=qi||visit.probe_rank!=rank)throw std::runtime_error("visit order mismatch");
            const auto& list=index.lists[visit.list_id];total+=list.list_size;
            triangle+=visit.tri+visit.tri_large;c.n+=visit.scan_end-visit.scan_begin;
            for(size_t dim=0;dim<m;++dim) {
                const double delta=c.qz[dim]-index.projection_list_means[visit.list_id*m+dim];
                mm[dim]+=list.list_size*delta*delta;
            }
        }
        if(total!=triangle+c.n)throw std::runtime_error("Triangle accounting mismatch");
        c.z.resize(m*c.n);c.norm.reserve(c.n);c.tau.reserve(c.n);c.exact.reserve(c.n);
        size_t offset=0;
        for(size_t rank=0;rank<probes;++rank) {
            const auto& v=stats.list_visits[qi*probes+rank];const auto& list=index.lists[v.list_id];
            if(!list.list_size)continue;
            const auto& meta=list.pivots;
            if(meta.signature_precision!=SignaturePrecision::FLOAT32||!meta.usable_signature(m+1))
                throw std::runtime_error("missing float32 candidate signature");
            for(size_t i=v.scan_begin;i<v.scan_end;++i) {
                c.norm.push_back(meta.candidate_centroid_squared[i]);c.tau.push_back(v.entry_tau2);
                const float* x=list.get_candidate_codes()+i*index.d;double dist=0;
                for(size_t a=0;a<index.d;++a){double delta=double(q[a])-x[a];dist+=delta*delta;}
                c.exact.push_back(dist);
            }
            for(size_t dim=0;dim<m;++dim)
                std::copy(meta.signature_data.begin()+dim*list.list_size+v.scan_begin,
                          meta.signature_data.begin()+dim*list.list_size+v.scan_end,
                          c.z.begin()+dim*c.n+offset);
            offset+=v.scan_end-v.scan_begin;
        }
        std::cout<<"query="<<qi<<" candidates="<<total<<" triangle_survivors="<<c.n<<" M="<<m<<std::endl;
        for(const std::string method:{"pca","mm","greedy"}) {
            const auto start=Clock::now();State state{std::vector<double>(c.n),std::vector<double>(c.n),0};
            std::vector<size_t> order,ranking(m);std::iota(ranking.begin(),ranking.end(),0);
            if(method=="mm")std::stable_sort(ranking.begin(),ranking.end(),[&](size_t a,size_t b){return mm[a]>mm[b];});
            std::vector<uint8_t> used(m);
            for(size_t step=0;step<budgets.back();++step) {
                size_t dim=ranking[step];
                if(method=="greedy") {
                    std::vector<std::pair<size_t,double>> values(m);
#pragma omp parallel for schedule(dynamic,1)
                    for(size_t row=0;row<m;++row)if(!used[row])values[row]=evaluate(c,state,row);
                    dim=m;
                    for(size_t row=0;row<m;++row)if(!used[row]&&(dim==m||values[row]>values[dim]))dim=row;
                }
                used[dim]=1;order.push_back(dim);extend(c,state,dim);
                if(std::binary_search(budgets.begin(),budgets.end(),step+1)) {
                    emit(c,state,qi,total,triangle,method,order,
                         std::chrono::duration<double>(Clock::now()-start).count(),results,orders);
                    std::cout<<"query="<<qi<<" method="<<method<<" L="<<step+1<<std::endl;
                    results.flush();orders.flush();
                }
            }
        }
    }
    if(!results||!orders)throw std::runtime_error("failed to write results");
    std::ofstream(out/"complete")<<"ok\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
