#include "Index.h"
#include "IVFScan.hpp"
#include <omp.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <atomic>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <cstdlib>

using namespace tribase;
using Clock=std::chrono::steady_clock;

static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;std::stringstream stream(line);std::string value;
    while (std::getline(stream,value,',')) {if (!value.empty()&&value.back()=='\r')value.pop_back();out.push_back(value);}
    return out;
}
struct Visit {
    size_t qid,rank,lid,begin,end,selected,ready,configured,nprobe;
    float tau,coarse;
    idx_t heap_id;
    bool full_ready,valid,used_prefix,forced;
    size_t scan_limit,seed_dims,block_size;int scan_mode;double prior;
    double predicted;
};
struct Timing {size_t p;double ns,cv;size_t checks,pruned;idx_t id;size_t final_p=0;};

int main(int argc,char** argv) {
    try {
        if(argc!=6) {std::cerr<<"Usage: replay_pivot_choice INDEX QUERY_FVECS VISITS_CSV OUTPUT_CSV RANK_SAMPLES\n";return 2;}
        Index index;index.load_index(argv[1]);
        if(index.metric!=MetricType::METRIC_L2 || index.multipivot_method!="pca")
            throw std::invalid_argument("Replay requires a PCA/L2 index");
        auto [queries,nq,dimension]=loadXvecs(argv[2]);
        if(size_t(dimension)!=index.d)throw std::invalid_argument("query dimension mismatch");
        const size_t rank_samples=std::stoull(argv[5]);
        if(rank_samples<2)throw std::invalid_argument("need at least two rank samples");
        std::ifstream input(argv[3]);std::string line;
        if(!std::getline(input,line))throw std::invalid_argument("missing visits CSV");
        auto header=split(line);std::map<std::string,size_t> column;
        for(size_t i=0;i<header.size();++i)column[header[i]]=i;
        std::vector<Visit> visits;
        // Optional query-global MM permutation captured by the production search.
        // Coordinates remain in the persisted bank's canonical row numbering.
        std::map<size_t,std::vector<size_t>> orders;
        if(const char* path=std::getenv("TRIBASE_REPLAY_ORDERS")) {
            std::ifstream order_input(path);std::string row;
            if(!std::getline(order_input,row))throw std::runtime_error("missing orders file");
            const std::string method=std::getenv("TRIBASE_REPLAY_METHOD")?std::getenv("TRIBASE_REPLAY_METHOD"):"qpivot";
            while(std::getline(order_input,row)) {
                const auto fields=split(row);
                if(fields.at(0)!=method || fields.at(1)!="0")continue;
                auto& permutation=orders[std::stoull(fields.at(2))];
                std::istringstream values(fields.at(4));size_t dim;
                while(values>>dim)permutation.push_back(dim);
                auto sorted=permutation;std::sort(sorted.begin(),sorted.end());
                if(sorted.size()+1!=index.global_pivots.count)throw std::runtime_error("incomplete MM permutation");
                for(size_t i=0;i<sorted.size();++i)if(sorted[i]!=i)throw std::runtime_error("invalid MM permutation");
            }
            if(orders.size()!=size_t(nq))throw std::runtime_error("missing query orders");
        }
        const bool exhaustive=std::getenv("TRIBASE_REPLAY_EXHAUSTIVE")!=nullptr;
        while(std::getline(input,line)) {
            auto cells=split(line);
            auto value=[&](const char* name)->const std::string& {return cells.at(column.at(name));};
            const size_t rank=std::stoull(value("probe_rank")),np=std::stoull(value("nprobe"));
            bool take=false;
            for(size_t s=0;s<rank_samples;++s)take|=rank==s*(np-1)/(rank_samples-1);
            if(!take)continue;
            Visit v;
            v.qid=std::stoull(value("query_id"));v.rank=rank;v.nprobe=np;v.lid=std::stoull(value("list_id"));
            v.begin=std::stoull(value("scan_begin"));v.end=std::stoull(value("scan_end"));
            v.selected=std::stoull(value("active_pivots"));v.ready=std::stoull(value("entry_ready_dims"));
            v.configured=std::stoull(value("pivot_count"));v.full_ready=std::stoul(value("entry_full_signature_ready"));
            v.tau=std::bit_cast<float>(static_cast<uint32_t>(std::stoul(value("entry_tau2_bits"))));
            v.coarse=std::bit_cast<float>(static_cast<uint32_t>(std::stoul(value("centroid2query_bits"))));
            v.heap_id=std::stoll(value("entry_heap_id"));v.valid=std::stoul(value("entry_prior_valid"));
            v.predicted=std::stod(value("predicted_prune_rate"));
            v.scan_limit=std::stoull(value("scan_pivot_limit"));v.seed_dims=std::stoull(value("progressive_seed_dims"));
            v.scan_mode=std::stoi(value("effective_progressive_mode"));v.used_prefix=std::stoul(value("used_dynamic_prefix"));
            v.prior=std::stod(value("entry_log_survival"));
            v.block_size=std::stoull(value("projection_block_size"));
            v.forced=std::stoul(value("forced_exploration"));
            if(v.qid>=size_t(nq)||v.lid>=index.nlist||v.begin>v.end||v.end>index.lists[v.lid].list_size)
                throw std::invalid_argument("invalid visit bounds");
            if(v.end>v.begin)visits.push_back(v);
        }
        std::ofstream output(argv[4]);std::ofstream curves(std::string(argv[4])+".costs.csv");
        if(!output||!curves)throw std::runtime_error("cannot open replay output");
        output<<"nlist,nprobe,query_id,probe_rank,list_id,list_size,survivors,entry_tau2,selected_P,best_P,best_P_within_5pct_min,best_P_within_5pct_max,selected_ns,best_ns,time_regret,selected_time_cv,best_time_cv,pivot_gap,predicted_prune_rate,actual_entry_prune_rate,actual_replay_prune_rate,prediction_error,entry_ready_dims,entry_prior_valid,neighbor_match,actual_policy_ns,actual_policy_time_cv,actual_policy_regret,actual_policy_selected_P,actual_policy_prune_rate,forced_exploration\n";
        curves<<"nprobe,query_id,probe_rank,list_id,P,median_ns,time_cv,pruned,checks,actual_entry_prune_rate,timing_level\n";
        std::mutex mutex;
        std::atomic<size_t> errors{0};
#pragma omp parallel
        {
            auto scanner=std::make_unique<IVFScan<MetricType::METRIC_L2,OptLevel::OPT_TRIANGLE,EdgeDevice::EDGEDEVIVE_DISABLED>>(index.d,1);
#pragma omp for schedule(dynamic,1)
            for(size_t vi=0;vi<visits.size();++vi) {
                const Visit& v=visits[vi];auto& list=index.lists[v.lid];
                const auto& geometry=index.multipivot_scope==MultiPivotScope::GLOBAL?index.global_pivots:list.pivots;
                const size_t maximum=std::min(v.configured,geometry.count);
                if(!geometry.has_U(maximum)||maximum<2||v.selected>maximum) {++errors;continue;}
                const float* q=queries.get()+v.qid*index.d;
                const size_t* direction_order=orders.empty()?nullptr:orders.at(v.qid).data();
                scanner->projection_order=direction_order;
                scanner->set_query(q);scanner->progressive_mode=0;scanner->progressive_geometry=&geometry;
                std::vector<float> full_z(maximum),z(maximum),centered(index.d);
                std::vector<double> qsq(maximum);
                float full_rho=0;
                computePcaQuerySignature(q,index.d,geometry,maximum,qsq.data(),full_z.data(),&full_rho,centered.data());
                if(direction_order) {
                    // QPivot materializes the complete bank and its ordered
                    // full residual before entering the first list. Cache it
                    // here too; prefix residuals are computed by scan_codes.
                    float remaining=static_cast<float>(qsq[0]);
                    for(size_t k=0;k+1<maximum;++k) {
                        const float qz=full_z[direction_order[k]];
                        remaining=std::max(0.0f,remaining-qz*qz);
                    }
                    full_rho=std::sqrt(remaining);
                }
                if(index.multipivot_scope==MultiPivotScope::PER_LIST)qsq[0]=v.coarse;
                std::unique_ptr<bool[]> skip(new bool[list.list_size+1]());
                std::vector<size_t> frozen_pruned(maximum+1,0);
                // One incremental pass gives the frozen-entry LB curve for every P.
                for(size_t ci=v.begin;ci<v.end;++ci) {
                    float hx2=list.pivots.candidate_centroid_squared[ci],hq2=qsq[0],score=0;
                    for(size_t p=1;p<=maximum;++p) {
                        if(p>1) {
                            const size_t dim=direction_order?direction_order[p-2]:p-2;
                            const float xz=list.pivots.signature_data[dim*list.list_size+ci];
                            const float delta=full_z[dim]-xz;score+=delta*delta;
                            hq2=std::max(0.0f,hq2-full_z[dim]*full_z[dim]);
                            hx2=std::max(0.0f,hx2-xz*xz);
                        }
                        const float gap=std::sqrt(hq2)-std::sqrt(hx2);
                        if(score+gap*gap>v.tau)++frozen_pruned[p];
                    }
                }
                auto once=[&](size_t p,bool policy=false)->Timing {
                    scanner->progressive_mode=policy?v.scan_mode:0;
                    scanner->progressive_prior=v.prior;scanner->progressive_seed_dims=policy?v.seed_dims:0;
                    std::copy(full_z.begin(),full_z.end(),z.begin());
                    size_t ready=index.multipivot_scope==MultiPivotScope::GLOBAL?v.ready:0;
                    float radius=v.tau;idx_t id=v.heap_id;float rho=full_rho;
                    const auto start=Clock::now();
                    const bool prefix=policy?v.used_prefix:(p && p<maximum);
                    if(p && (!policy || !v.scan_mode) && (!prefix || index.multipivot_scope==MultiPivotScope::GLOBAL)) {
                        if(!prefix && (index.multipivot_scope==MultiPivotScope::PER_LIST || !v.full_ready)) {
                            computePcaQuerySignature(q,index.d,geometry,p,qsq.data(),z.data(),&rho,centered.data());
                        } else if(prefix) {
                            for(size_t dim=ready;dim+1<p;++dim) {
                                float projection=0;
#pragma omp simd reduction(+:projection)
                                for(size_t axis=0;axis<index.d;++axis)projection+=centered[axis]*geometry.U[dim*index.d+axis];
                                z[dim]=projection;
                            }
                        }
                        ready=std::max(ready,p-1);
                    }
                    if(index.multipivot_scope==MultiPivotScope::PER_LIST && prefix)qsq[0]=v.coarse;
                    scanner->scan_codes(v.begin,v.end,list.list_size,list.get_candidate_codes(),list.get_candidate_id(),list.get_candidate_norms(),
                        v.coarse,list.get_candidate2centroid(),list.get_sqrt_candidate2centroid(),index.sub_k,
                        list.get_sub_nearest_IP_id(),list.get_sub_nearest_IP_dis(),list.get_sub_farest_IP_id(),list.get_sub_farest_IP_dis(),
                        list.get_sub_nearest_L2_id(),list.get_sub_nearest_L2_dis(),&list.pivots,qsq.data(),p,
                        p?MultiPivotMode::PROJECTION:MultiPivotMode::NONE,z.data(),rho,0,prefix?v.block_size:0,prefix,&ready,true,
                        skip.get(),&radius,&id,nullptr,index.centroid_codes.get()+v.lid*index.d,1,1);
                    const double ns=std::chrono::duration<double,std::nano>(Clock::now()-start).count();
                    return {p,ns,0,scanner->last_multipivot_checks,scanner->last_multipivot_pruned,id,scanner->progressive_mode?(scanner->last_progressive_dims?scanner->last_progressive_dims+1:0):p};
                };
                std::map<size_t,Timing> measured;
                const idx_t expected_id=once(0).id;
                auto measure=[&](size_t p) {
                    if(measured.contains(p))return;
                    const Timing warm=once(p);
                    const size_t repeats=std::clamp<size_t>(size_t(30000/std::max(warm.ns,1.0)),3,9);
                    std::vector<double> samples;Timing last=warm;
                    for(size_t i=0;i<repeats;++i) {last=once(p);samples.push_back(last.ns);}
                    const double mean=std::accumulate(samples.begin(),samples.end(),0.0)/samples.size();
                    double variance=0;for(double n:samples)variance+=(n-mean)*(n-mean);
                    std::sort(samples.begin(),samples.end());last.ns=samples.size()%2?samples[samples.size()/2]:.5*(samples[samples.size()/2-1]+samples[samples.size()/2]);
                    last.cv=std::sqrt(variance/samples.size())/mean;
                    if(last.id!=expected_id) {
                        const size_t number=errors.fetch_add(1);
                        if(number<5) {std::lock_guard<std::mutex> guard(mutex);std::cerr<<"Prefix mismatch q="<<v.qid<<" rank="<<v.rank<<" list="<<v.lid<<" P="<<p<<" expected="<<expected_id<<" actual="<<last.id<<'\n';}
                    }
                    measured.emplace(p,last);
                };
                std::vector<size_t> coarse{0,1,maximum,v.selected};
                const size_t step=std::max<size_t>(8,(maximum+31)/32);
                for(size_t p=9;p<maximum;p+=step)coarse.push_back(p);
                if(exhaustive)for(size_t p=0;p<=maximum;++p)coarse.push_back(p);
                std::mt19937 random(vi+19);std::shuffle(coarse.begin(),coarse.end(),random);
                for(size_t p:coarse)measure(p);
                const auto compare=[](const auto& a,const auto& b){return a.second.ns<b.second.ns;};
                const size_t peak=std::min_element(measured.begin(),measured.end(),compare)->first;
                const size_t lo=peak>step?peak-step:0,hi=std::min(maximum,peak+step);
                std::vector<size_t> refined;
                for(size_t p=lo;p<=hi;++p)refined.push_back(p);
                std::shuffle(refined.begin(),refined.end(),random);for(size_t p:refined)measure(p);
                // Final choice uses longer batches, independent of noisy screening
                // minima. Only the five best screened P plus chosen P/zero are
                // retimed, preserving the low-volume audit budget.
                std::vector<Timing> screened;
                for(const auto& [p,t]:measured)screened.push_back(t);
                std::sort(screened.begin(),screened.end(),[](const Timing& a,const Timing& b){return a.ns<b.ns;});
                std::set<size_t> shortlist{0,v.selected};
                if(exhaustive)for(size_t p=0;p<=maximum;++p)shortlist.insert(p);
                for(size_t i=0;i<std::min<size_t>(5,screened.size());++i)shortlist.insert(screened[i].p);
                auto batched=[&](size_t p,bool actual)->Timing {
                    Timing last=once(p,actual);
                    const size_t batch=std::clamp<size_t>(size_t(100000/std::max(last.ns,1.0)),1,64);
                    std::vector<double> means;
                    for(size_t block=0;block<7;++block) {
                        double sum=0;
                        for(size_t r=0;r<batch;++r){last=once(p,actual);sum+=last.ns;}
                        means.push_back(sum/batch);
                    }
                    const double mean=std::accumulate(means.begin(),means.end(),0.0)/means.size();
                    double variance=0;for(double t:means)variance+=(t-mean)*(t-mean);
                    std::sort(means.begin(),means.end());last.ns=means[means.size()/2];
                    last.cv=std::sqrt(variance/means.size())/mean;
                    return last;
                };
                std::map<size_t,Timing> final_times;
                std::vector<size_t> final_order(shortlist.begin(),shortlist.end());
                std::shuffle(final_order.begin(),final_order.end(),random);
                for(size_t p:final_order) {
                    const auto t=batched(p,false);final_times[p]=t;measured[p]=t;
                    if(t.id!=expected_id)++errors;
                }
                const Timing best=std::min_element(final_times.begin(),final_times.end(),compare)->second;
                const Timing chosen=final_times.at(v.selected);
                Timing policy=batched(v.scan_limit,true);
                if(policy.id!=expected_id || policy.final_p!=v.selected) {
                    const size_t number=errors.fetch_add(1);
                    if(number<5) {std::lock_guard<std::mutex> guard(mutex);std::cerr<<"Policy mismatch q="<<v.qid<<" rank="<<v.rank<<" list="<<v.lid<<" selected="<<v.selected<<" actualP="<<policy.final_p<<" expectedID="<<expected_id<<" actualID="<<policy.id<<'\n';}
                }
                size_t epsilon_min=maximum,epsilon_max=0;
                for(auto& [p,t]:final_times)if(t.ns<=best.ns*1.05){epsilon_min=std::min(epsilon_min,p);epsilon_max=std::max(epsilon_max,p);}
                const double R=v.end-v.begin,actual=frozen_pruned[v.selected]/R;
                const double replay_rate=chosen.checks?double(chosen.pruned)/chosen.checks:0;
                std::ostringstream rows,curve;rows<<std::setprecision(17);curve<<std::setprecision(17);
                rows<<index.nlist<<','<<v.nprobe<<','<<v.qid<<','<<v.rank<<','<<v.lid<<','<<list.list_size<<','<<R<<','<<v.tau<<','
                    <<v.selected<<','<<best.p<<','<<epsilon_min<<','<<epsilon_max<<','<<chosen.ns<<','<<best.ns<<','
                    <<chosen.ns/best.ns-1<<','<<chosen.cv<<','<<best.cv<<','<<long(v.selected)-long(best.p)<<','
                    <<v.predicted<<','<<actual<<','<<replay_rate<<','<<(v.valid?v.predicted-actual:std::numeric_limits<double>::quiet_NaN())<<','
                    <<v.ready<<','<<v.valid<<','<<(chosen.id==expected_id)<<','<<policy.ns<<','<<policy.cv<<','<<policy.ns/best.ns-1<<','<<policy.final_p<<','<<(policy.checks?double(policy.pruned)/policy.checks:0)<<','<<v.forced<<'\n';
                for(auto& [p,t]:measured)curve<<v.nprobe<<','<<v.qid<<','<<v.rank<<','<<v.lid<<','<<p<<','<<t.ns<<','<<t.cv<<','<<t.pruned<<','<<t.checks<<','<<frozen_pruned[p]/R<<','<<(shortlist.contains(p)?"batched_final":"screening")<<'\n';
                std::lock_guard<std::mutex> guard(mutex);output<<rows.str();curves<<curve.str();
            }
        }
        if(errors)throw std::runtime_error("replay metadata or neighbor-ID validation failed: "+std::to_string(errors.load()));
        std::cout<<"Replayed "<<visits.size()<<" complete query-list visits; every tested prefix matches the exact replay heap ID\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
