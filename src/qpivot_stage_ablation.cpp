// Controlled query-only ablation on one resident Global PCA/IVF index.
#include "Index.h"
#include "ExternalGroundTruth.h"
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

using namespace tribase;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
 try {
    if (argc != 15) throw std::invalid_argument(
        "Usage: qpivot_stage_ablation INDEX QUERIES GT_I32BIN OUTPUT NQ NPROBES_CSV "
        "POOL_DIMS FIXED_DIMS BLOCK REPEATS LOOPS THREADS NORMALIZE WARMUPS");
    const size_t nq=std::stoull(argv[5]),pool=std::stoull(argv[7]),budget=std::stoull(argv[8]),
        block=std::stoull(argv[9]),repeats=std::stoull(argv[10]),loops=std::stoull(argv[11]),
        warmups=std::stoull(argv[14]);
    const int threads=std::stoi(argv[12]);
    if (!nq||!pool||!budget||budget>pool||!block||!repeats||!loops||threads<1)
        throw std::invalid_argument("invalid positive parameter / budget exceeds pool");
    std::vector<size_t> probes;std::stringstream stream(argv[6]);std::string cell;
    while(std::getline(stream,cell,','))probes.push_back(std::stoull(cell));
    if(probes.empty())throw std::invalid_argument("empty nprobe list");
    std::sort(probes.begin(),probes.end());
    if(!probes.front()||std::adjacent_find(probes.begin(),probes.end())!=probes.end())
        throw std::invalid_argument("zero/duplicate nprobe");
    omp_set_num_threads(threads);
    Index index;index.load_index(argv[1]);
    if(index.metric!=METRIC_L2||!(index.added_opt_level&OPT_TRIANGLE)||
       index.multipivot_scope!=MultiPivotScope::GLOBAL||index.multipivot_method!="pca"||
       pool>=index.global_pivots.count||!index.global_pivots.has_U(pool+1)||probes.back()>index.nlist)
        throw std::invalid_argument("requires compatible Global PCA + Triangle bank covering requested pool");
    for(size_t lid=0;lid<index.nlist;++lid)if(index.lists[lid].list_size&&
       (index.lists[lid].pivots.signature_precision!=SignaturePrecision::FLOAT32||
        !index.lists[lid].pivots.usable_signature(pool+1)))
        throw std::invalid_argument("requires persisted float32 signatures");
    auto [queries,available,d]=loadXvecs(argv[2]);
    if(size_t(d)!=index.d||nq>size_t(available))throw std::invalid_argument("query shape mismatch");
    if(std::stoi(argv[13]))normalizeL2InPlace(queries.get(),nq,index.d);
    auto groundtruth=load_external_groundtruth(argv[3],nq);
    index.opt_level=OPT_TRIANGLE;index.signature_precision=SignaturePrecision::FLOAT32;
    index.set_projection_prefix_length(0);
    index.build_projection_moments(); // Offline, excluded identically for every query variant.
    const std::vector<std::string> methods={"no_signature","pd_only","residual","selection","adaptive"};
    auto configure=[&](const std::string& method) {
        const bool adaptive=method=="adaptive",selection=adaptive||method=="selection";
        index.projection_distance_only=method=="pd_only";
        index.projection_fixed_block_checkpoints=true;
        index.projection_block_size=block;
        index.projection_dynamic=adaptive;
        index.projection_progressive=0;
        index.projection_mm_dynamic=adaptive;
        index.projection_allow_budget_cap=adaptive;
        index.projection_moment_score=selection?2:0;
        index.projection_selection_pool_dims=selection?pool:0;
        index.projection_query_energy=false;index.projection_uniform_prefix=false;
        index.projection_query_adaptive=false;index.projection_scan_ascending=false;
        index.set_multipivot_search(method=="no_signature"?MultiPivotMode::NONE:MultiPivotMode::PROJECTION,
                                   method=="no_signature"?0:budget+1);
    };
    std::ofstream output(argv[4]);
    output<<std::setprecision(17)<<"method,nprobe,nq,threads,repeat,loops,pool_directions,fixed_directions,block_size,stats_enabled,"
        "wall_seconds,latency_ms,qps,recall,result_mismatches,total_candidates,triangle_pruned,mp_checks,mp_pruned,full_distances,"
        "overall_prune_rate,mean_active_directions,zero_budget_lists,list_visits,projection_worker_us,selection_worker_us,"
        "bound_worker_us,verification_worker_us,other_worker_us,total_worker_us\n";
    std::ofstream histogram(std::string(argv[4])+".budgets.csv");
    histogram<<"method,nprobe,repeat,pivots,directions,list_visits\n";
    if(!output||!histogram)throw std::runtime_error("cannot create output CSV");
    std::vector<float> refdist(nq),dist(nq);std::vector<idx_t> refids(nq),ids(nq);
    std::mt19937 rng(20261018);
    for(size_t probe:probes) {
        index.nprobe=probe;configure("no_signature");
        index.search(nq,queries.get(),1,refdist.data(),refids.data());
        auto verify=[&]() {
            for(size_t qi=0;qi<nq;++qi)if(ids[qi]!=refids[qi]||dist[qi]!=refdist[qi])
                throw std::runtime_error("result differs from same-index Triangle: query="+std::to_string(qi));
        };
        for(size_t rep=0;rep<repeats;++rep) {
            auto order=methods;std::shuffle(order.begin(),order.end(),rng);
            for(const auto& method:order) {
                configure(method);
                for(size_t w=0;w<warmups;++w)index.search(nq,queries.get(),1,dist.data(),ids.data());
                if(warmups)verify();
                Stats stats;const auto start=Clock::now();
                for(size_t loop=0;loop<loops;++loop)
                    stats=index.search(nq,queries.get(),1,dist.data(),ids.data());
                const double seconds=std::chrono::duration<double>(Clock::now()-start).count()/loops;
                verify();stats.summary();
                const double recall=calculate_id_recall(ids.data(),groundtruth.ids.get(),nq,1,groundtruth.width);
                const size_t tri=stats.skip_triangle_count+stats.skip_triangle_large_count;
                size_t visits=0,dim_sum=0;
                for(size_t p=0;p<stats.active_pivot_histogram.size();++p)if(stats.active_pivot_histogram[p]) {
                    const size_t count=stats.active_pivot_histogram[p];visits+=count;dim_sum+=(p?p-1:0)*count;
                    histogram<<method<<','<<probe<<','<<rep<<','<<p<<','<<(p?p-1:0)<<','<<count<<'\n';
                }
                const double selection=stats.query_energy_selection_seconds;
                const double projection=std::max(0.,stats.query_signature_seconds-selection)+stats.candidate_projection_seconds;
                const double verification=stats.candidate_exact_seconds;
                const double bound=stats.candidate_lb_seconds;
                const double other=std::max(0.,stats.search_worker_seconds-projection-selection-verification-bound);
#ifdef TRIBASE_ENABLE_STATS
                constexpr bool instrumented=true;
                if(stats.total_count!=tri+stats.multipivot_pruned+stats.candidate_distance_computations||stats.multipivot_invalid)
                    throw std::runtime_error("invalid candidate accounting");
#else
                constexpr bool instrumented=false;
#endif
                const double scale=1e6/nq;
                output<<method<<','<<probe<<','<<nq<<','<<threads<<','<<rep<<','<<loops<<','<<pool<<','<<budget<<','<<block<<','<<instrumented<<','
                    <<seconds<<','<<1000*seconds/nq<<','<<nq/seconds<<','<<recall<<",0,"<<stats.total_count<<','<<tri<<','
                    <<stats.multipivot_checks<<','<<stats.multipivot_pruned<<','<<stats.candidate_distance_computations<<','
                    <<(stats.total_count?double(tri+stats.multipivot_pruned)/stats.total_count:0.)<<','
                    <<(visits?double(dim_sum)/visits:0.)<<','<<stats.active_pivot_histogram[0]<<','<<visits<<','
                    <<projection*scale<<','<<selection*scale<<','<<bound*scale<<','<<verification*scale<<','<<other*scale<<','
                    <<stats.search_worker_seconds*scale<<'\n';output.flush();histogram.flush();
                std::cout<<"nprobe="<<probe<<" method="<<method<<" repeat="<<rep<<" ms/query="<<1000*seconds/nq
                         <<" recall="<<recall<<std::endl;
            }
        }
    }
    if(!output||!histogram)throw std::runtime_error("CSV write failed");
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
