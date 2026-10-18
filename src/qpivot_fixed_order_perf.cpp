#include "Index.h"
#include "Ppd.h"
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <limits>
#include <Eigen/Dense>

using namespace tribase;
using Clock=std::chrono::steady_clock;

int main(int argc,char** argv) {
 try {
#ifdef TRIBASE_ENABLE_STATS
    throw std::invalid_argument("qpivot_fixed_order_perf requires ENABLE_STATS=OFF");
#endif
    if(argc<13||argc>15)throw std::invalid_argument(
        "Usage: qpivot_fixed_order_perf INDEX QUERIES OUTPUT NQ NPROBE POOL BUDGETS_CSV REPEATS LOOPS THREADS NORMALIZE WARMUPS [--ppd-in-memory] [--ppd-basis-control]");
    bool with_ppd=false,basis_control=false;
    for(int arg=13;arg<argc;++arg) {
        if(std::string(argv[arg])=="--ppd-in-memory")with_ppd=true;
        else if(std::string(argv[arg])=="--ppd-basis-control")basis_control=true;
        else throw std::invalid_argument("unknown optional argument");
    }
    if(basis_control&&!with_ppd)throw std::invalid_argument("basis control requires in-memory PPD");
    const size_t nq=std::stoull(argv[4]),probe=std::stoull(argv[5]),pool=std::stoull(argv[6]);
    const size_t repeats=std::stoull(argv[8]),loops=std::stoull(argv[9]),warmups=std::stoull(argv[12]);
    const int threads=std::stoi(argv[10]);
    if(!nq||!probe||!pool||!repeats||!loops||!warmups||threads<1)throw std::invalid_argument("invalid positive parameter");
    std::vector<size_t> budgets;std::stringstream cells(argv[7]);std::string cell;
    while(std::getline(cells,cell,','))budgets.push_back(std::stoull(cell));
    std::sort(budgets.begin(),budgets.end());
    if(budgets.empty()||!budgets.front()||budgets.back()>pool||
       std::adjacent_find(budgets.begin(),budgets.end())!=budgets.end())throw std::invalid_argument("invalid budgets");
    omp_set_num_threads(threads);
    Index index;index.load_index(argv[1]);
    if(index.metric!=METRIC_L2||!(index.added_opt_level&OPT_TRIANGLE)||
       index.multipivot_scope!=MultiPivotScope::GLOBAL||index.multipivot_method!="pca"||
       pool>=index.global_pivots.count||!index.global_pivots.has_U(pool+1)||probe>index.nlist)
        throw std::invalid_argument("requires original rich Global PCA Triangle bank and valid pool/nprobe");
    for(size_t lid=0;lid<index.nlist;++lid)if(index.lists[lid].list_size &&
       (index.lists[lid].pivots.signature_precision!=SignaturePrecision::FLOAT32||
        !index.lists[lid].pivots.usable_signature(pool+1)))
        throw std::invalid_argument("missing persisted float32 candidate signatures");
    auto [queries,available,d]=loadXvecs(argv[2]);
    if(size_t(d)!=index.d||size_t(available)!=nq)throw std::invalid_argument("query count/dimension mismatch");
    if(std::stoi(argv[11]))normalizeL2InPlace(queries.get(),nq,index.d);
    index.nprobe=probe;index.opt_level=OPT_TRIANGLE;
    index.signature_precision=SignaturePrecision::FLOAT32;
    index.projection_dynamic=false;index.projection_progressive=0;index.projection_block_size=0;
    index.set_projection_prefix_length(0);
    index.build_projection_moments(); // Offline list statistics, excluded from query timing.
    PpdIndex ppd;
    if(with_ppd) {
        std::cout<<"Building full-training PPD in RAM; no sidecar is written"<<std::endl;
        const auto start=Clock::now();
        ppd.build(index,std::numeric_limits<size_t>::max(),0);
        std::cout<<"PPD ready; build_seconds="<<std::chrono::duration<double>(Clock::now()-start).count()<<std::endl;
    }
    PivotMetadata alternate_global;
    std::vector<PivotMetadata> alternate_lists;
    std::vector<double> alternate_means,alternate_variances;
    bool shared_basis_active=false;
    auto swap_basis=[&]() {
        std::swap(index.global_pivots,alternate_global);
        for(size_t lid=0;lid<index.nlist;++lid)std::swap(index.lists[lid].pivots,alternate_lists[lid]);
        std::swap(index.projection_list_means,alternate_means);
        std::swap(index.projection_list_variances,alternate_variances);
        shared_basis_active=!shared_basis_active;
    };
    if(basis_control) {
        // Match physical signature width, so prefix-residual preparation follows
        // the same native scan path in both basis families.
        const size_t P=index.global_pivots.count,M=P-1;
        alternate_global.d=index.d;alternate_global.count=alternate_global.requested_count=P;
        alternate_global.rank=M;alternate_global.condition=1.;alternate_global.codes=ppd.mean;
        alternate_global.U.assign(ppd.basis.begin(),ppd.basis.begin()+M*index.d);
        alternate_lists.resize(index.nlist);
#pragma omp parallel for schedule(dynamic,1)
        for(size_t lid=0;lid<index.nlist;++lid) {
            const auto& list=index.lists[lid];auto& meta=alternate_lists[lid];
            meta.d=index.d;meta.count=meta.requested_count=P;meta.rank=M;meta.condition=1.;
            meta.external_projection_geometry=true;meta.signature_precision=SignaturePrecision::FLOAT32;
            meta.signature_candidate_count=list.list_size;meta.signature_data.resize(P*list.list_size);
            meta.candidate_centroid_squared.resize(list.list_size);
            for(size_t i=0;i<list.list_size;++i) {
                const float* x=list.get_candidate_codes(i);double norm=0;float energy=0;
                for(size_t axis=0;axis<index.d;++axis){double v=double(x[axis])-ppd.mean[axis];norm+=v*v;}
                meta.candidate_centroid_squared[i]=static_cast<float>(norm);
                for(size_t dim=0;dim<M;++dim) {
                    const float z=ppd.transformed_lists[lid][i*index.d+dim];
                    meta.signature_data[dim*list.list_size+i]=z;energy+=z*z;
                }
                meta.signature_data[M*list.list_size+i]=std::sqrt(std::max(0.,norm-energy));
            }
        }
        // Compute shared-basis list moments offline, then restore original basis.
        alternate_means=index.projection_list_means;alternate_variances=index.projection_list_variances;
        swap_basis();index.build_projection_moments();swap_basis();
        using Matrix=Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
        const Eigen::Map<const Matrix> A(index.global_pivots.U.data(),M,index.d);
        const Eigen::Map<const Matrix> B(ppd.basis.data(),M,index.d);
        const Eigen::MatrixXf overlaps=A.topRows(pool)*B.topRows(pool).transpose();
        std::ofstream diagnostic(std::string(argv[3])+".basis.csv");
        diagnostic<<std::setprecision(17)<<"directions,subspace_overlap_fraction,physical_signature_directions\n";
        for(size_t L:budgets)diagnostic<<L<<','<<overlaps.topLeftCorner(L,L).squaredNorm()/L<<','<<M<<'\n';
        std::cout<<"Shared PPD basis/coordinates ready; physical signature dimensions="<<M<<std::endl;
    }
    std::vector<float> reference_dist(nq),dist(nq);std::vector<idx_t> reference_ids(nq),ids(nq);
    index.set_multipivot_search(MultiPivotMode::NONE,0);
    index.search(nq,queries.get(),1,reference_dist.data(),reference_ids.data());
    struct Plan {size_t budget;std::string method;};
    std::vector<Plan> plans;
    for(size_t budget:budgets)for(const std::string method:{"pca","mm"})plans.push_back({budget,method});
    if(basis_control)for(size_t budget:budgets)for(const std::string method:{"pca_ppd_basis","mm_ppd_basis"})plans.push_back({budget,method});
    if(with_ppd)plans.push_back({0,"ppd_B16"});
    auto select=[&](const Plan& plan) {
        const bool want_shared=plan.method.ends_with("_ppd_basis");
        if(basis_control&&want_shared!=shared_basis_active)swap_basis();
        index.projection_moment_score=plan.method.starts_with("mm")?2:0;
        index.projection_uniform_prefix=false;
        index.projection_selection_pool_dims=plan.budget?pool:0;
        index.set_multipivot_search(plan.budget?MultiPivotMode::PROJECTION:MultiPivotMode::NONE,plan.budget?plan.budget+1:0);
    };
    auto verify=[&](bool rotated) {
        for(size_t i=0;i<nq;++i)if(ids[i]!=reference_ids[i]||
            (rotated?std::abs(dist[i]-reference_dist[i])>2e-5f*std::max(1.f,std::abs(reference_dist[i])):dist[i]!=reference_dist[i]))
            throw std::runtime_error("search result differs from native Triangle at query="+std::to_string(i));
    };
    std::ofstream output(argv[3]);
    output<<std::setprecision(17)<<"method,directions,pivots,pool_directions,nprobe,nq,threads,repeat,loops,warmups,wall_seconds,latency_ms,qps,result_mismatches,selection_included,ppd_candidates,ppd_triangle_pruned,ppd_checked,ppd_pruned,ppd_full_distances,ppd_dimensions\n";
    if(!output)throw std::runtime_error("cannot create output CSV");
    std::mt19937 rng(20261018);
    for(size_t rep=0;rep<repeats;++rep) {
        std::shuffle(plans.begin(),plans.end(),rng);
        for(const auto& plan:plans) {
            select(plan);
            PpdSearchStats ppd_stats;
            auto search=[&]() {
                if(plan.method=="ppd_B16")ppd_stats=searchPpd(index,ppd,nq,queries.get(),1,probe,16,true,dist.data(),ids.data());
                else index.search(nq,queries.get(),1,dist.data(),ids.data());
            };
            for(size_t w=0;w<warmups;++w)search();
            verify(plan.method=="ppd_B16");
            const auto start=Clock::now();
            for(size_t loop=0;loop<loops;++loop)search();
            const double seconds=std::chrono::duration<double>(Clock::now()-start).count()/loops;
            verify(plan.method=="ppd_B16");
            output<<plan.method<<','<<plan.budget<<','<<(plan.budget?plan.budget+1:0)<<','<<pool<<','<<probe<<','<<nq<<','<<threads<<','
                  <<rep<<','<<loops<<','<<warmups<<','<<seconds<<','<<1000*seconds/nq<<','<<nq/seconds<<",0,"
                  <<1<<','<<ppd_stats.candidates<<','<<ppd_stats.triangle_pruned<<','<<ppd_stats.ppd_checked<<','
                  <<ppd_stats.ppd_pruned<<','<<ppd_stats.full_distances<<','<<ppd_stats.dimensions<<'\n';
            output.flush();
            std::cout<<"method="<<plan.method<<" L="<<plan.budget<<" repeat="<<rep<<" wall="<<seconds<<std::endl;
        }
    }
    if(!output)throw std::runtime_error("failed writing timing output");
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
