#include "Index.h"
#include "BenchmarkAllocations.h"
#include <Eigen/Dense>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <cstdlib>

using namespace tribase;
using Clock=std::chrono::steady_clock;
using Matrix=Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
struct Plan {size_t p;bool energy;int moments=0;bool dynamic=false,adaptive=false,perlist=false,ascending=false,uniform=false;int mm_policy=0;};

int main(int argc,char** argv) {
    try {
        if(argc<9||argc>14)throw std::invalid_argument("Usage: benchmark_query_energy INDEX QUERIES DICTIONARY OUTPUT NQ REPEATS LOOPS THREADS [moments|calibrate|adaptive|scan_order|mm_policy|mm_compare [NPROBE [BUDGETS_CSV [MODEL_OR_DASH PERLIST_OR_DASH]]]]");
        const bool with_moments=argc>=10;
        const std::string mode=with_moments?argv[9]:"energy";
        const bool calibrate=mode=="calibrate",adaptive=mode=="adaptive",scan_order=mode=="scan_order";
        const bool mm_compare=mode=="mm_compare";
        const bool mm_policy=mode=="mm_policy"||mm_compare;
        if(with_moments&&mode!="moments"&&!calibrate&&!adaptive&&!scan_order&&!mm_policy)throw std::invalid_argument("invalid benchmark mode");
        if(mm_policy&&(argc!=14||std::stoi(argv[8])!=1))throw std::invalid_argument("mm_policy requires model-or-dash, per-list-index-or-dash and one query thread");
        if(mm_compare&&(std::string(argv[12])!="-"||std::string(argv[13])!="-"))
            throw std::invalid_argument("mm_compare accepts only the two MM policies; pass - -");
        const char* offline_env=std::getenv("TRIBASE_OFFLINE_THREADS");
        const int offline_threads=offline_env?std::stoi(offline_env):32;
        if(offline_threads<1)throw std::invalid_argument("TRIBASE_OFFLINE_THREADS must be positive");
        if(scan_order&&(argc<12||argc>13))throw std::invalid_argument("scan_order requires nprobe and budgets, optional per-list index");
        if(adaptive&&argc!=14)throw std::invalid_argument("adaptive requires model directory and original per-list index");
        const size_t nq=std::stoull(argv[5]),repeats=std::stoull(argv[6]),loops=std::stoull(argv[7]);
        const int threads=std::stoi(argv[8]);
        const size_t nprobe=argc>=11?std::stoull(argv[10]):160;
        std::vector<size_t> budgets{32,65,96,129};
        if(argc>=12) {
            budgets.clear();std::stringstream stream(argv[11]);std::string cell;
            while(std::getline(stream,cell,','))budgets.push_back(std::stoull(cell));
        }
        if(!nq||!repeats||!loops||threads<1)throw std::invalid_argument("invalid benchmark configuration");
        Index index;index.load_index(argv[1]);
        if(index.metric!=METRIC_L2||!(index.added_opt_level&OPT_TRIANGLE)||!nprobe||nprobe>index.nlist)
            throw std::invalid_argument("expected L2 source with Triangle metadata and valid nprobe");
        auto [queries,available,d]=loadXvecs(argv[2]);
        if(size_t(d)!=index.d||nq>size_t(available))throw std::invalid_argument("query mismatch");
        const size_t bytes=std::filesystem::file_size(argv[3]);
        if(bytes%(index.d*sizeof(double)))throw std::invalid_argument("dictionary shape mismatch");
        const size_t rows=bytes/(index.d*sizeof(double));
        if(rows<2||rows>index.d+1||rows>MULTIPIVOT_MAX_PIVOTS)throw std::invalid_argument("invalid dictionary dimensions");
        const size_t m=rows-1;
        if(budgets.empty())throw std::invalid_argument("empty budgets");
        for(size_t p:budgets)if(!p||p>=m)throw std::invalid_argument("budget must be in [1,M), full prefix is added separately");
        std::vector<double> raw((m+1)*index.d);
        std::ifstream input(argv[3],std::ios::binary);input.read(reinterpret_cast<char*>(raw.data()),raw.size()*sizeof(double));
        if(!input)throw std::runtime_error("cannot read dictionary");
        PivotMetadata geometry;geometry.d=index.d;geometry.count=geometry.requested_count=m+1;
        geometry.rank=m;geometry.condition=1;geometry.codes.resize(index.d);geometry.U.resize(m*index.d);
        std::copy_n(raw.begin(),index.d,geometry.codes.begin());
        std::copy(raw.begin()+index.d,raw.end(),geometry.U.begin());
        index.configure_multipivot(MultiPivotScope::GLOBAL,"pca",m+1,0);
        index.global_pivots=std::move(geometry);
        const Eigen::Map<const Matrix> basis(index.global_pivots.U.data(),m,index.d);
        Eigen::setNbThreads(1);
        std::cout<<"Installing frozen dictionary and float32 SoA signatures in memory"<<std::endl;
        omp_set_num_threads(offline_threads);
#pragma omp parallel for schedule(dynamic,1)
        for(size_t lid=0;lid<index.nlist;++lid) {
            auto& list=index.lists[lid];const size_t n=list.list_size;
            Matrix centered(n,index.d);
            PivotMetadata meta;meta.d=index.d;meta.count=meta.requested_count=m+1;meta.rank=m;meta.condition=1;
            meta.external_projection_geometry=true;meta.signature_candidate_count=n;
            meta.signature_data.resize((m+1)*n);meta.candidate_centroid_squared.resize(n);
            for(size_t i=0;i<n;++i) {
                double norm=0;for(size_t a=0;a<index.d;++a){double delta=double(list.get_candidate_codes()[i*index.d+a])-index.global_pivots.codes[a];centered(i,a)=float(delta);norm+=delta*delta;}
                meta.candidate_centroid_squared[i]=float(norm);
            }
            Eigen::Map<Matrix> destination(meta.signature_data.data(),m,n);
            destination.noalias()=basis*centered.transpose();
            for(size_t i=0;i<n;++i){float energy=0;for(size_t j=0;j<m;++j){float z=meta.signature_data[j*n+i];energy+=z*z;}
                meta.signature_data[m*n+i]=std::sqrt(std::max(0.f,meta.candidate_centroid_squared[i]-energy));}
            list.pivots=std::move(meta);
        }
        if(with_moments) {
          const auto sidecar=std::filesystem::path(argv[4]).parent_path()/"projection_moments.bin";
          if(std::filesystem::exists(sidecar)) {
            index.load_projection_moments(sidecar.string());
            std::cout<<"Loaded offline list moments "<<sidecar<<std::endl;
          } else {
            const auto start=Clock::now();index.build_projection_moments();
            index.save_projection_moments(sidecar.string());
            index.projection_list_means.clear();index.projection_list_variances.clear();index.projection_moment_dims=0;
            index.load_projection_moments(sidecar.string());
            std::cout<<"Built, saved and reloaded offline list moments; bytes="<<std::filesystem::file_size(sidecar)
                <<" seconds="<<std::chrono::duration<double>(Clock::now()-start).count()<<std::endl;
          }
        }
        index.nprobe=nprobe;index.opt_level=OPT_TRIANGLE;index.signature_precision=SignaturePrecision::FLOAT32;
        index.microblock_enabled=false;index.projection_dynamic=false;index.projection_progressive=0;
        index.projection_block_size=0;index.set_projection_prefix_length(0);
        omp_set_num_threads(threads);
        std::vector<idx_t> reference(nq),ids(nq);std::vector<float> rd(nq),dist(nq);
        index.set_multipivot_search(MultiPivotMode::NONE,0);
        index.search(nq,queries.get(),1,rd.data(),reference.data());
        auto validate=[&]() {
            for(size_t i=0;i<nq;++i)if(ids[i]!=reference[i]||dist[i]!=rd[i])
                throw std::runtime_error("result mismatch at query "+std::to_string(i));
        };
        std::vector<Plan> plans{{0,false}};
        if(adaptive)budgets={16,24,32};
        for(size_t p:budgets) {
            if(!calibrate)plans.push_back({p,false});plans.push_back({p,true});
            if(with_moments){if(!calibrate&&!adaptive&&!scan_order)plans.push_back({p,false,1});plans.push_back({p,false,2});}
            if(scan_order){plans.push_back({p,false,0,false,false,false,false,true});
                plans.push_back({p,true,0,false,false,false,true});plans.push_back({p,false,2,false,false,false,true});}
        }
        plans.push_back({m,false});
        if(mm_policy) {
            plans={{0,false},{m,false},{budgets.back(),false,2},{m,false,0,true}};
            for(int policy:{1,2,3,4})plans.push_back({m,false,2,true,false,false,false,false,policy});
        }
        if(mm_compare)plans={{m,false,2,true,false,false,false,false,1},
                             {m,false,2,true,false,false,false,false,2}};
        std::unique_ptr<Index> perlist;
        QueryPlanModel energy_model,mean_model;
        if(adaptive) {
            index.load_query_plan_model((std::filesystem::path(argv[12])/"energy.model").string());energy_model=index.query_plan_model;
            index.load_query_plan_model((std::filesystem::path(argv[12])/"mean.model").string());mean_model=index.query_plan_model;
        }
        if(mm_policy&&std::string(argv[12])!="-") {
            index.load_query_plan_model(argv[12]);mean_model=index.query_plan_model;
            plans.push_back({m,false,2,false,true});
        }
        if(adaptive||(scan_order&&argc==13)||(mm_policy&&std::string(argv[13])!="-")) {
            perlist=std::make_unique<Index>();perlist->load_index(argv[(adaptive||mm_policy)?13:12]);perlist->nprobe=nprobe;perlist->opt_level=OPT_TRIANGLE;
            perlist->projection_dynamic=true;perlist->projection_block_size=8;
            if(perlist->d!=index.d||perlist->nlist!=index.nlist||perlist->multipivot_scope!=MultiPivotScope::PER_LIST||
               perlist->multipivot_built_count!=40||perlist->multipivot_method!="pca")throw std::invalid_argument("unexpected original dynamic index");
            // Ensure both system baselines use exactly the same IVF assignment.
            for(size_t lid=0;lid<index.nlist;++lid) {
                if(perlist->lists[lid].list_size!=index.lists[lid].list_size)throw std::invalid_argument("IVF partitions differ");
                for(size_t i=0;i<index.lists[lid].list_size;++i)if(perlist->lists[lid].get_candidate_id()[i]!=index.lists[lid].get_candidate_id()[i])throw std::invalid_argument("IVF candidate order differs");
            }
            for(size_t i=0;i<index.nlist*index.d;++i)if(perlist->centroid_codes[i]!=index.centroid_codes[i])throw std::invalid_argument("IVF centroids differ");
            plans.push_back({39,false,0,true,false,true});
        }
        if(adaptive||scan_order)plans.push_back({m,false,0,true});
        if(adaptive){plans.push_back({m,true,0,false,true});plans.push_back({m,false,2,false,true});}
        Index* selected_index=&index;
        auto select=[&](const Plan& plan) {
            selected_index=plan.perlist?perlist.get():&index;
            auto& target=*selected_index;
            target.projection_query_energy=plan.energy;target.projection_moment_score=plan.moments;
            target.projection_scan_ascending=plan.ascending;target.projection_uniform_prefix=plan.uniform;
            target.collect_projection_orders=scan_order;
            target.projection_mm_dynamic=plan.mm_policy!=0;
            target.projection_reuse_workspace=plan.mm_policy==2||plan.mm_policy==3;
            target.projection_progressive=plan.mm_policy==2?1:plan.mm_policy>=3?2:0;
            target.projection_progressive_keep_exploration=plan.mm_policy!=0;
            target.projection_dynamic=plan.dynamic;target.projection_block_size=plan.dynamic?8:0;
            target.projection_query_adaptive=plan.adaptive;
            if(plan.adaptive)target.query_plan_model=plan.energy?energy_model:mean_model;
            target.set_multipivot_search(plan.p?MultiPivotMode::PROJECTION:MultiPivotMode::NONE,plan.p?plan.p+1:0);
        };
        for(auto& plan:plans){select(plan);selected_index->search(nq,queries.get(),1,dist.data(),ids.data());validate();}
        std::filesystem::create_directories(std::filesystem::path(argv[4]).parent_path());
        std::ofstream output(argv[4]);output<<std::setprecision(17);
        if(calibrate) {
            output<<"query_id,directions,method,repeat,ms_per_query";
            for(size_t f=0;f<QUERY_PLAN_FEATURES;++f)output<<",f"<<f;output<<'\n';
            std::mt19937 rng(20261018);
            std::vector<std::array<QueryPlanFeatures,3>> all_features(nq);
            for(size_t qi=0;qi<nq;++qi) {
                const float* q=queries.get()+qi*index.d;
                IVFScan<METRIC_L2,OPT_NONE,EdgeDevice::EDGEDEVIVE_DISABLED> coarse(index.d,nprobe);
                std::vector<float> cd(nprobe);std::vector<idx_t> lids(nprobe);init_result(METRIC_L2,nprobe,cd.data(),lids.data());
                coarse.set_query(q);coarse.lite_scan_codes(index.nlist,index.centroid_codes.get(),reinterpret_cast<const size_t*>(index.centroid_ids.get()),cd.data(),lids.data());sort_result(METRIC_L2,nprobe,cd.data(),lids.data());
                std::vector<float> z(m+1),centered(index.d);std::vector<double> sq(m+1);float rho;
                computePcaQuerySignature(q,index.d,index.global_pivots,m+1,sq.data(),z.data(),&rho,centered.data());
                std::array<QueryPlanFeatures,3> features;
                for(int family:{0,2}) {
                    std::vector<size_t> order(m);std::iota(order.begin(),order.end(),0);std::vector<double> scores(m);
                    if(!family)for(size_t dim=0;dim<m;++dim)scores[dim]=z[dim]*z[dim];
                    else for(size_t j=0;j<nprobe;++j)for(size_t dim=0;dim<m;++dim){double diff=double(z[dim])-index.projection_list_means[lids[j]*m+dim];scores[dim]+=index.lists[lids[j]].list_size*diff*diff;}
                    std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return scores[a]>scores[b]||(scores[a]==scores[b]&&a<b);});
                    features[family]=makeQueryPlanFeatures(float(sq[0]),z.data(),m,order.data(),cd.data(),nprobe,index.d,[&](size_t j){return index.lists[lids[j]].list_size;});
                }
                all_features[qi]=features;
            }
            // Interleave query/action pairs: repeated calibration of one query
            // back-to-back would artificially favor a hot candidate working set.
            std::vector<std::pair<size_t,size_t>> events;
            for(size_t qi=0;qi<nq;++qi)for(size_t pi=0;pi<plans.size();++pi)events.emplace_back(qi,pi);
            for(size_t r=0;r<repeats;++r) {
                std::shuffle(events.begin(),events.end(),rng);
                for(const auto& [qi,pi]:events) {
                        const auto& plan=plans[pi];const float* q=queries.get()+qi*index.d;
                        const auto& features=all_features[qi];
                        select(plan);std::vector<idx_t> one_id(1);std::vector<float> one_dist(1);
                        const auto start=Clock::now();
                        for(size_t loop=0;loop<loops;++loop)index.search(1,q,1,one_dist.data(),one_id.data());
                        const double ms=1000*std::chrono::duration<double>(Clock::now()-start).count()/loops;
                        if(one_id[0]!=reference[qi]||one_dist[0]!=rd[qi])throw std::runtime_error("calibration result mismatch");
                        const char* method=plan.p?(plan.moments?"moment_mean":plan.energy?"query_energy":"prefix"):"triangle";
                        // The baseline/full actions are shared across order families.
                        for(int family:{0,2})if(!plan.p||(!plan.energy&&!plan.moments)||family==(plan.energy?0:2)) {
                            output<<qi<<','<<plan.p<<','<<(family==0?"query_energy":"moment_mean")<<','<<r<<','<<ms;
                            for(double value:features[family])output<<','<<value;output<<'\n';
                        }
                    if(qi%16==0)output.flush();
                }
                output.flush();std::cout<<"calibration repeat="<<r<<std::endl;
            }
            return 0;
        }
        std::ofstream choices(std::string(argv[4])+".choices.csv");choices<<"method,repeat,query_id,directions,canonical,predicted_ms\n";
        std::ofstream order_out(std::string(argv[4])+".orders.csv");order_out<<"method,repeat,query_id,directions,order\n";
        output<<"directions,method,repeat,loops,nq,threads,wall_seconds,ms_per_query,mp_checks,mp_pruned,candidate_distances,total_candidates,query_signature_us_per_query,selection_us_per_query,result_mismatches,stats_enabled,triangle_pruned,candidate_decision_us,candidate_lb_us,candidate_exact_us,other_us,worker_us,mp_invalid,active_copy_us,progressive_control_us,progressive_stages,progressive_candidate_updates,workspace_growths,workspace_bytes,new_calls_per_query,new_bytes_per_query,delete_calls_per_query,mean_active_directions\n";
        std::ofstream hist(std::string(argv[4])+".budgets.csv");hist<<"method,repeat,native_pivots,directions,list_visits\n";
        std::mt19937 rng(20261016);
        for(size_t repeat=0;repeat<repeats;++repeat) {
            std::shuffle(plans.begin(),plans.end(),rng);
            for(auto& plan:plans) {
                select(plan);
                benchmark_allocations::start();
                const auto start=Clock::now();Stats stats;
                for(size_t loop=0;loop<loops;++loop)stats=selected_index->search(nq,queries.get(),1,dist.data(),ids.data());
                const double seconds=std::chrono::duration<double>(Clock::now()-start).count();
                const auto allocations=benchmark_allocations::stop();validate();
                // Derived stage totals are finalized outside the measured interval.
                stats.summary();
                const char* method=plan.mm_policy==1?"mm_once":plan.mm_policy==2?"mm_stage_frozen_reuse":plan.mm_policy==3?"mm_stage_updated_reuse":plan.mm_policy==4?"mm_stage_updated_alloc":
                    plan.perlist?"original_perlist_dynamic":plan.dynamic?"original_global_dynamic":plan.adaptive?(mm_policy?"mm_query_model":plan.energy?"adaptive_energy":"adaptive_mean"):
                    plan.uniform?"prefix_uniform":plan.ascending?(plan.energy?"query_energy_ascending":"moment_mean_ascending"):
                    plan.p?(plan.moments==1?"moment_separation":plan.moments==2?"moment_mean":plan.energy?"query_energy":"prefix"):"triangle";
                for(const auto& choice:stats.query_plan_choices)choices<<method<<','<<repeat<<','<<choice.query_id<<','<<choice.directions<<','<<choice.canonical<<','<<choice.predicted_ms<<'\n';
                for(const auto& order:stats.query_direction_orders){order_out<<method<<','<<repeat<<','<<order.query_id<<','<<order.directions.size()<<',';
                    for(size_t i=0;i<order.directions.size();++i){if(i)order_out<<' ';order_out<<order.directions[i];}order_out<<'\n';}
                size_t budget_visits=0,direction_sum=0;
                for(size_t p=0;p<stats.active_pivot_histogram.size();++p)if(stats.active_pivot_histogram[p]) {
                    hist<<method<<','<<repeat<<','<<p<<','<<(p?p-1:0)<<','<<stats.active_pivot_histogram[p]<<'\n';
                    budget_visits+=stats.active_pivot_histogram[p];direction_sum+=(p?p-1:0)*stats.active_pivot_histogram[p];
                }
                output<<plan.p<<','<<method<<','<<repeat<<','<<loops<<','<<nq<<','<<threads<<','<<seconds<<','<<1000*seconds/(loops*nq)<<','
                    <<stats.multipivot_checks<<','<<stats.multipivot_pruned<<','<<stats.candidate_distance_computations<<','<<stats.total_count<<','
                    <<1e6*stats.query_signature_seconds/nq<<','<<1e6*stats.query_energy_selection_seconds/nq<<",0,"
#ifdef TRIBASE_ENABLE_STATS
                    <<1
#else
                    <<0
#endif
                    <<','<<stats.skip_triangle_count+stats.skip_triangle_large_count<<','<<1e6*stats.candidate_decision_seconds/nq<<','
                    <<1e6*stats.candidate_lb_seconds/nq<<','<<1e6*stats.candidate_exact_seconds/nq<<','<<1e6*stats.other_seconds/nq<<','
                    <<1e6*stats.search_worker_seconds/nq<<','<<stats.multipivot_invalid<<','<<1e6*stats.candidate_active_copy_seconds/nq<<','
                    <<1e6*stats.progressive_control_seconds/nq<<','<<stats.progressive_stages<<','<<stats.progressive_candidate_updates<<','
                    <<stats.progressive_workspace_growths<<','<<stats.progressive_workspace_bytes<<','<<double(allocations.calls)/(nq*loops)<<','
                    <<double(allocations.bytes)/(nq*loops)<<','<<double(allocations.deletes)/(nq*loops)<<','
                    <<(budget_visits?double(direction_sum)/budget_visits:0.)<<'\n';output.flush();
                std::cout<<"repeat="<<repeat<<" P="<<plan.p<<" method="<<method<<" ms/query="<<1000*seconds/(loops*nq)<<std::endl;
            }
        }
        if(!output)throw std::runtime_error("benchmark output failed");
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
