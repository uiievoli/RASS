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
using Matrix = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
struct Visit {size_t q,lid,rank,n;double tau,coarse;};
struct Cache {Matrix z;std::vector<double> norm,local_norm;};
static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> r;std::stringstream ss(line);std::string s;
    while(std::getline(ss,s,','))r.push_back(s);return r;
}
static std::string join(const std::vector<size_t>& values) {
    std::ostringstream out;for(size_t i=0;i<values.size();++i){if(i)out<<' ';out<<values[i];}return out.str();
}

int main(int argc,char** argv) {
    try {
        if(argc!=9)throw std::invalid_argument("Usage: query_subset_oracle INDEX QUERIES VISITS OUTPUT NQ M RANDOM_REPEATS SEED");
        const size_t nq=std::stoull(argv[5]),m=std::stoull(argv[6]),repeats=std::stoull(argv[7]);
        const uint64_t seed=std::stoull(argv[8]);
        if(nq<1||m<129||m>511||!repeats)throw std::invalid_argument("invalid configuration");
        std::filesystem::create_directories(argv[4]);const std::filesystem::path out=argv[4];
        Index index;index.load_index(argv[1]);
        if(index.metric!=MetricType::METRIC_L2||index.multipivot_scope!=MultiPivotScope::GLOBAL||index.multipivot_method!="pca")
            throw std::invalid_argument("requires global PCA L2 index");
        auto [queries,available,d]=loadXvecs(argv[2]);
        if(size_t(d)!=index.d||nq>size_t(available)||m>index.d)throw std::invalid_argument("query/dictionary mismatch");
        std::ifstream input(argv[3]);std::string line;if(!std::getline(input,line))throw std::runtime_error("missing captures");
        auto header=split(line);std::map<std::string,size_t> columns;
        for(size_t i=0;i<header.size();++i)columns[header[i]]=i;
        std::vector<std::vector<Visit>> visits(nq);std::vector<bool> needed(index.nlist,false);
        while(std::getline(input,line)) {
            auto row=split(line);auto get=[&](const char* key)->const std::string&{return row.at(columns.at(key));};
            if(std::stoul(get("nprobe"))!=160||std::stoul(get("pivot_count"))!=index.multipivot_built_count)continue;
            size_t q=std::stoull(get("query_id"));if(q>=nq)continue;
            Visit v{q,std::stoull(get("list_id")),std::stoull(get("probe_rank")),std::stoull(get("list_size")),
                std::bit_cast<float>(uint32_t(std::stoul(get("entry_tau2_bits")))),
                std::bit_cast<float>(uint32_t(std::stoul(get("centroid2query_bits"))))};
            if(v.lid>=index.nlist||v.n!=index.lists[v.lid].list_size)throw std::runtime_error("capture/index mismatch");
            visits[q].push_back(v);needed[v.lid]=true;
        }
        for(auto& vv:visits) {
            std::sort(vv.begin(),vv.end(),[](auto& a,auto& b){return a.rank<b.rank;});
            if(vv.size()!=160)throw std::runtime_error("need all 160 visits per query");
            for(size_t i=0;i<vv.size();++i)if(vv[i].rank!=i)throw std::runtime_error("duplicate/missing probe rank");
        }
        std::cout<<"Building frozen global PCA dictionary from IVF centroids, M="<<m<<std::endl;
        Eigen::setNbThreads(1);
        auto selected=selectPcaPivots(index.centroid_codes.get(),index.nlist,index.d,m+1,nullptr,-1);
        if(selected.metadata.count!=m+1)throw std::runtime_error("incomplete PCA dictionary");
        Eigen::VectorXd center(index.d);for(size_t a=0;a<index.d;++a)center[a]=selected.metadata.codes[a];
        Matrix basis(m,index.d);
        for(size_t j=0;j<m;++j) {
            for(size_t a=0;a<index.d;++a)basis(j,a)=selected.metadata.U[j*index.d+a];
            // Double reorthogonalization removes float32 storage error while
            // retaining the PCA order and each nested prefix subspace.
            for(int pass=0;pass<2;++pass)for(size_t k=0;k<j;++k)basis.row(j)-=basis.row(j).dot(basis.row(k))*basis.row(k);
            basis.row(j).normalize();
        }
        const Matrix gram=basis*basis.transpose();
        const double orth_error=(gram-Matrix::Identity(m,m)).cwiseAbs().maxCoeff();
        if(orth_error>1e-10)throw std::runtime_error("dictionary not orthonormal");
        double old_alignment=1;
        for(size_t j=0;j<std::min(m,index.global_pivots.count-1);++j) {
            double dot=0;for(size_t a=0;a<index.d;++a)dot+=basis(j,a)*index.global_pivots.U[j*index.d+a];
            old_alignment=std::min(old_alignment,std::abs(dot));
        }
        std::ofstream geometry(out/"geometry.csv");geometry<<std::setprecision(17)<<"directions,orthogonality_max_error,old_prefix_min_abs_alignment\n"<<m<<','<<orth_error<<','<<old_alignment<<'\n';
        std::ofstream bank(out/"dictionary.bin",std::ios::binary);
        bank.write(reinterpret_cast<const char*>(center.data()),center.size()*sizeof(double));
        bank.write(reinterpret_cast<const char*>(basis.data()),basis.size()*sizeof(double));bank.close();
        std::vector<Cache> cache(index.nlist);
        std::cout<<"Projecting candidates in "<<std::count(needed.begin(),needed.end(),true)<<" visited lists"<<std::endl;
#pragma omp parallel for schedule(dynamic,1)
        for(size_t lid=0;lid<index.nlist;++lid)if(needed[lid]) {
            auto& list=index.lists[lid];auto& c=cache[lid];const size_t n=list.list_size;
            Matrix centered(n,index.d);c.norm.resize(n);c.local_norm.resize(n);
            for(size_t i=0;i<n;++i) {
                double norm=0,local=0;const float* x=list.get_candidate_codes()+i*index.d;
                for(size_t a=0;a<index.d;++a){double delta=double(x[a])-center[a];centered(i,a)=delta;norm+=delta*delta;
                    double ld=double(x[a])-index.centroid_codes[lid*index.d+a];local+=ld*ld;}
                c.norm[i]=norm;c.local_norm[i]=local;
            }
            c.z.noalias()=basis*centered.transpose();
        }
        std::ofstream query_out(out/"queries.csv"),list_out(out/"lists.csv"),orders(out/"orders.csv"),safety(out/"safety.csv");
        const char* suffix="directions,native_P,method,trial,candidates,triangle_pruned,mp_checked,mp_pruned,overall_pruned\n";
        query_out<<"query_id,"<<suffix;list_out<<"query_id,probe_rank,list_id,entry_tau2,"<<suffix;
        orders<<"query_id,directions,method,trial,order\n";
        safety<<"query_id,final_plan_checks,false_prunes,max_lb_excess\n";
        query_out<<std::setprecision(17);list_out<<std::setprecision(17);
        const std::vector<size_t> budgets{32,65,96,129};
        for(size_t qi=0;qi<nq;++qi) {
            const auto& vv=visits[qi];size_t n=0;for(auto& v:vv)n+=v.n;
            Eigen::VectorXd qvec(index.d);for(size_t a=0;a<index.d;++a)qvec[a]=double(queries[qi*index.d+a])-center[a];
            const double qnorm=qvec.squaredNorm();const Eigen::VectorXd qz=basis*qvec;
            std::vector<double> x2(m*n),delta2(m*n),norm(n),tau(n),exact(n);
            std::vector<uint8_t> survivor(n);std::vector<size_t> offsets(vv.size()+1,0);
            for(size_t j=0;j<vv.size();++j)offsets[j+1]=offsets[j]+vv[j].n;
#pragma omp parallel for schedule(dynamic,1)
            for(size_t j=0;j<vv.size();++j) {
                const auto& v=vv[j];const auto& c=cache[v.lid];const auto& list=index.lists[v.lid];
                double local_q=0;for(size_t a=0;a<index.d;++a){double d=double(queries[qi*index.d+a])-index.centroid_codes[v.lid*index.d+a];local_q+=d*d;}
                for(size_t i=0;i<v.n;++i) {
                    const size_t t=offsets[j]+i;norm[t]=c.norm[i];tau[t]=v.tau;
                    const double gap=std::sqrt(local_q)-std::sqrt(c.local_norm[i]);survivor[t]=gap*gap<=v.tau;
                    double dist=0;const float* x=list.get_candidate_codes()+i*index.d;
                    for(size_t a=0;a<index.d;++a){double d=double(queries[qi*index.d+a])-x[a];dist+=d*d;}exact[t]=dist;
                }
                for(size_t dim=0;dim<m;++dim)for(size_t i=0;i<v.n;++i){size_t t=offsets[j]+i;double b=c.z(dim,i),d=qz[dim]-b;x2[dim*n+t]=b*b;delta2[dim*n+t]=d*d;}
            }
            size_t checked=std::accumulate(survivor.begin(),survivor.end(),size_t(0));
            std::vector<size_t> active;active.reserve(checked);for(size_t i=0;i<n;++i)if(survivor[i])active.push_back(i);
            // Compact Triangle survivors. Query-global directions are shared;
            // per-list thresholds remain attached to individual candidates.
            const size_t an=active.size();std::vector<double> ax2(m*an),adelta(m*an),anorm(an),atau(an);
#pragma omp parallel for
            for(size_t dim=0;dim<m;++dim)for(size_t j=0;j<an;++j){ax2[dim*an+j]=x2[dim*n+active[j]];adelta[dim*an+j]=delta2[dim*n+active[j]];}
            for(size_t j=0;j<an;++j){anorm[j]=norm[active[j]];atau[j]=tau[active[j]];}
            std::vector<double> spent(an,0),score(an,0);double spent_q=0;
            std::vector<size_t> order;std::vector<bool> used(m,false);
            std::vector<size_t> counts(m);std::vector<double> strengths(m);
            size_t false_prunes=0,plan_checks=0;double max_excess=0;
            auto evaluate=[&](const std::vector<size_t>& plan,const std::string& method,int trial) {
                std::vector<double> ps(n,0),px(n,0);double pq=0;
                for(size_t dim:plan){pq+=qz[dim]*qz[dim];for(size_t i=0;i<n;++i){ps[i]+=delta2[dim*n+i];px[i]+=x2[dim*n+i];}}
                size_t mp_total=0,tri_total=0;
                for(size_t j=0;j<vv.size();++j) {
                    size_t mp=0,tri=0;for(size_t t=offsets[j];t<offsets[j+1];++t) {
                        const double gap=std::sqrt(std::max(0.,qnorm-pq))-std::sqrt(std::max(0.,norm[t]-px[t]));
                        const double lb=ps[t]+gap*gap;max_excess=std::max(max_excess,lb-exact[t]);
                        if(lb>tau[t]&&exact[t]<=tau[t])++false_prunes;
                        if(!survivor[t]){++tri;if(exact[t]<=tau[t])++false_prunes;}
                        else if(lb>tau[t])++mp;
                    }
                    list_out<<qi<<','<<vv[j].rank<<','<<vv[j].lid<<','<<vv[j].tau<<','<<plan.size()<<','<<plan.size()+1<<','<<method<<','<<trial<<','<<vv[j].n<<','<<tri<<','<<vv[j].n-tri<<','<<mp<<','<<tri+mp<<'\n';
                    mp_total+=mp;tri_total+=tri;
                }
                query_out<<qi<<','<<plan.size()<<','<<plan.size()+1<<','<<method<<','<<trial<<','<<n<<','<<tri_total<<','<<n-tri_total<<','<<mp_total<<','<<tri_total+mp_total<<'\n';
                orders<<qi<<','<<plan.size()<<','<<method<<','<<trial<<','<<join(plan)<<'\n';++plan_checks;
            };
            for(size_t t=1;t<=129;++t) {
#pragma omp parallel for schedule(static)
                for(size_t dim=0;dim<m;++dim)if(!used[dim]) {
                    size_t count=0;double strength=0;const double qr=std::sqrt(std::max(0.,qnorm-spent_q-qz[dim]*qz[dim]));
                    const double* xp=ax2.data()+dim*an;const double* dp=adelta.data()+dim*an;
#pragma omp simd reduction(+:count,strength)
                    for(size_t i=0;i<an;++i) {
                        const double gap=qr-std::sqrt(std::max(0.,anorm[i]-spent[i]-xp[i]));
                        const double lb=score[i]+dp[i]+gap*gap;count+=lb>atau[i];strength+=lb;
                    }
                    counts[dim]=count;strengths[dim]=strength;
                }
                size_t best=m;
                for(size_t dim=0;dim<m;++dim)if(!used[dim]&&(best==m||counts[dim]>counts[best]||(counts[dim]==counts[best]&&strengths[dim]>strengths[best])))best=dim;
                used[best]=true;order.push_back(best);spent_q+=qz[best]*qz[best];
                for(size_t i=0;i<an;++i){spent[i]+=ax2[best*an+i];score[i]+=adelta[best*an+i];}
                if(std::find(budgets.begin(),budgets.end(),t)!=budgets.end())evaluate(order,"greedy",-1);
            }
            std::vector<size_t> energy(m);std::iota(energy.begin(),energy.end(),0);
            std::stable_sort(energy.begin(),energy.end(),[&](size_t a,size_t b){return qz[a]*qz[a]>qz[b]*qz[b];});
            for(size_t p:budgets) {
                std::vector<size_t> prefix(p);std::iota(prefix.begin(),prefix.end(),0);evaluate(prefix,"prefix",-1);
                evaluate(std::vector<size_t>(energy.begin(),energy.begin()+p),"query_energy",-1);
                for(size_t r=0;r<repeats;++r) {
                    std::vector<size_t> random(m);std::iota(random.begin(),random.end(),0);
                    std::mt19937_64 rng(seed+qi*1000003+r*1009);std::shuffle(random.begin(),random.end(),rng);
                    random.resize(p);evaluate(random,"random",int(r));
                }
            }
            std::vector<size_t> full(m);std::iota(full.begin(),full.end(),0);evaluate(full,"prefix",-1);
            safety<<qi<<','<<plan_checks<<','<<false_prunes<<','<<std::setprecision(17)<<max_excess<<'\n';
            query_out.flush();list_out.flush();orders.flush();safety.flush();
            std::cout<<"query="<<qi<<" candidates="<<n<<" triangle_survivors="<<checked<<" false_prunes="<<false_prunes<<std::endl;
            if(false_prunes)throw std::runtime_error("false pruning detected");
        }
        if(!query_out||!list_out||!orders||!safety)throw std::runtime_error("output write failed");
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
