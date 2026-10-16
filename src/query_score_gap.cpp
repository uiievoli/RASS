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
struct Cache {Matrix z;Eigen::VectorXd mean,variance;std::vector<double> norm,local_norm;};
static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> r;std::stringstream ss(line);std::string s;
    while(std::getline(ss,s,','))r.push_back(s);return r;
}
static std::string join(const std::vector<size_t>& values) {
    std::ostringstream out;for(size_t i=0;i<values.size();++i){if(i)out<<' ';out<<values[i];}return out.str();
}

int main(int argc,char** argv) {
    try {
        if(argc!=14)throw std::invalid_argument("Usage: query_score_gap INDEX QUERIES VISITS OUTPUT NQ M RANDOM_REPEATS SEED BUDGETS NPROBE DICTIONARY GREEDY_ORDERS_OR_DASH CAPTURE_P");
        const size_t nq=std::stoull(argv[5]),m=std::stoull(argv[6]),repeats=std::stoull(argv[7]);
        const uint64_t seed=std::stoull(argv[8]);
        const size_t nprobe=std::stoull(argv[10]),capture_p=std::stoull(argv[13]);
        std::vector<size_t> budgets;for(const auto& s:split(argv[9]))budgets.push_back(std::stoull(s));
        if(nq<1||m<1||m>511||!nprobe||budgets.empty()||!std::is_sorted(budgets.begin(),budgets.end())||
           budgets.front()<1||budgets.back()>=m||std::adjacent_find(budgets.begin(),budgets.end())!=budgets.end())throw std::invalid_argument("invalid configuration");
        std::filesystem::create_directories(argv[4]);const std::filesystem::path out=argv[4];
        Index index;index.load_index(argv[1]);
        if(index.metric!=MetricType::METRIC_L2||nprobe>index.nlist)throw std::invalid_argument("requires L2 IVF");
        auto [queries,available,d]=loadXvecs(argv[2]);
        if(size_t(d)!=index.d||nq>size_t(available)||m>index.d)throw std::invalid_argument("query/dictionary mismatch");
        std::ifstream input(argv[3]);std::string line;if(!std::getline(input,line))throw std::runtime_error("missing captures");
        auto header=split(line);std::map<std::string,size_t> columns;
        for(size_t i=0;i<header.size();++i)columns[header[i]]=i;
        std::vector<std::vector<Visit>> visits(nq);std::vector<bool> needed(index.nlist,false);
        while(std::getline(input,line)) {
            auto row=split(line);auto get=[&](const char* key)->const std::string&{return row.at(columns.at(key));};
            if(std::stoul(get("nprobe"))!=nprobe||std::stoul(get("pivot_count"))!=capture_p)continue;
            size_t q=std::stoull(get("query_id"));if(q>=nq)continue;
            Visit v{q,std::stoull(get("list_id")),std::stoull(get("probe_rank")),std::stoull(get("list_size")),
                std::bit_cast<float>(uint32_t(std::stoul(get("entry_tau2_bits")))),
                std::bit_cast<float>(uint32_t(std::stoul(get("centroid2query_bits"))))};
            if(v.lid>=index.nlist||v.n!=index.lists[v.lid].list_size)throw std::runtime_error("capture/index mismatch");
            visits[q].push_back(v);needed[v.lid]=true;
        }
        for(auto& vv:visits) {
            std::sort(vv.begin(),vv.end(),[](auto& a,auto& b){return a.rank<b.rank;});
            if(vv.size()!=nprobe)throw std::runtime_error("missing visits per query");
            for(size_t i=0;i<vv.size();++i)if(vv[i].rank!=i)throw std::runtime_error("duplicate/missing probe rank");
        }
        std::cout<<"Loading frozen global PCA dictionary, M="<<m<<std::endl;
        Eigen::setNbThreads(1);
        std::ifstream frozen(argv[11],std::ios::binary);Eigen::VectorXd center(index.d);
        Matrix basis(m,index.d);
        frozen.read(reinterpret_cast<char*>(center.data()),index.d*sizeof(double));
        frozen.read(reinterpret_cast<char*>(basis.data()),m*index.d*sizeof(double));
        if(!frozen||frozen.peek()!=std::char_traits<char>::eof()||!center.allFinite()||!basis.allFinite())throw std::runtime_error("invalid dictionary");
        const Matrix gram=basis*basis.transpose();
        const double orth_error=(gram-Matrix::Identity(m,m)).cwiseAbs().maxCoeff();
        if(orth_error>1e-10)throw std::runtime_error("dictionary not orthonormal");
        double old_alignment=1;
        for(size_t j=0;j<std::min(m,index.global_pivots.count?index.global_pivots.count-1:0);++j) {
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
            if(n){c.mean=c.z.rowwise().mean();c.variance=(c.z.colwise()-c.mean).array().square().rowwise().mean();}
            else {c.mean=Eigen::VectorXd::Zero(m);c.variance=Eigen::VectorXd::Zero(m);}
        }
        std::ofstream moments(out/"list_moments.csv");moments<<std::setprecision(17)<<"list_id,candidates,direction,mean,variance\n";
        for(size_t lid=0;lid<index.nlist;++lid)if(needed[lid])for(size_t dim=0;dim<m;++dim)
            moments<<lid<<','<<index.lists[lid].list_size<<','<<dim<<','<<cache[lid].mean[dim]<<','<<cache[lid].variance[dim]<<'\n';
        moments.close();
        std::map<size_t,std::vector<size_t>> reused;
        if(std::string(argv[12])!="-") {
            std::ifstream previous(argv[12]);if(!previous)throw std::runtime_error("cannot read greedy orders");
            std::getline(previous,line);
            while(std::getline(previous,line)) {auto row=split(line);if(row.size()!=5||row[2]!="greedy")continue;
                size_t q=std::stoull(row[0]);std::stringstream values(row[4]);std::vector<size_t> plan;size_t value;
                while(values>>value)plan.push_back(value);
                if(plan.size()>reused[q].size())reused[q]=std::move(plan);
            }
            for(size_t q=0;q<nq;++q){const auto& plan=reused.at(q);if(plan.size()<budgets.back())throw std::runtime_error("short greedy path");
                std::vector<bool> seen(m,false);for(size_t dim:plan){if(dim>=m||seen[dim])throw std::runtime_error("invalid greedy direction");seen[dim]=true;}}
        }
        std::filesystem::create_directories(out/"validation_samples");
        std::ofstream samples(out/"validation_samples.csv"),scores_out(out/"query_scores.csv");
        samples<<std::setprecision(17)<<"query_id,list_id,rank,candidates,entry_tau2,dimension\n";
        scores_out<<std::setprecision(17)<<"query_id,direction,query_projection,pooled_mean,pooled_variance,epsilon,query_energy,moment_mean,z_score,z_score_eps1e8,z_score_eps1e4,moment_mean_listweighted\n";
        std::ofstream query_out(out/"queries.csv"),list_out(out/"lists.csv"),orders(out/"orders.csv"),safety(out/"safety.csv");
        const char* suffix="directions,native_P,method,trial,candidates,triangle_pruned,mp_checked,mp_pruned,overall_pruned\n";
        query_out<<"query_id,"<<suffix;list_out<<"query_id,probe_rank,list_id,entry_tau2,"<<suffix;
        orders<<"query_id,directions,method,trial,order\n";
        safety<<"query_id,final_plan_checks,false_prunes,max_lb_excess\n";
        query_out<<std::setprecision(17);list_out<<std::setprecision(17);
        for(size_t qi=0;qi<nq;++qi) {
            const auto& vv=visits[qi];size_t n=0;for(auto& v:vv)n+=v.n;
            Eigen::VectorXd qvec(index.d);for(size_t a=0;a<index.d;++a)qvec[a]=double(queries[qi*index.d+a])-center[a];
            const double qnorm=qvec.squaredNorm();const Eigen::VectorXd qz=basis*qvec;
            Eigen::VectorXd mean=Eigen::VectorXd::Zero(m),variance=Eigen::VectorXd::Zero(m),list_mm=Eigen::VectorXd::Zero(m);
            for(const auto& v:vv)mean+=double(v.n)*cache[v.lid].mean;mean/=double(n);
            for(const auto& v:vv) {
                const auto& c=cache[v.lid];
                variance+=double(v.n)*(c.variance.array()+(c.mean-mean).array().square()).matrix();
                list_mm+=double(v.n)*(qz-c.mean).array().square().matrix();
            }
            variance/=double(n);list_mm/=double(n);
            const double variance_scale=std::max(variance.mean(),1e-30),epsilon=1e-6*variance_scale;
            for(size_t dim=0;dim<m;++dim) {
                const double mm=std::pow(qz[dim]-mean[dim],2);
                scores_out<<qi<<','<<dim<<','<<qz[dim]<<','<<mean[dim]<<','<<variance[dim]<<','<<epsilon<<','<<qz[dim]*qz[dim]<<','<<mm<<','
                    <<mm/(variance[dim]+epsilon)<<','<<mm/(variance[dim]+1e-8*variance_scale)<<','<<mm/(variance[dim]+1e-4*variance_scale)<<','<<list_mm[dim]<<'\n';
            }
            if(qi==0)for(size_t rank:{size_t(0),nprobe/2,nprobe-1}) {
                const auto& v=vv[rank];const auto& list=index.lists[v.lid];
                std::ofstream raw(out/"validation_samples"/(std::to_string(qi)+"_"+std::to_string(v.lid)+".f32"),std::ios::binary);
                raw.write(reinterpret_cast<const char*>(index.centroid_codes.get()+v.lid*index.d),index.d*sizeof(float));
                raw.write(reinterpret_cast<const char*>(list.get_candidate_codes()),v.n*index.d*sizeof(float));
                if(!raw)throw std::runtime_error("sample write failed");
                samples<<qi<<','<<v.lid<<','<<rank<<','<<v.n<<','<<v.tau<<','<<index.d<<'\n';
            }
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
            if(reused.empty())for(size_t t=1;t<=budgets.back();++t) {
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
            else for(size_t p:budgets)evaluate(std::vector<size_t>(reused.at(qi).begin(),reused.at(qi).begin()+p),"greedy",-1);
            std::map<std::string,std::vector<size_t>> ranked;
            for(const std::string method:{"query_energy","moment_mean","z_score","z_score_eps1e8","z_score_eps1e4","moment_mean_listweighted"}) {
                auto& order=ranked[method];order.resize(m);std::iota(order.begin(),order.end(),0);
                auto value=[&](size_t dim) {double mm=std::pow(qz[dim]-mean[dim],2);
                    if(method=="query_energy")return qz[dim]*qz[dim];
                    if(method=="moment_mean_listweighted")return list_mm[dim];
                    if(method=="moment_mean")return mm;
                    double ratio=method=="z_score_eps1e8"?1e-8:method=="z_score_eps1e4"?1e-4:1e-6;
                    return mm/(variance[dim]+ratio*variance_scale);};
                std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return value(a)>value(b);});
            }
            for(size_t p:budgets) {
                std::vector<size_t> prefix(p);std::iota(prefix.begin(),prefix.end(),0);evaluate(prefix,"prefix",-1);
                for(const auto& [method,order]:ranked)evaluate(std::vector<size_t>(order.begin(),order.begin()+p),method,-1);
                for(size_t r=0;r<repeats;++r) {
                    std::vector<size_t> random(m);std::iota(random.begin(),random.end(),0);
                    std::mt19937_64 rng(seed+qi*1000003+r*1009);std::shuffle(random.begin(),random.end(),rng);
                    random.resize(p);evaluate(random,"random",int(r));
                }
            }
            safety<<qi<<','<<plan_checks<<','<<false_prunes<<','<<std::setprecision(17)<<max_excess<<'\n';
            query_out.flush();list_out.flush();orders.flush();safety.flush();
            std::cout<<"query="<<qi<<" candidates="<<n<<" triangle_survivors="<<checked<<" false_prunes="<<false_prunes<<std::endl;
            if(false_prunes)throw std::runtime_error("false pruning detected");
        }
        if(!query_out||!list_out||!orders||!safety||!scores_out||!samples)throw std::runtime_error("output write failed");
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
