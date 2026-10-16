#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace tribase {
constexpr size_t QUERY_PLAN_FEATURES=20;
using QueryPlanFeatures=std::array<double,QUERY_PLAN_FEATURES>;
struct QueryPlanAction {size_t directions=0;bool canonical=false;};
struct QueryPlanChoice {size_t query_id=0,directions=0;bool canonical=false;double predicted_ms=0;};
struct QueryDirectionOrder {size_t query_id=0;std::vector<size_t> directions;};
struct QueryPlanModel {
    size_t nprobe=0,d=0,m=0;int score_family=-1;
    QueryPlanFeatures mean{},scale{};
    std::vector<QueryPlanAction> actions;
    std::vector<double> coefficients;
    QueryPlanChoice choose(const QueryPlanFeatures& raw) const {
        QueryPlanFeatures z{};
        for(size_t i=0;i<z.size();++i)z[i]=std::clamp((raw[i]-mean[i])/scale[i],-6.,6.);
        double best=std::numeric_limits<double>::infinity();size_t winner=0;
        for(size_t a=0;a<actions.size();++a) {
            const double* c=coefficients.data()+a*(QUERY_PLAN_FEATURES+1);double log_ms=c[0];
            for(size_t i=0;i<z.size();++i)log_ms+=c[i+1]*z[i];
            if(log_ms<best){best=log_ms;winner=a;}
        }
        return {0,actions[winner].directions,actions[winner].canonical,std::exp(std::clamp(best,-20.,20.))};
    }
};

// Feature extraction scans only query projections and coarse-list statistics.
// It never reads candidate vectors/signatures or the search heap.
template<class ListSize>
QueryPlanFeatures makeQueryPlanFeatures(float qnorm,const float* z,size_t m,
                                        const size_t* order,const float* coarse,size_t np,
                                        size_t d,ListSize list_size) {
    QueryPlanFeatures f{};const double norm=std::max(double(qnorm),1e-20);
    double sum=0,sum2=0,n=0,n2=0,maxn=0;
    for(size_t j=0;j<np;++j){double v=coarse[j],s=list_size(j);sum+=v;sum2+=v*v;n+=s;n2+=s*s;maxn=std::max(maxn,s);}
    const double avg=sum/np,avgn=n/np;
    f[0]=std::log1p(norm);f[1]=coarse[0]/norm;f[2]=coarse[np-1]/norm;f[3]=avg/norm;
    f[4]=std::sqrt(std::max(0.,sum2/np-avg*avg))/norm;f[5]=std::log1p(n);
    f[6]=std::sqrt(std::max(0.,n2/np-avgn*avgn))/std::max(avgn,1.);
    f[7]=maxn/std::max(avgn,1.);
    double energy=0,maxe=0,canonical16=0;
    for(size_t i=0;i<m;++i){double e=double(z[i])*z[i];energy+=e;maxe=std::max(maxe,e);if(i<16)canonical16+=e;}
    const double denom=std::max(energy,1e-20);f[8]=energy/norm;
    const size_t checkpoints[]{8,16,32,64};double cumulative=0;size_t checkpoint=0;
    for(size_t i=0;i<m;++i){double v=z[order[i]];cumulative+=v*v;
        while(checkpoint<4&&i+1>=std::min(checkpoints[checkpoint],m))f[9+checkpoint++]=cumulative/denom;}
    f[13]=canonical16/denom;f[14]=maxe/denom;
    double entropy=0;for(size_t i=0;i<m;++i){double p=double(z[i])*z[i]/denom;if(p>0)entropy-=p*std::log(p);}
    f[15]=entropy/std::log(std::max<size_t>(m,2));f[16]=std::log1p(std::max(0.f,coarse[0]));
    f[17]=(coarse[np-1]-coarse[0])/std::max(double(coarse[0]),1e-20);
    f[18]=std::log1p(np);f[19]=std::log1p(d);return f;
}
} // namespace tribase
