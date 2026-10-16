#include "Index.h"
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

using namespace tribase;
static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out; std::stringstream stream(line); std::string s;
    while (std::getline(stream,s,',')) out.push_back(s);
    return out;
}

// Export persisted PCA signatures and exact distances for controlled offline
// subset experiments. Input rows are selected real list-entry snapshots.
int main(int argc,char** argv) {
    try {
        if(argc!=5) throw std::invalid_argument("Usage: export_pivot_subsets INDEX QUERIES SELECTED_CSV OUTPUT_DIR");
        Index index; index.load_index(argv[1]);
        if(index.metric!=MetricType::METRIC_L2 || index.multipivot_method!="pca")
            throw std::invalid_argument("requires PCA/L2");
        auto [queries,nq,d]=loadXvecs(argv[2]);
        if(size_t(d)!=index.d) throw std::invalid_argument("dimension mismatch");
        std::ifstream input(argv[3]); std::string line;
        if(!std::getline(input,line)) throw std::runtime_error("missing snapshots");
        auto header=split(line); std::map<std::string,size_t> cols;
        for(size_t i=0;i<header.size();++i) cols[header[i]]=i;
        std::filesystem::create_directories(argv[4]);
        size_t exported=0;
        while(std::getline(input,line)) {
            auto row=split(line);
            auto get=[&](const char* key)->const std::string& {return row.at(cols.at(key));};
            const size_t qid=std::stoull(get("query_id")),lid=std::stoull(get("list_id"));
            if(qid>=size_t(nq)||lid>=index.nlist) throw std::runtime_error("invalid snapshot");
            auto& list=index.lists[lid]; auto& g=list.pivots;
            const auto& geometry=index.multipivot_scope==MultiPivotScope::GLOBAL?index.global_pivots:g;
            const size_t begin=std::stoull(get("scan_begin")),end=std::stoull(get("scan_end"));
            if(begin>=end || end>list.list_size || g.count<2 ||
               g.signature_data.size()!=g.count*list.list_size) throw std::runtime_error("invalid signatures/bounds");
            const size_t m=g.count-1,n=end-begin;
            std::vector<float> z(g.count),centered(index.d); std::vector<double> sq(g.count); float rho;
            const float* query=queries.get()+qid*index.d;
            if(!computePcaQuerySignature(query,index.d,geometry,g.count,sq.data(),z.data(),&rho,centered.data()))
                throw std::runtime_error("signature failed");
            const double qnorm=index.multipivot_scope==MultiPivotScope::GLOBAL?sq[0]:
                std::bit_cast<float>(uint32_t(std::stoul(get("centroid2query_bits"))));
            const float tau=std::bit_cast<float>(uint32_t(std::stoul(get("entry_tau2_bits"))));
            std::vector<double> data{double(n),double(m),qnorm,tau};
            data.insert(data.end(),z.begin(),z.begin()+m);
            for(size_t i=begin;i<end;++i) data.push_back(g.candidate_centroid_squared.at(i));
            for(size_t j=0;j<m;++j) for(size_t i=begin;i<end;++i)
                data.push_back(g.signature_data[j*list.list_size+i]);
            for(size_t i=begin;i<end;++i) {
                const float* x=list.get_candidate_codes()+i*index.d; double distance=0;
                for(size_t j=0;j<index.d;++j) {const double delta=double(query[j])-x[j];distance+=delta*delta;}
                data.push_back(distance);
            }
            auto path=std::filesystem::path(argv[4])/(std::to_string(qid)+"_"+std::to_string(lid)+".bin");
            std::ofstream output(path,std::ios::binary);
            output.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(double));
            if(!output) throw std::runtime_error("snapshot write failed");
            ++exported;
        }
        std::cout<<"Exported "<<exported<<" snapshots\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
