#include "Index.h"
#include "Ppd.h"
#include "utils.h"
#include <argparse/argparse.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace tribase;
using Clock = std::chrono::steady_clock;
template<class F> double timed(F&& f) {
    const auto begin=Clock::now(); f();
    return std::chrono::duration<double>(Clock::now()-begin).count();
}

int main(int argc, char** argv) {
    argparse::ArgumentParser args("index_build_benchmark");
    args.add_argument("--base").required();
    args.add_argument("--out").required();
    args.add_argument("--nlist").scan<'u',size_t>().default_value(size_t{1000});
    args.add_argument("--pivot-count").scan<'u',size_t>().default_value(size_t{16});
    args.add_argument("--global-pivot-count").scan<'u',size_t>().default_value(size_t{0});
    args.add_argument("--per-list-pivot-count").scan<'u',size_t>().default_value(size_t{0});
    args.add_argument("--seed").scan<'u',uint64_t>().default_value(uint64_t{0});
    args.add_argument("--ppd-train-samples").scan<'u',size_t>().default_value(size_t{0});
    args.add_argument("--pca-without-triangle").flag();
    try {
        args.parse_args(argc,argv);
        const std::filesystem::path out(args.get<std::string>("--out"));
        if (std::filesystem::exists(out) && !std::filesystem::is_empty(out))
            throw std::runtime_error("Output must be a new or empty repeat directory");
        std::filesystem::create_directories(out);
        const size_t nlist=args.get<size_t>("--nlist"), P=args.get<size_t>("--pivot-count");
        const size_t global_P=args.get<size_t>("--global-pivot-count")?args.get<size_t>("--global-pivot-count"):P;
        const size_t per_list_P=args.get<size_t>("--per-list-pivot-count")?args.get<size_t>("--per-list-pivot-count"):P;
        size_t n=0; int dim=0; std::unique_ptr<float[]> base;
        const double input_seconds=timed([&]{std::tie(base,n,dim)=loadXvecs(args.get<std::string>("--base"));});
        const auto valid_P=[&](size_t p){return p>=2 && p<=512 && p<=static_cast<size_t>(dim)+1;};
        if (nlist==0 || n<nlist || !valid_P(global_P) || !valid_P(per_list_P))
            throw std::invalid_argument("Require 0<nlist<=N, 2<=P<=min(512,D+1)");
        const bool triangle=!args.get<bool>("--pca-without-triangle");
        const auto source=out/"ivf.index";
        double train=0,add=0,ivf_save=0;
        {
            Index index(dim,nlist,0,MetricType::METRIC_L2,OptLevel::OPT_NONE);
            std::cout<<"START shared IVF clustering"<<std::endl;
            train=timed([&]{index.train(n,base.get());});
            std::cout<<"START shared IVF assignment and list construction"<<std::endl;
            add=timed([&]{index.add(n,base.get());});
            ivf_save=timed([&]{index.save_index(source.string());});
        }
        base.reset();
        const double common=train+add;
        std::ofstream csv(out/"build_times.csv");
        csv.precision(17);
        csv<<"method,n,d,nlist,pivot_count,ppd_training_samples,pca_triangle,input_load_seconds,ivf_train_seconds,ivf_add_seconds,common_ivf_build_seconds,index_reload_seconds,triangle_build_seconds,pca_selection_seconds,signature_build_seconds,ppd_build_seconds,extra_build_seconds,total_build_seconds,base_save_seconds,method_save_seconds,total_save_seconds,build_plus_save_seconds,logical_index_bytes\n";
        auto row=[&](const std::string& method,size_t pivots,size_t samples,int tri,double reload,
                     double triangle_time,double selection,double signatures,double ppd,double extra,
                     double save,uintmax_t bytes) {
            const double save_total=method=="ivf"?ivf_save:method=="ppd"?ivf_save+save:save;
            csv<<method<<','<<n<<','<<dim<<','<<nlist<<','<<pivots<<','<<samples<<','<<tri<<','
               <<input_seconds<<','<<train<<','<<add<<','<<common<<','<<reload<<','
               <<triangle_time<<','<<selection<<','<<signatures<<','<<ppd<<','<<extra<<','
               <<common+extra<<','<<ivf_save<<','<<save<<','<<save_total<<','
               <<common+extra+save_total<<','<<bytes<<'\n';
            csv.flush();
            std::cout<<method<<": common="<<common<<" extra="<<extra<<" build="<<common+extra
                     <<" save="<<save_total<<" seconds"<<std::endl;
        };
        const auto base_bytes=std::filesystem::file_size(source);
        row("ivf",0,0,0,0,0,0,0,0,0,0,base_bytes);
        for (const std::string method:{"tribase_triangle","ppd","pca_per_list","pca_global"}) {
            const size_t method_P=method=="pca_global"?global_P:per_list_P;
            std::cout<<"START "<<method<<std::endl;
            Index index;
            const double reload=timed([&]{index.load_index(source.string());});
            double tri_time=0,selection=0,signature=0,ppd_time=0,extra=0,save=0;
            size_t ppd_samples=0; uintmax_t bytes=0;
            const auto destination=out/(method=="ppd"?"ppd.sidecar":method+".index");
            if (method=="ppd") {
                PpdIndex ppd;
                size_t requested=args.get<size_t>("--ppd-train-samples");
                if (requested==0) requested=n;
                ppd_time=timed([&]{ppd.build(index,requested,args.get<uint64_t>("--seed"));});
                extra=ppd_time; ppd_samples=ppd.training_samples;
                save=timed([&]{ppd.save(destination.string());});
                bytes=base_bytes+std::filesystem::file_size(destination);
            } else {
                const auto begin=Clock::now();
                if (method=="tribase_triangle" || triangle)
                    tri_time=timed([&]{index.ensure_triangle_radii();index.opt_level=OptLevel::OPT_TRIANGLE;});
                if (method!="tribase_triangle") {
                    index.configure_multipivot(method=="pca_global"?MultiPivotScope::GLOBAL:MultiPivotScope::PER_LIST,
                                              "pca",method_P,args.get<uint64_t>("--seed"));
                    index.rebuild_multipivot_metadata();
                    selection=index.multipivot_selection_seconds;
                    signature=index.multipivot_candidate_distance_seconds;
                }
                extra=std::chrono::duration<double>(Clock::now()-begin).count();
                save=timed([&]{index.save_index(destination.string());});
                bytes=std::filesystem::file_size(destination);
            }
            row(method,method=="tribase_triangle"?1:method=="ppd"?0:method_P,ppd_samples,
                method=="tribase_triangle"?1:method=="ppd"?0:triangle,reload,
                tri_time,selection,signature,ppd_time,extra,save,bytes);
        }
        if (!csv) throw std::runtime_error("Failed to write timing CSV");
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n'; return 1;
    }
}
