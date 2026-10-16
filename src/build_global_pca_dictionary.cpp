#include "Index.h"
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace tribase;
using Matrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::invalid_argument("Usage: build_global_pca_dictionary INDEX OUTPUT DIRECTIONS");
        const size_t m=std::stoull(argv[3]);
        if(std::filesystem::exists(argv[2]))throw std::invalid_argument("dictionary already exists; choose a new output");
        Index index;index.load_index(argv[1]);
        if(index.metric!=METRIC_L2||!m||m>index.d||m+1>MULTIPIVOT_MAX_PIVOTS)
            throw std::invalid_argument("invalid L2 dictionary dimensions");
        Eigen::setNbThreads(1);
        auto selected=selectPcaPivots(index.centroid_codes.get(),index.nlist,index.d,m+1,nullptr,-1);
        if(selected.metadata.count!=m+1)throw std::runtime_error("incomplete dictionary");
        Eigen::VectorXd center(index.d);Matrix basis(m,index.d);
        for(size_t a=0;a<index.d;++a)center[a]=selected.metadata.codes[a];
        for(size_t j=0;j<m;++j) {
            for(size_t a=0;a<index.d;++a)basis(j,a)=selected.metadata.U[j*index.d+a];
            for(int pass=0;pass<2;++pass)for(size_t k=0;k<j;++k)basis.row(j)-=basis.row(j).dot(basis.row(k))*basis.row(k);
            basis.row(j).normalize();
        }
        const double error=(basis*basis.transpose()-Matrix::Identity(m,m)).cwiseAbs().maxCoeff();
        if(!basis.allFinite()||error>1e-10)throw std::runtime_error("invalid orthonormal dictionary");
        const std::filesystem::path path(argv[2]);
        if(path.has_parent_path())std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path,std::ios::binary);
        out.write(reinterpret_cast<const char*>(center.data()),center.size()*sizeof(double));
        out.write(reinterpret_cast<const char*>(basis.data()),basis.size()*sizeof(double));
        out.close();if(!out)throw std::runtime_error("dictionary write failed");
        std::ofstream geometry(path.string()+".geometry.csv");
        geometry<<std::setprecision(17)<<"dimension,directions,training_ivf_centroids,orthogonality_max_error\n"
            <<index.d<<','<<m<<','<<index.nlist<<','<<error<<'\n';
        std::cout<<"Frozen global PCA dictionary: D="<<index.d<<" M="<<m<<" centroid_training="<<index.nlist<<" orthogonality_error="<<error<<std::endl;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
