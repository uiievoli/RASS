include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_FILES
    Clustering.cpp
    Index.cpp
    IVF.cpp
    Microblock.cpp
    MultiPivot.cpp
    Ppd.cpp
    # utils.cpp
)

add_library(Tribase STATIC ${SRC_FILES})

target_link_libraries(Tribase PUBLIC OpenMP::OpenMP_CXX
                                    Eigen3::Eigen
                                    MKL::MKL)

add_executable(main main.cpp)
target_link_libraries(main Tribase faiss)

add_executable(query query.cpp)
target_link_libraries(query Tribase faiss)

add_executable(ppd_query ppd_query.cpp)
target_link_libraries(ppd_query Tribase faiss)

# Offline controlled experiment: copy the exact PPD PCA basis into a native bank.
add_executable(import_ppd_basis import_ppd_basis.cpp)
target_link_libraries(import_ppd_basis Tribase faiss)

add_executable(replay_pivot_choice replay_pivot_choice.cpp)
target_link_libraries(replay_pivot_choice Tribase faiss)

add_executable(export_pivot_subsets export_pivot_subsets.cpp)
target_link_libraries(export_pivot_subsets Tribase faiss)

add_executable(query_subset_oracle query_subset_oracle.cpp)
target_link_libraries(query_subset_oracle Tribase faiss)
add_executable(query_score_gap query_score_gap.cpp)
target_link_libraries(query_score_gap Tribase faiss)

add_executable(benchmark_query_energy benchmark_query_energy.cpp)
target_link_libraries(benchmark_query_energy Tribase faiss)

add_executable(build_global_pca_dictionary build_global_pca_dictionary.cpp)
target_link_libraries(build_global_pca_dictionary Tribase faiss)

add_executable(prune_comparison prune_comparison.cpp)
target_link_libraries(prune_comparison Tribase faiss)

add_executable(index_build_benchmark index_build_benchmark.cpp)
target_link_libraries(index_build_benchmark Tribase faiss)

add_executable(import_legacy_index import_legacy_index.cpp)
target_link_libraries(import_legacy_index Tribase faiss)

add_executable(uniform_dataset_generate uniform_dataset_generate.cpp)

add_subdirectory(tests)
