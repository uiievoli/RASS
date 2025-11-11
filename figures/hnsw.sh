# datasets=("nuswide" "fasion_mnist_784" "msong" "sift1m" "glove25") # 
# for dataset in ${datasets[@]}; do
#     ./release/bin/hnswlib_test --dataset $dataset --tag "release"
# done

./release/bin/hnswlib_test --dataset nuswide --tag "release" --maxef 1000
./release/bin/hnswlib_test --dataset fasion_mnist_784 --tag "release" --maxef 100
./release/bin/hnswlib_test --dataset msong --tag "release" --maxef 100
./release/bin/hnswlib_test --dataset sift1m --tag "release" --maxef 1000
./release/bin/hnswlib_test --dataset glove25 --tag "release" --maxef 1000