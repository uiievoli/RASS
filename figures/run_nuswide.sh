# just `bash figures/run_nuswide.sh` to run


if [ ! -f "benchmarks/nuswide/origin/nuswide_base.fvecs" ]; then
    pipx install gdown
    gdown https://drive.google.com/file/d/12wFLDNStJU02pEn7VcAs00LyS7uzcAbl/view?usp=sharing --fuzzy
    unzip -o benchmarks.zip
fi

if docker ps -a --format '{{.Names}}' | grep -qw tribase-dev; then
    echo "container tribase-dev already exists. Starting the container..."
    docker start tribase-dev
else
    docker pull panjd123/tribase-env:latest
    docker run -d \
    --user "$(id -u):$(id -g)" \
    --name tribase-dev \
    -v .:/app/tribase \
    --restart always \
    panjd123/tribase-env \
    tail -f /dev/null
fi

docker exec -it tribase-dev bash script/build.sh

dataset="nuswide"
nprobes="1 3 5 7 10 30 50 70 100 150 200 250 300 350 400 450 500 510 513 516 0"

faiss_output_csv_file="./logs/edge-recall-qps-faiss_tmp.csv"
rm $faiss_output_csv_file
docker exec -it -e EDGE_DEVICE_ENABLED=1 tribase-dev ./release/bin/query --benchmarks_path ./benchmarks --dataset $dataset \
    --nprobes $nprobes \
    --run_faiss \
    --loop 3 \
    --csv $faiss_output_csv_file \
    --cache \
    --verbose

output_csv_file="./logs/edge-recall-qps-tribase_tmp.csv"
rm $output_csv_file
docker exec -it -e EDGE_DEVICE_ENABLED=1 tribase-dev ./release/bin/query --benchmarks_path ./benchmarks --dataset $dataset \
    --nprobes $nprobes \
    --sub_nprobe_ratio 1 \
    --opt_levels OPT_NONE OPT_TRIANGLE OPT_SUBNN_L2 OPT_TRI_SUBNN_L2 OPT_SUBNN_IP OPT_TRI_SUBNN_IP OPT_ALL \
    --loop 3 \
    --csv $output_csv_file \
    --verbose

output_csv_file="./logs/standard-recall-qps-tribase_tmp.csv"
rm $output_csv_file
docker exec -it tribase-dev ./build/bin/query --benchmarks_path ./benchmarks --dataset $dataset \
    --nprobes $nprobes \
    --sub_nprobe_ratio 1 \
    --opt_levels OPT_NONE OPT_TRIANGLE OPT_SUBNN_L2 OPT_TRI_SUBNN_L2 OPT_SUBNN_IP OPT_TRI_SUBNN_IP OPT_ALL \
    --loop 1 \
    --csv $output_csv_file \
    --verbose --cache

docker exec -it tribase-dev ./release/bin/hnswlib_test --dataset nuswide --tag "release" --maxef 1000 --tag tmp # logs/hnswlib_add_time_tmp.csv

docker exec -it tribase-dev python figures/fig7.py --faiss_log "./logs/edge-recall-qps-faiss_tmp.csv" --tribase_log "./logs/edge-recall-qps-tribase_tmp.csv"
