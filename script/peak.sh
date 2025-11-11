#!/bin/bash

# 需要监控的查询脚本
script="./release/bin/query"

# 构建文件名
output_file="logs/mem.txt"

# 启动查询脚本并传递参数
taskset -c 0 $script &
pid=$!

# 初始化一个变量来存储峰值内存使用
peak_memory_usage=0

# 监控内存使用情况
while true; do
    # 获取当前时间戳
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")

    # 使用 ps 获取内存使用情况 (RSS)
    memory_usage=$(ps -p $pid -o rss=)

    # 检查脚本是否仍在运行
    if ! ps -p $pid > /dev/null; then
        echo "$script with query TEST_$(printf "%02d" $i) has exited"
        # 在进程退出时记录最后一次内存使用
        echo "$timestamp Memory Usage: $memory_usage KB" >> "$output_file"
        break
    fi

    # 更新峰值内存使用
    if (( memory_usage > peak_memory_usage )); then
        peak_memory_usage=$memory_usage
    fi

    # 输出内存使用情况到文件
    echo "$timestamp Memory Usage: $memory_usage KB" >> "$output_file"

    # 每 1 毫秒记录一次
    sleep 0.001
done

# 在文件末尾记录峰值内存使用
echo "Peak Memory Usage: $peak_memory_usage KB" >> "$output_file"
