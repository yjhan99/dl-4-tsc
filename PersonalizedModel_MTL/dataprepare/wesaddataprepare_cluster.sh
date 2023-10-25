for dataset in wesad; do
    python tune_one_cluster.py "$1" "$dataset"
done
