# This script was created because of probable memory leakage in version of tensorflow/keras used in this project
# for clas in fcnM resnetM cnnM mlpLstmM; do
for clas in fcnM; do
  for dataset in wesad; do
    for max_eval in $(seq 1 1); do
      for i_fold in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14; do
      # for i_fold in 00 01 02; do
        python tune_one_cluster.py "$1" "$dataset"_fold_15_"$i_fold" $clas "$max_eval"
        # python tune_one.py "$1" "$dataset"_fold_13_"$i_fold" $clas "$max_eval"
      done
    done
  done
done