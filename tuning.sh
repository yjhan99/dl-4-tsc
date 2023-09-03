# This script was created because of probable memory leakage in version of tensorflow/keras used in this project
# for clas in fcnM resnetM cnnM mlpLstmM; do
for clas in resnetM mlpLstmM; do
  for dataset in wesad; do
    for max_eval in $(seq 1 10); do
      for i_fold in 00 01 02 03 04; do
        python tune_one.py "$1" "$dataset"_fold_05_"$i_fold" $clas "$max_eval"
      done
    done
  done
done