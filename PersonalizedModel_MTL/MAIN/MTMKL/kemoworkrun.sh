# for pnum in 1 2 3 4 8 10 12 13 14 16 18 19 20 21 22 23 25 26 27; do
# for pnum in 1 2 4 16 18 21 23; do
for pnum in 1; do
  # python MTMKLWrapper.py KEmoWork/Cluster/Trait/datasetClusterTasks-dataset_test_[${pnum}]-y_ clusters
  python MTMKLWrapper.py KEmoWork/Cluster/Feature/datasetClusterTasks-dataset_test_[${pnum}]-y_ clusters
done
