for pnum in 1 2 3 4 8 10 12 13 14 16 18 19 20 21 22 23 25 26 27; do
  # can add True at the end to restore previous results
  # python tensorFlowWrapper.py KEmoWork/User/datasetClusterTasks-dataset_test_[${pnum}]-y_ multitask users
  python tensorFlowWrapper.py KEmoWork/Cluster/Trait/datasetClusterTasks-dataset_test_[${pnum}]-y_ multitask users
  # python tensorFlowWrapper.py KEmoWork/Cluster/Feature/datasetClusterTasks-dataset_test_[${pnum}]-y_ multitask users
done
