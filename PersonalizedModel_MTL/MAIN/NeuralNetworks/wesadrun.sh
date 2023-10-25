for pnum in 2 3 4 5 6 7 8 9 10 11 13 14 15 16 17; do
  # can add True at the end to restore previous results
  # python tensorFlowWrapper.py WESAD/User/datasetClusterTasks-dataset_test_[${pnum}]-y_ multitask users
  python tensorFlowWrapper.py WESAD/Cluster/Trait/datasetClusterTasks-dataset_test_[${pnum}]-y_ multitask users
  # python tensorFlowWrapper.py WESAD/Cluster/Feature/datasetClusterTasks-dataset_test_[${pnum}]-y_ multitask users
done
