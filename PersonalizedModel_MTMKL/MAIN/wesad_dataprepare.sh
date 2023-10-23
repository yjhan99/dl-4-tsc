for pnum in 2 3 4 5 6 7 8 9 10 11 13 14 15 16 17; do
  # python make_datasets.py --datafile="./Dataset/WESAD/Feature/dataset_test_[${pnum}].csv" --task_type="users"
  python make_datasets.py --datafile="./Dataset/WESAD/Trait/dataset_test_[${pnum}].csv" --task_type="users"
done
