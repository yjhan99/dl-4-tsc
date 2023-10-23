for pnum in 1 2 3 4 8 10 12 13 14 16 18 19 20 21 22 23 25 26 27; do
  # python make_datasets.py --datafile="./Dataset/KEmoWork/Feature/dataset_test_[${pnum}].csv" --task_type="users"
  python make_datasets.py --datafile="./Dataset/KEmoWork/Trait/dataset_test_[${pnum}].csv" --task_type="users"
done
