Step 1: Download the dataset files "preprocessing/uniformly_sampled.tsv", "preprocessing/recall_oriented.tsv" and "preprocessing/precision_oriented.tsv" from https://blog.twitter.com/engineering/en_us/a/2015/evaluating-language-identification-performance

Step 2: Run "preprocessing/1_merge_uniform_precision.py" and "preprocessing/2_create_ids_only_files.py"

Step 3: Download the tweet data using the created id lists "preprocessing/input/uniform_precision_ids.tsv" and "preprocessing/input/recall_oriented_ids.tsv" and the hydrator program https://github.com/DocNow/hydrator

Step 4: Configure and run "preprocessing/preprocessing.py" to preprocess the data and encode the dataset into integer format

Step 5: Configure and run classifier/main.py to train a model