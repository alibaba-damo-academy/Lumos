############################# Public code for pre-tokenization ##############################
# cd to Project

### Example code for pre-tokenize images
# Step 1: pre-tokenize
python pre_tokenize/parallel_tokenization_image.py

# Step 2: obtain data json
python -u pre_tokenize/concat_record.py  \
--sub_record_dir  pre_tokenize/data/test_image  \
--save_path       pre_tokenize/data/test_image/merge-record.json \
--merge_sub_tasks \


### Example code for pre-tokenize videos
# Step 1: pre-tokenize
python pre_tokenize/parallel_tokenization_video.py

# Step 2: obtain data json
python -u pre_tokenize/concat_record.py  \
--sub_record_dir  pre_tokenize/data/test_video  \
--save_path       pre_tokenize/data/test_video/merge-record.json \
--merge_sub_tasks \