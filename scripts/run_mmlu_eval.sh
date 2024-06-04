export CUDA_VISIBLE_DEVICES=0

python mmlu_test/eval_loqa.py \
    --eval_directory your_path/checkpoint-10000/ \
    --results_file mmlu_results.csv \
    --data_dir your_path/data/