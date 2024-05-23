export CUDA_VISIBLE_DEVICES=0
output_dir=./path_to_your_output_dir
dataset=alpaca

if [ "$dataset" = "alpaca" ]; then
    max_steps=10000
elif [ "$dataset" = "flan-v2" ]; then
    max_steps=20000
else
    echo "Unknown dataset"
    exit 1
fi

python loqa.py \
    --model_path path_to_your_quantized_model \
    --output_dir $output_dir \
    --dataset $dataset \
    --max_steps $max_steps \
    --do_eval True \
    --max_eval_samples 1000 