# LoQA
<div align="center">
  <img src="image/LoQA.png" width="600"/>
</div>
Low rank Quantization Adaptation (LoQA) is a novel approach that effectively fine-tunes holistic quantization parameters. 

## Installation
```bash
conda create -n loqa python=3.8
conda activate loqa
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
cd AutoGPTQ 
pip install -e .
cd ..
pip install bitsandbytes
pip install -r requirements.txt
pip install protobuf==3.20.*
pip uninstall triton
```

## Quantization
We use "./AutoGPTQ/examples/quantization/basic_usage_wikitext2.py" for quantization.
For example, you can use:
```bash
python basic_usage_wikitext2.py --bits 4 -group_size 32 --pretrained_model_dir <model_path> --quantized_model_dir <save_path>
```
If you change the group-size, you need to change the group_size in `./AutoGPTQ/auto_gptq/utils/peft_utils.py` accordingly.


## Training

Before running `./scripts/run_loqa.sh`, you need to replace `model_path` and `output_dir`:
- `model_path` should be set to the output path from the Quantization step, i.e., the `<save_path>` in `--quantized_model_dir <save_path>`.
- `output_dir` should be set to your desired output path, e.g., `./output/llama-7b-w4a16g32/`.

```bash
./scripts/run_loqa.sh
```


## Evaluation
You can use the following steps to evaluate the trained model on MMLU:

1. The MMLU data is available for download [**here**](https://people.eecs.berkeley.edu/~hendrycks/data.tar).

2. Replace the `data_dir` and `eval_directory` in `scripts/run_mmlu_eval.sh`:
   - `data_dir` should be set to the path where you extracted the data from step 1.
   - `eval_directory` should be set to your output path, e.g., `output/llama-7b-w4a16g32/checkpoint-10000/`. In LoQA, we always use the last checkpoint as the final evaluation result.

3. Run `scripts/run_mmlu_eval.sh`.
