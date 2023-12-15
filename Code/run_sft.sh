CUDA_VISIBLE_DEVICES=4,5,6 \
accelerate launch --main_process_port 29502 sft_llama2.py \
--output_dir=sft_roscoe_gsm8k_small \
--model_name mistralai/Mistral-7B-v0.1 \
--seq_length 512 \
--data_files datasets/sft_roscoe_gsmk8k_small.jsonl \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 8 \
--learning_rate 1e-5 \
--max_steps 5000 \
--warmup_steps 1000 \