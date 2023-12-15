# accelerate launch --main_process_port 30000 
CUDA_VISIBLE_DEVICES=1 \
python3 dpo_fine_tune.py \
--model_name_or_path=sft_roscoe_gsm8k_small/final_merged_checkpoint  \
--hf_model_name mistralai/Mistral-7B-v0.1 \
--learning_rate 1e-5 \
--max_steps 3500 \
--logging_steps 10 \
--warmup_steps 500 \
--train_data_dir "datasets/dpo_train" \
--test_data_dir "datasets/dpo_test" \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 12 \
--max_length 512 \
--data_files roscoe_gsmk8k_small.jsonl \
--loss_type hinge \
--output_dir=dpo_gsm8k_roscoe_v3.1 \
--beta 0.4