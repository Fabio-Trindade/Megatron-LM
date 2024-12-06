import pandas as pd
from datetime import datetime
import ast

# train_step
df = pd.read_csv("train_time_log.csv")
df["init_datetime"] = df["init_datetime"].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df["final_datetime"] = df["final_datetime"].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

first_row = df.iloc[0]

# train config
num_gpus = 8
seq_len = first_row["seq_length"]
num_micro_batches =first_row["num_microbatches"]
micro_batch_size =  first_row["micro_batch_size"]
global_batch_size = num_micro_batches * micro_batch_size
num_tokens_per_iter =  seq_len * num_micro_batches * micro_batch_size
print(f"Micro batches: {first_row['num_microbatches']}, Micro batch size: {first_row['micro_batch_size']}, Sequence len: {first_row['seq_length']}, Num tokens per iter:  {num_tokens_per_iter}")
print('-'*50)

# for train_step calculation, tokens/s are computed assuming that each row has the same values for num_microbatches, micro_batch_size, and seq_length.
# therefore, we must ensure all data have consistent values.
assert(all(df["num_microbatches"]  == first_row["num_microbatches"] ))
assert(all(df["micro_batch_size"]  == first_row["micro_batch_size"] ))
assert(all(df["seq_length"]  == first_row["seq_length"] ))

# each GPU processes the same data in the same iteration in parallel,
# resulting in the data being written "|GPU|" times
assert(len(df) % num_gpus == 0)


# calculates the time for each iteration by subtracting the maximum and minimum times,
# i.e., the minimum and maximum values in each set of |GPU| rows
new_df = pd.DataFrame({"times": []})
for i in range(0, len(df), num_gpus):
    temp_df = df.iloc[i:i + num_gpus]
    elapsed_time = (temp_df["final_datetime"].max() - temp_df["init_datetime"].min()).total_seconds()
    new_df = pd.concat([new_df, pd.DataFrame({"times": [elapsed_time]})], ignore_index=True)

number_of_iters = 43 - 1
total_tokens = num_tokens_per_iter * len(new_df)
# train_step time: collected from forward, backward and optimizer operations (zero_grad and step)
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py#L741
print(f"(train_step): {((num_tokens_per_iter) / new_df['times'].mean()):.2f} tokens/s")
print('-'*50)

# train time: start before line 376 and finish after line 381
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py#L217
final_training_time = 312.984607 # collected from terminal, being more precise
print(f"(train): {((number_of_iters*num_tokens_per_iter) / final_training_time):.2f} tokens/s")
print('-'*50)

# finetuning time: start before line 278 and finish after line 284
# https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py
final_finetuning_time = 333.07548 # collected from terminal
print(f"(finetune): {((number_of_iters*num_tokens_per_iter) / final_finetuning_time):.2f} tokens/s")
print('-'*50)

# calculating total tokens from other perspectives to ensure the total tokens above are correct
print("Total tokens (train_step): ", total_tokens)
print('-'*50)
# 1) by number of samples
# each training sample has 'seq_len' tokens. therefore, the number of processed tokens is seq_len * train_samples
total_train_samples = 691  # collected from terminal
print("Total tokens (train_samples): ", total_train_samples * seq_len)
reason_for_inconsistency = (
    "reason for inconsistency: The number of iterations goes from the initial iter (in this case, 1) "
    "until train-iters (specified from the terminal, in this case, 43 - 1 ), i.e., iterations run 42 times.\n"
    "this way, we can consider that the number of samples equals 42 * 8 (num_microbatches) * 2 (micro_batch_size)\n"
    "this results in a number of tokens equal to 672 (num_samples) * 8192 (seq_len) = 5.505.024"
)
print(reason_for_inconsistency)
print('-'*50)


# 2) by the shapes of the outputs collected directly from the forward function
# for the same reasons as calculations during train_step, the number of real outputs equals the number of collected data 
# normalized by the number of GPUs.
# in this approach, the number of processed tokens equals sum(shape[0] * shape[1] / |GPU|) for each row.
# shapes at indices 0 and 1 correspond to seq_len or micro_batch_size.
df_2 = pd.read_csv("forward_info.csv")
df_2['tokens_out_shape'] = df_2['tokens_out_shape'].map(lambda x: x.strip("[]").split(','))
df_2['processed_tokens'] = df_2['tokens_out_shape'].map(lambda x: int(x[0]) * int(x[1]))
total_tokens_forward = df_2['processed_tokens'].sum() / num_gpus
print("Total tokens (forward_step): ", int(total_tokens_forward))
print('-'*50)


