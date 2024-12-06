# Steps to finetune Llama 3.1-8B on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) Dataset:
1) Download the PyTorch Docker image:
```
docker pull fabiotrindaderamos/megatron_llama
```

2) Clone the repository:
```
git clone https://github.com/Fabio-Trindade/Megatron-LM.git
```

3) Run and execute the container, replacing the path to the cloned repository:
```
docker run --shm-size=150g --gpus all --rm -d -v /path/to/Megatron-LM/:/workspace/Megatron-LM --network host --name train-llama --entrypoint sleep fabiotrindaderamos/megatron_llama infinity

docker exec -it train-llama /bin/bash
```


5) Request access to the [Llama-3.1-8B model](https://huggingface.co/meta-llama/Llama-3.1-8B).
Then, download the model using your Hugging Face token, and parallelize it:
```
cd /workspace/Megatron-LM && chmod +x config_llama_for_finetuning.sh
./config_llama_for_finetuning.sh
```

6) Finetune the parallelized model:
```
./finetune_llama_3_1_8B.sh
```

# Steps for throughput calculation

After finetuning, run:
```
python3 calc_tokens_per_sec.py
```