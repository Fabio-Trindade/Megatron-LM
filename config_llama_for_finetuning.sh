# downloading llama
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir Llama-3.1-8B

echo
echo "Converting to parallel model..."
python tools/checkpoint/convert.py \
--bf16 \
--model-type GPT \
--loader llama_mistral \
--saver mcore \
--target-tensor-parallel-size 2 \
--target-pipeline-parallel-size 4 \
--checkpoint-type hf \
--load-dir Llama-3.1-8B/ \
--save-dir ./m-Llama-3.1-8B-2t-4p \
--tokenizer-model ./Llama-3.1-8B/ \
--model-size llama3-8B 

