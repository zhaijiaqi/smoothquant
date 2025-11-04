# Llama-2-7B
CUDA_VISIBLE_DEVICES=0 python -m smoothquant.ppl_eval \
    --alpha 0.85 \
    --model_path meta-llama/Llama-2-7b-hf \
    --act_scales_path act_scales/llama-2-7b.pt \
    --smooth \
    --quantize

# Mistral-7B
CUDA_VISIBLE_DEVICES=0 python -m smoothquant.ppl_eval \
    --alpha 0.8 \
    --model_path mistralai/Mistral-7B-v0.1 \
    --act_scales_path act_scales/Mistral-7B-v0.1.pt \
    --smooth \
    --quantize



# Falcon-7B
CUDA_VISIBLE_DEVICES=0 python -m smoothquant.ppl_eval \
    --alpha 0.6 \
    --model_path tiiuae/falcon-7b \
    --act_scales_path act_scales/falcon-7b.pt \
    --smooth \
    --quantize


# Meta-Llama-3-8B
CUDA_VISIBLE_DEVICES=0 python -m smoothquant.ppl_eval \
    --alpha 0.85 \
    --model_path meta-llama/Meta-Llama-3-8B \
    --act_scales_path act_scales/Meta-Llama-3-8B.pt \
    --smooth \
    --quantize



