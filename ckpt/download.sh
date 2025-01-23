python download_script.py \
    --source hf \
    --repo_id microsoft/wavlm-large \
    --filename pytorch_model.bin \
    --save_path ./WavLM-Large.pt

python download_script.py \
    --source hf \
    --repo_id facebook/w2v-bert-2.0 \
    --filename model.safetensors \
    --save_path \
    ./codec_ckpt/hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b/model.safetensors

python download_script.py \
     --source hf \
     --repo_id HKUSTAudio/xcodec2 \
     --filename ckpt/epoch=4-step=1400000.ckpt \
     --save_path ./codec_ckpt/epoch=4-step=1400000.ckpt

python download_script.py \
    --source hf \
    --repo_id BeauKang01/LLaSE \
    --filename best.pt.tar \
    --save_path ./best.pt.tar