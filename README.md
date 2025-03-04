# **LLaSE: Maximizing Acoustic Preservation for LLaMA-based Speech Enhancement**  

Boyi Kang\*¹, Xinfa Zhu\*¹, Zihan Zhang¹, Zhen Ye², Ziqian Wang¹, Wei Xue², Lei Xie¹  
¹ **Audio, Speech and Language Processing Group (ASLP@NPU)**,  
School of Computer Science, Northwestern Polytechnical University, Xi’an, China  
² **The Hong Kong University of Science and Technology**

---
## News
LLaSE-G1 have been released.

Github: https://github.com/Kevin-naticl/LLaSE-G1

Huggingface: https://huggingface.co/ASLP-lab/LLaSE-G1

Paper: https://arxiv.org/abs/2503.00493

## Abstract
Language Models (LMs) have shown strong semantic understanding and contextual modeling capabilities, which have recently flourished in generative speech enhancement. However, most LM-based speech enhancement approaches focus on semantic information while ignoring the key vital of acoustic information, which leads to acoustic inconsistency after enhancement, including speaker timbre varaitions and intonation. This paper proposes LLaSE, a LLaMA-based language model for Speech Enhancement. To address the challenge of acoustic inconsistency, LLaSE takes continuous representations from WavLM as input and predicts speech tokens from XCodec2, a recently released efficient Codec, maximizing acoustic preservation. Experimental results demonstrate that LLaSE achieves state-of-the-art performance on speech enhancement, offering a robust and scalable solution for speech denoising and quality improvement.

## Demo Page

Demo Page: https://kevin-naticl.github.io/LLaSE-Demopage/

![Overall Architecture of LLaSE](LLaSE.png)

## DNSMOS results on DNS Challenge testset

| Model       | Type          | Testset          | SIG     | BAK     | OVRL    |
|-------------|---------------|------------------|---------|---------|---------|
| Unprocessed | -             | syn_with_reverb  | 1.76    | 1.50    | 1.39    |
|             |               | syn_no_reverb    | 3.39    | 2.62    | 2.48    |
|             |               | real_recording   | 3.05    | 2.51    | 2.26    |
| Conv-TasNet | Discriminative | syn_with_reverb | 2.42    | 2.71    | 2.01    |
|             |               | syn_no_reverb    | 3.09    | 3.34    | 3.00    |
|             |               | real_recording   | 3.10    | 2.98    | 2.41    |
| DEMUCS      | Discriminative | syn_with_reverb | 2.86    | 3.90    | 2.55    |
|             |               | syn_no_reverb    | 3.58    | 4.15    | 3.35    |
|             |               | real_recording   | 3.26    | 4.03    | 2.99    |
| FRCRN       | Discriminative | syn_with_reverb | 2.93    | 2.92    | 2.28    |
|             |               | syn_no_reverb    | 3.58    | 4.13    | 3.34    |
|             |               | real_recording   | 3.37    | 3.98    | 3.04    |
| SELM        | Generative    | syn_with_reverb  | 3.16    | 3.58    | 2.70    |
|             |               | syn_no_reverb    | 3.51    | 4.10    | 3.26    |
|             |               | real_recording   | **3.59**| 3.44    | 3.12    |
| MaskSR      | Generative    | syn_with_reverb  | 3.53    | 4.07    | 3.25    |
|             |               | syn_no_reverb    | 3.59    | 4.12    | 3.34    |
|             |               | real_recording   | 3.43    | 4.03    | 3.14    |
| GENSE       | Generative    | syn_with_reverb  | 3.49    | 3.73    | 3.19    |
|             |               | syn_no_reverb    | **3.65**| **4.18**| **3.43**|
|             |               | real_recording   | -       | -       | -       |
| LLaSE       | Generative    | syn_with_reverb  | **3.59**| **4.10**| **3.33**|
|             |               | syn_no_reverb    | **3.65**| 4.17    | **3.43**|
|             |               | real_recording   | 3.50    | **4.10**| **3.24**|

## Usage

### 1. Clone the Repo
```bash
git clone https://github.com/Kevin-naticl/LLaSE.git
cd LLaSE
```

### 2. Install Requirements
```bash
conda create -n LLaSE python=3.10
conda activate LLaSE
pip install -r requirements.txt
```

### 3. Download the Checkpoint from Hugging Face
You can use the provided shell script to download the checkpoint or manually download it from [Hugging Face](https://huggingface.co/).

```bash
cd ckpt
bash download.sh
```

### 4. Inference
1. Provide the file list in `./config/test.yml`.
2. Run the inference script:

```bash
bash inference.sh
```

The processed `.wav` files will be saved in `./decode/wav` by default (16k sample rate).

---

### Future Updates
- A Python module will be available in the future.
