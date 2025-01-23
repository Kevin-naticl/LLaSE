# **LLaSE: Maximizing Acoustic Preservation for LLaMA-based Speech Enhancement**  

Boyi Kang\*¹, Xinfa Zhu\*¹, Zihan Zhang¹, Zhen Ye², Ziqian Wang¹, Lei Xie¹  
¹ **Audio, Speech and Language Processing Group (ASLP@NPU)**,  
School of Computer Science, Northwestern Polytechnical University, Xi’an, China  
² **The Hong Kong University of Science and Technology**

---

## Abstract
Language Models (LMs) have shown strong capabilities in semantic understanding and contextual modeling, making them promising for speech enhancement. Building on SELM, our previous work that first introduced LMs to speech enhancement, we note that SELM and other existing generative speech enhancement approaches still face challenges, such as variations in timbre and content before and after enhancement. To address these limitations, we propose LLaSE, which utilizes continuous representations from WavLM and integrates a LLaMA backbone combined with the more powerful Xcodec2 decoder, significantly improving contextual modeling capabilities and enabling more accurate and stable enhancement. Experimental results demonstrate that LLaSE achieves state-of-the-art performance on speech enhancement, offering a robust and scalable solution for speech enhancement.

## Demo Page

Demo Page: https://kevin-naticl.github.io/LLaSE-Demopage/

![Overall Architecture of LLaSE](LLaSE.png)

## DNSMOS results on DNS Challenge testset
| Model       | Type          | Testset          | SIG   | BAK   | OVRL  |
|-------------|---------------|------------------|-------|-------|-------|
| Unprocessed | -             | syn_with_reverb  | 1.760 | 1.497 | 1.392 |
|             |               | syn_no_reverb    | 3.392 | 2.618 | 2.483 |
|             |               | real_recording   | 3.053 | 2.509 | 2.255 |
| Conv-TasNet | Discriminative | syn_with_reverb  | 2.415 | 2.710 | 2.010 |
|             |               | syn_no_reverb    | 3.092 | 3.341 | 3.001 |
|             |               | real_recording   | 3.102 | 2.975 | 2.410 |
| DEMUCS      | Discriminative | syn_with_reverb  | 2.856 | 3.897 | 2.553 |
|             |               | syn_no_reverb    | 3.575 | 4.153 | 3.345 |
|             |               | real_recording   | 3.263 | 4.027 | 2.988 |
| FRCRN       | Discriminative | syn_with_reverb  | 2.934 | 2.924 | 2.279 |
|             |               | syn_no_reverb    | 3.578 | 4.133 | 3.335 |
|             |               | real_recording   | 3.370 | 3.977 | 3.037 |
| SELM        | Generative    | syn_with_reverb  | 3.160 | 3.577 | 2.695 |
|             |               | syn_no_reverb    | 3.508 | 4.096 | 3.258 |
|             |               | real_recording   | 3.591 | 3.435 | 3.124 |
| MaskSR      | Generative    | syn_with_reverb  | 3.531 | 4.065 | 3.253 |
|             |               | syn_no_reverb    | 3.586 | 4.116 | 3.339 |
|             |               | real_recording   | 3.430 | 4.025 | 3.136 |
| GENSE       | Generative    | syn_with_reverb  | 3.49  | 3.73  | 3.19  |
|             |               | syn_no_reverb    | 3.65  | 4.18  | 3.43  |
|             |               | real_recording   | -     | -     | -     |
| LLaSE       | Generative    | syn_with_reverb  | 3.5933| 4.0958| 3.3272|
|             |               | syn_no_reverb    | 3.6532| 4.1695| 3.4284|
|             |               | real_recording   | 3.4998| 4.1002| 3.2369|

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