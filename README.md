# REVEAL: Reasoning-Enhanced Forensic Evidence Analysis for Explainable AI-Generated Image Detection[ECCV 2026]

[ 📄[**Paper**](https://arxiv.org/abs/2511.23158) | 🔗[**Data**](https://huggingface.co/datasets/sanmu29/REVEAL-Bench) | 🚀[**Model**](https://huggingface.co/sanmu29/REVEAL/tree/main/REVEAL) ]


In this work, we introduce:

> 📍**REVEAL-Bench**: An explainable AI-generated image detection benchmark featuring multi-view forensic evidence, reasoning traces, and rigorous in-domain and out-of-distribution evaluation protocols.
>
> 📍**REVEAL**: A reasoning-enhanced forensic framework that aggregates multi-view evidence through large vision-language models, achieving strong generalization across unseen generators while providing transparent, human-interpretable decision processes.


## Installation

```bash
conda create -n REVEAL python=3.10
conda activate REVEAL
```

Please follow the official Swift v3.6 installation guide to install the required dependencies:

https://swift.readthedocs.io/en/v3.6/


## 🔥 Training
### 0. Prepare the REVEAL-Bench Dataset
Download training data [here]([https://www.modelscope.cn/datasets/EricTanh/HydraFake/tree/master/jsons/train](https://huggingface.co/datasets/sanmu29/REVEAL-Bench)).

### 1. supervised fine-tuning on a consolidated Chainof-Evidence dataset (CoE Tuning)
```bash
sh examples/train/liger/sft_llava_13b_single.sh
```

### 2. Reasoning-Enhanced GRPO (R-GRPO)
```bash
sh examples/train/grpo/plugin/grpollava_stepthink13b.sh
```

### 3. Test REVEAL
```bash
# Effort for example
python3 test/inference.py
```



## 🛡️ REVEAL-Bench

📍 **Overview:**

<p align="center">
    <img src="./dataset_pipline_01.png" alt="Dataset" width="95%">
</p>


(a) Overview of the REVEAL-Bench construction pipeline. Starting from large-scale real and synthetic image collections, we perform data curation and pre-filtering, expert-grounded evidence annotation, and Chain-of-Evidence (CoE) synthesis to generate explainable forensic supervision.

(b) Illustration of the eight specialized forensic experts used for evidence collection. Each expert focuses on a distinct forensic perspective, including local artifacts, frequency-domain anomalies, pixel-level noise, spatial consistency, geometry reasoning, shadow coherence, texture-frequency fusion, and high-pass semantic analysis, producing structured diagnostic evidence for downstream reasoning.

(c) Chain-of-Evidence generation process. A large vision-language model consolidates fragmented multi-view forensic evidence into coherent and auditable reasoning trajectories, explicitly linking low-level observations to high-level authenticity judgments.

(d) Dataset statistics of REVEAL-Bench. The benchmark contains 60K high-quality explainable samples (30K real and 30K synthetic), each equipped with expert-grounded forensic evidence and structured reasoning traces, spanning diverse image categories, visual qualities, resolutions, and generative models.





## 🛰️ REVEAL
🔍 We introduce REVEAL, a reasoning-enhanced forensic framework...

🧠 Two-stage training pipeline:

(1) **Chain-of-Evidence Tuning (CoE-Tuning)**: Learn multi-view forensic reasoning trajectories by supervising evidence discovery, evidence verification, and evidence summarization across diverse forensic cues.

(2) **Reasoning-Enhanced GRPO (R-GRPO)**: Optimize reasoning robustness and evidence aggregation through reinforcement learning, improving generalization to unseen generators while maintaining explanation fidelity.

<p align="center">
    <img src="./method_pipline.drawio_01.png" alt="Training pipeline" width="70%">
</p>




## Citation
If you find our work useful, please cite our paper:
```
@article{cao2025reveal,
  title={REVEAL: Reasoning-Enhanced Forensic Evidence Analysis for Explainable AI-Generated Image Detection},
  author={Cao, Huangsen and Mei, Qin and Li, Zhiheng and Li, Yuxi and Meng, Zhan and Zhang, Ying and Li, Chen and Zhang, Zhimeng and Ding, Xin and Wang, Yongwei and others},
  journal={arXiv preprint arXiv:2511.23158},
  year={2025}
}
```


## Acknowledgements

This repo benefits from [ms-swift](https://github.com/modelscope/ms-swift). Thanks for their great works!
