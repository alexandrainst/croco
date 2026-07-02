# References

Bibliography for all CroCo research experiments.

---

## Core Methods

### DPO (Direct Preference Optimization)

```bibtex
@article{rafailov2023direct,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Manning, Christopher D and Ermon, Stefano and Finn, Chelsea},
  journal={arXiv preprint arXiv:2305.18290},
  year={2023},
  url={https://arxiv.org/abs/2305.18290}
}
```

### SimPO (Simple Preference Optimization)

```bibtex
@article{meng2024simpo,
  title={SimPO: Simple Preference Optimization with a Reference-Free Reward},
  author={Meng, Yu and Xia, Mengzhou and Chen, Danqi},
  journal={arXiv preprint arXiv:2405.14734},
  year={2024},
  url={https://arxiv.org/abs/2405.14734}
}
```

### GRPO (Group Relative Policy Optimization)

```bibtex
@article{shao2024deepseekmath,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, YK and Wu, Y and others},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024},
  url={https://arxiv.org/abs/2402.03300}
}
```

**Note:** GRPO was introduced in the DeepSeekMath paper. The method is described in
Appendix A.3.

### ORPO (Odds Ratio Preference Optimization)

```bibtex
@article{hong2024orpo,
  title={ORPO: Monolithic Preference Optimization without Reference Model},
  author={Hong, Jiwoo and Lee, Noah and Thorne, James},
  journal={arXiv preprint arXiv:2403.07691},
  year={2024},
  url={https://arxiv.org/abs/2403.07691}
}
```

### CPO (Contrastive Preference Optimization)

```bibtex
@article{xu2024cpo,
  title={CPO: Contrastive Preference Optimization for Language Models},
  author={Xu, Yiding and others},
  journal={arXiv preprint},
  year={2024}
}
```

### KTO (Kahneman-Tversky Optimization)

```bibtex
@article{ethayarajh2024kto,
  title={KTO: Model Alignment as Prospect Theoretic Optimization},
  author={Ethayarajh, Kawin and Xu, Winnie and Muennighoff, Niklas and Jurafsky, Dan and Kiela, Douwe},
  journal={arXiv preprint},
  year={2024},
  url={https://arxiv.org/abs/2402.01306}
}
```

### IPO (Identity Preference Optimization)

```bibtex
@article{azar2024general,
  title={A General Theoretical Paradigm to Understand Learning from Human Preferences},
  author={Azar, Mohammad Gheshlaghi and Munos, Remi and Ghavamzadeh, Mohammad and others},
  journal={arXiv preprint arXiv:2310.12036},
  year={2024},
  url={https://arxiv.org/abs/2310.12036}
}
```

### RRHF (Rejection Reweighting Fine-tuning)

```bibtex
@article{yuan2023rrhf,
  title={RRHF: Rank Responses to Align Language Models with Human Feedback without Tears},
  author={Yuan, Zheng and others},
  journal={arXiv preprint arXiv:2304.05302},
  year={2023},
  url={https://arxiv.org/abs/2304.05302}
}
```

### SLiC-HF (Sequence Likelihood Calibration)

```bibtex
@article{zhao2023slic,
  title={SLiC-HF: Sequence Likelihood Calibration with Human Feedback},
  author={Zhao, Yao and Durrett, Greg and others},
  journal={arXiv preprint},
  year={2023}
}
```

---

## Techniques

### LoRA (Low-Rank Adaptation)

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021},
  url={https://arxiv.org/abs/2106.09685}
}
```

### Curriculum Learning

```bibtex
@article{bengio2009curriculum,
  title={Curriculum Learning},
  author={Bengio, Yoshua and Louradour, Jérôme and Collobert, Ronan and Weston, Jason},
  journal={Proceedings of the 26th Annual International Conference on Machine Learning},
  pages={41--48},
  year={2009},
  url={https://doi.org/10.1145/1553374.1553380}
}
```

### Length Normalization in Language Models

```bibtex
@article{koehn2017six,
  title={Six Challenges for NMT},
  author={Koehn, Philipp and Knowles, Rebecca},
  journal={Proceedings of the First Workshop on Neural Machine Translation},
  pages={54--59},
  year={2017},
  url={https://aclanthology.org/W17-3206/}
}
```

**Note:** Length normalization is a standard technique to counter verbosity bias in
sequence models. Discussed in context of reward models in the SimPO paper.

---

## Models

### Munin-Apertus-8B

```bibtex
@model{munin_apertus_8b,
  title={Munin-Apertus-8B},
  author={Danish Foundation Models},
  organization={Hugging Face},
  year={2024},
  url={https://huggingface.co/danish-foundation-models/munin-apertus-8b}
}
```

### Skywork-Reward-V2-Qwen3-8B

```bibtex
@model{skywork_reward_v2,
  title={Skywork-Reward-V2-Qwen3-8B},
  author={Skywork AI},
  organization={Hugging Face},
  year={2024},
  url={https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B}
}
```

### Llama-3-8B

```bibtex
@model{llama3_8b,
  title={Llama 3 Model Card},
  author={Meta AI},
  organization={Hugging Face},
  year={2024},
  url={https://huggingface.co/meta-llama/Meta-Llama-3-8B}
}
```

---

## Datasets

### Laerebogen

```bibtex
@dataset{laerebogen,
  title={Laerebogen: A Danish Textbook Dataset for Language Model Training},
  author={Danish Foundation Models},
  organization={Hugging Face},
  year={2024},
  url={https://huggingface.co/datasets/danish-foundation-models/laerebogen}
}
```

### UltraChat / UltraFeedback

```bibtex
@dataset{ultrachat,
  title={UltraChat: An Open-Source Multi-Turn Instruction Dataset},
  author={Ding, Ning and others},
  year={2023},
  url={https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k}
}

@dataset{ultrafeedback,
  title={UltraFeedback: Boosting Language Models with High-Quality Feedback},
  author={Cui, Ganqu and others},
  year={2023},
  url={https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized}
}
```

---

## Evaluation Benchmarks

### AlpacaEval 2

```bibtex
@article{alpaca_eval,
  title={AlpacaEval: An Automatic Evaluator of Instruction-following Models},
  author={Li, Xuechen and others},
  year={2023},
  url={https://github.com/tatsu-lab/alpaca_eval}
}
```

### MT-Bench

```bibtex
@article{zheng2023judging,
  title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
  author={Zheng, Lianmin and others},
  journal={arXiv preprint arXiv:2306.05685},
  year={2023},
  url={https://arxiv.org/abs/2306.05685}
}
```

### Arena-Hard

```bibtex
@article{arenahard,
  title={From Crowdsourced Data to High-Quality Benchmarks: Introducing Arena-Hard},
  author={Li, Tianle and others},
  journal={arXiv preprint},
  year={2024},
  url={https://huggingface.co/spaces/lmsys/arena-hard}
}
```

---

## Tools & Frameworks

### vLLM

```bibtex
@article{kwon2023vllm,
  title={vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Sifei and Sheng, Ying and others},
  year={2023},
  url={https://github.com/vllm-project/vllm}
}
```

### TRL (Transformer Reinforcement Learning)

```bibtex
@misc{trl,
  title={TRL: Transformer Reinforcement Learning},
  author={Hugging Face},
  year={2024},
  url={https://github.com/huggingface/trl}
}
```

### EuroEval

```bibtex
@misc{euroeval,
  title={EuroEval: Evaluation Benchmark for European Languages},
  author={Danish Foundation Models},
  year={2024},
  url={https://github.com/danish-foundation-models/euroeval}
}
```

---

*Last updated: 2026-07-02*
