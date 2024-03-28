# GPTVQ

This repository contains the code for the paper [GPTVQ: The Blessing of Dimensionality in LLM Quantization](https://arxiv.org/abs/2402.15319) (under review).
This codebase is based upon the codebase for for the ICLR 2023 paper [GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://arxiv.org/abs/2210.17323),
downloaded [The GPTQ GitHub page](https://github.com/IST-DASLab/gptq/).


## Abstract
In this work we show that the accuracy and efficiency of neural network quantization can be significantly improved by increasing the quantization dimensionality. We propose the GPTVQ~method, a new fast method for post-training vector quantization (VQ) that scales well to Large Language Models (LLMs). 
Our method interleaves quantization of one or more columns with updates to the remaining unquantized weights, using information from the Hessian of the per-layer output reconstruction MSE.
Quantization codebooks are initialized using an efficient data-aware version of the EM algorithm. The codebooks are then updated, and further compressed by using integer quantization and SVD-based compression. 
GPTVQ establishes a new state-of-the art in the size vs accuracy trade-offs on a wide range of LLMs such as Llama-v2 and Mistral. 
Furthermore, our method is efficient: on a single H100 it takes between 3 and 11 hours to process a Llamav2-70B model, depending on quantization setting.
Lastly, with on-device timings for VQ decompression on a mobile CPU we show that VQ leads to improved latency compared to using a 4-bit integer format.


## Dependencies
See `requirements.txt`.

All experiments were run on a single 80GB NVIDIA H100. However, most experiments will work on a GPU with a lot less memory as well.
In case experiments run out of memory, the `--assignment-chunk-size` argument can be used to reduce memory requirements.
A lower value for this argument will reduce memory requirements at the expense of longer run times.


## Reproducibility

### Installation
Requirements are listed in `requirements.txt`. Install these in your environment using
```
pip install -r requirements.txt
```

Modify your `PYTHONPATH` to include the root directory of this repository.

All experiments were run using `python3.9`.


### Experiments
Scripts to reproduce results in the paper are included as shell scripts. 

### Models
To run these scripts, the following environment variables need to be set to point to the relevant models:

```
LLAMA1_13B_PATH
LLAMA1_30B_PATH
LLAMA1_65B_PATH
LLAMA1_7B_PATH
LLAMA2_13B_PATH
LLAMA2_70B_PATH
LLAMA2_7B_PATH
MISTRAL_7B_PATH
MIXTRAL_PATH
```

These can point to either models on the HuggingFace model hub, or to local checkpoints for the corresponding architectures.

NB1: For some of these models a HuggingFace authorization token is required. If this is the case, please modify line 18 in `llama.py` with a valid HuggingFace authorization token.
NB2: The HuggingFace checkpoints in these scripts were not necessarily the same as the model checkpoints used for the experiments in the paper. As a result, minor differences might occur. 


### Datasets
For calibration the `wikitext2` training set from the HuggingFace datasets hub is used. Perplexity results are generated using the `wikitext2` test set is used.

To generate zero-shot results, add the `--output-dir` argument to the command for an experiment. The VQ quantized model will be saved in this directory.
This command is by default *NOT* included in the experiment scripts in this repository, to avoid excessive file storage requirements.
Afterwards, run the [`llm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) on the checkpoint stored in this directory.

```
lm_eval --model hf \
    --model_args pretrained=/PATH/TO/CHECKPOINT \
    --tasks piqa,boolq,winogrande,hellaswag,arc_easy,arc_challenge \
    --device cuda:0
    --batch_size auto
```

## Cite

If you found this work useful, please consider citing:

```
@article{vanbaalen-gptvq,
  title={GPTVQ: The Blessing of Dimensionality in LLM Quantization}, 
  author={Mart van Baalen and Andrey Kuzmin and Markus Nagel and Peter Couperus and Cedric Bastoul and Eric Mahurin and Tijmen Blankevoort and Paul Whatmough},
  year={2024},
  journal={arXiv preprint arXiv:2402.15319}
}
```
