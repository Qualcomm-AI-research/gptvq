# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/IST-DASLab/gptq
# Copyright 2022 IST-DASLab, Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import os
import time

import torch
import torch.nn as nn

import transformers

from gptq import *
from modelutils import *
from quant import *
from vq_quant import *


HF_TOKEN = None  # TODO add token here


def get_llama(model, model_type):
    import torch

    def skip(*_, **__):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    token_auth_kwargs = {}
    if HF_TOKEN is not None:
        token_auth_kwargs["use_auth_token"] = HF_TOKEN

    if model_type == "mistral":
        from transformers import MistralForCausalLM

        model = MistralForCausalLM.from_pretrained(model, torch_dtype="auto", **token_auth_kwargs)
    elif model_type == "mixtral":
        from transformers import MixtralForCausalLM

        model = MixtralForCausalLM.from_pretrained(model, torch_dtype="auto", **token_auth_kwargs)
    else:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto", **token_auth_kwargs)
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev, args):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    if args.use_vq:
        QClass = lambda: VQQuantizer(
            vq_dim=args.vq_dim,
            columns_per_group=args.columns_per_group,
            vq_scaling_blocksize=args.vq_scaling_blocksize,
            vq_scaling_norm=args.vq_scaling_norm,
            vq_scaling_n_bits=args.vq_scaling_n_bits,
            vq_scaling_domain=args.vq_scaling_domain,
            kmeans_init_method=args.kmeans_init_method,
            assignment_chunk_size=args.assignment_chunk_size,
            kmeans_iters=args.kmeans_iters,
            codebook_bitwidth=args.codebook_bitwidth,
            quantize_per_codebook=args.quantize_per_codebook,
            quantize_during_kmeans=args.quantize_during_kmeans,
            n_subsample=args.kpp_n_subsample,
        )
    else:
        QClass = Quantizer

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [[k for k in list(full.keys()) if "block_sparse_moe.gate" not in k]]

        for names in sequential:

            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = QClass()
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Quantizing ...")
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    static_groups=args.static_groups,
                    include_m_step=args.include_m_step,
                    use_vq=args.use_vq,
                    svd_rank=args.svd_rank,
                    hessian_weighted_lookups=args.hessian_weighted_lookups,
                    only_init_kmeans=args.only_init_kmeans,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids
            )[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev, no_quant):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest and not no_quant:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                orig_shape = W.shape
                if args.groupsize > -1:
                    W = W.view(-1, args.groupsize)

                quantizer.find_params(W, weight=True)
                subset[name].weight.data = (
                    quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq)
                    .to(next(iter(layer.parameters())).dtype)
                    .view(orig_shape)
                )

        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="LlaMa model to load; pass location of hugginface converted checkpoint.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
    parser.add_argument(
        "--wbits",
        type=float,
        default=16,
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--sym", action="store_true", help="Whether to perform symmetric quantization."
    )
    parser.add_argument(
        "--save", type=str, default="", help="Save quantized checkpoint under this name."
    )
    parser.add_argument(
        "--new-eval", action="store_true", help="Whether to use the new PTB and C4 eval."
    )
    parser.add_argument(
        "--no-quant", action="store_true", help="If set, run FP16 model without quantization"
    )
    parser.add_argument(
        "--act-order",
        action="store_true",
        help="Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument(
        "--true-sequential", action="store_true", help="Whether to run in true sequential model."
    )
    parser.add_argument(
        "--static-groups",
        action="store_true",
        help="Whether to use static groups; recommended when using `--actorder` for more efficient inference.",
    )
    parser.add_argument(
        "--use-vq", action="store_true", help="If set, use VQ (multi-dim non-uniform) quantization"
    )
    parser.add_argument("--vq-dim", type=int, default=2, help="Dimensionality of VQ (if using)")
    parser.add_argument(
        "--vq-scaling-blocksize", type=int, default=-1, help="VQ scaling block size"
    )

    parser.add_argument("--vq-scaling-n-bits", type=int, default=4, help="VQ scaling bit-width")

    parser.add_argument("--vq-scaling-norm", type=str, default="max", help="VQ scaling norm")
    parser.add_argument(
        "--vq-scaling-domain",
        type=str,
        default="log",
        choices=["log", "linear"],
        help="VQ scaling domain",
    )

    parser.add_argument(
        "--include-m-step",
        action="store_true",
        help="If set, perform an M-step (centroid updating) after GPTQ with VQ",
    )
    parser.add_argument(
        "--columns-per-group",
        type=int,
        default=None,
        help="For group-/blockwise quant: force number of columns each group spans (rest is absorbed in rows)",
    )
    parser.add_argument(
        "--kmeans-init-method",
        type=str,
        default="cdf",
        choices=["cdf", "kpp", "mahalanobis"],
        help="init method for Kmeans",
    )
    parser.add_argument(
        "--assignment-chunk-size",
        type=int,
        default=None,
        help="Chunk assignment step for better memory management",
    )
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument(
        "--codebook-bitwidth", type=int, default=None, help="Bitwidth for codebook quantization"
    )
    parser.add_argument(
        "--quantize-per-codebook",
        action="store_true",
        default=False,
        help="Quantize codebooks individually (more overhead) or per column block",
    )
    parser.add_argument(
        "--quantize-during-kmeans",
        action="store_true",
        default=False,
        help="Quantize codebooks after every M-step. If not set: only quantize after k-means",
    )
    parser.add_argument(
        "--model-type",
        choices=["llama", "mistral", "mixtral"],
        default="llama",
        help="In case this is a Mistral model (GPTQ layerwise remains the same)",
    )
    parser.add_argument("--kpp-n-subsample", type=int, default=10000)
    parser.add_argument("--svd-rank", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save model in")
    parser.add_argument("--hessian-weighted-lookups", action="store_true", default=False)
    parser.add_argument("--only-init-kmeans", action="store_true", default=False)

    args = parser.parse_args()

    if not args.use_vq:
        args.wbits = int(args.wbits)

    model = get_llama(args.model, args.model_type)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest and not args.no_quant:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV, args)
        print(time.time() - tick)

    datasets = ["wikitext2"]
    if args.new_eval:
        datasets = ["wikitext2", "ptb-new", "c4-new"]
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV, no_quant=args.no_quant)

    if args.output_dir is not None:
        from transformers import AutoTokenizer
        from transformers.training_args import TrainingArguments

        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        new_args = TrainingArguments(no_cuda=True, output_dir=args.output_dir)
        trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=new_args)
        trainer.save_model(args.output_dir)
