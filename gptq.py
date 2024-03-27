# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/IST-DASLab/gptq
# Copyright 2022 IST-DASLab, Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution


import torch
from torch import nn
import numpy as np

import math
import time

import transformers

from quant import *
from vq_quant import vq_quantize, quantize_centroids

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def quad_loss(w_q, G, v, offset):
    """
    A generic function for computing the quadratic loss:
    L = 1/2 (G w_q, w_q) + (v, w_q) + offset

    Parameters
    ----------
    w_q : (c_out, m) or (m, 1)
        Quantized weights to be optimized.
    G : (m, m)
        Matrix part.
    v : shape(w_q)
        Linear part.
    offset : ()
        Scalar part.
    """
    # Quadratic loss: 1/2 wGw^T
    loss = 0.5 * (w_q.mm(G) * w_q).sum()
    # Add linear term and offset
    loss += (v * w_q).sum()
    loss += offset
    return loss


def quad_loss_2(W, Q, G):
    Werr = W - Q
    return (Werr.mm(G) * Werr).sum()


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def lut_m_step(self, Q_orig, groupsize, quantizer, scale=None, svd_rank=None):
        with torch.enable_grad():
            W = self.layer.weight.data.clone().float()
            G = self.G
            del self.G
            if scale is not None:
                scale.detach()

            offset = (W.mm(G) * W).sum()

            all_centroids = quantizer.all_centroids
            all_assignments = self.assignments
            vq_dim = quantizer.vq_dim

            if svd_rank is not None:
                assert vq_dim == 1, "In this implementation, SVD only works on 1D VQ"
                r = int(all_centroids[0].shape[1] * svd_rank)
                print(f"Effective SVD rank: {r}")
                Groups = all_centroids[0].shape[0]

                C = torch.concat(all_centroids, dim=0).squeeze()  # G x K
                C, new_idx = torch.sort(C, dim=1)
                new_idx = torch.argsort(new_idx, dim=1).split(Groups)

                U, S, V = torch.linalg.svd(C, full_matrices=False)
                all_centroids, V = (U * S[None])[:, :r].split(Groups), V[:r]

                new_assignments = []
                for idx, a in zip(new_idx, all_assignments):
                    new_assignments.append([])
                    for a_ in a:
                        remapped_a = torch.gather(idx, dim=1, index=a_)
                        new_assignments[-1].append(remapped_a)
                all_assignments = new_assignments

            def make_quantized_weight(centroids, assignments, scale=None):
                all_values = []
                for c, a in zip(centroids, assignments):
                    if svd_rank is not None:
                        c = (c @ V).unsqueeze(-1)
                    for a_ in a:
                        values = torch.gather(
                            c, dim=1, index=a_.unsqueeze(-1).expand(-1, -1, vq_dim)
                        )
                        all_values.append(values.view(W.shape[0], -1))
                Q = torch.concat(all_values, dim=1)
                if scale is not None:
                    Q = torch.mul(Q, scale)
                return Q

            with torch.no_grad():
                Q = make_quantized_weight(all_centroids, all_assignments, scale)

                orig_loss = quad_loss_2(W, Q, G)
                snr_before = 10 * np.log10(offset.item() / orig_loss.item())

            must_restart = True
            lr = 1e-3

            while must_restart:
                orig_centroids = [c.data.clone() for c in all_centroids]
                [c.requires_grad_() for c in all_centroids]
                param_list = list(all_centroids) + ([] if svd_rank is None else [V])
                o = torch.optim.Adam(param_list, lr=lr)
                for _ in range(25):
                    must_restart = False
                    o.zero_grad()
                    Q = make_quantized_weight(all_centroids, all_assignments, scale)
                    loss = quad_loss_2(W, Q, G)
                    if loss > orig_loss or torch.isnan(loss):
                        lr *= 1e-1
                        print(f"Inner loop: Restarting M-step with lr={lr:.2e}")
                        must_restart = True
                        all_centroids = orig_centroids
                        break
                    loss.backward()
                    o.step()

                if not must_restart:
                    if quantizer.codebook_bitwidth is not None:
                        new_all_centroids = [
                            quantize_centroids(
                                c.requires_grad_(False),
                                quantizer.codebook_bitwidth,
                                per_codebook=quantizer.quantize_per_codebook,
                            )
                            for c in all_centroids
                        ]
                    else:
                        new_all_centroids = all_centroids
                    Q = make_quantized_weight(new_all_centroids, all_assignments, scale)
                    loss = quad_loss_2(W, Q, G)
                    if torch.isnan(loss):
                        lr *= 1e-1
                        print(f"Outer loop: Restarting M-step with lr={lr:.2e}")
                        must_restart = True
                        all_centroids = orig_centroids
                        continue

                    del orig_centroids
                    print(
                        f"time M-step SGD {(time.time() - self.tick):.2f}; final loss: {loss.item():.4f}"
                    )
                    orig_loss = quad_loss_2(W, Q, G)
                    snr_after = 10 * np.log10(offset.item() / orig_loss.item())

                    print(f"improvement: {snr_before:.2f} -> {snr_after:.2f}")

        return Q

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        include_m_step=False,
        use_vq=False,
        svd_rank=None,
        hessian_weighted_lookups=False,
        only_init_kmeans=False,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        self.tick = time.time()

        if not self.quantizer.ready() and not use_vq:
            self.quantizer.find_params(W, weight=True)

        H = self.H
        self.G = self.H.clone()
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            raise NotImplementedError("Static groups are not supported in this repo")

        if actorder:
            raise NotImplementedError("Activation (re)-ordering is not supported in this repo")

        vq_dim = self.assignments = None
        S = vq_scaling_blocksize = vq_scaling_n_bits = None
        if use_vq:
            vq_dim = self.quantizer.vq_dim
            groupsize = self.quantizer.get_groupsize(W, groupsize)
            self.assignments = []
            assert blocksize % vq_dim == 0

            vq_scaling_blocksize = self.quantizer.vq_scaling_blocksize
            vq_scaling_n_bits = self.quantizer.vq_scaling_n_bits
            if vq_scaling_blocksize > 0:
                assert vq_scaling_blocksize % vq_dim == 0
                S = torch.ones_like(W)

            print(W.shape)
            print(
                f"VQ scaling BS {vq_scaling_blocksize} @ {vq_scaling_n_bits}b "
                f"({self.quantizer.vq_scaling_domain} domain)"
            )
            print(f"Using Hessian-aware K-means {hessian_weighted_lookups}")

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            if use_vq and vq_scaling_blocksize > 0:
                W1_scaled, S1 = self.quantizer.blockwise_normalize_data(
                    W1,
                    vq_scaling_blocksize,
                    self.quantizer.vq_scaling_norm,
                    vq_scaling_n_bits,
                    self.quantizer.vq_scaling_domain,
                )
                S[:, i1:i2] = S1
            else:
                W1_scaled = W1
                S1 = torch.ones_like(W1)

            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        extra_args = {}
                        if use_vq and vq_dim > 1 and hessian_weighted_lookups:
                            H_inv_diag = torch.diag(Hinv)[i1 + i : i1 + i + groupsize]
                            extra_args["H_inv_diag"] = H_inv_diag

                        W_group = W[:, (i1 + i) : (i1 + i + groupsize)]

                        W_group_scaled = W_group
                        if use_vq:
                            self.assignments.append([])
                            if vq_scaling_blocksize > 0:
                                assert vq_scaling_blocksize % vq_dim == 0
                                W_group_scaled, S_group = self.quantizer.blockwise_normalize_data(
                                    W_group,
                                    vq_scaling_blocksize,
                                    self.quantizer.vq_scaling_norm,
                                    self.quantizer.vq_scaling_n_bits,
                                    self.quantizer.vq_scaling_domain,
                                )

                        self.quantizer.find_params(W_group_scaled, weight=True, **extra_args)

                if not use_vq:
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = quantize(
                        w.unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq,
                    ).flatten()

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2

                    err1 = (w - q) / d
                    # (R x 1).matmul(1 x C') --> R x C' (C': remaining (unquantized) columns)
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                elif i % vq_dim == 0:
                    w = W1[:, i : i + vq_dim]  # R x D
                    d = torch.diag(Hinv1)[i : i + vq_dim].unsqueeze(0)  # 1 x D
                    w_scaled = W1_scaled[:, i : i + vq_dim]  # R x D
                    s = S1[:, i : i + vq_dim]

                    H_inv_diag = None
                    if vq_dim > 1 and hessian_weighted_lookups:
                        H_inv_diag = 1.0 / d.to(w.device)

                    q, assmt = vq_quantize(
                        w_scaled, self.quantizer, H_inv_diag=H_inv_diag
                    )  # R x 1 x D, R x 1
                    q = torch.mul(q, s)  # de-scaling

                    self.assignments[-1].append(assmt)

                    Q1[:, i : i + vq_dim] = q
                    Losses1[:, i : i + vq_dim] = (w - q) ** 2 / d**2  # R x D / 1 x D

                    err1 = (w - q) / d  # R x D
                    # batch matmul solution: (D x R x 1).matmul(D x 1 x C').sum(0) --> R x C'
                    if not only_init_kmeans:
                        update = torch.bmm(
                            err1.transpose(0, 1).unsqueeze(-1),
                            Hinv1[i : i + vq_dim, i + vq_dim :].unsqueeze(1),
                        ).sum(0)
                        W1[:, i + vq_dim :] -= update
                        Err1[:, i : i + vq_dim] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            if not only_init_kmeans:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - self.tick))
        print("error", torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        if include_m_step:
            Q = self.lut_m_step(Q, groupsize, self.quantizer, scale=S, svd_rank=svd_rank)

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
