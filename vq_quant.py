# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import torch
from torch import nn
import time

from uniform_quantizers import SymmetricUniformQuantizer


def get_assignments(X, centroids, chunk_size=None, H_inv_diag=None):
    """
    X: G x N x D
    centroids: G x K x D
    """
    if H_inv_diag is None:
        H_inv_diag = torch.ones(X.shape[-1]).to(X.device)
    elif H_inv_diag.ndim > 2:  # should then be 1 x N x D
        assert (
            H_inv_diag.shape[0] == 1
            and H_inv_diag.shape[1] == X.shape[1]
            and H_inv_diag.shape[2] == X.shape[2]
        ), f"{H_inv_diag.shape, X.shape}"
        H_inv_diag = H_inv_diag.unsqueeze(2)  # 1 x N x 1 x D

    if chunk_size is None:
        X_chunks = [X]
        H_inv_diag_chunks = [H_inv_diag]
    else:
        X_chunks = torch.split(X, chunk_size, dim=1)
        if H_inv_diag.ndim > 1:
            H_inv_diag_chunks = torch.split(H_inv_diag, chunk_size, dim=1)
        else:
            H_inv_diag_chunks = [H_inv_diag] * len(X_chunks)

    centroids = centroids.unsqueeze(1)  # G x 1 x K x D

    assignments = []
    for X, H_inv_diag in zip(X_chunks, H_inv_diag_chunks):
        X = X.unsqueeze(2)  # G x N' x 1 x D

        dist = ((X - centroids).pow(2) * H_inv_diag).sum(-1)

        assignments.append(dist.argmin(-1))  # G x N'
    assignments = torch.concat(assignments, dim=1)

    return assignments  # G x N


def vq_quantize(X, quantizer, H_inv_diag=None, centroids=None):
    assert len(X.shape) == 2
    orig_shape = X.shape

    vq_dim = quantizer.vq_dim

    X = X.reshape(quantizer.groups_per_column, -1, vq_dim)  # G x N x D
    if centroids is None:
        centroids = quantizer.all_centroids[-1]  # G x K x D
    idx = get_assignments(
        X, centroids, chunk_size=quantizer.assignment_chunk_size, H_inv_diag=H_inv_diag
    )  # G x N

    # below, idx expanded to G x N x D
    values = torch.gather(centroids, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, vq_dim))

    # return shapes: G x N x D, G x N
    return values.view(orig_shape), idx


def kmeans_m_step_3(
    centroids: torch.Tensor,
    n_centroids: int,
    assignments: torch.LongTensor,
    X: torch.Tensor,
    H_inv_diag=None,
):
    """
    X: G x N x D
    centroids: G x K x D
    assignments: G x N
    H_inv_diag: 1 x N x D
    """
    crange = torch.arange(0, n_centroids).to(centroids.device)

    # G x N x 1 == 1 x 1 x K --> G x N x K
    assignments_expanded = (assignments.unsqueeze(-1) == crange.view(1, 1, -1)).to(X.dtype)

    if H_inv_diag is None:
        norm = 1.0 / torch.clip(assignments_expanded.sum(1), min=1)  # G x K
        clusters_for_centroid = torch.einsum("gnd,gnk,gk->gkd", X, assignments_expanded, norm)
    else:
        norm = 1.0 / torch.clip(
            torch.einsum("gnk,nd->gkd", assignments_expanded, H_inv_diag[0]), min=1e-10
        )
        clusters_for_centroid = torch.einsum(
            "gnd,nd,gnk,gkd->gkd", X, H_inv_diag[0], assignments_expanded, norm
        )

    centroids.copy_(clusters_for_centroid)


def kmeans_vq(
    X,
    centroids,
    iters=10,
    assignment_chunk_size=None,
    H_inv_diag=None,
    codebook_bitwidth=None,
    per_codebook=False,
):
    n_centroids = centroids.shape[1]
    for iter in range(iters):
        # E-step
        assignments = get_assignments(
            X, centroids, chunk_size=assignment_chunk_size, H_inv_diag=H_inv_diag
        )

        # M-step: gather all values for each centroid and compute means
        # Centroids is shape G x D x K; assignments is shape G x N
        kmeans_m_step_3(centroids, n_centroids, assignments, X, H_inv_diag=H_inv_diag)

        if codebook_bitwidth is not None:
            quantize_centroids(centroids, codebook_bitwidth, per_codebook=per_codebook)


def kpp_parallel_sampled(data: torch.Tensor, k: int):
    G, N, D = data.shape

    if N * D < 32768 * 2:
        split_data = data.split(16)
    elif N * D * k < 32768 * 2 * 16:
        split_data = data.split(4)
    else:
        split_data = data.split(1)

    all_init = []

    for data in split_data:
        init = torch.zeros((data.shape[0], k, data.shape[-1]), dtype=torch.float16).to(
            data.device
        )  # G, K, D
        all_dists = torch.cdist(data.to(torch.float16), data.to(torch.float16), p=2)  # G, N, N
        init[:, 0] = data[:, 0]

        D2 = torch.zeros(data.shape[0], k, N).to(data.device)
        D2[:, 0] = all_dists[:, 0]

        for i in range(1, k):
            dists = D2[:, :i].amin(dim=1)  # G, N
            dists = (dists / dists.sum(-1, keepdims=True)).cumsum(-1)  # G, N

            v = torch.rand_like(dists[:, :1])  # G, 1

            idx = torch.clip(torch.searchsorted(dists, v).unsqueeze(-1), 0, N - 1)  # G, 1, 1

            D2[:, i : i + 1] = torch.gather(all_dists, dim=1, index=idx.expand(-1, 1, N))
            init[:, i : i + 1] = torch.gather(data, dim=1, index=idx.expand(-1, 1, D))
        all_init.append(init)
    return torch.concatenate(all_init)


def mahalanobis_init(X, n_centroids):
    """
    X: G x N x D
    centroids: G x K x D
    """
    vq_dim = X.shape[-1]
    mu = X.mean(1).unsqueeze(1)
    Xcentered = X - mu

    Sigma = torch.bmm(Xcentered.transpose(1, 2), Xcentered)  # G x D x D
    Lambda = torch.linalg.inv(Sigma)

    dists = (torch.bmm(Xcentered, Lambda) * Xcentered).sum(-1)  # G x N
    sorted_dists = torch.argsort(dists, dim=1)  # G x N
    idx = torch.round(torch.linspace(0, Xcentered.shape[1] - 1, n_centroids)).long()  # K
    idx = (
        sorted_dists[:, idx].unsqueeze(-1).expand(-1, -1, vq_dim)
    )  # G x K --> G x K x 1 --> G x K x D

    return torch.gather(X, dim=1, index=idx)


def quantize_centroids(centroids, bitwidth, per_codebook=True):
    orig_shape = centroids.shape
    if not per_codebook:
        centroids_ = centroids.view(1, -1)
    else:
        centroids_ = centroids.flatten(start_dim=1)

    imin, imax = -(2 ** (bitwidth - 1)), 2 ** (bitwidth - 1) - 1
    qmin, qmax = centroids_.min(dim=1)[0].abs(), centroids_.max(dim=1)[0]

    qmax = torch.max(qmin, qmax).unsqueeze(1)

    scale = qmax / imax

    qcentroids = torch.clip(torch.round(centroids_ / scale), imin, imax) * scale
    centroids.copy_(qcentroids.view(orig_shape))
    return centroids


class VQQuantizer(nn.Module):

    def __init__(
        self,
        vq_dim=2,
        n_subsample=100000,
        columns_per_group=None,
        kmeans_init_method="mahalanobis",
        assignment_chunk_size=None,
        kmeans_iters=10,
        codebook_bitwidth=None,
        quantize_per_codebook=True,
        vq_scaling_blocksize=-1,
        vq_scaling_norm="max",
        vq_scaling_n_bits=4,
        vq_scaling_domain="log",
        quantize_during_kmeans=False,
    ):
        super().__init__()
        self.vq_dim = vq_dim
        self.n_centroids = None
        self.scale = self.maxq = self.zero = None
        self.all_centroids = []
        self.columns_per_group = columns_per_group
        self.rows_per_group = None
        self.kpp_subsamples = n_subsample
        self.kmeans_init_method = kmeans_init_method
        self.assignment_chunk_size = assignment_chunk_size
        self.kmeans_iters = kmeans_iters
        self.codebook_bitwidth = codebook_bitwidth
        self.quantize_per_codebook = quantize_per_codebook
        self.quantize_during_kmeans = quantize_during_kmeans
        self.vq_scaling_blocksize = vq_scaling_blocksize
        self.vq_scaling_norm = vq_scaling_norm
        self.vq_scaling_n_bits = vq_scaling_n_bits
        self.vq_scaling_domain = vq_scaling_domain

    def get_groupsize(self, X, groupsize):
        if self.columns_per_group is not None:
            if groupsize < self.columns_per_group:
                assert self.columns_per_group % groupsize == 0
                self.columns_per_group = groupsize

            assert groupsize % self.columns_per_group == 0
            assert X.shape[1] % self.columns_per_group == 0

            self.rows_per_group = groupsize // self.columns_per_group
            assert X.shape[0] % self.rows_per_group == 0

            self.groups_per_column = X.shape[0] // self.rows_per_group

            return self.columns_per_group

        if groupsize < X.shape[1]:
            assert X.shape[1] % groupsize == 0
            self.groups_per_column = X.shape[0]
            return groupsize

        if groupsize % X.shape[1] != 0:
            print(
                f"Requested groupsize {groupsize} doesn't fit tensor shape[0] {X.shape[0]}. "
                f"Upscaling to {int(np.ceil(groupsize / X.shape[0]) * X.shape[0])}"
            )

        rows_per_group = int(np.ceil(groupsize / X.shape[1]))
        self.groups_per_column = X.shape[0] // rows_per_group
        return X.shape[1]

    def ready(self):
        return self.n_centroids != None

    def configure(self, wbits, **_):
        self.wbits = int(wbits * self.vq_dim)
        self.n_centroids = int(2**self.wbits)

    def find_params(self, X: torch.Tensor, weight=True, H_inv_diag=None):
        assert weight
        assert len(X.shape) == 2

        X = X.reshape(self.groups_per_column, -1, self.vq_dim)  # G x N x D
        if H_inv_diag is not None:
            H_inv_diag = H_inv_diag.reshape(1, -1, self.vq_dim)  # 1 x N x D
            if self.rows_per_group > 1:
                H_inv_diag = H_inv_diag.tile(1, self.rows_per_group, 1)

        if self.kmeans_init_method == "cdf":
            assert self.vq_dim == 1
            X, _ = torch.sort(X, 1)
            idx = torch.round(torch.linspace(0, X.shape[1] - 1, self.n_centroids)).long()
            centroids = X[:, idx].clone()  # G x K x 1
        elif self.kmeans_init_method == "kpp":
            centroids = kpp_parallel_sampled(X, self.n_centroids)
        elif self.kmeans_init_method == "mahalanobis":
            centroids = mahalanobis_init(X, self.n_centroids)
        else:
            raise ValueError(f"Unkown k-means init method: {self.kmeans_init_method}")

        # At this point, centroids should be shape G x K x D
        extra_args = {}
        if self.quantize_during_kmeans and self.codebook_bitwidth is not None:
            extra_args = dict(
                codebook_bitwidth=self.codebook_bitwidth, per_codebook=self.quantize_per_codebook
            )

        kmeans_vq(
            X,
            centroids,
            iters=self.kmeans_iters,
            assignment_chunk_size=self.assignment_chunk_size,
            H_inv_diag=H_inv_diag,
            **extra_args,
        )

        if self.codebook_bitwidth is not None and not self.quantize_during_kmeans:
            quantize_centroids(
                centroids, self.codebook_bitwidth, per_codebook=self.quantize_per_codebook
            )

        self.all_centroids.append(centroids)

    def blockwise_normalize_data(
        self,
        x_float,
        vq_scaling_blocksize,
        vq_scaling_norm="max",
        n_bits_scales=4,
        vq_scaling_domain="log",
    ):
        self.vq_scaling_blocksize = vq_scaling_blocksize
        orig_shape = x_float.shape
        if self.vq_scaling_blocksize > 0:
            x_float = x_float.view(
                x_float.shape[0],
                int(x_float.shape[1] // self.vq_scaling_blocksize),
                self.vq_scaling_blocksize,
            )
            if vq_scaling_norm == "L2":
                self.scales = torch.sqrt((torch.sum(x_float**2, dim=2)))
            elif vq_scaling_norm == "L1":
                self.scales = torch.sum(torch.abs(x_float), dim=2)
            elif vq_scaling_norm == "max":
                self.scales = torch.abs(x_float).max(dim=2).values
            else:
                raise NotImplementedError("This type of norm is not supported")

            if vq_scaling_domain == "log":
                self.log_scales = torch.log10(self.scales)
                self.log_scales[torch.abs(self.scales) < 1.0e-8] = (
                    0.0  # don't scale zeros, keep them as it is
                )

                self.min_log_scale, _ = torch.min(self.log_scales, dim=0, keepdim=True)
                self.log_scales -= self.min_log_scale

                if n_bits_scales < 16:
                    quant = SymmetricUniformQuantizer(n_bits=n_bits_scales, per_channel=True)
                    quant_range_min, _ = torch.min(self.log_scales, dim=0, keepdim=True)
                    quant_range_max, _ = torch.max(self.log_scales, dim=0, keepdim=True)
                    quant.set_quant_range(quant_range_min, quant_range_max)

                    self.log_scales = quant.forward(self.log_scales)

                log_scales = (self.log_scales + self.min_log_scale).unsqueeze(2)
                self.scales = torch.pow(10.0, log_scales)
            elif vq_scaling_domain == "linear":
                self.scales[torch.abs(self.scales) < 1.0e-8] = 1.0

                if n_bits_scales < 16:
                    quant = SymmetricUniformQuantizer(n_bits=n_bits_scales, per_channel=True)
                    self.min_scale, _ = torch.min(self.scales, dim=0, keepdim=True)
                    self.scales -= self.min_scale

                    quant_range_min, _ = torch.min(self.scales, dim=0, keepdim=True)
                    quant_range_max, _ = torch.max(self.scales, dim=0, keepdim=True)
                    quant.set_quant_range(quant_range_min, quant_range_max)

                    self.scales = quant.forward(self.scales)
                    self.scales += self.min_scale
                self.scales = self.scales.unsqueeze(2)
            else:
                raise NotImplementedError

            scales_repeated = self.scales.squeeze(-1).repeat_interleave(vq_scaling_blocksize, dim=1)
            x_float = torch.div(x_float, self.scales)
            x_float = x_float.view(orig_shape)

        return x_float, scales_repeated
