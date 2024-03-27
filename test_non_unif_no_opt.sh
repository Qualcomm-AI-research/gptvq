#!/bin/bash

# Ensure PYTHONPATH points to the root gptvq directory

python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --only-init-kmeans --wbits 2 --vq-dim 1 --groupsize 512 $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --only-init-kmeans --wbits 2 --vq-dim 1 --groupsize 512 --hessian-weighted-lookups $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --only-init-kmeans --wbits 3 --vq-dim 1 --groupsize 1024 $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --only-init-kmeans --wbits 3 --vq-dim 1 --groupsize 1024 --hessian-weighted-lookups $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --only-init-kmeans --wbits 4 --vq-dim 1 --groupsize 2048 $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --only-init-kmeans --wbits 4 --vq-dim 1 --groupsize 2048 --hessian-weighted-lookups $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --only-init-kmeans --wbits 2 --vq-dim 2 --groupsize 4096 $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --only-init-kmeans --wbits 2 --vq-dim 2 --groupsize 4096 --hessian-weighted-lookups $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --only-init-kmeans --wbits 3 --vq-dim 2 --groupsize 16384 $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --only-init-kmeans --wbits 3 --vq-dim 2 --groupsize 16384 --hessian-weighted-lookups $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --only-init-kmeans --wbits 4 --vq-dim 2 --groupsize 65536 $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --only-init-kmeans --wbits 4 --vq-dim 2 --groupsize 65536 --hessian-weighted-lookups $LLAMA2_7B_PATH wikitext2
