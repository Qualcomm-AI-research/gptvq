#!/bin/bash

# Ensure PYTHONPATH points to the root gptvq directory

python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --hessian-weighted-lookups --wbits 3 --groupsize 1024 --vq-dim 1 --kmeans-init-method mahalanobis $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --hessian-weighted-lookups --wbits 3 --groupsize 1024 --vq-dim 1 --kmeans-init-method kpp $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --hessian-weighted-lookups --wbits 4 --groupsize 2048 --vq-dim 1 --kmeans-init-method mahalanobis $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --hessian-weighted-lookups --wbits 4 --groupsize 2048 --vq-dim 1 --kmeans-init-method kpp $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --hessian-weighted-lookups --wbits 3 --groupsize 16384 --vq-dim 2 --kmeans-init-method mahalanobis $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --hessian-weighted-lookups --wbits 3 --groupsize 16384 --vq-dim 2 --kmeans-init-method kpp $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --hessian-weighted-lookups --wbits 4 --groupsize 65536 --vq-dim 2 --kmeans-init-method mahalanobis $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --hessian-weighted-lookups --wbits 4 --groupsize 65536 --vq-dim 2 --kmeans-init-method kpp $LLAMA2_7B_PATH wikitext2
