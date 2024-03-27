# Code taken from https://github.com/IST-DASLab/gptq
# Copyright 2022 IST-DASLab, Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution


import torch
import torch.nn as nn


DEV = torch.device("cuda:0")


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1)
        )
    return res
