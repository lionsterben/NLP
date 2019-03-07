import numpy as np
import math


def position_embedding(seq_len, d_model):
    res = np.zeros((seq_len, d_model))
    even_start = 0
    pos = [i for i in range(seq_len)]
    while even_start < d_model:
        res[:, even_start] = list(map(lambda i:math.sin(i/math.pow(10000, even_start/d_model)), pos)) 
        even_start += 2
    odd_start = 1
    while odd_start < d_model:
        res[:, odd_start] = list(map(lambda i:math.cos(i/math.pow(10000, (odd_start-1)/d_model)), pos))
        odd_start += 2
    return res


def predict_position_embedding(pos, d_model):
    res = np.zeros((1, d_model))
    even_start = 0
    while even_start < d_model:
        res[0, even_start] = math.sin(pos/math.pow(10000, even_start/d_model))
        even_start += 2
    odd_start = 1
    while odd_start < d_model:
        res[0, odd_start] = math.cos(pos/math.pow(10000, (odd_start-1)/d_model))
        odd_start += 2
    return res


def optimizer_lr_rate(d_model, step_num, warmup_steps=20000):
    return math.pow(d_model, -0.3)*min(math.pow(step_num, -0.5), step_num*math.pow(warmup_steps, -1.5))
