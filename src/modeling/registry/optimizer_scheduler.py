import math


def cosine_annealing_with_linear_warmup(T_max, eta_min, eta_max, warmup_epochs, lr):
    return lambda T_cur: (eta_min + 0.5 * (eta_max - eta_min) * (
                        1.0 + math.cos((T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr