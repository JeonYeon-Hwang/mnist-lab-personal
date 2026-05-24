# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    batch_size = y_pred.shape[0]

    answer_probs = y_pred[np.arange(batch_size), y_true]
    log_probs = np.log(np.clip(answer_probs, 1e-7, 1.0))
    sum_probs = -np.sum(log_probs);

    return sum_probs / batch_size

    raise NotImplementedError("cross_entropy_loss를 구현하세요.")
