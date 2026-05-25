# -*- coding: utf-8 -*-
"""
신경망 layer 모음.

학생 구현 대상:
- Affine.forward, Affine.backward
- BatchNorm.forward, BatchNorm.backward
- Dropout.forward, Dropout.backward
"""

import numpy as np


class Affine:
    """
    완전연결층(Fully Connected Layer).

    수식은 y = xW + b 입니다.
    MNIST에서는 784개 픽셀 입력을 은닉층/출력층 차원으로 선형 변환하는 역할을 합니다.
    """

    def __init__(self, W, b):
        """가중치 W와 편향 b를 외부 params dict와 같은 배열 객체로 공유합니다."""
        self.W = W
        self.b = b
        
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim)
        """
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
        raise NotImplementedError("Affine.forward를 구현하세요.")

    def backward(self, dout):
        """
        Args:
            dout: (batch_size, output_dim)

        Returns:
            dx: (batch_size, input_dim)

        Side effects:
            self.dW, self.db에 optimizer가 사용할 gradient를 저장합니다.
        """
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)

        return dx

        raise NotImplementedError("Affine.backward를 구현하세요.")


class BatchNorm:
    """
    Batch Normalization.

    미니배치 단위로 각 feature의 평균과 분산을 맞춰 학습을 안정화합니다.
    train=True일 때는 현재 배치 통계를 쓰고, 추론 때는 누적 running_mean/running_var를 사용합니다.
    """

    def __init__(self, gamma, beta, momentum=0.9):
        """
        Args:
            gamma: 정규화된 값을 다시 scale하는 학습 파라미터
            beta: 정규화된 값에 더하는 shift 학습 파라미터
            momentum: running_mean/running_var 이동평균 비율
        """
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = np.zeros_like(beta)
        self.running_var = np.zeros_like(beta)
        self.eps = 1e-7

        self.x_hat = None
        self.dgamma = None
        self.dbeta = None
        self.std = None


    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, feature_dim)
            train: True면 배치 통계, False면 running 통계 사용

        Returns:
            정규화 후 gamma, beta가 적용된 배열
        """
        if train:
            # affine에서 받아온 데이터를 정규화
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            x_hat = (x - mu) / np.sqrt(var + self.eps)
            out = self.gamma * x_hat + self.beta
            self.x_hat = x_hat
            self.std = np.sqrt(var + 1e-7)
            
            # 관성으로 running 반영
            ratio = self.momentum
            self.running_mean = ratio * self.running_mean + (1 - ratio) * mu
            self.running_var = ratio * self.running_var + (1 - ratio) * var

            return out
        else:
            # 생성된 running을 그대로 사용
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta

            return out
        

    def backward(self, dout):
        """
        BatchNorm 입력 x, scale gamma, shift beta에 대한 gradient를 계산합니다.

        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            dx: BatchNorm 입력 x에 대한 gradient
        """
        # 힌트: 먼저 dbeta와 dgamma shape가 beta/gamma와 같은지 확인합니다.
        self.dgamma = np.sum(dout * self.x_hat, axis=0)
        self.dbeta = np.sum(dout, axis=0)

        # 도저히 모르겠다.. 일단 복붙
        N = dout.shape[0]
        dx = (1.0 / N) * (self.gamma / self.std) * (
            N * dout - np.sum(dout, axis=0) - self.x_hat * np.sum(dout * self.x_hat, axis=0)
        )

        return dx


class Dropout:
    """
    Dropout.

    학습 중 일부 뉴런 출력을 무작위로 0으로 만들어 과적합을 줄입니다.
    이 구현은 추론 시 출력에 (1 - drop_ratio)를 곱하는 기본 dropout 방식을 사용합니다.
    """

    def __init__(self, drop_ratio=0.5):
        """Args: drop_ratio: 학습 중 0으로 만들 뉴런 비율."""
        self.drop_ratio = drop_ratio
        self.mask = None

    def forward(self, x, train=True):
        """
        Args:
            x: 입력 배열
            train: True면 무작위 mask 적용, False면 평균적인 출력 크기로 scale
        """
        if train:
            self.mask = np.random.rand(*x.shape) > self.drop_ratio
            return x * self.mask
        else:
            return x *(1.0 - self.drop_ratio)

        raise NotImplementedError("Dropout.forward를 구현하세요.")

    def backward(self, dout):
        """forward에서 꺼졌던 뉴런 위치에는 gradient도 흘리지 않습니다."""
        return dout * self.mask
    
        raise NotImplementedError("Dropout.backward를 구현하세요.")
