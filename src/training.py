# -*- coding: utf-8 -*-
"""학습 루프, 평가, 시각화 함수 모음."""

import matplotlib.pyplot as plt
import numpy as np

from losses import cross_entropy_loss


def train(model, optimizer, x_train, y_train, epochs=20, batch_size=128):
    """
    미니배치 학습 루프.

    한 배치마다 Forward -> Loss -> Backward -> Optimizer 업데이트 순서로 진행합니다.
    교육생은 이 함수에서 "예측값을 만들고, 손실을 계산하고, gradient로 파라미터를 바꾸는"
    전체 흐름을 확인할 수 있습니다.

    Returns:
        loss_history: epoch별 평균 손실 리스트
    """
    # TODO: epoch마다 데이터를 섞고, batch 단위로 forward/loss/backward/update를 수행하세요.
    # 힌트: Softmax + CrossEntropy 결합 gradient는 y_pred copy에서 정답 위치에 1을 빼서 만듭니다.
    
    # 오차 점수 기록
    loss_history = []

    # 먼저 묶음 단위를 만든다
    N = x_train.shape[0]

    # epochs 만큼 반복
    for _ in range(epochs):
        indices = np.random.permutation(N)
        batch_loss = []

        for i in range(0, N, batch_size): 
            batch_indices = indices[i: i + batch_size]
            x_sample = x_train[batch_indices]
            y_answer = y_train[batch_indices]

            # 이제 forward로 트레이닝을 합니다 => 예측 점수 산출
            # 예측 점수와 실제 점수와 비교하여 오차 점수 산출
            probs = model.forward(x_sample, train=True)
            loss = cross_entropy_loss(probs, y_answer)
            batch_loss.append(loss)

            # 다음 backward를 수행하여 gradient를 산출
            dout = probs.copy()
            dout[np.arange(x_sample.shape[0]), y_answer] -= 1
            dout /= x_sample.shape[0]

            # gradient를 기반으로 모델의 가중치를 수정
            model.backward(dout)
            optimizer.update(model.params, model.grads)

        epoch_loss = np.mean(batch_loss)
        loss_history.append(epoch_loss)


    return loss_history


def evaluate(model, x, y):
    """정확도(%)와 총 파라미터 수 반환."""
    y_pred = model.predict(x)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(p.size for p in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history):
    """손실 커브 그래프."""
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
