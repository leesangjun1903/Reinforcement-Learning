
# Conservative Q-Learning for Offline Reinforcement Learning
## 1. 핵심 주장 및 기여도 요약
"Conservative Q-Learning for Offline Reinforcement Learning"은 오프라인 강화학습 분야에서 근본적인 도전과제인 **분포 이동(distributional shift)** 문제를 해결하기 위한 혁신적인 방법론을 제시합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

### 핵심 주장
논문의 주요 주장은 다음과 같습니다:

1. **보수적 Q-함수 학습의 필요성**: 오프라인 환경에서 표준 오프-정책(off-policy) 강화학습 방법들이 학습 데이터셋에 포함되지 않은 액션(행동)에 대해 Q-값을 과대추정하는 문제가 발생하므로, 학습된 Q-함수가 진정한 정책 가치의 하한(lower bound)을 제공하도록 강제해야 한다는 원리입니다.

2. **완화된 하한의 충분성**: 모든 상태-액션 쌍에 대한 점별(pointwise) 하한이 아닌, **정책 하에서의 기댓값만 하한**이면 충분하다는 통찰력입니다. 이는 불필요한 과도한 비관주의를 피하면서도 안전한 정책 개선을 보장합니다.

3. **간단한 정규화를 통한 실현**: 이러한 이론적 보장을 달성하기 위해 단순한 Q-값 정규화항을 표준 Bellman 오류 목적함수에 추가하는 것으로 충분하며, 기존 알고리즘(SAC, DQN)에 20줄 이하의 코드 수정으로 구현 가능합니다.

### 주요 기여
**이론적 기여**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- **정리 3.1-3.2**: 보수적 Q-함수가 진정한 정책 가치를 하한하는 것의 엄밀한 증명
- **정리 3.3-3.4**: CQL이 "간격 확장(gap-expanding)" 특성을 가지며, 이로 인해 분포 내 액션과 분포 외 액션 간의 Q-값 차이가 증가함을 보여줌
- **정리 3.5-3.6**: 안전한 정책 개선의 보장과 동료 정책 대비 성능 개선의 이론적 경계(bound) 제시

**실증적 기여**:
- D4RL 벤치마크에서 기존 오프라인 RL 방법 대비 2-5배의 성능 향상
- 복잡한 다중 정책 분포(multi-modal data distribution) 데이터셋에서 특히 강력한 성능
- AntMaze, Adroit, Kitchen 등의 고차원 과제에서 유일하게 의미 있는 성능 달성

***

## 2. 해결하려는 문제 및 제안 방법 상세 분석
### 2.1 오프라인 강화학습의 핵심 문제 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
오프라인 강화학습은 환경과의 상호작용 없이 사전 수집된 데이터셋 $D$로부터 최적 정책을 학습하려는 문제입니다. 이는 로봇공학, 의료, 자율주행 등 실제 환경과의 상호작용이 위험하거나 비용이 높은 분야에서 중요한 과제입니다.

**문제의 근본**: 표준 오프-정책 강화학습 알고리즘이 오프라인 환경에서 실패하는 이유는 **행동 분포 편이(action distribution shift)**입니다.

표준 액터-크리틱 정책 개선 단계:

$$\pi_{k+1} = \arg\max_\pi \mathbb{E}_{s \sim D, a \sim \pi(a|s)} [\hat{Q}_k(s,a)]$$

이 식에서 $\hat{Q}_k$는 데이터셋의 행동 분포 $\beta(a|s)$에서만 학습되었는데, 새로운 정책 $\pi$의 행동에 대해 평가됩니다. 함수 근사와 유한 샘플로 인해, 데이터셋에 포함되지 않은 행동(분포 외 행동, OOD action)에 대한 Q-값이 심각하게 과대추정될 수 있습니다 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf).

**결과**:
- 정책이 잘못된 높은 Q-값을 가진 OOD 행동을 선택
- 실제 환경에서 성능 붕괴 또는 안전 문제 발생
- 기존 online RL에서는 실제 행동 결과 관찰로 이러한 오류를 수정 가능하지만, 오프라인 환경에서는 불가능

### 2.2 CQL 해결책: 보수적 Q-함수 학습
CQL은 이 문제를 두 가지 방식으로 해결합니다:

#### 방식 1: 점별 하한 (Equation 1) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

$$\min_Q \lambda \mathbb{E}_{s \sim D, a \sim \rho(a|s)} [Q(s,a)] + \frac{1}{2}\mathbb{E}_{s,a,s' \sim D} \left[\left(Q(s,a) - \mathcal{B}^{\pi_k}Q(s,a)\right)^2\right]$$

여기서:
- 첫 번째 항: 임의의 분포 $\rho(a|s)$에서 Q-값을 최소화하여 OOD 액션에 대한 과대추정 억제
- 두 번째 항: 표준 Bellman 오류 (TD 손실)
- $\lambda$: 정규화 강도 파라미터

**직관**: 모든 가능한 행동에서 Q-값을 낮게 유지하면, 데이터셋 분포에서의 Q-값이 과도하게 낮아질 수 있습니다(과소추정).

#### 방식 2: 정책 가치 하한 (Equation 2, CQL의 최종 형태) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

$$\min_Q \lambda \left(\mathbb{E}_{s \sim D, a \sim \rho(a|s)} [Q(s,a)] - \mathbb{E}_{s \sim D, a \sim \beta(a|s)} [Q(s,a)]\right) + \frac{1}{2}\mathbb{E}_{s,a,s' \sim D} \left[\left(Q(s,a) - \mathcal{B}^{\pi_k}Q(s,a)\right)^2\right]$$

**개선점**:
- OOD 분포 $\rho$에서의 Q-값을 최소화
- 데이터 분포 $\beta$에서의 Q-값을 **최대화**
- 결과: 데이터셋 내 액션에 대해서는 과소추정이 덜함
- 정리 3.2에 의해, $\mathbb{E}_{a \sim \pi}[Q(s,a)]$만 진정한 정책 가치를 하한 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

**수학적 정당화**: 정리 3.2는 다음을 증명합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

$`\mathbb{E}_{a \sim \pi} [Q^*(s,a)] \geq \mathbb{E}_{a \sim \pi} [Q^*(s,a)] - \left\|I - P^*\right\|_\infty \cdot C_{r,T,\delta} \cdot \lambda^{-1} \cdot \mathbb{E}_{s \sim d^*} \left[\sum_a \left|\rho(a|s) - \beta(a|s)\right| \cdot \mathbb{1}_{n_{s,a} > 0}\right]`$

$\lambda$가 충분히 크면, 우변이 음수가 되어 하한 성질이 보장됩니다.

#### 방식 3: CQL 동족 (Equation 3-4) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

$$\min_Q \max_\pi \mathbb{E}_{s \sim D, a \sim \pi(a|s)} [Q(s,a)] - \mathbb{E}_{s \sim D, a \sim \beta(a|s)} [Q(s,a)] + \frac{1}{2}\mathbb{E}_{s,a,s' \sim D} \left[\left(Q(s,a) - \mathcal{B}^{\pi_k}Q(s,a)\right)^2\right] + R(\pi | \beta)$$

**CQL^H (Entropy 정규화)**:

$$\min_Q \mathbb{E}_{s \sim D} [\log \sum_a e^{Q(s,a)}] - \mathbb{E}_{s,a \sim D} [Q(s,a)] + \frac{1}{2}\mathbb{E}_{s,a,s' \sim D} \left[\left(Q(s,a) - \mathcal{B}^{\pi_k}Q(s,a)\right)^2\right]$$

여기서 $\log \sum_a e^{Q(s,a)}$는 soft-maximum으로 작동하며, 고차원 액션 공간에서 더 안정적입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

### 2.3 핵심 이론 결과
**정리 3.3 (보수적 Q-값)**: 정책 업데이트가 천천히 일어나면 ($D_{TV}(\pi_{k-1}, \pi_k^*)$ 작음), 임의의 CQL 변종에서: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

$`V^*(s) \geq \mathbb{E}_{\pi_k}[Q^*(s,a)]`$

증명 직관: 보수적 정규화항으로 인한 과소추정이 다음 반복에서의 과대추정을 상쇄합니다.

**정리 3.4 (간격 확장 특성)**: CQL 백업은 다음을 만족합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

$`\mathbb{E}_{\pi_\beta} [\hat{Q}^{k+1}(s,a)] - \mathbb{E}_{\pi_k} [\hat{Q}^{k+1}(s,a)] \geq \mathbb{E}_{\pi_\beta} [Q^*(s,a)] - \mathbb{E}_{\pi_k} [Q^*(s,a)]`$

즉, Q-함수가 데이터 분포 내 액션과 분포 외 액션 간의 차이를 자동으로 확대하여, 정책을 데이터셋 내로 자동으로 제약합니다.

***

## 3. 모델 구조 및 알고리즘
### 3.1 알고리즘 1: Conservative Q-Learning [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
```
초기화: Q-함수 Q_θ, 정책 π_φ (액터-크리틱의 경우)

for 각 학습 단계 t = 1, ..., N do:
  - G_Q 경사 스텝: CQL 목적함수로 Q-함수 업데이트 (Equation 4 또는 변종)
  - G_π 경사 스텝: 정책 업데이트 (엔트로피 정규화 SAC 스타일)
end for
```

### 3.2 Q-함수 업데이트 상세
**손실 함수** (CQLH 변종): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

$$L_Q(\theta) = \mathbb{E}_{s \sim D} [\log \sum_a e^{Q_\theta(s,a)}] - \mathbb{E}_{(s,a) \sim D} [Q_\theta(s,a)] + \frac{1}{2}\mathbb{E}_{(s,a,s',r) \sim D} \left[(r + \gamma \mathbb{E}_{a' \sim \pi_\phi} [Q_{\bar{\theta}}(s',a')] - Q_\theta(s,a))^2\right]$$

구성 요소:

1. **Soft-Max 정규화** ($\mathbb{E}_{s \sim D} [\log \sum_a e^{Q(s,a)}]$):
   - OOD 액션들에 대한 Q-값을 낮춤
   - 상태 $s$에서 모든 가능한 액션의 "평균" Q-값에 대한 상한 제공

2. **데이터 최대화** ($-\mathbb{E}_{(s,a) \sim D} [Q(s,a)]$):
   - 실제 데이터셋에 포함된 (상태, 액션) 쌍의 Q-값을 높임
   - 점별 하한의 과도한 비관주의 완화

3. **Bellman TD 손실**:
   - 표준 시간차 학습(Temporal Difference) 목적
   - 동적 프로그래밍 원리 유지

### 3.3 정책 업데이트
액터-크리틱 설정에서의 정책 개선:

$$\max_\phi \mathbb{E}_{s \sim D, a \sim \pi_\phi(a|s)} [Q_\theta(s,a)] - \alpha \mathbb{E}_{s \sim D, a \sim \pi_\phi(a|s)} [\log \pi_\phi(a|s)]$$

여기서 $\alpha$는 엔트로피 계수(SAC로부터). 정책은 보수적 Q-함수에 대해서만 최적화되므로, 자동으로 데이터 분포에 가까운 액션을 선호합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

### 3.4 네트워크 구조
**구성 요소**:
- **Critic (Q-함수)**: 2개의 독립 신경망 (쌍 Q-러닝) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
  - 입력: 상태 벡터
  - 출력: Q-값 (스칼라)
  - 구조: 표준 SAC/DQN 아키텍처 (특별한 수정 없음)

- **Actor (정책)**: 확률론적 정책 신경망
  - 입력: 상태 벡터
  - 출력: 액션 분포의 파라미터 (평균, 분산)
  - 액터-크리틱에서만 사용 (Q-러닝 변종에서는 불필요)

- **Target 네트워크**: 안정성을 위한 천천한 업데이트

**하이퍼파라미터**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- 학습률: 정책 3e-5, Q-함수 1e-4 (정책 천천한 업데이트)
- 배치 크기: 256
- $\lambda$: 자동 라그랑주 조정 또는 고정값 (도메인 의존)
- 할인 계수 $\gamma$: 0.99

***

## 4. 성능 향상 및 한계
### 4.1 경험적 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
#### D4RL 벤치마크 결과

| 데이터셋 유형 | CQL 성능 개선 | 주요 특성 |
|---------|----------|---------|
| Expert | 경미한 개선 | 좋은 데이터이므로 모든 방법이 잘 작동 |
| Medium | 작은 개선 | 단일 정책 데이터 |
| Mixed | **2-3배 개선** | 여러 정책의 혼합 분포 |
| Medium-Expert | **2-3배 개선** | 다중 정책 분포 |
| Random-Expert | **최대 5배 개선** | 극단적으로 다양한 분포 |

**구체적 예시** (표 1): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- Hopper-Random: 11.3 (SAC) → 10.8 (CQL) (낮은 분산 데이터)
- Hopper-Random-Expert: 5.6 (SAC) → 110.5 (CQL) (**~20배 개선**)
- Walker2d-Mixed: 1.9 (SAC) → 26.7 (CQL) (**14배 개선**)

#### Adroit 과제 (고차원 조작) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

표 2의 Adroit 결과:
- Pen-Human: 34.4 (BC) → 55.8 (CQL) (62% 개선)
- Hammer-Human: 1.5 (BC) → 4.4 (CQL) (193% 개선)
- Door-Human: 0.5 (BC) → 9.9 (CQL) (**20배 개선**)

**주목**: 기존 오프라인 RL 방법은 모두 실패하고 행동 복제(BC)만이 기준을 삼음. CQL만이 BC를 능가. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

#### AntMaze 과제 (경로 계획) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

표 2의 AntMaze 결과:
- Umaze: 모든 방법이 유사 성능
- Medium-Diverse: 0 (BEAR, BRAC) vs 53.7 (CQL) (**유일한 성공**)
- Large-Diverse: 0 (모든 베이스라인) vs 14.9 (CQL)

**의미**: 경로를 "연결(stitch)"하기 위해 서브-최적 궤적의 조합이 필요한 과제에서 CQL의 우수성 증명. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

#### Atari 게임 (이미지 입력) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

표 3의 결과 (1% 데이터만 사용):
- Qbert: 383.6 (QR-DQN) → 14,012 (CQL) (**36배 개선**)
- Breakout: 7.9 (QR-DQN) → 61.1 (CQL) (**7.7배 개선**)
- Seaquest: 672.9 (QR-DQN) → 779.4 (CQL) (약간의 개선)

### 4.2 이론적 보장 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
**정리 3.5**: CQL은 다음의 페널티가 있는 경험적 MDP 목적함수를 최적화합니다:

$`\arg\max_\pi \mathbb{E}_{s \sim d^\pi_M} [V^*(s)] - \frac{1}{\lambda(1-\gamma)}\mathbb{E}_{s \sim d^M} [D^{CQL}_\lambda(s)]`$

여기서 $D^{CQL}\_\lambda(s) = \sum_a |\pi(a|s) - \beta(a|s)| / (1 + \mathbb{1}\_{n_{s,a} > 0})$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf).

**정리 3.6 (안전한 정책 개선)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

$$J(\pi^*, M) - J(\pi_\beta, M) \geq -\left(2C_{r,T,\delta} \lambda^{-1} R_{max} C_{T,\delta}^{-1} - \mathbb{E}_{s \sim d^M} \left[\text{개선 용어}\right]\right)$$

즉, 적절한 $\lambda$ 선택 하에서 학습된 정책은 행동 정책보다 보장되게 나음. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

### 4.3 성능 한계 및 문제점
#### 1. 이론-실행 간극 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

- **이론**: 점별 또는 선형 함수 근사 가정
- **실행**: 깊은 신경망 사용
- **문제**: 신경망의 일반화 오류를 정확히 특성화하기 어려움
- **결과**: 매우 깊은 Q-함수 추정 오류에 대한 이론적 보장 미흡

#### 2. 과도한 비관주의 (Over-conservatism)

- **현상**: 복잡한 과제에서 Q-값이 진정한 값 대비 너무 낮아질 수 있음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- **결과**: 정책 개선이 제한되고 성능 향상이 둔화
- **원인**: 모든 상태에서 균등한 정규화 강도 $\lambda$ 사용

#### 3. 하이퍼파라미터 민감성

- **문제**: $\lambda$ (또는 라그랑주 제약 $\nu$) 선택이 성능에 큰 영향 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- **어려움**: 도메인마다 다른 최적값 필요
- **자동 조정**: Lagrangian 이중 경사법 도입하였으나, 여전히 초기 $\nu$ 값에 민감 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

#### 4. 검증 문제 (No Offline Validation)

- **감독학습**: 검증 집합을 사용하여 과적합 방지 가능
- **오프라인 RL**: 검증 데이터에서의 정책 성능을 정확히 추정하기 어려움
- **결과**: 조기 종료(early stopping) 어려움, 과적합 위험

#### 5. 계산 비용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

- **Log-sum-exp 계산**: $\log \sum_a e^{Q(s,a)}$는 고차원 액션 공간에서 고비용
- **대안**: CQL^π (이전 정책 사용) 또는 샘플링 기반 근사 사용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- **트레이드오프**: 근사는 성능 저하 유발 가능

#### 6. 분포 외 정책 검증의 어려움

- **문제**: 행동 정책이 불충분하거나 알려지지 않은 경우
- **결과**: 올바른 정규화 강도 판단 불가능

***

## 5. 모델 일반화 성능 향상 가능성
### 5.1 일반화의 정의 및 분석
오프라인 RL에서의 일반화는 여러 차원을 포함합니다:

#### 1. **데이터 분포 내 일반화 (In-distribution Generalization)**

데이터셋에 포함된 상태-액션 쌍에 대한 일반화:

$$\text{성능} = \mathbb{E}_{(s,a) \sim D}[V(s)] \text{ vs } V^*(s)$$

**CQL의 강점**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- Q-함수의 점별 하한 보장으로 점진적 개선 가능
- 표 4에서 CQL은 기준 정책의 평가값을 정확히 하한
- BEAR 및 앙상블 방법은 과대평가하여 신뢰성 부족 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

#### 2. **분포 외 일반화 (Out-of-distribution Generalization)**

데이터셋에 없는 상태-액션에 대한 성능:

**CQL의 방법론**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- **간격 확장**: 정리 3.4의 gap-expanding 특성으로, OOD 액션은 자동으로 낮은 Q-값 할당
- **암묵적 정책 제약**: 정규화 항으로 인해 정책이 데이터 분포 내로 자동 제약
- **온순한 일반화**: 점별 하한이 아닌 기댓값 하한이므로, OOD 지역에서의 적절한 일반화 가능

**실증적 증거** (그림 2): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- Hopper-Expert: CQL의 $\Delta_k$ (in-distribution vs OOD Q-값 차이) ≈ -10 (음수 = 보수적)
- BEAR의 $\Delta_k$ ≈ +30 (양수 = OOD 과대평가 위험)

#### 3. **다중 정책 분포에서의 일반화**

표 1에서 "Mixed" 및 "Random-Expert" 데이터셋: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- 서로 다른 정책으로 수집된 궤적의 혼합
- 기존 방법 (BEAR, BRAC): 성능 악화
- CQL: 2-5배 개선

**원인**: CQL의 보수적 정규화가 복잡한 멀티-모달 분포에서도 안정적

#### 4. **차원 확장성 (Scalability)**

표 2의 Adroit (고차원 조작) 결과: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- 24-DoF 로봇 손 제어
- CQL^π (이전 정책 사용): CQLH (log-sum-exp) 보다 안정
- 이유: 고차원에서 importance 가중치의 분산이 높기 때문

### 5.2 일반화 성능 향상 메커니즘
#### 메커니즘 1: 보수적 추정의 안정성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

CQL의 하한 성질은 다음의 이점을 제공합니다:

$$\text{정책 성능} = \mathbb{E}[\text{누적 보상}] \geq \mathbb{E}[Q_{CQL}(s, a)]$$

따라서 $Q_{CQL}$이 낮으면 실제 성능은 이보다 높을 가능성이 높습니다. 이는:
- 과도한 최적화 위험 감소
- 실제 환경에서의 안전한 배포 가능
- 하이퍼파라미터 튜닝의 안정성 증가

#### 메커니즘 2: 자동 정책 제약 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

정리 3.4의 gap-expanding 특성으로부터: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

$$\mathbb{E}_{a \sim \pi_\beta}[Q] - \mathbb{E}_{a \sim \pi_k}[Q] \text{이 증가}$$

이는 자동으로 정책이 데이터 분포로 수렴하도록 강제하여:
- 명시적 정책 제약 필요 없음
- BCQ, BEAR의 행동 정책 추정 필요 없음
- 복잡한 다중 모드 분포에서도 안정적

#### 메커니즘 3: 유연한 낮은 경계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

점별 하한이 아닌 기댓값 하한이므로:

$$\mathbb{E}_\pi[Q] \geq V^* \text{ (보장)}$$

그러나:

$$Q(s, a) \text{는 개별 } (s, a) \text{에서 } Q^* \text{보다 낮을 수 있음}$$

이는 다음을 의미합니다:
- 정책은 기댓값 하에서는 안전 (과대평가 없음)
- 그러나 개별 행동은 일반화하여 높은 Q-값 가능 (합리적 정책 개선)
- 과도한 비관주의 회피

### 5.3 최근 일반화 개선 (2020 이후)
의 비교 차트에서 보이는 바와 같이, CQL 이후의 방법들은 다음과 같이 일반화를 개선했습니다:
#### IQL (2021) [arxiv](https://arxiv.org/abs/2110.06169v1)

**개선점**:
- In-sample 값 추정만 사용
- 기댓값 회귀(expectile regression)로 OOD 쿼리 완전 회피
- CQL보다 더욱 보수적이나, 많은 과제에서 동등 또는 우수 성능

**일반화 향상**:
- 분포 내 행동에만 의존하므로 OOD 오류 근본 배제
- 대신 과도한 비관주의 가능성

#### MCQ (2022) [arxiv](https://arxiv.org/html/2206.04745v3)

**개선점**:
- "온순한 보수주의(mild conservatism)" 개념 도입
- OOD 행동에 의사 값(pseudo value) 할당
- 과도한 비관주의 완화

**일반화 향상**:
- 정책이 데이터 분포 밖으로 더 많이 일반화 가능
- Offline-to-Online 미세조정에서 우수한 성능
- 실제 성능 개선과 보수주의 간의 더 나은 균형

#### SA-CQL (2022) [emergentmind](https://www.emergentmind.com/topics/conservative-q-learning-cql-model)

**개선점**:
- 상태별 적응형 정규화 강도
- 상태 밀도(state density)에 따라 $\lambda$ 조정
- DualDICE를 사용한 밀도 비율 추정

**일반화 향상**:
- 데이터가 충분한 상태에서는 덜 보수적
- 데이터가 희소한 상태에서는 더 보수적
- 효율적인 탐사-착취 균형

#### DOGE (2023) [arxiv](https://arxiv.org/pdf/2205.11027.pdf)

**개선점**:
- 데이터셋 기하학을 활용
- 깊은 함수 근사의 볼록 껍질 내에서의 좋은 근사 특성 활용
- OOD 지역의 일반화 가능성 증대

**일반화 향상**:
- CQL의 "일반화 외부 엄격한 제약" 완화
- 데이터 기하학적으로 합리적인 OOD 지역에서 정책 개선 가능
- CQL보다 더 나은 최종 성능

#### GTP (2025) [arxiv](https://arxiv.org/abs/2510.11499)

**개선점**:
- 생성 모델 활용 (확산 정책 등)
- 정규화가 아닌 생성 모델의 표현력 활용
- 복잡한 다중 정책 분포의 명시적 모델링

**일반화 향상**:
- AntMaze hard 과제에서 완벽한 점수 달성
- CQL 불가능한 극단적 분포 외 일반화

### 5.4 일반화 성능의 한계
그러나 2024년 연구에서 지적된 바: [proceedings.iclr](https://proceedings.iclr.cc/paper_files/paper/2024/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf)

**감독 학습 대비 일반화 격차**:
- Behavior Cloning (BC)이 train/test 분할에서 더 나은 일반화
- Offline RL은 in-distribution 성능에 최적화 (기댓값 하한)
- 분포 외 테스트 집합에서는 BC 우월

**결론**: CQL 포함 모든 오프라인 RL 방법의 일반화는:
- **강점**: 동일 분포(데이터셋 내) 일반화 우수
- **약점**: 진정한 분포 외 일반화 (new test distribution) 어려움
- **이유**: 오프라인 학습의 근본적 특성 (상호작용 불가)

***

## 6. 이론 및 실행 간의 격차
### 6.1 이론적 분석의 범위 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
CQL 논문의 이론적 결과: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- **정리 3.1-3.4**: 함수 근사 없는 tabular 설정
- **정리 D.1**: 선형 함수 근사 (linear FA)
- **정리 D.2**: 신경 탄젠트 커널(NTK) 하 단계 경사 업데이트

### 6.2 실행 시 차이
실제 알고리즘: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- 깊은 신경망 (multiple hidden layers)
- SGD/Adam 최적화 (완전 수렴 보장 없음)
- 고차원 상태 및 행동 공간
- 비볼록 최적화 문제

**문제점**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- NTK 가정 (작은 파라미터 범위) 위반
- 신경망의 실제 일반화 오류 특성화 불가능
- 실제 구현에서의 수렴 보장 없음

### 6.3 경험적 검증 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
논문은 표 4를 통해 이론적 보장의 실증 검증: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

| 과제 | CQLH | CQL (Eq. 1) | Ensemble (2) | Ensemble (4) |
|-----|------|----------|-------------|-------------|
| Hopper-Expert | -43.20 | -151.36 | 3.71e6 | 2.93e6 |
| Hopper-Mixed | -10.93 | -22.87 | 15.00e6 | 59.93e3 |

**해석**:
- 음수 값: 추정 Q-값 < 실제 성능 (하한 만족)
- CQLH: 대부분 하한 만족
- Ensemble: 심각한 과대평가 (상한 위반)
- 이는 이론이 실제로 작동함을 시사 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

***

## 7. 최근 연구와의 비교 분석 (2020-2026)
차트와 검색 결과를 기반으로 한 체계적 비교:

### 7.1 근본적 접근 방식 진화
**1세대 (2019-2020): 정책 제약 패러다임**
- BCQ (2019): 생성 모델로 액션 제약
- BEAR (2019): KL 발산으로 정책 제약
- **문제**: 행동 정책 추정의 어려움

**2세대 (2020): 값 정규화 패러다임**
- **CQL (2020)**: Q-함수 정규화 도입
  - 명시적 정책 제약 불필요
  - 행동 정책 추정 불필요
  - 이론적 보장 제공
  
**3세대 (2021-2022): 개선된 보수주의**
- IQL (2021): In-sample 값 추정
- TD3+BC (2021): 행동 복제 정규화
- MCQ (2022): 온순한 보수주의
- SA-CQL (2022): 적응형 정규화

**4세대 (2023-2025): 다양한 패러다임**
- DOGE (2023): 기하학 기반 접근
- GTP (2025): 생성 모델 활용
- BAQ (2025): Offline-to-online 전환

### 7.2 성능 비교 매트릭스
**D4RL Gym (MuJoCo) 평균 점수 비교**:

| 메서드 | 단순 데이터셋 | 복합 데이터셋 | AntMaze (어려움) |
|-------|-----------|-----------|--------------|
| BCQ | 좋음 | 나쁨 | 0 |
| CQL | 좋음 | **매우 좋음** | **좋음** |
| IQL | 좋음 | **매우 좋음** | **좋음** |
| TD3+BC | 좋음 | 좋음 | 0 |
| MCQ | 중간 | **매우 좋음** | 중간 |
| SA-CQL | 좋음 | **최상** | **최상** |
| DOGE | 좋음 | **매우 좋음** | 좋음 |
| GTP | 매우 좋음 | **최상** | **완벽** |

### 7.3 특성별 비교
#### 이론적 우수성
- **CQL**: 가장 견고한 이론 (Theorems 3.1-3.6)
- **IQL**: 강한 이론 (in-sample 보장)
- **MCQ**: 완화된 이론 (경험적)
- **GTP**: 이론 약함 (경험적 중심)

#### 계산 효율성
- **TD3+BC**: 최고 효율 (간단한 정규화)
- **CQL**: 중간 (log-sum-exp 비용)
- **IQL**: 중간 (기댓값 회귀)
- **GTP**: 낮음 (생성 모델 오버헤드)

#### 하이퍼파라미터 민감성
- **CQL**: 중간 ($\lambda$ 조정 필요)
- **IQL**: 낮음 (τ, α 직관적)
- **TD3+BC**: 낮음 (α = 0.4 효과적)
- **MCQ**: 낮음 (점진적 감소)
- **GTP**: 높음 (생성 모델 파라미터)

#### 실무 적용성
- **TD3+BC**: 최고 (간단, 효과적)
- **CQL**: 높음 (안정적, 이론적)
- **IQL**: 높음 (안정적, 효율적)
- **MCQ**: 중간 (개선됨, 덜 확립됨)
- **GTP**: 낮음 (복잡, 최신)

### 7.4 애플리케이션별 권장사항
**자율주행**: [arxiv](https://arxiv.org/abs/2508.07029v2)
- CQL 기본 사용 (3.2배 성공률 개선)
- 데이터 큐레이션 중요 (불확실성 기반 가중치)

**로봇 조작**: [roboticsproceedings](http://www.roboticsproceedings.org/rss19/p019.pdf)
- CQL (Adroit에서 우수) 또는 IQL
- 사전학습 + 미세조정 패러다임

**의료 정책**: [arxiv](https://arxiv.org/abs/2505.16242)
- OGSRL (안전 제약 포함)
- CQL 기반이나 상태 궤적 제약 추가

**금융 거래**: [semanticscholar](https://www.semanticscholar.org/paper/aeaa9990567617f8325fa035824c1521cfe86178)
- TD3+BC 또는 CQL (검증됨)
- Decision Transformer 비교

***

## 8. 이후 연구에 미치는 영향
### 8.1 학술적 영향
**인용도**:
- 2,918+ 인용 (높은 영향력)
- 오프라인 RL의 표준 베이스라인
- ICML, NeurIPS, ICLR 등에서 후속 연구 다수

**개념적 기여**:
- "보수적 Q-함수" 개념의 정립
- 오프라인 RL의 이론적 기초 확립
- 이후 방법들의 출발점

### 8.2 기술적 영향
**알고리즘 설계**:
- CQL 정규화 항의 여러 변종 개발 [emergentmind](https://www.emergentmind.com/topics/conservative-q-learning-cql-model)
- 적응형 $\lambda$ 선택 메커니즘
- 다양한 정규화 분포 탐색

**표준화**:
- D4RL 벤치마크의 기준 방법
- NeurIPS Offline RL Workshop의 핵심 주제
- 실무 구현의 표준 기준

### 8.3 응용 분야의 확대
**초기 (2020-2021)**:
- 로봇공학 시뮬레이션 (MuJoCo, Atari)

**중기 (2021-2023)**:
- 로봇 조작 (Adroit, Kitchen) [roboticsproceedings](http://www.roboticsproceedings.org/rss19/p019.pdf)
- 네비게이션 (AntMaze) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- 시뮬레이션 기반 현실 세계 전이 [roboticsproceedings](http://www.roboticsproceedings.org/rss19/p019.pdf)

**최근 (2023-2026)**:
- 자율주행 [arxiv](https://arxiv.org/abs/2508.07029v2)
- 의료 정책 학습 [arxiv](https://arxiv.org/abs/2505.16242)
- 금융 거래 [semanticscholar](https://www.semanticscholar.org/paper/aeaa9990567617f8325fa035824c1521cfe86178)
- 칩 설계 (상세 라우팅) [arxiv](https://arxiv.org/abs/2512.03594)
- DeFi 프로토콜 최적화 [arxiv](https://arxiv.org/abs/2506.00505)

### 8.4 이론적 진전
**분포 이동 이론**:
- CQL 이전: 경험적 해결책 중심
- CQL 이후: 정책 가치의 "하한" 개념 정립
- 최근: 적응형 하한, 기하학 기반 접근

**함수 근사 이론**:
- CQL: NTK 하 선형 근사 분석 ] [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
- 최근: 신경망의 실제 일반화 메커니즘 이해 시도

***

## 9. 향후 연구 시 고려사항
### 9.1 이론적 고려사항
#### 1. 깊은 신경망 이론화

**현재 상황**:
- 정리 3.1-3.6은 선형 함수 근사 또는 NTK 가정
- 실제 구현은 깊은 비선형 신경망

**필요한 작업**:
- 신경망의 실제 일반화 오류 특성화
- 과적합 방지 메커니즘의 이론화
- 학습 동역학의 수렴 분석

#### 2. 적응형 정규화 이론

**문제**:
- $\lambda$의 최적값이 도메인 및 상태별로 다름
- 고정 $\lambda$는 과도한 비관주의 초래 가능

**미래 방향**:
- 상태별/연속적 적응형 $\lambda$ 선택의 이론 (SA-CQL 확장) [emergentmind](https://www.emergentmind.com/topics/conservative-q-learning-cql-model)
- 데이터 분포 학습과 정규화 강도의 통합

#### 3. 복합 도메인의 이론화

**현재 한계**:
- 대부분의 이론은 단일 도메인 가정
- 다중 과제/도메인 설정의 이론 부족

**미래 방향**:
- 메타-학습과 오프라인 RL의 통합 이론
- 전이 학습의 정식화

### 9.2 알고리즘 설계 고려사항
#### 1. 과도한 비관주의 완화

**추세**: MCQ, DOGE, DMG 등이 온순한 보수주의 제시 [arxiv](https://arxiv.org/html/2206.04745v3)

**향후 방향**:
- 자동 보수주의 수준 조절 메커니즘
- 데이터셋 특성(다중 모드, 분포 범위)에 기반한 동적 $\lambda$
- 과소추정과 과대추정의 균형 (둘 다 위험)

#### 2. 불확실성 정량화

**현재**: CQL은 점 추정 (점별 Q-값)

**발전 방향**:
- 분산 기반 오프라인 RL (분포 RL 활용)
- 알레아토릭(aleatoric) vs 인식적(epistemic) 불확실성 분해
- 불확실성 기반 탐사 (선택적 미세조정 시)

#### 3. 온라인 미세조정 전이

**문제**: Offline-to-online 전이 중 성능 붕괴 가능 [offline-rl-neurips.github](https://offline-rl-neurips.github.io/2021/pdf/30.pdf)

**해결책**:
- 적응형 행동 복제 정규화 (AdaptiveBC) [arxiv](https://arxiv.org/pdf/2210.13846.pdf)
- 행동 적응 정책 (BAQ) [github](https://github.com/sfujim/TD3_BC)
- 동적 정규화 강도 조정

#### 4. 다중 에이전트 확장

**진행**: SA-CQL for MARL, CFCQL 등 [emergentmind](https://www.emergentmind.com/topics/conservative-q-learning-cql-model)

**고려사항**:
- 에이전트 수 증가에 따른 확장성
- 에이전트 간 분포 이동
- 협력 vs 경쟁 설정에서의 오프라인 RL

### 9.3 실무적 고려사항
#### 1. 벤치마킹 및 평가

**현재**:
- D4RL이 표준이나, 실제 응용과의 괴리 존재
- 새로운 벤치마크 필요 (자율주행], 로봇) [arxiv](https://arxiv.org/html/2503.19267v1)

**향후**:
- 다양한 도메인의 실제 데이터셋 공개 필요
- 오프라인 평가 메트릭의 표준화 (현재 문제) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

#### 2. 데이터 큐레이션

**최근 발견**: [arxiv](https://arxiv.org/html/2405.18187v2)
- 균등한 샘플링보다 중요도 가중치(criticality weighting) 우수
- 불확실성 기반 큐레이션이 3배 안전성 개선 [arxiv](https://arxiv.org/html/2405.18187v2)

**향후 방향**:
- 오프라인 RL과 데이터 큐레이션의 통합 프레임워크
- 자동 데이터 정제 기법
- 낮은 품질 데이터 처리 메커니즘

#### 3. 안전 보장

**응용**: 의료, 자율주행 등에서 중요 [arxiv](https://arxiv.org/abs/2505.16242)

**현재 CQL의 한계**:
- 정책 안전성만 보장 (가치 하한)
- 상태 궤적의 안전성 보장 안 함

**향후 방향**:
- 제약 조건 있는 오프라인 RL (OGSRL) [arxiv](https://arxiv.org/abs/2505.16242)
- 안전 임계값의 확률적 보장
- 인간 피드백의 통합

### 9.4 연구 방향의 구체적 제안
#### A. 이론-실행 간 격차 해소

**과제**: 신경망 기반 CQL의 수렴 및 일반화 분석

**제안**:
- 과적합 경계 분석 (Rademacher complexity 등)
- 정규화 강도와 일반화 오류의 트레이드오프
- 온라인 미세조정 시 CQL의 수렴 속도

#### B. 적응형 보수주의

**과제**: 데이터셋 의존 최적 $\lambda$ 선택

**제안**:
- 데이터 엔트로피, 다중 모드 수준, 분포 범위로부터 자동 $\lambda$ 선택
- 학습 진행 중 $\lambda$ 일정 감소 스케줄
- 메타-학습 기반 $\lambda$ 최적화

#### C. 생성 모델 통합

**과제**: GTP가 완벽한 점수 달성하는 메커니즘 이해 [arxiv](https://arxiv.org/abs/2510.11499)

**제안**:
- CQL 정규화와 생성 모델의 표현력 결합
- 확산 모델 기반 정책의 오프라인 RL
- 트랜스포머 기반 시퀀스 모델과 CQL의 통합

#### D. 실시간 시스템 적용

**과제**: 레이턴시 제약이 있는 자율주행, 로봇 등에 적용

**제안**:
- 경량 신경망 아키텍처 (모바일 배포)
- 계산 그래프 최적화 (log-sum-exp 근사)
- 하드웨어 가속 (GPU/TPU) 통합

#### E. 비정상(Nonstationary) 오프라인 RL

**과제**: 실제 데이터는 여러 기간에 수집, 분포 변화 가능

**제안**:
- 시간-가중 오프라인 RL (최근 데이터에 높은 가중치)
- 변화 감지 메커니즘
- 온라인-오프라인 혼합 학습 (연속 데이터 수집 상황)

***

## 10. 결론
Conservative Q-Learning은 오프라인 강화학습 분야에서 이론과 실용성을 모두 갖춘 핵심 알고리즘으로, **분포 이동 문제의 해결책**을 명확하게 제시했습니다. 그 주요 성공 요인은: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)

1. **이론적 견고성**: 정책 가치의 하한 보장을 통한 수학적 정당화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
2. **실무적 단순성**: 기존 알고리즘에 최소한의 수정으로 구현 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
3. **경험적 강력함**: 2-5배 성능 향상, 특히 복잡한 데이터셋에서 우수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3610ccc0-e9e0-4483-9c2a-ceadf259f246/2006.04779v3.pdf)
4. **광범위한 적용성**: 로봇공학부터 자율주행, 의료에 이르기까지 검증 [roboticsproceedings](http://www.roboticsproceedings.org/rss19/p019.pdf)

2020년 이후의 연구는 CQL의 **기본 개념을 계승하면서 개선**하고 있습니다: [emergentmind](https://www.emergentmind.com/topics/conservative-q-learning-cql-model)

- **온순한 보수주의**: 과도한 비관주의 완화
- **적응형 정규화**: 상태별/데이터 의존 조정
- **기하학 기반 접근**: 일반화 가능 OOD 지역 활용
- **생성 모델 통합**: 더 표현력 있는 정책 클래스

향후 연구의 핵심 과제는  **신경망 이론의 강화**,  **자동 하이퍼파라미터 선택**,  **데이터 큐레이션 통합**, 그리고  **안전 보장의 확대**입니다. CQL은 단순히 하나의 알고리즘을 넘어 **오프라인 RL의 패러다임 전환**을 일으켰으며, 그 영향은 앞으로도 지속될 것으로 예상됩니다. [arxiv](https://arxiv.org/abs/2505.16242)

***

**주요 참고문헌 (Research IDs)**:

<span style="display:none">[^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92]</span>

<div align="center">⁂</div>

[^1_1]: 2006.04779v3.pdf

[^1_2]: https://arxiv.org/abs/2110.06169v1

[^1_3]: https://www.emergentmind.com/topics/implicit-q-learning-iql

[^1_4]: https://arxiv.org/html/2206.04745v3

[^1_5]: https://arxiv.org/html/2406.13961

[^1_6]: https://www.emergentmind.com/topics/conservative-q-learning-cql-model

[^1_7]: https://arxiv.org/pdf/2205.11027.pdf

[^1_8]: https://arxiv.org/abs/2510.11499

[^1_9]: https://arxiv.org/html/2510.11499v1

[^1_10]: https://proceedings.iclr.cc/paper_files/paper/2024/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf

[^1_11]: https://arxiv.org/abs/2508.07029v2

[^1_12]: https://arxiv.org/html/2405.18187v2

[^1_13]: https://arxiv.org/html/2501.18138v1

[^1_14]: http://www.roboticsproceedings.org/rss19/p019.pdf

[^1_15]: https://arxiv.org/abs/2505.16242

[^1_16]: https://www.arxiv.org/abs/2505.16242

[^1_17]: https://www.semanticscholar.org/paper/aeaa9990567617f8325fa035824c1521cfe86178

[^1_18]: https://arxiv.org/abs/2512.03594

[^1_19]: https://arxiv.org/abs/2506.00505

[^1_20]: http://arxiv.org/pdf/2411.07934.pdf

[^1_21]: https://offline-rl-neurips.github.io/2021/pdf/30.pdf

[^1_22]: https://arxiv.org/pdf/2210.13846.pdf

[^1_23]: https://github.com/sfujim/TD3_BC

[^1_24]: https://arxiv.org/html/2503.19267v1

[^1_25]: https://www.semanticscholar.org/paper/19d944455c07bbcc28faa7a893233a0326f51749

[^1_26]: https://journaljerr.com/index.php/JERR/article/view/1779

[^1_27]: https://dl.acm.org/doi/10.1145/3383313.3411536

[^1_28]: https://www.semanticscholar.org/paper/0e1aa954b5852a0093503d9c5c163d68b5865355

[^1_29]: https://www.semanticscholar.org/paper/568c6c8ca820a161c1097fe3bcc947f436075f57

[^1_30]: https://www.semanticscholar.org/paper/cb2f9420dfb371ec866637c5b9a92a076fd746bd

[^1_31]: https://www.semanticscholar.org/paper/23f9e787c9a5c8b39c21ba110e0a58763b02ba3b

[^1_32]: https://arxiv.org/abs/2306.03680

[^1_33]: https://www.semanticscholar.org/paper/9b8444f3fba46f861740808fc20bf90ae791b478

[^1_34]: https://arxiv.org/abs/2211.14827

[^1_35]: http://arxiv.org/pdf/2409.16830.pdf

[^1_36]: https://arxiv.org/pdf/2204.12581.pdf

[^1_37]: https://arxiv.org/pdf/2102.08363.pdf

[^1_38]: http://arxiv.org/pdf/2312.12191.pdf

[^1_39]: https://arxiv.org/html/2309.16973

[^1_40]: http://arxiv.org/pdf/2211.14827.pdf

[^1_41]: http://arxiv.org/pdf/2406.09486.pdf

[^1_42]: https://openreview.net/forum?id=qkddTMfmdn

[^1_43]: https://offline-rl-neurips.github.io/2021/pdf/24.pdf

[^1_44]: https://icml.cc/virtual/2022/session/20061

[^1_45]: http://bair.berkeley.edu/blog/2022/04/25/rl-or-bc/

[^1_46]: https://papers.neurips.cc/paper_files/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf

[^1_47]: https://dl.acm.org/doi/10.5555/3545946.3598767

[^1_48]: https://arxiv.org/abs/2006.04779

[^1_49]: https://apxml.com/courses/advanced-reinforcement-learning/chapter-7-offline-reinforcement-learning/cql-offline

[^1_50]: https://www.youtube.com/watch?v=Q3z88ZgQ-IU

[^1_51]: https://ropiens.tistory.com/144

[^1_52]: https://arxiv.org/html/2509.10303v1

[^1_53]: https://velog.io/@jaeheon-lee/Paper-Review-Conservative-Q-Learning-for-Offline-Reinforcement-Learning-kblpdcap

[^1_54]: https://pdfs.semanticscholar.org/438e/42826afe267e0da0716076e27c90cd6aed82.pdf

[^1_55]: https://arxiv.org/html/2511.16475v1

[^1_56]: https://arxiv.org/pdf/2006.01844.pdf

[^1_57]: https://pdfs.semanticscholar.org/38e0/b3c5aee11894d110ed3d189d825daea26897.pdf

[^1_58]: https://arxiv.org/pdf/2006.04779.pdf

[^1_59]: https://arxiv.org/html/2407.17877v1

[^1_60]: https://arxiv.org/html/2511.02567v1

[^1_61]: https://arxiv.org/html/2406.04534v1

[^1_62]: https://pdfs.semanticscholar.org/aaa1/447f49ef8e6b5dd08822ed43427e5740b888.pdf

[^1_63]: https://arxiv.org/html/2510.19562v2

[^1_64]: https://arxiv.org/html/2412.16848v1

[^1_65]: https://arxiv.org/abs/2510.00358

[^1_66]: https://arxiv.org/abs/2511.22210

[^1_67]: https://arxiv.org/abs/2509.15099

[^1_68]: https://arxiv.org/abs/2505.04231

[^1_69]: https://arxiv.org/abs/2409.07606

[^1_70]: https://arxiv.org/pdf/1911.11361.pdf

[^1_71]: https://arxiv.org/pdf/2106.09119.pdf

[^1_72]: http://arxiv.org/pdf/2407.00699.pdf

[^1_73]: http://arxiv.org/pdf/2502.08985.pdf

[^1_74]: https://arxiv.org/abs/2306.00972

[^1_75]: https://www.ijcai.org/proceedings/2025/0642.pdf

[^1_76]: https://openreview.net/pdf/68a0713a71679e2a82d9b8b9cb139bd5f6c3f963.pdf

[^1_77]: https://openreview.net/pdf/571b9bfdd61b88b93e53a926e523ad77c287b531.pdf

[^1_78]: https://www.scitepress.org/Papers/2025/130972/130972.pdf

[^1_79]: https://hajim.rochester.edu/ece/sites/gsharma/papers/LiuSelfBehavCloneRL_ECAI2024.pdf

[^1_80]: https://proceedings.iclr.cc/paper_files/paper/2025/file/c06f788963f0ce069f5b2dbf83fe7822-Paper-Conference.pdf

[^1_81]: https://openreview.net/pdf?id=68n2s9ZJWF8

[^1_82]: https://arxiv.org/pdf/2106.06860.pdf

[^1_83]: https://openreview.net/forum?id=EBT0oymkZb

[^1_84]: https://transferlab.ai/pills/2023/implicit-q-learning/

[^1_85]: https://arxiv.org/pdf/2511.03695.pdf

[^1_86]: http://arxiv.org/abs/2301.01298

[^1_87]: https://arxiv.org/pdf/2405.18187.pdf

[^1_88]: https://www.arxiv.org/pdf/2408.02165.pdf

[^1_89]: https://arxiv.org/abs/2110.06169

[^1_90]: https://www.arxiv.org/abs/2508.18397

[^1_91]: https://arxiv.org/html/2501.08907v1

[^1_92]: https://arxiv.org/html/2510.12638v1
