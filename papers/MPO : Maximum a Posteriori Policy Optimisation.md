# MPO : Maximum a Posteriori Policy Optimisation

## 1. 핵심 주장과 주요 기여

Maximum a Posteriori Policy Optimisation (MPO)는 상대 엔트로피(relative entropy) 목적 함수에 대한 좌표 상승법(coordinate ascent)을 기반으로 한 강화학습 알고리즘입니다. 이 논문의 핵심 주장은 다음과 같습니다:[1]

**주요 기여:**
- **Off-policy 학습의 효율성**: MPO는 on-policy 알고리즘의 확장성(scalability), 강건성(robustness), 하이퍼파라미터 둔감성(hyperparameter insensitivity)을 유지하면서도 off-policy 가치 기반 방법의 데이터 효율성(data efficiency)을 제공합니다.[1]
- **EM 기반 프레임워크**: 강화학습을 확률적 추론(probabilistic inference) 문제로 재구성하여, Expectation-Maximization (EM) 스타일의 교대 최적화를 통해 정책을 개선합니다.[1]
- **탁월한 샘플 효율성**: 연속 제어 문제에서 기존 최신 알고리즘보다 **약 10배 빠른 샘플 효율성**을 보였으며, 조기 수렴(premature convergence)과 하이퍼파라미터 설정에 대한 강건성을 개선했습니다.[1]

## 2. 문제 정의 및 제안 방법

### 해결하고자 하는 문제

기존 강화학습 알고리즘들은 다음과 같은 한계를 가지고 있었습니다:[1]

1. **On-policy 알고리즘 (TRPO, PPO)**: 안정적이지만 샘플 효율성이 낮음
2. **Off-policy 알고리즘 (DDPG, SVG)**: 데이터 효율적이지만 고차원 문제에서 하이퍼파라미터 튜닝이 어렵고 불안정함
3. **연속 행동 공간에서의 높은 기울기 분산(gradient variance)**

### 제안하는 방법: MPO 알고리즘

MPO는 강화학습을 추론 문제로 변환하는 접근법을 사용합니다. 핵심 아이디어는 "어떤 행동이 미래 보상을 최대화하는가?"라는 질문을 "미래 보상 최대화에 성공했다고 가정할 때, 어떤 행동을 선택했을 가능성이 가장 높은가?"로 바꾸는 것입니다.[1]

#### 수식적 정의

**KL-정규화된 목적 함수:**

$$
J(q, \theta) = \mathbb{E}_q\left[\sum_{t=0}^{\infty} \gamma^t\left(r_t - \alpha \text{KL}(q(a|s_t)\|\pi(a|s_t, \theta))\right)\right] + \log p(\theta)
$$

여기서:
- $$q(a|s)$$: 변분 분포(variational distribution)
- $$\pi(a|s, \theta)$$: 파라미터 $$\theta$$를 가진 정책
- $$\alpha$$: 온도 파라미터
- $$p(\theta)$$: 정책 파라미터에 대한 사전 분포[1]

**정규화된 Q-함수:**

$$
Q_\theta^q(s, a) = r_0 + \mathbb{E}_{q(\tau), s_0=s, a_0=a}\left[\sum_{t\geq1} \gamma^t(r_t - \alpha \text{KL}(q_t\|\pi_t))\right]
$$

#### E-Step: 정책 평가 및 개선

제약 조건이 있는 E-step은 다음과 같이 정의됩니다:[1]

$$
\max_q \mathbb{E}_{\mu(s)}\left[\mathbb{E}_{q(a|s)}[Q_{\theta_i}(s, a)]\right] \quad \text{s.t.} \quad \mathbb{E}_{\mu(s)}[\text{KL}(q(a|s), \pi(a|s, \theta_i))] < \epsilon
$$

**비모수적 변분 분포의 최적 해:**

$$
q_i(a|s) \propto \pi(a|s, \theta_i) \exp\left(\frac{Q_{\theta_i}(s, a)}{\eta^*}\right)
$$

여기서 $$\eta^*$$는 다음 볼록 쌍대 함수(convex dual function)를 최소화하여 얻습니다:[1]

$$
g(\eta) = \eta\epsilon + \eta\int \mu(s) \log\left(\int \pi(a|s, \theta_i) \exp\left(\frac{Q_{\theta_i}(s, a)}{\eta}\right) da\right) ds
$$

#### M-Step: 정책 업데이트

M-step에서는 E-step에서 얻은 변분 분포 $$q_i$$를 사용하여 파라미터 $$\theta$$를 업데이트합니다:[1]

$$
\max_\theta \mathbb{E}_{\mu_q(s)}\left[\mathbb{E}_{q(a|s)}[\log \pi(a|s, \theta)]\right] \quad \text{s.t.} \quad \mathbb{E}_{\mu_q(s)}[\text{KL}(\pi(a|s, \theta_i), \pi(a|s, \theta))] < \epsilon
$$

이는 가중치가 적용된 maximum a-posteriori (MAP) 추정 문제로, E-step의 변분 분포가 샘플에 가중치를 부여합니다.[1]

### 모델 구조

MPO는 다음과 같은 구성 요소를 가집니다:[1]

1. **정책 네트워크** $$\pi(a|s, \theta)$$: 연속 제어를 위해 가우시안 분포 사용 (평균과 공분산을 신경망으로 파라미터화)
2. **Q-함수 네트워크** $$Q_\theta(s, a, \phi)$$: Retrace 알고리즘을 사용한 안정적인 정책 평가
3. **타겟 Q-네트워크**: M-step 후 복사되는 파라미터 $$\phi'$$

**Retrace를 사용한 Q-함수 학습:**

$$
\min_\phi \mathbb{E}_{\mu_b(s), b(a|s)}\left[(Q_{\theta_i}(s_t, a_t, \phi) - Q_t^{\text{ret}})^2\right]
$$

여기서:

$$
Q_t^{\text{ret}} = Q_{\phi'}(s_t, a_t) + \sum_{j=t}^{\infty} \gamma^{j-t}\left(\prod_{k=t+1}^{j} c_k\right)\left[r(s_j, a_j) + \mathbb{E}_{\pi(a|s_{j+1})}[Q_{\phi'}(s_{j+1}, a)] - Q_{\phi'}(s_j, a_j)\right]
$$

$$
c_k = \min\left(1, \frac{\pi(a_k|s_k)}{b(a_k|s_k)}\right)
$$

## 3. 성능 향상 및 일반화 능력

### 실험 성능

**DeepMind Control Suite에서의 결과:**
- MPO는 18개의 연속 제어 태스크에서 평가되었으며, **평균 1000개 미만의 궤적(또는 $$10^6$$ 샘플)**으로 최고 성능에 도달했습니다.[1]
- PPO 대비 **약 10배** 빠른 샘플 효율성을 보였으며, DDPG보다도 우수한 데이터 효율성을 달성했습니다.[1]

**고차원 제어 문제:**
- **Walker-2D Parkour 도메인**: PPO가 약 100만 궤적이 필요한 반면, MPO는 약 **7만 궤적(6천만 샘플)**만으로 유사한 성능을 달성했습니다.[1]
- **56 자유도 휴머노이드**: 모든 실험에서 동일한 하이퍼파라미터를 사용했음에도 불구하고 안정적인 학습을 보였습니다.[1]

### 일반화 성능 향상 메커니즘

MPO의 일반화 성능 향상은 다음 요소들에서 비롯됩니다:

**1. KL 제약을 통한 강건한 업데이트:**
- **E-step에서의 역방향 KL(mode-seeking KL)**: $$\text{KL}(q\|\pi)$$를 사용하여 변분 분포가 정책의 높은 확률 영역에 집중하도록 합니다[1].
- **M-step에서의 정방향 KL(moment-matching KL)**: $$\text{KL}(\pi_i\|\pi)$$를 사용하여 파라미터 정책의 엔트로피 붕괴를 방지합니다[1].
- 이 이중 KL 제약 전략은 과적합을 최소화하고 샘플을 넘어선 일반화를 촉진합니다.[1]

**2. 평균과 공분산의 분리된 제약:**
가우시안 정책의 경우, KL을 평균($$C_\mu$$)과 공분산($$C_\Sigma$$)으로 분리하여 다른 학습률을 적용합니다:[1]

$$
\int \mu_q(s) \text{KL}(\pi_i(a|s), \pi(a|s, \theta)) ds = C_\mu + C_\Sigma
$$

공분산에 더 작은 $$\epsilon$$을 설정하여 탐색을 유지하면서 조기 수렴을 방지합니다.[1]

**3. Off-policy 학습과 경험 재생:**
- Retrace 알고리즘을 사용한 안정적인 off-policy 정책 평가로 샘플 재사용이 가능합니다.[1]
- 이는 다양한 상태-행동 분포에서 학습하여 일반화를 개선합니다.

**4. 하이퍼파라미터 강건성:**
- 모든 실험에서 **동일한 하이퍼파라미터** 설정을 사용했으며, 이는 알고리즘이 태스크별 튜닝 없이도 일반화됨을 보여줍니다.[1]
- 엔트로피 정규화를 수동으로 설정하는 방법(EPG + Retrace)과 달리, MPO의 자동 조정된 KL 제약은 환경 간 전이가 더 우수합니다.[1]

### 한계점

논문에서 명시적으로 언급된 한계점은 다음과 같습니다:

1. **Retrace의 안정성**: Retrace가 함수 근사와 함께 사용될 때 안정성이 보장되지 않을 수 있습니다.[1]
2. **이산 제어에서의 성능**: Atari 환경에서는 경쟁력은 있지만 최신 알고리즘만큼 우수하지는 않았습니다. 저자들은 최근 이산 행동을 위한 발전과 결합하면 개선될 수 있다고 제안합니다.[1]
3. **부분 E-step**: E-step이 $$Q_{\theta_i}$$를 상수로 취급하여 $$J$$를 완전히 최적화하지 못합니다. 이는 빠른 수렴을 위한 실용적 선택이지만 이론적 최적성을 희생합니다.[1]

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 연구에 미치는 영향

**1. RL과 추론의 통합:**
MPO는 강화학습을 확률적 추론으로 재구성하는 강력한 사례를 제시하며, EM 기반 방법이 현대 심층 강화학습에서 실용적이고 효과적임을 보여줍니다.[1]

**2. 신뢰 영역 방법에 대한 새로운 관점:**
- TRPO와 PPO를 "명시적 M-step이 없는 파라미터적 E-step"으로 해석할 수 있음을 보였습니다.[1]
- 이는 기존 알고리즘들 간의 이론적 연결을 명확히 합니다.

**3. Off-policy 알고리즘 설계:**
MPO는 off-policy 학습에서 **Q-함수의 기울기 없이도 정책을 업데이트**할 수 있음을 보여주었으며, 대신 여러 행동 샘플을 비교하여 더 나은 행동의 확률을 높입니다.[1]

**4. 하이퍼파라미터 자동 조정:**
KL 제약의 자동 조정(라그랑주 승수를 통한)은 수동 엔트로피 스케일 튜닝의 필요성을 줄여, 더 범용적인 알고리즘 개발의 길을 열었습니다.[1]

### 향후 연구 시 고려 사항

**1. 이산 행동 공간으로의 확장:**
- Atari 실험은 가능성을 보였지만, 이산 제어를 위한 최근 발전(예: distributional RL)과의 통합이 필요합니다.[1]

**2. 모델 기반 접근과의 결합:**
- 현재 MPO는 모델 프리(model-free) 방법입니다. 전이 모델을 통합하면 샘플 효율성을 더욱 향상시킬 수 있습니다.

**3. 계층적 및 다중 태스크 학습:**
- MPO의 프레임워크는 계층적 정책 또는 다중 태스크 설정으로 확장될 가능성이 있습니다.

**4. 이론적 보장 강화:**
- 단조 개선 보증이 비정보적 사전에 대해서만 증명되었습니다. 더 일반적인 설정에 대한 이론적 분석이 필요합니다.[1]

**5. 함수 근사의 안정성:**
- Retrace의 안정성 문제를 해결하거나, 더 강력한 보장을 가진 대안적 정책 평가 방법을 탐구해야 합니다.[1]

**6. 실제 로봇 공학 응용:**
- MPO의 샘플 효율성은 실제 로봇 실험에 유망하지만, 안전 제약과 부분 관측성을 다루는 연구가 필요합니다.

**7. 탐색 전략:**
- 현재 탐색은 정책의 확률적 특성에 의존합니다. 호기심 주도 탐색이나 내재적 동기 부여와 같은 명시적 탐색 메커니즘이 성능을 더욱 향상시킬 수 있습니다.

MPO는 강화학습 알고리즘 설계에 있어 확률적 추론과 신뢰 영역 최적화의 원칙적 결합을 제공하며, 샘플 효율성, 안정성, 일반화 능력에서 중요한 진전을 이루었습니다. 이러한 기여는 향후 더 강건하고 효율적인 강화학습 시스템 개발의 기반이 될 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c2f6383a-07b3-4f01-9951-4d977cc0c0b0/1806.06920v1.pdf)
