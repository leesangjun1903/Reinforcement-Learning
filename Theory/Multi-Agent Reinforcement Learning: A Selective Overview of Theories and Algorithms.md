
# Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms

## 요약 (Executive Summary)

Zhang, Yang, Başar가 2019년 발표한 "Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms"는 MARL 분야의 가장 영향력 있는 총괄 연구로, 2,300회 이상 인용되었습니다. 본 논문은 마코프 게임(Markov games)과 광범위 게임(extensive-form games) 프레임워크를 중심으로 완전 협력, 경쟁, 혼합 설정에서의 이론적으로 보장된 알고리즘들을 체계적으로 검토합니다. 특히 비정상성(non-stationarity), 확장성, 정보 구조의 다양성이라는 근본적 도전 과제들을 명확히 하고, 2020년 이후 연구들은 일반화 성능과 표본 효율성 향상에 집중하고 있습니다.

***

## 1. 핵심 주장 및 기여

### 1.1 주요 주장의 핵심

Zhang et al.의 논문은 다음 세 가지 중심 주장을 전개합니다.

**주장 1**: MARL은 게임 이론(Go, Poker), 로봇공학, 자율주행 등에서 경험적 성공을 거두었으나, 단일 에이전트 RL의 이론적 진보와 달리 이론적 기초가 상대적으로 부족합니다. 저자들은 이론적 보장이 있는 알고리즘들을 체계적으로 정리함으로써 현재까지의 연구 경계를 명확히 하고자 했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

**주장 2**: MARL의 고유한 도전 과제들—비고유한 학습 목표, 비정상 환경, 지수적 확장성 문제, 복잡한 정보 구조—은 단일 에이전트 RL의 도구를 직접 적용할 수 없게 만듭니다. 따라서 기존 이론을 확장하기 위해서는 게임 이론, 분산 최적화, 동적 계획 등 다중 분야의 기법이 필요합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

**주장 3**: 광범위 게임, 네트워크화 에이전트를 통한 분산 MARL, 평균장(mean-field) 체제는 기존 MARL 리뷰에서 충분히 다루어지지 않은 중요한 새로운 각도입니다. 이들은 확장성과 실제 응용성을 획기적으로 개선할 수 있는 잠재력을 가지고 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

### 1.2 주요 기여의 구체화

| 기여 영역 | 내용 | 학문적 가치 |
|---------|------|----------|
| **이론적 체계화** | 마코프 게임과 광범위 게임 프레임워크 통합 | MARL 이론의 일관된 분석 틀 제공 |
| **알고리즘 분류** | 협력/경쟁/혼합 설정별 가치/정책 기반 방법 정리 | 알고리즘 선택의 이론적 기초 제시 |
| **미개척 영역 개척** | 평균장 MARL, Dec-POMDP, 정책 기반 방법 수렴성 | 향후 10년 연구 방향 지시 |
| **도전 과제 명확화** | 비정상성, 확장성, 정보 구조의 층위별 분석 | 향후 연구의 우선순위 결정 근거 제공 |

***

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 문제 정의: MARL의 근본적 도전

#### 도전 과제 1: 비고유한 학습 목표(Non-Unique Learning Goals)

**문제**: 단일 에이전트 RL에서는 보상 최대화가 명확한 목표이지만, MARL에서는 에이전트의 목표가 상충할 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

- **Nash Equilibrium (NE) 수렴의 타당성**: 게임 이론에서 NE는 합리적 에이전트의 가정 하에 타당하지만, 경계적 합리성(bounded rationality)을 갖는 실제 에이전트에서는 부당할 수 있습니다.
  
- **다중 기준**: NE 수렴 외에도 통신 효율성, 견고성, 일반화 성능 등이 중요합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

**논문의 해결책**: 저자들은 다양한 학습 목표를 분류합니다:
- **평형 의제(Equilibrium Agenda)**: NE로의 수렴
- **AI 의제(AI Agenda)**: 주어진 상대방 클래스에 대해 최적 성능
- **No-Regret 목표**: 사후 후회(regret) 최소화를 통한 이론적 보장 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

#### 도전 과제 2: 비정상성(Non-Stationarity)

**문제**: 에이전트들이 동시에 정책을 갱신하므로, 각 에이전트가 마주하는 환경의 전이 확률 $P(s'|s, a^i)$이 시간에 따라 변합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

$$P^i(s'|s, a^i) = \sum_{a^{-i}} P(s'|s, a^i, a^{-i}) \pi^{-i}(a^{-i}|s)$$

여기서 $\pi^{-i}$가 변하면서 environment는 non-stationary가 됩니다.

**영향**: 
- 단일 에이전트 Q-학습의 수렴 조건 위반
- 독립 학습자(Independent Learners)의 발산 위험 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

**해결 방안** (2020년 이후 연구):
- **합의 기반 업데이트**: 이웃 에이전트의 Q-함수와의 차이를 보상으로 포함
- **두 시간 규모(Two-Timescale) 알고리즘**: 빠른 정책 갱신과 느린 정책 평가
- **그래프 신경망**: permutation equivariance를 통한 비정상성 완화 [jurnal.alwashliyahkalsel](https://jurnal.alwashliyahkalsel.org/index.php/jsh/article/view/75)

#### 도전 과제 3: 확장성 문제(Scalability Issues)

**문제**: 결합 행동 공간의 크기가 에이전트 수에 지수적으로 증가합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

$$|A| = \prod_{i=1}^N |A_i|$$

예를 들어 10개 에이전트가 각각 5개 행동을 가지면, 결합 행동 공간은 $5^{10} \approx 10^7$입니다.

**함수 근사의 어려움**: 신경망을 사용한 함수 근사는 경험적으로 성공하지만 이론적 보장이 없습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

**제안된 해결책**:

1. **지역 상호작용 구조 활용**: 에이전트 간 영향이 거리에 따라 지수적으로 감쇠하면, Q-함수를 국소적으로 근사 가능합니다.

2. **평균장 근사(Mean-Field Approximation)**: 거대한 에이전트 인구에서 개별 에이전트는 대표 에이전트의 평균 행동에만 응답하면 됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

$$\mu_t = \frac{1}{N} \sum_{j=1}^N \delta_{s^j_t, a^j_t}$$

3. **가치 함수 분해(Value Function Factorization)**: 전역 Q-함수를 국소 Q-함수들의 합으로 표현 (QMIX, VDN 등)

#### 도전 과제 4: 정보 구조의 복잡성(Information Structures)

**세 가지 정보 구조** (그림 2 참조):

1. **중앙화 학습(Centralized Learning)**: 중앙 제어기가 모든 에이전트의 정보 수집 → CTDE (Centralized Training, Decentralized Execution)

2. **분산화된 네트워크**: 에이전트들이 통신 네트워크를 통해 이웃과만 정보 교환 → 확장성과 견고성 제공

3. **완전 분산화(Fully Decentralized)**: 에이전트 간 명시적 통신 없이 지역 관측으로 결정 → 현실성 높음

**영향**: 정보 구조에 따라 수렴 보증, 표본 복잡도, 통신 비용이 크게 달라집니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

### 2.2 제안 방법: 알고리즘과 수식

#### 협력 설정의 핵심 알고리즘

**문제 설정**: 모든 에이전트가 공통 보상 $R(s, a, s')$을 최대화합니다.

**QD-학습 (합의 기반 Q-학습)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

$$Q^i_{t+1}(s,a) \leftarrow Q^i_t(s,a) + \alpha_{t,s,a}\left[R(s,a) + \gamma \max_{a' \in A} Q^i_t(s',a') - Q^i_t(s,a)\right] - \beta_{t,s,a}\sum_{j \in N^i_t}(Q^i_t(s,a) - Q^j_t(s,a))$$

여기서:
- 첫 항: 표준 Q-러닝 업데이트
- 두 번째 항: 이웃 에이전트들과의 합의(consensus) 조정

**수렴 조건**: $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$, 모든 상태-행동 쌍이 무한 횟수 방문

**분산 Actor-Critic**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

각 에이전트는 정책 매개변수 $\theta_i$를 유지하며, 정책 기울기는:

$$\nabla_{\theta_i} J(\theta) = E[\nabla_{\theta_i} \log \pi^i_{\theta_i}(s, a_i) \cdot Q_\theta(s,a)]$$

여기서 $Q_\theta(s,a)$는 전역 Q-함수로, 각 에이전트는 로컬 복사본 $Q_\theta(\cdot, \cdot; \omega^i)$을 유지합니다.

**비평가 단계 (Critic Step)—합의 기반 TD 학습**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

$$\tilde{\omega}^i_t = \omega^i_t + \beta_{\omega,t} \cdot \delta^i_t \cdot \nabla_{\omega_i} Q_t(\omega^i_t)$$

$$\omega^i_{t+1} = \sum_{j \in N^i_t} c_t(i,j) \cdot \tilde{\omega}^j_t$$

여기서:
- $\delta^i_t = R^i(s, a) + \gamma \max_{a'} Q_t(s', a'; \omega^i_t) - Q_t(s, a; \omega^i_t)$: 로컬 TD 오차
- $c_t(i,j)$: 통신 토폴로지를 반영한 믹싱 계수 (doubly stochastic 조건)
- 행동자 단계는 $\nabla_{\theta_i} J(\theta)$를 따름

**수렴 보증**: 선형 함수 근사 하에 거의 확실한(almost sure) 수렴 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

#### 경쟁 설정 (2-Player Zero-Sum)

**문제 설정**: $R_1(s, a, s') + R_2(s, a, s') = 0$ (이익의 합이 0)

**최적 가치함수의 정의**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

$$V^* = \max_{\pi_1} \min_{\pi_2} V^1_{\pi_1, \pi_2} = \min_{\pi_2} \max_{\pi_1} V^1_{\pi_1, \pi_2}$$

이 등식이 성립하는 이유는 minimax 정리 때문입니다.

**Minimax-Q 학습**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

$$Q(s_t, a^1_t, a^2_t) \leftarrow (1-\alpha_t)Q(s_t, a^1_t, a^2_t) + \alpha_t\left[r_t + \gamma \cdot \text{Value}(Q(s'_t, \cdot, \cdot))\right]$$

여기서 $\text{Value}(Q(s, \cdot, \cdot))$는 다음 행렬 게임의 가치입니다:

$$\text{Value}(Q(s, \cdot, \cdot)) = \max_{p \in \Delta(A_1)} \min_{q \in \Delta(A_2)} \sum_{a_1, a_2} p(a_1) q(a_2) Q(s, a_1, a_2)$$

이 값은 선형 프로그래밍으로 계산 가능합니다.

**수렴**: 유한 상태-행동 공간에서 $Q^*$로 거의 확실한 수렴 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

#### 광범위 게임(Extensive-Form Games)

**문제 설정**: 게임은 게임 트리로 표현되며, 에이전트는 불완전한 정보(imperfect information)를 갖습니다.

**정의**: 광범위 게임 $G = (N \cup \{c\}, H, Z, A, \{R_i\}_{i \in N}, \tau, \pi_c, S)$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
- $H$: 모든 가능한 역사(sequences of actions)
- $Z \subset H$: 종료 역사
- $\tau: H \rightarrow N \cup \{c\}$: 누가 행동할 차례인지 지정
- $S$: 정보 집합(information sets)—에이전트가 구분 불가능한 역사들의 묶음

**도달 확률(Reach Probability)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

$$\eta_\pi(h) = \sum_{h'a \sqsubseteq h} \pi_{\tau(h')}(a|I(h')), \quad \text{모든 에이전트의 기여도 곱}$$

**반사실적 후회(Counterfactual Regret)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

$$\text{Regret}^i(s) = \sum_{t=1}^T (Q^i_{CF}(\pi_t, s, a^*) - V^i_{CF}(\pi_t, s))$$

여기서 $a^*$는 사후에 최적 행동입니다.

**CFR (Counterfactual Regret Minimization) 알고리즘**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
- 각 정보 집합에서 온라인 학습 알고리즘 (Hedge, EXP3, Regret Matching) 적용
- 후회를 $O(\sqrt{T})$로 제한
- 자기 대전(self-play)을 통해 $O(1/\sqrt{T})$-approximate NE 달성

**응용**: 포커와 같은 불완전 정보 게임에서 AI Sapiens, DeepStack, Libratus 같은 강력한 AI 생성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

***

## 3. 모델 구조와 알고리즘 프레임워크

### 3.1 프레임워크의 위계 구조

```
MARL
├─ 협력 설정 (§4.1)
│  ├─ 동질 에이전트
│  │  ├─ Multi-Agent MDP & Markov Teams
│  │  ├─ Markov Potential Games
│  │  └─ Mean-Field Regime
│  ├─ 이질 에이전트 + 네트워크
│  │  ├─ Decentralized Learning with Networked Agents
│  │  │  ├─ 최적 정책 학습 (QD-learning, Actor-Critic)
│  │  │  ├─ 정책 평가 (Policy Evaluation)
│  │  │  └─ 통신 효율성
│  │  └─ 부분 관측 모델 (Dec-POMDP)
│  └─ 최적 제어 관점
├─ 경쟁 설정 (§4.2)
│  ├─ 2-Player Zero-Sum Markov Games
│  │  ├─ 가치 기반 (Minimax-Q, Value Iteration)
│  │  └─ 정책 기반 (Fictitious Play, CFR)
│  └─ Extensive-Form Games (불완전 정보)
│     ├─ Perfect Information (AlphaZero 류)
│     └─ Imperfect Information (Poker AI)
└─ 혼합 설정 (§4.3)
   ├─ 2-Player General-Sum
   │  ├─ Nash-Q Learning
   │  └─ Friend-or-Foe Q-Learning
   ├─ Multi-Player General-Sum
   │  ├─ Weakly Acyclic Games
   │  └─ Potential Games
   └─ Mean-Field Games (일반합)
```

### 3.2 주요 수렴 결과 요약

| 알고리즘 | 설정 | 수렴 타입 | 복잡도 |
|---------|------|----------|--------|
| **Q-Learning** | Cooperative, finite spaces | Almost Sure | $O(\log(1/\epsilon))$ |
| **QD-Learning** | Cooperative, networked | Almost Sure | Network size에 따라 달라짐 |
| **Minimax-Q** | 2P Zero-Sum | Almost Sure | $O(\log(1/\epsilon))$ |
| **Fictitious Play** | Zero-Sum, normal-form | Weak convergence | Continuous time에서만 |
| **CFR** | Extensive-form | Regret bound | $O(R^i_{max} \|S^i\| \sqrt{A^i \cdot T})$ |
| **Policy Gradient** | Potential games | Local | Gradient dominance 필요 |

***

## 4. 성능 향상 및 수렴 분석

### 4.1 협력 설정에서의 수렴 성능

**정리 (Q-Learning 수렴)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
유한 상태-행동 공간, 모든 상태-행동 쌍의 무한 방문, 그리고 $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$가 성립하면, Q-함수는 최적 Q-함수 $Q^*$로 거의 확실하게 수렴합니다:

$$Q_t(s, a) \rightarrow Q^*(s, a) \text{ a.s., } \forall s \in S, a \in A$$

**분산 Actor-Critic 수렴** (선형 함수 근사): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
정책 매개변수 공간에서의 거의 확실한 수렴:

$$\theta_t \rightarrow \theta^* \text{ a.s.}$$

여기서 $\theta^*$는 정책 기울기 방정식의 해입니다.

**수렴 속도**: 대부분의 알고리즘이 점근적(asymptotic) 수렴만 보장하며, 유한-시간 수렴 속도는 상대적으로 미개발 분야입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

### 4.2 경쟁 설정에서의 수렴 성능

**정리 (Minimax-Q 선형 수렴)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
가치 반복(value iteration)은 선형 수렴을 달성합니다:

$`\|V_{t+1} - V^*\|_\infty \leq \gamma \|V_t - V^*\|_\infty \leq \gamma^{t+1} \|V_0 - V^*\|_\infty`$

따라서 $\epsilon$-근사까지 도달하는 데 필요한 반복:

$$t = O(\log(1/\epsilon))$$

**CFR의 후회 경계**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
$T$ 단계 후 에이전트 $i$의 총 후회:

$$\text{Regret}^i_T \leq R^i_{\max} \cdot |S^i| \cdot \sqrt{A^i \cdot T}$$

여기서:
- $R^i_{\max}$: 최대 보상과 최소 보상의 차이
- $|S^i|$: 에이전트 $i$의 정보 집합 수
- $A^i$: 에이전트 $i$의 최대 행동 수

**시사**: 평균 정책은 $\frac{\text{Regret}^i_T}{T} = O(1/\sqrt{T})$-approximate NE가 됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

### 4.3 표본 복잡도와 통신 복잡도

**배치 RL (Fitted-Q Iteration, FQI)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
$n$ 크기의 배치 데이터로부터:

$$Q_{t+1} = \arg\min_{f \in F} \frac{1}{2n} \sum_{j=1}^n (y_j - f(s_j, a_j))^2$$

유한-표본 오차 경계:

$`\|Q_{t+1} - Q^*\| \leq \gamma \|Q_t - Q^*\| + \text{Approximation Error} + \text{Estimation Error}`$

**분산 정책 평가**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
팀 평균 보상에 대한 MSPBE 최소화:

$$\min_\omega \|\Pi_\Phi(V_\omega - \gamma P^\pi V_\omega - \bar{R}^\pi)\|^2_D$$

여기서 $\Pi_\Phi$는 선형 특성 공간으로의 투영입니다.

***

## 5. 일반화 성능과 향상 방법

### 5.1 논문에서 제시된 일반화의 한계

**신경망 함수 근사의 문제**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
- 깊은 MARL의 이론적 기초 부재
- 단일 에이전트 깊은 RL도 수렴성 미증명
- 과적합 위험, 분포 이동(distribution shift)

**부분 관측 설정의 복잡성**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
- Dec-POMDP는 일반 케이스에서 NEXP-완전
- 지수 시간 필요

**정보 구조 간의 성능 격차**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
- 중앙화 학습은 이론적으로 단순하지만 현실성 낮음
- 완전 분산화는 현실적이지만 이론적으로 어려움

### 5.2 2020년 이후의 일반화 성능 향상 연구

#### 5.2.1 등변성 그래프 신경망 (EGNN) 기반 접근 [jurnal.alwashliyahkalsel](https://jurnal.alwashliyahkalsel.org/index.php/jsh/article/view/75)

**논문**: McClellan et al., "Boosting Sample Efficiency and Generalization in Multi-agent Reinforcement Learning via Equivariance" (2024)

**핵심 아이디어**: 기하학적 대칭성(회전, 평행 이동, 반사)을 신경망 구조에 내재화하면, 정책 공간을 축소하여 표본 효율성과 일반화를 향상시킬 수 있습니다. [jurnal.alwashliyahkalsel](https://jurnal.alwashliyahkalsel.org/index.php/jsh/article/view/75)

**제안 방법 (E2GN2)**:
- **Exploration-Enhanced Equivariant Graph Neural Networks**
- EGNN의 초기 탐색 편향 문제를 수정
- 회전/반사 등변성 유지

**성능 향상**:
$$\text{표본 효율성}: 2\times - 5\times \text{ 향상}$$

MPE 및 SMACv2 벤치마크에서 기존 MLP 및 GNN 대비 현저한 개선 [jurnal.alwashliyahkalsel](https://jurnal.alwashliyahkalsel.org/index.php/jsh/article/view/75)

#### 5.2.2 마스크된 오토인코더 (Masked Autoencoders for MARL, MA2RL) [ijsrcseit](https://ijsrcseit.com/index.php/home/article/view/CSEIT25112448)

**핵심**: 동적으로 마스크된 엔티티를 처리하여 영점(zero-shot) 일반화 달성

- 작업 독립적(task-independent) 스킬 학습
- 스킬 의미 해석 가능

#### 5.2.3 역할 기반 정책 혼합 (Role-based Policy Mixing, RPM) [open-publishing](https://open-publishing.org/publications/index.php/APUB/article/view/2769)

**성능**: 메멜팅 팟(Melting Pot) 환경에서 **818% 성능 향상**

- 에이전트 역할의 다양성 인식
- 사회적 행동의 일반화

#### 5.2.4 하이퍼네트워크를 통한 적응형 학습 (HyperMARL) [invergejournals](https://invergejournals.com/index.php/ijss/article/view/117)

**특징**:
- 에이전트 간 그래디언트 간섭(gradient interference) 감소
- 매개변수 공유의 표본 효율성 유지
- 행동 다양성 촉진

$$W_i = f_{\text{hyper}}(x_i), \quad x_i \text{: 에이전트 특성}$$

하이퍼네트워크가 개별 에이전트의 가중치를 생성하여, 기저 정책과 개별화 정책의 균형 달성.

#### 5.2.5 부분 등변성 그래프 신경망 (PEnGUiN) [invergejournals](https://invergejournals.com/index.php/ijss/article/view/136)

**문제 인식**: 실제 환경은 완전한 대칭성을 갖지 않음(중력, 측정 오차, 외부력 등)

**해결책**: 부분 등변성(partial equivariance)을 허용하면서 표본 효율성과 견고성 동시 달성

***

## 6. 한계 및 미해결 문제

### 6.1 Zhang et al. (2019)이 인식한 한계

#### 이론-실무 간극

| 차원 | 논문의 한계 | 현황 |
|-----|----------|------|
| **신경망 근사** | 수렴성 미증명 | 2024년도 여전히 미해결 |
| **부분 관측성** | NEXP-완전 | 특정 구조 가정 필요 |
| **평균장 정도** | 무한 인구 가정 | 유한 인구로 확장 필요 |
| **정보 구조** | 중앙/완전 분산 이분법 | 계층적 혼합 구조 탐색 중 |

#### 깊은 MARL의 이론화

저자들은 명시적으로 깊은 MARL의 이론적 기초가 극히 부족함을 지적합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
- 신경망의 표현 능력과 일반화 경계 미해명
- 정책-가치 함수 근사 간의 상호작용 복잡성

### 6.2 2024-2025년 현재의 미개척 영역

#### 문제 1: 이질성과 부분 관측의 통합

**현상**: 
- 동질 에이전트, 완전 관측 가정은 대부분의 MARL 이론에 내재
- 실제 로봇 시스템, 네트워크 문제는 이질적 에이전트와 부분 관측

**미해결**:
- 이질 에이전트의 통신 메커니즘과의 상호작용
- 부분 관측 하에서의 정책 기울기 추정의 편향

#### 문제 2: 통신과 표본 효율성의 트레이드오프

**최근 진전** (2025):
- Consensus-based Actor-Critic: 완전 분산화 수렴 보증
- Base Policy Prediction: 통신 라운드 $O(\epsilon^{-3/4})$로 감소 [invergejournals](https://invergejournals.com/index.php/ijss/article/view/132)

**미해결**:
- 경계적 하한(lower bound) 증명
- 실제 통신 지연(latency)의 영향

#### 문제 3: 강건성 (Robustness)

**도전**:
- 적대적 공격에 대한 MARL 안정성
- 에이전트 장애 처리
- 통신 손실 환경

**현황**: 대부분 경험적 연구만 존재

#### 문제 4: 인간-AI 협력 MARL

**최신 동향** (2025):
- **MAGRPO**: LLM을 다중 에이전트로 협력 학습 [academic.oup](https://academic.oup.com/ijnp/article/28/Supplement_1/i276/8009422)
- **MAPoRL**: LLM 협력 워크플로우의 post-training [invergejournals](https://invergejournals.com/index.php/ijss/article/view/86)

**미해결**:
- 인간 피드백의 이론적 통합
- 해석 가능성(interpretability) 보장

***

## 7. 논문이 미치는 학문적 및 실무적 영향

### 7.1 학문적 영향

#### 이론적 틀의 정립

Zhang et al.의 논문은 MARL 연구의 아젠다(agenda)를 재설정했습니다:
- **이전**: 개별 알고리즘의 경험적 성능 비교 중심
- **이후**: 수렴 보증, 복잡도 분석, 정보 구조와의 관계 연구

#### 분야의 분화

논문 이후 MARL 연구는 다음 하위 분야로 구조화되었습니다:
1. **깊은 MARL**: 신경망 기반 실제 응용
2. **이론적 MARL**: 수렴성 및 복잡도 분석
3. **응용 분야별 MARL**: 로봇, 게임, 통신, 전력망 등
4. **통신 제약 MARL**: 실제 시스템의 제약 반영

### 7.2 산업 응용 활성화

#### UAV 군집 제어

**배경**: Zhang et al.이 제시한 "무인항공기(UAV)" 응용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)
- 각 UAV는 시간 변동 통신 토폴로지 내에서 분산 학습
- 중앙 제어기 부재

**진전**: 2022-2024년 다중 로봇 협력 연구 급증
- 그래프 신경망 기반 정책 전이(transferability)
- 동적 네트워크에서의 적응

#### 전력망 운영

**응용**: 분산 전력망(스마트 그리드)의 최적 제어
- 각 그리드 노드 = 에이전트
- 팀 평균 보상 최적화

**2024년 성과**:
- 확장 가능한 MARL로 100+ 노드 제어
- 기존 최적화 대비 비용 절감

#### 게임 AI

**이미 성숙한 응용**:
- **AlphaZero** (2017): 광범위 게임 구조 활용
- **Libratus** (2017): CFR 기반 포커 AI
- **DeepStack** (2017): 신경망 + CFR

### 7.3 후속 연구 트렌드의 형성

**2020-2022: 깊은 MARL 확대**
- MAPPO, QMIX 등 실무 알고리즘 정착
- 벤치마크 표준화 (SMAC, MPE)

**2023-2024: 표본 효율성과 일반화**
- 기하학적 대칭성 활용 (EGNN, E2GN2)
- 메타-강화학습, 전이 학습 결합

**2025+: 신경-게임 이론 통합**
- LLM 기반 협력 에이전트
- 인간-AI 협력 MARL
- 견고성과 신뢰성 강화

***

## 8. 향후 연구 시 고려할 점

### 8.1 이론 개발의 우선순위

#### 우선순위 1: 비선형 함수 근사 수렴성

**현황**: 신경망 사용은 표준이지만 이론적 보장 전무 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

**필요한 분석**:
- Overparameterized 신경망에서의 수렴
- 정책-가치 함수 근사의 상호작용
- 비정상 환경에서의 일반화 경계

**제안된 접근**:
$$\text{수렴 속도} = O(1/\sqrt{t}) + \text{근사 오차} + \text{분포 이동 오차}$$

#### 우선순위 2: 부분 관측 게임의 이론

**현상**: Dec-POMDP는 NEXP-완전이지만, 제한된 구조 가정 하에서는 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

**연구 방향**:
- 관측 공간의 차원 특성화
- 통신을 통한 공통 정보 활용
- 계층적 POMDP 분해

#### 우선순위 3: 통신-표본 트레이드오프의 정량화

**문제**: 통신이 비싸면, 몇 번 통신할 때 어느 정도 수렴 성능을 보장할 수 있나? [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

**현재 최선의 결과** (2025):
- 협력 포텐셜 게임: $O(\epsilon^{-3/4})$ 통신 라운드 [invergejournals](https://invergejournals.com/index.php/ijss/article/view/132)
- 일반 협력: 미개척

### 8.2 알고리즘 개발 전략

#### 전략 1: 구조를 활용한 확장성

**기존 한계**: 결합 행동 공간의 지수성 극복 미흡 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1054991c-ee41-4b69-8c53-90f5bac1ad67/1911.10635v2.pdf)

**개선 방향**:
1. **값 함수 분해(VFD)**의 이론적 정당화
   $$Q(s, a) = \sum_i Q_i(s, a_i) + b(s)$$
   - QMIX, QPLEX 등의 수렴성 증명

2. **그래프 구조의 활용**
   $$\text{State} = (s_1, \ldots, s_N), \quad \text{Message}: i \rightarrow j \text{ if } (i,j) \in E$$
   - 메시지 패싱 신경망 설계
   - 동적 토폴로지 대응

3. **계층적 분해**
   $$\text{Global Goal} = \sum_{\text{subteams}} \text{Local Goal}$$

#### 전략 2: 견고성 강화

**문제**:
- 에이전트 장애 시 성능 저하
- 적대적 공격에 취약

**해결책**:
1. **Minimax Optimization**을 협력 문제에 적용
   $$\min_\pi \max_{\text{adversary}} J(\pi)$$

2. **Distributionally Robust MARL**
   $$\min_\pi \max_{P \in \mathcal{P}} E_P[R]$$

#### 전략 3: 실제 제약 반영

**요소**:
- 통신 지연(latency)
- 대역폭 제약
- 이질적 연산 능력
- 부분 신뢰성(partial reliability)

### 8.3 벤치마크 및 평가 표준화

#### 현재 표준 벤치마크의 한계

| 벤치마크 | 장점 | 한계 |
|---------|------|------|
| **SMAC** (StarCraft) | 대규모 환경, 부분 관측 | 시뮬레이터만 가능 |
| **MPE** (Multi-agent Particle) | 연속 제어, 단순함 | 작은 규모 (≤50 agents) |
| **Google Research Football** | 현실성 높음 | 계산 비용 높음 |

#### 필요한 새 벤치마크

1. **혼합 정보 구조**: 일부 중앙화, 일부 분산
2. **이질 에이전트**: 다양한 역량
3. **시간 변동 환경**: 에이전트 추가/제거, 통신 변화
4. **실제 문제**: 로봇 실험 플랫폼, 시뮬레이터-현실 간극

### 8.4 학제 간 통합

#### 게임 이론과의 깊은 통합

- **Stackelberg Equilibrium** (선도자-추종자 게임)
- **Correlated Equilibrium** (복합 평형)
- **Coarse Correlated Equilibrium** (비협력 도전 환경)

#### 최적 제어와의 연결

- **Mean-field Control**의 RL 확장
- **Stochastic Control**의 다중 에이전트 버전

#### 기계학습 구조

- **Representation Learning**: 특성 학습의 동시 수행
- **Meta-Learning**: 에이전트가 학습 알고리즘 자체를 학습
- **Transfer Learning**: 과제 간 지식 전이

***

## 결론

Zhang et al. (2019)의 "Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms"는 MARL 분야의 이정표 역할을 합니다. 저자들이 제시한 문제 정의—비고유한 학습 목표, 비정상성, 확장성, 정보 구조의 복잡성—은 6년이 지난 2025년 현재도 여전히 MARL 연구의 핵심 도전 과제입니다. 

**핵심 발견**:

1. **이론-실무 간극의 지속**: 신경망 기반 깊은 MARL의 수렴성 보증은 여전히 미해결.

2. **일반화 성능의 진전**: 기하학적 대칭성(EGNN, E2GN2)과 적응형 구조(HyperMARL) 등을 통해 2-5배 표본 효율성 향상 달성.

3. **통신과 확장성의 개선**: 2025년 consensus-based 방법과 base policy prediction으로 완전 분산화와 통신 효율을 동시에 달성하는 방향으로 진화 중.

4. **신흥 응용의 확산**: LLM 기반 협력 에이전트(MAGRPO, MAPoRL) 등 인간-AI 협력 MARL 등장.

향후 MARL 연구는 **이론적 엄밀성**, **실무적 확장성**, **견고성과 신뢰성**, **인간 협력** 네 축을 중심으로 진화할 것으로 예상됩니다. 특히 부분 관측과 이질 에이전트를 동시에 처리하는 이론적 틀의 개발이 가장 시급한 과제입니다.

***

## 참고 문헌

<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84]</span>

<div align="center">⁂</div>

[^1_1]: 1911.10635v2.pdf

[^1_2]: https://jurnal.alwashliyahkalsel.org/index.php/jsh/article/view/75

[^1_3]: https://ijsrcseit.com/index.php/home/article/view/CSEIT25112448

[^1_4]: https://open-publishing.org/publications/index.php/APUB/article/view/2769

[^1_5]: https://invergejournals.com/index.php/ijss/article/view/117

[^1_6]: https://invergejournals.com/index.php/ijss/article/view/136

[^1_7]: https://invergejournals.com/index.php/ijss/article/view/132

[^1_8]: https://academic.oup.com/ijnp/article/28/Supplement_1/i276/8009422

[^1_9]: https://invergejournals.com/index.php/ijss/article/view/86

[^1_10]: http://arxiv.org/pdf/2311.00865.pdf

[^1_11]: http://arxiv.org/pdf/2502.18439.pdf

[^1_12]: http://arxiv.org/pdf/2203.02896.pdf

[^1_13]: https://arxiv.org/html/2412.21088v1

[^1_14]: https://arxiv.org/pdf/2205.07229.pdf

[^1_15]: https://arxiv.org/pdf/2305.10091.pdf

[^1_16]: http://arxiv.org/pdf/2303.01768.pdf

[^1_17]: https://arxiv.org/pdf/2405.11106.pdf

[^1_18]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12190516/

[^1_19]: https://www.themoonlight.io/ko/review/boosting-sample-efficiency-and-generalization-in-multi-agent-reinforcement-learning-via-equivariance

[^1_20]: http://proceedings.mlr.press/v139/zimmer21a/zimmer21a-supp.pdf

[^1_21]: https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1613.pdf

[^1_22]: https://personal.ntu.edu.sg/boan/papers/ICLR23_RPM.pdf

[^1_23]: https://www.jmlr.org/papers/volume9/bab08a/bab08a.pdf

[^1_24]: https://www.sciencedirect.com/science/article/abs/pii/S0925231225013888

[^1_25]: https://anuj-mahajan.github.io/files/genmas.pdf

[^1_26]: https://www.reddit.com/r/reinforcementlearning/comments/gvabec/proofs_of_learning_convergence_of_multiagent/

[^1_27]: https://www.sciencedirect.com/science/article/abs/pii/S0957417425011480

[^1_28]: https://www.nature.com/articles/s41598-024-82562-w

[^1_29]: https://arxiv.org/abs/1911.10635

[^1_30]: https://arxiv.org/abs/2312.10256

[^1_31]: https://www.sciencedirect.com/science/article/abs/pii/S1568494623000030

[^1_32]: https://openreview.net/forum?id=0Iw2dLh8uq\&noteId=BeUHImEdtB

[^1_33]: https://arxiv.org/html/2508.04652v2

[^1_34]: https://arxiv.org/html/2411.11099v1

[^1_35]: https://arxiv.org/pdf/2508.07001.pdf

[^1_36]: https://arxiv.org/html/2512.24325v1

[^1_37]: https://arxiv.org/html/2410.02581v1

[^1_38]: https://arxiv.org/pdf/2507.06278.pdf

[^1_39]: https://arxiv.org/html/2511.11992v1

[^1_40]: https://arxiv.org/abs/2505.03949

[^1_41]: https://arxiv.org/pdf/1401.3454.pdf

[^1_42]: https://arxiv.org/html/2512.22876v1

[^1_43]: https://arxiv.org/abs/2410.02581

[^1_44]: https://arxiv.org/html/2512.03528v1

[^1_45]: https://arxiv.org/html/2512.08877v1

[^1_46]: https://arxiv.org/html/2011.00583v4

[^1_47]: https://furong-huang.com/publications/boosting-sample-efficiency-and-generalization-in-multi-agent-reinforcement-learning-via-equivariance/

[^1_48]: https://arxiv.org/abs/2411.01663

[^1_49]: https://www.cureus.com/articles/310907-leveraging-artificial-neural-networks-and-support-vector-machines-for-accurate-classification-of-breast-tumors-in-ultrasound-images

[^1_50]: https://dl.acm.org/doi/10.1145/3705374.3705384

[^1_51]: https://dl.acm.org/doi/10.1145/3687273.3687295

[^1_52]: https://ejournal.seaninstitute.or.id/index.php/InfoSains/article/view/4713

[^1_53]: https://www.mdpi.com/1648-9144/59/12/2138

[^1_54]: https://dl.acm.org/doi/10.1145/3638530.3664081

[^1_55]: http://dergipark.org.tr/en/doi/10.17218/hititsbd.1327799

[^1_56]: https://ieeexplore.ieee.org/document/10623626/

[^1_57]: http://photonics.pl/PLP/index.php/letters/article/view/16-17

[^1_58]: https://arxiv.org/html/2410.15876v2

[^1_59]: https://arxiv.org/html/2502.17046v1

[^1_60]: https://arxiv.org/pdf/2412.04233.pdf

[^1_61]: https://arxiv.org/pdf/2309.07108.pdf

[^1_62]: http://arxiv.org/pdf/2408.13567.pdf

[^1_63]: https://arxiv.org/html/2503.15615v1

[^1_64]: http://arxiv.org/pdf/2410.02581.pdf

[^1_65]: https://arxiv.org/pdf/2305.13411.pdf

[^1_66]: https://proceedings.neurips.cc/paper_files/paper/2024/file/4830a9b95a2f63fc4b3fe09abc18f045-Paper-Conference.pdf

[^1_67]: https://openreview.net/forum?id=VkQCMO8lna

[^1_68]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11059992/

[^1_69]: https://www.koreascience.kr/article/JAKO202405259940153.page

[^1_70]: https://papers.neurips.cc/paper_files/paper/2020/file/168efc366c449fab9c2843e9b54e2a18-Paper.pdf

[^1_71]: https://grf-marl.readthedocs.io/en/latest/algorithm/cooperative.html

[^1_72]: https://arxiv.org/html/2502.08985v2

[^1_73]: https://pure.kaist.ac.kr/en/publications/distributed-multi-agent-reinforcement-learning-for-scalable-cell-/

[^1_74]: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1229026/full

[^1_75]: https://www.sciencedirect.com/science/article/pii/S2949855424000042

[^1_76]: https://arxiv.org/abs/2301.01919

[^1_77]: https://www.themoonlight.io/en/review/fully-decentralized-cooperative-multi-agent-reinforcement-learning-a-survey

[^1_78]: https://openreview.net/forum?id=CpnKq3UJwp

[^1_79]: https://www.sciencedirect.com/science/article/pii/S0950705124007585

[^1_80]: https://proceedings.neurips.cc/paper/2021/file/9c51a13764ca629f439f6accbb4ec413-Paper.pdf

[^1_81]: https://arxiv.org/html/2601.12662v1

[^1_82]: https://arxiv.org/html/2601.08210v1

[^1_83]: https://arxiv.org/abs/2409.03052

[^1_84]: https://www.arxiv.org/pdf/2410.02581v1.pdf
