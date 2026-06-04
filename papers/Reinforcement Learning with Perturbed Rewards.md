
# Reinforcement Learning with Perturbed Rewards 

> **논문 정보**
> - **저자:** Jingkang Wang, Yang Liu, Bo Li
> - **소속:** University of Toronto & Vector Institute / UC Santa Cruz / UIUC
> - **발표:** AAAI 2020 (Spotlight)
> - **arXiv:** [1810.01032](https://arxiv.org/abs/1810.01032)
> - **출판:** Proceedings of the AAAI Conference on Artificial Intelligence, 34(04), pp. 6202–6209.
> - **GitHub:** [wangjksjtu/rl-perturbed-reward](https://github.com/wangjksjtu/rl-perturbed-reward)

---

## 1. 핵심 주장 및 주요 기여 요약

최근 연구들은 RL 모델이 다양한 노이즈 시나리오에서 취약하다는 사실을 보여 왔다. 예를 들어, 보상이 센서를 통해 수집될 경우 관측된 보상 채널은 노이즈에 취약하여 신뢰할 수 없으며, 로보틱스 등의 응용에서는 Deep RL 알고리즘이 corrupted reward를 받아 임의의 오류를 발생시키도록 조작될 수 있다.

이에 대한 핵심 주장과 기여는 다음과 같다.

### ✅ 핵심 주장

이 논문은 편향된(biased) 보상 설정에서의 강건한 RL을 다룬 **최초의 연구**로, 기존 연구들이 요구하던 노이즈 분포에 대한 가정(예: 제로-평균 가우시안 노이즈)을 두지 않는다. 노이즈 생성이 **reward confusion matrix**를 따른다는 사실 외에 true reward 분포나 적대적 전략에 대한 어떠한 가정도 요구하지 않는다.

### ✅ 주요 기여

논문의 기여는 다음과 같이 요약된다: (1) 관측된 perturbed reward를 이용해 true reward에 대한 **간단하고 효과적인 unbiased estimator**를 정의하는 아이디어를 RL 설정으로 적용 및 일반화.

핵심 아이디어는 **reward confusion matrix를 추정**하고, **unbiased surrogate reward 집합을 정의**하는 것이며, 수렴성 및 샘플 복잡도를 이론적으로 증명하였다.

또한 보상 confusion matrix를 알지 못하는 설정(각 (state, action) 쌍의 보상이 결정론적인 경우)으로도 솔루션을 일반화하며, 기존 지도학습 연구들과 달리 confusion matrix를 프레임워크 내에서 직접 추정한다.

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 🔴 2.1 해결하고자 하는 문제

이 논문은 RL 에이전트가 오직 **perturbed reward만 관측**할 수 있는 노이즈 환경에서 학습 가능하도록 하는 robust 프레임워크를 개발한다. 핵심 난제는 관측된 보상이 편향될 가능성이 높으며, RL/DRL에서는 누적 오류가 시간이 지남에 따라 보상 추정 오류를 증폭시킨다는 점이다.

이 프레임워크는 노이즈 데이터를 다루는 지도학습 분야의 접근법에서 영감을 얻었다.

### 🟠 2.2 MDP 형식화

환경은 MDP $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, R, P, \gamma \rangle$ 로 정형화된다.

- $\mathcal{S}$: 상태 공간(state space)
- $\mathcal{A}$: 행동 공간(action space)
- $R$: **실제(true) 보상 함수** (에이전트에게 직접 관측되지 않음)
- $P$: 전이 확률(transition probability)
- $\gamma \in [0, 1]$: 할인 계수(discount factor)

에이전트가 실제로 받는 보상은 **perturbed reward** $\tilde{r}$이며, 이는 true reward $r$이 confusion matrix $C$를 통해 변환된 것이다.

$$\tilde{r} = C \cdot r$$

예를 들어 이진 보상(binary reward, $r \in \{-1, +1\}$)의 경우, confusion matrix는 다음과 같이 정의된다:

$$C = \begin{pmatrix} 1 - \rho^+ & \rho^+ \\ \rho^- & 1 - \rho^- \end{pmatrix}$$

여기서 $\rho^+$는 positive reward가 negative로 바뀔 확률, $\rho^-$는 negative reward가 positive로 바뀔 확률이다.

### 🟡 2.3 Unbiased Surrogate Reward 정의

논문은 오류율이 알려진 경우 이진 보상에 대한 unbiased estimator를 먼저 소개하고, 이를 **다중 결과(multi-outcome) 및 연속 보상(continuous reward) 설정**으로 확장한다.

오류율 $\rho^+, \rho^-$가 알려진 경우, true reward에 대한 **unbiased surrogate reward** $\hat{r}$는 다음과 같이 정의된다:

$$\hat{r} = \frac{(1 - \rho^-) \tilde{r} - \rho^- \tilde{r}'}{1 - \rho^+ - \rho^-}$$

더 일반적으로, $k$개의 이산 보상 레벨 $\mathcal{R} = \{r_1, \ldots, r_k\}$가 있을 때 confusion matrix $C \in \mathbb{R}^{k \times k}$를 이용한 unbiased surrogate는:

$$\mathbb{E}[\hat{r}] = C^{-1} \tilde{r}$$

단, $C$가 invertible할 때 성립하며, 이 surrogate reward를 기반으로 Q-value 및 정책을 최적화한다:

$$\hat{Q}^\pi(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \hat{r}_t \mid s_0 = s, a_0 = a, \pi\right]$$

### 🟢 2.4 Confusion Matrix 추정 모듈

결정론적 보상 설정에서 confusion matrix를 효율적이고 유연하게 추정하는 모듈을 제안한다.

추정 절차는 다음과 같다:

1. 동일한 $(s, a)$ 쌍에 대해 **반복 관측**(repeated observations)으로 true reward의 추정치 $\bar{r}$을 정제(refine)한다.
2. 추정된 $\bar{r}$을 활용하여 confusion matrix $\tilde{C}$를 추정한다:

$$\tilde{C}_{ij} = \frac{\text{count}(\tilde{r} = r_j \mid r = r_i)}{\text{count}(r = r_i)}$$

3. 추정된 $\hat{C}$를 기반으로 surrogate reward $\dot{r}$을 재계산하고 정책을 업데이트한다.

이 추정 절차는 결정론적 보상에만 적용 가능하며, 반복 관측으로 추정된 ground truth reward를 precision 높게 정제하는 방식을 사용한다.

### 🔵 2.5 모델 구조

이 솔루션 프레임워크는 기존 RL/DRL 알고리즘 위에 구축되며, 제로-평균 가우시안 노이즈와 같은 true 분포에 대한 어떠한 가정도 없이 편향된 노이즈 보상 설정을 최초로 다룬다.

구체적으로 프레임워크는 다음 세 모듈로 구성된다:

| 모듈 | 역할 |
|------|------|
| **Reward Confusion Matrix Estimator** | $(s, a)$ 쌍의 반복 관측으로 $C$ 추정 |
| **Unbiased Surrogate Reward Generator** | $\hat{C}^{-1}$를 이용한 $\dot{r}$ 계산 |
| **Base RL/DRL Optimizer** | DQN, DDQN, PPO 등 기존 알고리즘 사용 |

이 구현은 **keras-rl** 및 **OpenAI Baselines** 프레임워크 기반으로 제공된다.

### 🟣 2.6 성능 향상

다양한 DRL 플랫폼에서의 실험 결과, 추정된 surrogate reward를 기반으로 훈련된 정책이 더 높은 기대 보상을 달성하고 기존 베이스라인보다 더 빠르게 수렴하였다. 특히 최신 PPO 알고리즘은 오류율 10% 및 30% 조건에서 5개 Atari 게임 평균 점수를 각각 **84.6%** 및 **80.8%** 향상시켰다.

놀랍게도 일부 경우에는 clean reward 기반 학습보다 더 높은 누적 보상을 달성하기도 하였는데, 이는 삽입된 노이즈와 unbiased estimator의 조합이 **추가적인 탐색(exploration) 레이어**로 작용했기 때문으로 추정된다.

### ⚫ 2.7 한계

추정 절차는 **결정론적 보상(deterministic reward)**에만 적용 가능하며, 확률론적 보상(stochastic rewards)에는 적용되지 않는다. true reward에 불확실성이 있는 경우, clean 케이스의 $C \cdot R$과 perturbed 케이스의 $R$+added noise를 구별하는 것이 불가능하기 때문이다.

추가적인 한계:
- Confusion matrix의 invertibility 가정이 필요하다
- 연속 상태/보상 공간에서의 추정 정확도가 이산 케이스보다 낮을 수 있다
- 오류율 $\rho^+, \rho^-$를 모르는 경우 추정 단계에서 추가 샘플이 요구된다

---

## 3. 모델의 일반화 성능 향상 가능성

이 프레임워크의 핵심 강점 중 하나는 **기존 RL/DRL 알고리즘 위에 plug-in 형태로 결합**되며, 기존 방법들이 요구하던 true reward 분포에 대한 가정(예: zero-mean Gaussian noise)을 전혀 요구하지 않는다는 점이다.

일반화 관점에서 다음 측면들이 중요하다:

**① 노이즈 환경에서의 정책 일반화**

후속 연구인 Distributional Reward Critic 프레임워크(arXiv:2401.05710)에서는 더욱 일반적인 미지의 perturbation 클래스를 연구하고, 훈련 중 보상 분포와 perturbation을 추정하는 방법을 제안하며, **어떤 RL 알고리즘과도 호환**되고 clean reward 환경을 포함한 다양한 환경에서 comparable하거나 더 나은 보상을 달성함을 보인다.

**② 탐색과 일반화의 상호작용**

방법이 일부 설정에서 더 높은 누적 보상을 달성하는 현상은, 노이즈 제거 과정이 **추가적인 탐색으로 기능**할 수 있음을 시사하며, 이는 향후 연구로 이어질 가능성이 있다.

**③ 다양한 도메인 적용 가능성**

현실 환경에서 수집된 보상은 적대자, 센서 오류, 또는 주관적인 인간 피드백으로 인해 변조(perturbed), 손상(corrupted), 또는 노이즈를 포함할 수 있으므로, 이러한 보상 하에서 학습 가능한 에이전트를 구성하는 것은 매우 중요하다.

**④ RLHF와의 연결**

실제 환경에서 RL 배포는 노이즈가 있는 불완전한 감독 신호를 수반하며, 특히 인간 피드백 기반 강화학습(RLHF)에서 두드러지게 나타난다. 이 논문의 프레임워크는 RLHF에서의 reward hacking 및 reward overoptimization 문제를 완화하는 데에도 응용 가능성이 있다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 🔭 4.1 연구에 미치는 영향

| 영역 | 영향 |
|------|------|
| **Robust RL** | 편향된 보상 설정에서의 이론적 토대 제공 |
| **RLHF** | 인간 피드백의 노이즈 처리 연구로 확장 |
| **Safe RL** | 적대적 보상 조작에 대한 방어 메커니즘 설계 |
| **도메인 일반화** | 다양한 플랫폼(Atari, MuJoCo 등)에 plug-in 적용 가능 |

기존 방법론들은 perturbation이 사전에 알려져 있거나, clean reward에 접근 가능하거나, perturbation이 최적 정책을 보존한다는 **강한 가정을 전제**하고 있다는 한계가 있었다. 이 논문은 그러한 가정 없이도 작동하는 프레임워크를 제시함으로써 후속 연구의 기반을 마련하였다.

2024년 AAAI에 발표된 후속 연구(Distributional Reward Critic, arXiv:2401.05710)는 이 논문에서 영감을 받아, 44/48개의 테스트 설정에서 최고 리턴을 달성함으로써(최선 베이스라인의 11/48 대비) 노이즈가 있는 보상 환경에서의 RL 연구를 대폭 심화·확장하였다.

### ⚠️ 4.2 앞으로 연구 시 고려할 점

1. **확률론적(stochastic) 보상 설정으로의 확장**
   현재 추정 절차는 결정론적 보상에만 적용 가능하므로, 반복 관측으로 추정 ground truth reward를 정제하는 방식을 확률론적 보상에도 적용할 수 있는 방법론 개발이 필요하다.

2. **비정상(non-stationary) 노이즈 환경 대응**
   실제 환경에서는 오류율 $\rho^+, \rho^-$가 시간에 따라 변할 수 있으므로, adaptive confusion matrix 추정 기법이 필요하다.

3. **Confusion Matrix의 가역성(invertibility) 가정 완화**
   $C$가 항상 invertible하지 않을 수 있으므로, 정규화(regularization) 또는 pseudo-inverse 기반 접근이 연구되어야 한다.

4. **RLHF와의 통합**
   보상 자체가 verifier에 의해 공급되어 노이즈를 포함하는 경우, 새로운 추론 시간 전략이나 재순위 방식을 제안하는 것이 아니라, **노이즈가 있는 보상 하에서 정책 최적화를 어떻게 수행할지**를 다루는 것이 핵심 과제이다.

5. **탐색(exploration)과 노이즈 제거의 상호작용 연구**
   노이즈 제거 과정이 일부 설정에서 추가 탐색으로 기능한다는 현상은 별도의 심층 연구가 필요하다.

6. **Large Language Model(LLM) 파인튜닝으로의 응용**
   VRPO 등 후속 연구에서 볼 수 있듯이, 노이즈가 포함된 피드백에서 관련 정보를 추출하는 모델의 능력은 다양한 도메인에서의 robust 일반화를 가능하게 한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 발표 | 핵심 방법 | Wang et al.(2020) 대비 특징 |
|------|------|-----------|---------------------------|
| **Distributional Reward Critic (DRC)** (Chen et al.) | AAAI 2025 | 보상 분포 및 perturbation을 훈련 중 추정하는 분산 보상 critic | 더 일반적인 unknown perturbation 클래스 처리, 모든 RL 알고리즘 호환 |
| **VRPO** (2025) | arXiv 2025 | 노이즈 감독 하 value model 중심 robust 정책 최적화 | RLHF/LLM 파인튜닝으로 응용 확장 |
| **Reward Machines for Noisy Environments** (Li et al.) | NeurIPS Workshop 2022 | 노이즈 레이블에서 reward machine 학습 | 비마르코프 보상 구조에서의 노이즈 처리 |

DRC(Distributional Reward Critic) 프레임워크는 새롭고 더 일반적인 클래스의 unknown perturbation을 연구하며, 훈련 중 보상 분포와 perturbation을 추정하는 방법을 도입한다. 이 방법은 어떤 RL 알고리즘과도 호환되며, clean reward 환경을 포함한 다양한 환경에서 comparable하거나 더 나은 보상을 달성한다.

그러나 DRC를 포함한 기존 방법론 대부분은 여전히 perturbation 정보가 사전에 알려지거나, clean reward에 접근 가능하거나, perturbation이 최적 정책을 보존한다는 가정에 의존한다는 한계가 있다. 이는 Wang et al.(2020)의 최초 가정 완화가 여전히 이 분야 연구의 핵심 방향임을 보여준다.

---

## 📚 참고 문헌 및 출처

1. **Wang, J., Liu, Y., & Li, B. (2020). Reinforcement Learning with Perturbed Rewards.** *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(04), 6202–6209.
   - arXiv: https://arxiv.org/abs/1810.01032
   - AAAI Official: https://ojs.aaai.org/index.php/AAAI/article/view/6086
   - PDF (AAAI CDN): https://cdn.aaai.org/ojs/6086/6086-13-9311-1-10-20200513.pdf

2. **GitHub 구현체:** https://github.com/wangjksjtu/rl-perturbed-reward

3. **OpenReview 페이지:** https://openreview.net/forum?id=BkMWx309FX

4. **Chen, X. et al. (2024). The Distributional Reward Critic Framework for Reinforcement Learning Under Perturbed Rewards.** *AAAI 2025.*
   - arXiv: https://arxiv.org/abs/2401.05710

5. **Korkmaz, E. (2024). A Survey Analyzing Generalization in Deep Reinforcement Learning.** arXiv:2401.02349.
   - https://arxiv.org/html/2401.02349v1

6. **VRPO: Rethinking Value Modeling for Robust RL Training under Noisy Supervision.** arXiv:2508.03058.
   - https://arxiv.org/pdf/2508.03058

7. **Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers.** arXiv:2510.00915.
   - https://arxiv.org/html/2510.00915v3

8. **ResearchGate 논문 페이지:** https://www.researchgate.net/publication/342543726_Reinforcement_Learning_with_Perturbed_Rewards
