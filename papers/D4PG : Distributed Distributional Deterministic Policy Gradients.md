# D4PG : Distributed Distributional Deterministic Policy Gradients

**핵심 요약**  
본 논문은 *분포적 강화학습(distributional RL)* 관점을 연속 제어(continuous control) 문제에 적용하고, 이를 대규모 분산 학습(framework)으로 확장한 **Distributed Distributional Deep Deterministic Policy Gradient**(D4PG) 알고리즘을 제안한다. 주요 기여는 다음과 같다.  
- **분포적 Critic 업데이트**: 상태-행동 가치의 분포를 모델링하고, 분포적 Bellman 연산자에 기반한 손실로 안정적 학습 신호를 획득.  
- **병렬 Actor를 통한 분산 경험 수집**: 다수의 병렬 환경에서 데이터를 수집해 학습 효율 및 wall-clock 시간 단축.  
- **N-step 수익**과 **Prioritized Experience Replay**의 결합.  
다양한 단순 제어, 조작, 장애물 회피 과제에서 SOTA 성능을 달성했다.[1]

## 1. 해결하고자 하는 문제  
전통적 DDPG 알고리즘은 기대값 기반 critic(Q-value)만을 학습하므로, 연속 행동 공간에서 불안정하거나 과대평가에 취약하다. 또한 단일 환경에 의존하는 경험 수집 방식으로 인해 학습 속도가 느리고 일반화 성능에 한계가 존재한다.

## 2. 제안하는 방법  
### 2.1 분포적 Critic 업데이트  
- 반환(Return)을 확률 변수 $$Z$$로 정의하고, 분포적 Bellman 연산자  

$$
    \mathcal{T}Z(x,a) \;\overset{D}{=}\; r(x,a) + \gamma Z(x',\pi(x'))
  $$  
  
를 사용해, 분포 간 거리 $$d$$를 최소화하는 손실  

$$
    L(w) = d\bigl(\mathcal{T}Z_{w^-},\,Z_w\bigr)
  $$  
  
을 최소화한다.[1]
- 실험에서는 **Categorical distribution**(51개 atom, $$[V_\mathrm{min},V_\mathrm{max}]$$ 구간)에 대한 교차 엔트로피 손실을 사용했다.

### 2.2 Actor 업데이트  
- 분포적 값을 기대값으로 변환하여 정책 기울기에 반영:  

$$
    \nabla_\theta J = \mathbb{E}\_{x\sim\mu}\bigl[\nabla_a \mathbb{E}[Z_w(x,a)]\rvert_{a=\pi_\theta(x)}\nabla_\theta\pi_\theta(x)\bigr].
  $$

### 2.3 N-step 수익  
- 표준 Bellman 연산자를 N스텝 연산자로 확장:  

$$
    \mathcal{T}^{(N)}Q(x_0,a_0) = \sum_{n=0}^{N-1}\gamma^n r_n + \gamma^N Q(x_N,\pi(x_N)),
  $$  
  
분포적 버전에도 동일하게 적용하여 더 긴 미래 정보를 활용해 안정성 및 수렴 속도 향상.

### 2.4 분산 경험 수집 및 Prioritized Replay  
- $$K$$개의 병렬 actor가 동일한 replay buffer에 경험을 저장하고, 단일 learner 프로세스가 샘플링 및 업데이트 수행.  
- 중요도 기반 샘플링(priority)을 통해 TD-error가 큰 transition에 더 높은 확률 부여.

### 2.5 모델 구조  
- **Actor/Critic 네트워크**: 표준 제어·조작 과제에서는 MLP 기반, parkour 도메인에서는 지형 정보(terrain)와 관절 정보(proprioception)를 각각 처리하는 torso를 두고 이후 결합하는 구조.[1]

## 3. 성능 향상 및 한계  
- **Ablation 실험**: N-step 수익이 가장 큰 성능 개선을 주며, 분포적 업데이트와 분산 학습도 유의미한 기여. Prioritized replay는 DDPG에 비해 D4PG에서 불안정성을 유발할 수 있음.  
- **제어 과제**: MuJoCo 단순 물리 제어, dexterous 조작, 장애물 기반 parkour까지 광범위하게 평가하여 D4PG가 일관되게 최상위 성능을 보임.[1]
- **한계**:  
  - Prioritized replay의 불안정성  
  - 분포 모형으로 Mixture of Gaussians는 Categorical 대비 성능 저하 발견  
  - 병렬 샘플링 위주로 샘플 효율성(sample efficiency)은 PPO 등 일부 온-폴리시 기법에 미흡

## 4. 일반화 성능 향상 관점  
분포적 critic은 반환의 전체 분포를 모델링하여 불확실성을 명시적으로 반영하므로, 학습 중 과대추정(bias)을 줄이고 안정적인 기울기 신호를 제공한다. 이로 인해 다양한 환경 변이(domain randomization)와 잡음에도 견고하게 동작하여 **모델의 일반화 성능**이 크게 향상된다. 또한 다중 actor로부터 얻은 경험은 상태-행동 공간을 광범위하게 탐색하게 하여 오버피팅을 방지하고, 새로운 과제 전이(transfer)에도 유리하다.

## 5. 향후 연구에 미치는 영향 및 고려사항  
D4PG는 분포적 강화학습과 대규모 분산 학습을 결합한 대표적인 사례로, 향후 다음 연구 방향에 영향을 줄 것이다.  
- **다양한 분포 모형 탐색**: Categorical 이외에도 안정적이면서 해상도가 높은 분포 모형 개발  
- **효율적 샘플링 기법**: 병렬 샘플링과 샘플 효율성 간 균형을 맞추기 위한 하이브리드 온-/오프폴리시 전략  
- **일반화 테스트 벤치마크 확장**: 더 복잡·비정형 환경에서의 일반화 성능 검증  
- **안정성 향상**: Prioritized replay의 불안정성 완화 기법 및 하이퍼파라미터 자동 조정 기법 연구

이와 같은 고려사항을 바탕으로, 분포적·분산 강화학습의 실용성과 견고성을 한층 더 끌어올릴 수 있을 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c09983d1-f2a1-4c93-9953-3f9231498cd2/1804.08617v1.pdf)
