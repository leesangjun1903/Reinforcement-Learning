# Diffusion Actor-Critic with Entropy Regulator (DACER)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

DACER는 기존 강화학습(RL)에서 정책(policy)을 **대각 가우시안 분포(diagonal Gaussian distribution)** 로 파라미터화하는 방식의 한계를 극복하고자 한다. 전통적인 단봉(unimodal) 가우시안 정책은 멀티모달(multimodal) 최적 정책을 표현하지 못해 지역 최적해(local optima)에 빠질 위험이 있다. 이를 해결하기 위해 **확산 모델(diffusion model)의 역방향 과정(reverse process)을 새로운 정책 함수로 활용**하는 온라인 RL 알고리즘 DACER를 제안한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **① 새로운 정책 함수 제안** | 확산 모델의 역방향 프로세스를 정책 근사기(policy approximator)로 재개념화 |
| **② 엔트로피 추정 방법** | 가우시안 혼합 모델(GMM)을 이용한 확산 정책의 엔트로피 추정 |
| **③ 탐색-착취 자동 조절** | 추정된 엔트로피 기반으로 파라미터 $\alpha$를 학습하여 탐색 수준 자동 조절 |
| **④ SOTA 성능 달성** | MuJoCo 8개 벤치마크에서 DDPG, TD3, PPO, SAC, DSAC, TRPO 대비 최고 성능 |
| **⑤ 코드 공개** | JAX 기반 구현 코드 오픈소스 공개 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

**문제 1: 정책 표현력의 한계**

기존 RL 알고리즘(SAC, TD3, DDPG 등)은 정책을 대각 가우시안 분포로 파라미터화한다:

$$\pi_\theta(a|s) = \mathcal{N}(a; \mu_\theta(s), \sigma_\theta^2(s) \cdot I)$$

이러한 단봉 분포는 동일 상태에서 두 개의 다른 행동이 유사한 Q값을 가지는 경우(멀티모달 상황), **모드 커버링(mode-covering) 현상**을 유발한다. 즉, 정책이 두 모드 사이의 중간 영역(낮은 Q값 영역)에 높은 밀도를 할당하게 된다.

**문제 2: 확산 정책의 엔트로피 결정 불가**

확산 정책의 분포는 해석적 표현식(analytical expression)이 없어 엔트로피를 직접 계산할 수 없다. 이는 최대 엔트로피 RL 프레임워크(SAC처럼)와의 결합을 어렵게 만든다.

**문제 3: 기존 온라인 RL + 확산 정책 연구의 한계**

- Yang et al. (2023): 행동 그래디언트 방법($\nabla_a Q$ 기반) → 추가 학습 시간 및 준최적 성능
- Psenka et al. (2023): Q-score matching(QSM) → $\nabla_a Q$를 정확히 학습해야 하는 어려움

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) 확산 정책 표현 (Diffusion Policy Representation)

DDPM(Denoising Diffusion Probabilistic Model)을 기반으로 조건부 확산 모델의 역방향 프로세스를 정책으로 사용한다:

$$\pi_\theta(\boldsymbol{a}|\boldsymbol{s}) = p_\theta(\boldsymbol{a}_{0:T}|\boldsymbol{s}) = p(\boldsymbol{a}_T) \prod_{t=1}^{T} p_\theta(\boldsymbol{a}_{t-1}|\boldsymbol{a}_t, \boldsymbol{s}) \tag{7}$$

여기서 $p(\boldsymbol{a}_T) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$이고, 역방향 체인의 최종 샘플 $\boldsymbol{a}_0$이 RL 평가에 사용되는 행동이다.

평균 $\boldsymbol{\mu}\_\theta$는 노이즈 예측 모델 $\epsilon_\theta$로부터 구성된다:

$$\boldsymbol{\mu}_\theta(\boldsymbol{a}_t, \boldsymbol{s}, t) = \frac{1}{\sqrt{\alpha_t}} \left( \boldsymbol{a}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\boldsymbol{a}_t, \boldsymbol{s}, t) \right) \tag{8}$$

여기서 $\alpha_t = 1 - \beta_t$, $\bar{\alpha}\_t = \prod_{k=1}^{t} \alpha_k$이다.

재파라미터화 트릭(reparametrization trick)을 이용한 샘플링 과정:

$$\boldsymbol{a}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \boldsymbol{a}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\boldsymbol{a}_t, \boldsymbol{s}, t) \right) + \sqrt{\beta_t} \boldsymbol{\epsilon} \tag{9}$$

여기서 $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$, $t$는 $T$에서 $0$까지의 역방향 타임스텝이다.

#### (B) 확산 정책 학습 (Diffusion Policy Learning)

행동 복제(behavior cloning) 없이, **기대 Q값 최대화**를 목적 함수로 직접 최적화한다:

$$\max_\theta \mathbb{E}_{\boldsymbol{s} \sim \mathcal{B}, \boldsymbol{a}_0 \sim \pi_\theta(\cdot|\boldsymbol{s})} \left[ Q_\phi(\boldsymbol{s}, \boldsymbol{a}_0) \right] \tag{10}$$

Q값 함수는 Double Q-learning 트릭과 Bellman 방정식으로 학습한다:

$$\min_{\phi_i} \mathbb{E}_{(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') \sim \mathcal{B}} \left[ \left( r(\boldsymbol{s}, \boldsymbol{a}) + \gamma \min_{i=1,2} Q_{\phi'_i}(\boldsymbol{s}', \boldsymbol{a}') - Q_{\phi_i}(\boldsymbol{s}, \boldsymbol{a}) \right)^2 \right] \tag{11}$$

#### (C) 엔트로피 추정 및 조절 (Entropy Estimation via GMM)

확산 정책의 엔트로피를 추정하기 위해 **가우시안 혼합 모델(GMM)** 을 사용한다:

$$\hat{f}(\boldsymbol{a}) = \sum_{k=1}^{K} w_k \cdot \mathcal{N}(\boldsymbol{a}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \tag{12}$$

**E-step** (사후 확률 계산):

$$\gamma(z_k^i) = \frac{w_k \cdot \mathcal{N}(\boldsymbol{a}^i|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} w_j \cdot \mathcal{N}(\boldsymbol{a}^i|\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)} \tag{13}$$

**M-step** (파라미터 업데이트):

$$w_k = \frac{1}{N} \sum_{i=1}^{N} \gamma(z_k^i), \quad \boldsymbol{\mu}_k = \frac{\sum_{i=1}^{N} \gamma(z_k^i) \cdot \boldsymbol{a}^i}{\sum_{i=1}^{N} \gamma(z_k^i)}, \quad \boldsymbol{\Sigma}_k = \frac{\sum_{i=1}^{N} \gamma(z_k^i)(\boldsymbol{a}^i - \boldsymbol{\mu}_k)(\boldsymbol{a}^i - \boldsymbol{\mu}_k)^T}{\sum_{i=1}^{N} \gamma(z_k^i)} \tag{14}$$

GMM 기반 엔트로피 추정:

$$\mathcal{H}_s \approx -\sum_{k=1}^{K} w_k \log w_k + \sum_{k=1}^{K} w_k \cdot \frac{1}{2} \log \left( (2\pi e)^d |\boldsymbol{\Sigma}_k| \right) \tag{15}$$

여기서 $d$는 행동의 차원이다.

**엔트로피 기반 파라미터 $\alpha$ 업데이트:**

$$\alpha \leftarrow \alpha - \beta_\alpha [\hat{\mathcal{H}} - \overline{\mathcal{H}}] \tag{16}$$

여기서 $\overline{\mathcal{H}}$는 목표 엔트로피, $\hat{\mathcal{H}} = \mathbb{E}_{s \sim \mathcal{B}}[\mathcal{H}_s]$는 추정된 확산 정책 엔트로피이다.

**노이즈 적응적 조절:**

$$\boldsymbol{a} = \boldsymbol{a} + \lambda\alpha \cdot \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$$

여기서 $\lambda$는 하이퍼파라미터이고, $\boldsymbol{a}$는 확산 모델의 출력이다. 평가(evaluation) 단계에서는 노이즈를 추가하지 않는다.

---

### 2-3. 모델 구조

```
[환경 상태 s]
      │
      ▼
┌─────────────────────────────────────────────────┐
│         Diffusion Policy Network (Actor)         │
│  a_T ~ N(0,I)                                    │
│  → Reverse Diffusion: T steps                   │
│     at-1 = μ_θ(at, s, t) + √βt · ε              │
│  → 노이즈 예측 네트워크 ε_θ: [256,256,256] MLP   │
│     (Humanoid: [512,512,512])                    │
│     활성화 함수: Mish                            │
│  → Sinusoidal Embedding: t → 16차원             │
│  → a_0 출력 후 λα·N(0,I) 노이즈 추가           │
└─────────────────────────────────────────────────┘
      │ a_0 (행동)
      ▼
┌─────────────────────────────────────────────────┐
│         Critic Networks (Double Q)               │
│  Q_φ1(s,a), Q_φ2(s,a)                           │
│  Target: Q_φ'1, Q_φ'2                           │
│  아키텍처: [256,256,256] MLP, GeLU              │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│         Entropy Regulator                        │
│  - 매 10,000 iteration마다 엔트로피 추정         │
│  - N=200 행동 샘플링                            │
│  - GMM 피팅 (K=3 가우시안 성분)                 │
│  - α 업데이트 (식 16)                           │
└─────────────────────────────────────────────────┘
```

**주요 하이퍼파라미터:**

| 파라미터 | 값 |
|---|---|
| 역방향 확산 스텝 수 $T$ | 20 |
| GMM 성분 수 $K$ | 3 |
| 엔트로피 추정 행동 샘플 수 | 200 |
| 노이즈 스케일 $\lambda$ | 0.1 (Humanoid, HalfCheetah: 0.15) |
| 목표 엔트로피 $\overline{\mathcal{H}}$ | $-0.9 \cdot \dim(\mathcal{A})$ |
| 배치 크기 | 256 |
| 리플레이 버퍼 크기 | $10^6$ |
| $\alpha$ 업데이트 주기 | 10,000 step마다 |

---

### 2-4. 성능 향상

#### MuJoCo 벤치마크 결과

| Task | DACER | DSAC | SAC | TD3 | DDPG | TRPO | PPO |
|---|---|---|---|---|---|---|---|
| Humanoid-v3 | **11888±244** | 10829±243 | 9335±695 | 5631±435 | 5291±662 | 965±555 | 6869±1563 |
| Ant-v3 | **9108±103** | 7086±261 | 6427±804 | 6184±486 | 4549±788 | 6203±578 | 6156±185 |
| HalfCheetah-v3 | **17177±176** | 17025±157 | 16573±224 | 8632±4041 | 13970±2083 | 4785±967 | 5789±2200 |
| Walker2d-v3 | **6701±62** | 6424±147 | 6200±263 | 5237±335 | 4095±68 | 5502±593 | 4831±637 |
| Hopper-v3 | **4104±49** | 3660±533 | 2483±943 | 3569±455 | 2644±659 | 3474±400 | 2647±482 |
| Swimmer-v3 | **152±7** | 138±6 | 140±14 | 134±5 | 146±4 | 70±38 | 130±2 |

Humanoid-v3에서 TRPO 대비 **1131.9%**, DDPG 대비 **124.7%** 향상을 달성하였다.

#### 멀티모달 태스크 (Multi-goal) 실험 결과

- DACER는 대칭적으로 배치된 4개 목표 지점에 대해 **4개의 대칭 피크를 갖는 가치 함수(value function)** 학습에 성공
- DSAC는 불완전한 피크를 학습, TD3·PPO는 피크 구조 자체를 학습하지 못함
- 5개의 멀티모달 요구 지점에서 각 100개의 궤적 샘플링 결과, DACER는 강한 멀티모달 특성을 보임

---

### 2-5. 한계 (Limitations)

1. **엔트로피 추정 계산 비용**: GMM 기반 엔트로피 추정에는 약 40ms가 소요되어, 10,000 iteration마다 한 번씩만 추정 가능 → 최대 엔트로피 RL과의 완전한 통합이 어려움
2. **그래디언트 전파 비용**: 역방향 확산 전체 체인에 걸쳐 그래디언트를 기록해야 하므로 메모리 및 계산 비용이 큼
3. **확산 스텝 수 민감성**: $T$가 너무 크면(예: $T=30$) 그래디언트 폭발(gradient explosion)이 발생하여 성능 저하
4. **MuJoCo 환경 한정**: 실세계 환경이나 이미지 기반 관측 공간에서의 검증은 이루어지지 않음
5. **오프라인 RL 미적용**: 행동 복제 항목을 제거하여 온라인 RL에 특화되었고, 오프라인 RL로의 직접 전환이 불가함

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 멀티모달 정책 표현을 통한 일반화

DACER의 가장 핵심적인 일반화 메커니즘은 확산 정책의 **멀티모달 분포 표현 능력**이다. 기존 가우시안 정책이 단일 모드만 표현 가능한 반면, 확산 정책은 복잡한 멀티모달 분포를 표현할 수 있다.

$$\pi_\theta(\boldsymbol{a}|\boldsymbol{s}) = p_\theta(\boldsymbol{a}_{0:T}|\boldsymbol{s}) = p(\boldsymbol{a}_T) \prod_{t=1}^{T} p_\theta(\boldsymbol{a}_{t-1}|\boldsymbol{a}_t, \boldsymbol{s})$$

이 구조는 이론적으로 임의의 복잡한 조건부 분포를 근사할 수 있으며, 이는 **다양한 환경에서의 일반화 능력의 기반**이 된다.

### 3-2. 적응적 탐색을 통한 일반화

엔트로피 레귤레이터를 통한 탐색-착취 자동 조절은 일반화에 직접적으로 기여한다:

$$\alpha \leftarrow \alpha - \beta_\alpha [\hat{\mathcal{H}} - \overline{\mathcal{H}}]$$

- 학습 초기: $\alpha$가 크게 유지되어 넓은 행동 공간 탐색 → 새로운 환경에서 다양한 전략 발견 가능
- 학습 후기: $\alpha$가 줄어들면서 학습된 정책 활용에 집중 → 수렴된 정책의 안정성 확보

이 적응적 탐색 메커니즘은 다양한 환경 설정에서도 자동으로 적절한 탐색-착취 균형을 찾기 때문에, **환경 특성에 덜 민감한 일반화 알고리즘**이 된다.

### 3-3. 행동 복제 없는 순수 RL 최적화

DACER는 기존 오프라인 확산 정책 방법들(Diffusion-QL 등)과 달리 행동 복제 항목을 제거하고 순수하게 Q값 최대화를 목표로 한다:

$$\max_\theta \mathbb{E}_{\boldsymbol{s} \sim \mathcal{B}, \boldsymbol{a}_0 \sim \pi_\theta(\cdot|\boldsymbol{s})} [Q_\phi(\boldsymbol{s}, \boldsymbol{a}_0)]$$

이는 **분포 이동(distribution shift) 문제 없이** 온라인 환경에서 직접 정책을 최적화하므로, 환경 변화에 더 강건한 일반화 성능을 가질 수 있다.

### 3-4. 다양한 Actor-Critic 프레임워크와의 호환성

DACER는 특정 RL 알고리즘에 종속되지 않고 **대부분의 Actor-Critic 프레임워크와 결합 가능**하도록 설계되었다. 이는 다양한 도메인의 태스크에 DACER의 확산 정책 표현 능력을 적용할 수 있음을 의미한다.

### 3-5. 일반화 성능의 한계와 개선 방향

다만 현재 DACER의 일반화 성능에는 다음과 같은 한계가 있다:

| 한계 요인 | 영향 |
|---|---|
| MuJoCo 환경에만 검증 | 이미지 기반 관측, 희소 보상 환경에서의 일반화 미검증 |
| $T$ 값 고정 ($T=20$) | 다른 환경 특성에 따라 최적 $T$ 값이 다를 수 있음 |
| GMM 성분 수 $K=3$ 고정 | 더 복잡한 멀티모달 환경에서는 더 많은 성분이 필요할 수 있음 |
| 고차원 연속 행동 공간에서만 검증 | 이산 행동 공간이나 계층적 행동 공간에서의 적용 미검증 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. 온라인 RL + 확산 정책 연구 비교

| 연구 | 방법 | 장점 | 단점 | DACER와의 차이 |
|---|---|---|---|---|
| **Yang et al. (2023)** [arXiv:2305.13122] | 행동 그래디언트($\nabla_a Q$) + 모방 학습 | 직접적인 Q 최적화 | 추가 학습 시간, 준최적 성능 | DACER는 단일 역방향 프로세스로 직접 Q값 최대화 |
| **Psenka et al. (2023)** [arXiv:2312.11752] QSM | Score-based 확산 정책을 $\nabla_a Q$에 정렬 | 행동 복제 한계 극복 | $\nabla_a Q$ 정확한 학습 어려움 | DACER는 $\nabla_a Q$ 직접 학습 불필요 |
| **DACER (2024)** | 역방향 프로세스 = 정책 + GMM 엔트로피 추정 | 일반성, SOTA 성능 | 엔트로피 추정 비용 | - |

### 4-2. 오프라인 RL + 확산 정책 연구 비교

| 연구 | 방법 | 핵심 특징 | DACER와의 차이 |
|---|---|---|---|
| **Diffusion-QL** Wang et al. (2023, ICLR) | 행동 복제 손실 + Q-learning | 오프라인 RL에서 강력한 성능 | DACER는 행동 복제 항목 제거, 온라인 RL 특화 |
| **Decision Diffusion** Ajay et al. (2023, ICLR) | Classifier-free guidance + 궤적 정보 통합 | 조건부 생성 모델로 의사결정 | DACER는 순수 Actor-Critic, 온라인 RL |
| **EDP** Kang et al. (NeurIPS 2023/2024) | One-step 샘플로 훈련 효율화 | 빠른 훈련 | DACER는 전체 역방향 체인 사용 |
| **Consistency Policy** Chen et al. (2023) [arXiv:2310.06343] | 일관성 모델로 1-step 생성 | 추론 속도 향상 | DACER는 다단계 역방향 프로세스 사용 |
| **Diffusion Policy** Chi et al. (2023) [arXiv:2303.04137] | 시각-운동 정책을 확산으로 학습 | 로봇 조작에서 우수한 성능 | DACER는 순수 상태 기반 RL |

### 4-3. 기존 온라인 RL 기준선과의 비교

| 알고리즘 | 정책 표현 | 탐색 메커니즘 | 멀티모달 지원 |
|---|---|---|---|
| **SAC** (Haarnoja et al., 2018, ICML) | 대각 가우시안 | 최대 엔트로피 | ✗ |
| **TD3** (Fujimoto et al., 2018, ICML) | 결정론적 | 고정 가우시안 노이즈 | ✗ |
| **DSAC** (Duan et al., 2021, IEEE TNNLS) | 분포 소프트 AC | 최대 엔트로피 | ✗ |
| **DACER** (2024, NeurIPS) | 확산 모델 (역방향 프로세스) | GMM 엔트로피 추정 기반 적응 | ✓ |

### 4-4. 확산 모델 자체 연구와의 관계

- **DDPM** (Ho et al., NeurIPS 2020): DACER의 기반 모델
- **Score-based SDE** (Song et al., arXiv 2020): 확산 모델의 이론적 기반 제공
- **Deconstructing DDPM** (Chen et al., arXiv:2401.14404): 확산 모델의 표현력이 역방향 프로세스에서 나온다는 통찰 → DACER의 핵심 동기 부여

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

**① 정책 표현 패러다임 전환**

DACER는 RL의 정책 함수를 가우시안 분포에서 생성 모델로 확장하는 패러다임 전환을 이끈다. 향후 정규화 흐름(normalizing flows), 에너지 기반 모델(EBM), 일관성 모델(consistency models) 등 다양한 생성 모델을 정책으로 활용하는 연구로 이어질 수 있다.

**② 엔트로피 추정 방법론의 발전**

GMM 기반 엔트로피 추정은 해석적 표현이 불가능한 복잡한 정책의 엔트로피를 근사하는 새로운 방법론을 제시한다. 이는 더 효율적인 엔트로피 추정(예: flow-based estimator, variational lower bound)으로 발전할 수 있다.

**③ 온라인 RL과 생성 모델의 통합**

오프라인 RL에서 주로 사용되던 확산 정책을 온라인 RL로 확장한 최초 시도 중 하나로, 이는 실시간 로봇 제어, 자율 주행, 게임 AI 등에 생성 모델을 적용하는 연구의 기초가 된다.

**④ 멀티모달 최적 정책 분야**

DACER는 멀티모달 최적 정책이 존재하는 문제들(로봇 조작, 게임, 복잡한 물리 제어)에서 가우시안 정책의 근본적 한계를 실험적으로 증명하고, 이를 극복하는 방향을 제시한다.

### 5-2. 앞으로 연구 시 고려할 점

**① 엔트로피 추정 효율화**

현재 40ms의 GMM 추정 비용이 실시간 적용의 병목이다. 향후 연구에서는 다음을 고려해야 한다:
- 경량 엔트로피 추정 방법 (예: variational entropy bound)
- 배치 크기 전체가 아닌 소수 상태만 사용하는 추정
- 신경망으로 직접 엔트로피를 근사하는 방법

**② 확산 스텝 수와 성능의 트레이드오프**

$T=20$이 최적임이 실험적으로 확인되었으나, 환경별 최적 $T$ 탐색이 필요하다. $T$가 너무 크면 그래디언트 폭발이 발생하므로, 그래디언트 클리핑이나 대안적 역방향 확산 방법(DDIM 등)의 도입을 고려해야 한다.

**③ 이미지 기반 관측 환경으로의 확장**

현재 DACER는 저차원 상태 벡터 기반 MuJoCo에서만 검증되었다. 시각 기반 RL(예: Atari, DeepMind Control Suite의 픽셀 관측)로 확장할 때 확산 모델의 입력 인코더 설계 및 계산 비용 증가를 고려해야 한다.

**④ 희소 보상(Sparse Reward) 환경에서의 검증**

MuJoCo는 밀집 보상(dense reward) 환경이다. DACER의 멀티모달 탐색 능력이 희소 보상 환경에서 더 큰 이점을 가질 수 있으나, 이에 대한 실험적 검증이 필요하다.

**⑤ 오프라인-온라인 하이브리드 RL과의 결합**

오프라인 데이터를 활용한 사전 훈련 후 온라인 파인튜닝(fine-tuning)하는 하이브리드 접근법과 DACER를 결합하면, 데이터 효율성과 정책 표현력을 모두 향상시킬 수 있다.

**⑥ 이론적 수렴 보장 연구**

현재 DACER는 이론적 수렴 분석이 부재하다. 확산 정책의 역방향 프로세스를 통한 정책 그래디언트의 수렴성, 편향(bias)-분산(variance) 트레이드오프 분석이 필요하다.

**⑦ 더 복잡한 멀티모달 환경 설계**

현재의 Multi-goal 태스크는 상대적으로 단순하다. 더 고차원, 더 복잡한 멀티모달 구조를 가진 벤치마크 환경 개발이 필요하다.

---

## 참고 자료 (출처)

1. **Wang, Y. et al. (2024).** "Diffusion Actor-Critic with Entropy Regulator." *38th Conference on Neural Information Processing Systems (NeurIPS 2024).* arXiv:2405.15177v5.

2. **Ho, J., Jain, A., & Abbeel, P. (2020).** "Denoising diffusion probabilistic models." *Advances in Neural Information Processing Systems, 33*, 6840–6851.

3. **Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).** "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." *ICML 2018*, 1861–1870.

4. **Wang, Z., Hunt, J. J., & Zhou, M. (2023).** "Diffusion policies as an expressive policy class for offline reinforcement learning." *ICLR 2023.*

5. **Yang, L. et al. (2023).** "Policy representation via diffusion probability model for reinforcement learning." arXiv:2305.13122.

6. **Psenka, M., Escontrela, A., Abbeel, P., & Ma, Y. (2023).** "Learning a diffusion model policy from rewards via Q-score matching." arXiv:2312.11752.

7. **Fujimoto, S., Hoof, H., & Meger, D. (2018).** "Addressing function approximation error in actor-critic methods." *ICML 2018*, 1587–1596.

8. **Duan, J. et al. (2021).** "Distributional soft actor-critic: Off-policy reinforcement learning for addressing value estimation errors." *IEEE Transactions on Neural Networks and Learning Systems, 33*(11), 6584–6598.

9. **Kang, B. et al. (NeurIPS 2023/2024).** "Efficient diffusion policies for offline reinforcement learning." *NeurIPS.*

10. **Chi, C. et al. (2023).** "Diffusion policy: Visuomotor policy learning via action diffusion." arXiv:2303.04137.

11. **Chen, Y., Li, H., & Zhao, D. (2023).** "Boosting continuous control with consistency policy." arXiv:2310.06343.

12. **Ajay, A. et al. (2023).** "Is conditional generative modeling all you need for decision-making?" *ICLR 2023.*

13. **Song, Y. et al. (2020).** "Score-based generative modeling through stochastic differential equations." arXiv:2011.13456.

14. **Chen, X. et al. (2024).** "Deconstructing denoising diffusion models for self-supervised learning." arXiv:2401.14404.

15. **Huber, M. F. et al. (2008).** "On entropy approximation for Gaussian mixture random vectors." *IEEE International Conference on Multisensor Fusion and Integration*, 181–188.

16. **DACER GitHub Repository:** https://github.com/happy-yan/DACER-Diffusion-with-Online-RL
