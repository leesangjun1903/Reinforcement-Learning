
# Flow Q-Learning (FQL) 

> **논문 정보**
> - **제목**: Flow Q-Learning (FQL)
> - **저자**: Seohong Park, Qiyang Li, Sergey Levine (UC Berkeley)
> - **발표**: ICML 2025
> - **arXiv**: [2502.02538](https://arxiv.org/abs/2502.02538) (2025년 2월)
> - **프로젝트 페이지**: https://seohong.me/projects/fql/

---

## 1. 핵심 주장 및 주요 기여 요약

FQL은 데이터 내의 임의로 복잡한 행동 분포(action distribution)를 모델링하기 위해 표현력이 풍부한 flow-matching 정책(policy)을 활용하는, 단순하고 성능이 우수한 오프라인 강화학습 방법이다.

### 핵심 주장

| 항목 | 내용 |
|------|------|
| **핵심 문제** | flow 정책을 RL로 직접 학습할 때 발생하는 불안정한 역전파 문제 |
| **핵심 해법** | 반복적 flow 정책이 아닌, 표현력 있는 **1-step 정책(one-step policy)**을 RL로 학습 |
| **핵심 기여** | 재귀적 역전파 제거 + 추론 속도 향상 + 표현력 유지 |

이 문제를 해결하기 위해, 반복적 flow 정책을 직접 가이드하는 대신 표현력 있는 1-step 정책을 RL로 학습한다. 이로써 불안정한 재귀적 역전파(recursive backpropagation)를 완전히 피하고, 테스트 타임에 비용이 큰 반복적 행동 생성을 제거하면서도 표현력을 거의 유지할 수 있다.

---

## 2. 해결하고자 하는 문제

### 2-1. 오프라인 RL과 멀티모달 행동 분포

오프라인 RL은 비용이 큰 환경 상호작용 없이 사전에 수집된 데이터셋으로부터 효과적인 의사결정 정책을 학습한다. 오프라인 RL의 핵심은 제약된 최적화(constrained optimization)로, 에이전트는 데이터셋의 상태-행동 분포 내에서 유지하면서 리턴을 최대화해야 한다.

데이터셋이 더 크고 다양해짐에 따라 행동 분포가 더 복잡하고 멀티모달(multimodal)해졌으며, 이는 이러한 복잡한 분포를 포착하고 더 정밀한 행동 제약을 구현할 수 있는 표현력 있는 정책 클래스를 필요로 한다.

### 2-2. flow 정책을 RL로 학습하는 어려움

flow 정책을 RL로 학습하는 것은, 행동 생성 과정의 반복적(iterative) 특성으로 인해 까다로운 문제이다.

기존 접근법인 **FBRAC(Flow Behavior-Regularized Actor-Critic)**의 문제점:
FBRAC에서는 BPTT(Backpropagation Through Time)가 수반되며, FQL은 추가적인 1-step 정책을 통해 Q 최대화와 행동 증류(behavioral distillation) 사이를 보간(interpolate)한다.

핵심 도전 과제는 반복적 프로세스의 출력값을 직접 최적화하려 할 때 BPTT(backpropagation through time)의 단점이다.

---

## 3. 제안하는 방법 (수식 포함)

### 3-1. 전체 알고리즘 구조

FQL은 크리틱 네트워크 $Q_{\phi_c}$, 행동 복제(behavioral cloning) 정책 $\pi_{\phi_b}$, 그리고 파라미터 $\phi_c, \phi_a, \phi_o$를 갖는 1-step 정책 $\pi_{\phi_o}$, 총 세 가지 네트워크로 구성된 actor-critic 구조를 사용한다.

핵심 아이디어는 flow 정책은 오직 행동 복제(behavioral cloning)로만 학습하면서, 별도의 표현력 있는 1-step 정책을 RL로 학습하는 것이다.

### 3-2. Flow Matching 배경

Flow matching은 노이즈 분포 $p_0$에서 데이터 분포 $p_1$으로의 매핑을 학습한다. 시간 $t \in [0,1]$에서 중간 상태를 $x_t = (1-t)x_0 + t x_1$로 정의하고, 속도 필드(velocity field) $v_\theta(x_t, t)$를 다음과 같이 학습한다:

$$\mathcal{L}_{\text{flow}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]$$

### 3-3. FQL의 3가지 학습 목표

**① 크리틱(Q-function) 학습**: 표준 Bellman MSE loss를 사용한다.

$$\mathcal{L}_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( Q_\phi(s, a) - \left( r + \gamma \bar{Q}(s', a') \right) \right)^2 \right]$$

여기서 $a' \sim \pi_{\phi_o}(\cdot | s')$이고, $\bar{Q}$는 타겟 네트워크이다.

**② 행동 flow 정책 학습 (Behavioral Cloning)**:
FQL은 두 개의 Q 함수 $Q_1, Q_2: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$, 시간 의존 벡터 필드 $v: \mathcal{S} \times \mathcal{A} \times [0,1] \to \mathcal{A}$로 정의된 행동 flow 정책, 그리고 표준 피드포워드 네트워크로 정의된 1-step 정책을 학습한다.

Flow 정책의 BC 손실:

$$\mathcal{L}_{\text{flow-BC}}(\phi_b) = \mathbb{E}_{(s,a) \sim \mathcal{D},\, x_0 \sim \mathcal{N}(0,I),\, t \sim \mathcal{U}[0,1]} \left[ \left\| v_{\phi_b}(s, x_t, t) - (a - x_0) \right\|^2 \right]$$

**③ 1-step 정책 학습 (핵심)**: BPTT 없이 Q를 최대화하며, flow 정책을 교사로 삼아 지식을 증류(distillation)한다.

$$\mathcal{L}_{\text{one-step}}(\phi_o) = -\mathbb{E}_{s \sim \mathcal{D},\, x_0 \sim \mathcal{N}(0,I)} \left[ Q(s, \pi_{\phi_o}(s, x_0)) \right] + \alpha \cdot \mathcal{L}_{\text{distill}}(\phi_o)$$

여기서 증류 손실(distillation loss)은:

$$\mathcal{L}_{\text{distill}}(\phi_o) = \mathbb{E}_{s, x_0} \left[ \left\| \pi_{\phi_o}(s, x_0) - \hat{a}_{\text{flow}}(s, x_0) \right\|^2 \right]$$

$\hat{a}_{\text{flow}}$는 flow 정책을 ODE로 적분하여 얻은 행동이며, $\alpha$는 BC 계수(hyperparameter)이다.

1-step 정책은 표준 피드포워드 네트워크이므로 Q 함수를 최대화할 때 BPTT를 피할 수 있으며, flow 정책의 표현력과 멀티모달성을 유지한다(1-step 정책은 flow 정책과 동일한 노이즈 변수에 조건화되므로 멀티모달 분포도 표현 가능).

---

## 4. 모델 구조

FQL은 단순하다. flow matching의 단순성 덕분에(특히 denoising diffusion에 비해), 표준 actor-critic 프레임워크 위에 몇 줄의 코드만으로 구현할 수 있다.

### 전체 모델 구조 도식

```
[오프라인 데이터셋 D]
        │
        ├──────────────────────────────┐
        ▼                              ▼
  [Flow 정책 π_flow]            [크리틱 Q_φ]
  (BC 손실로만 학습)             (Bellman 손실)
  시간-의존 벡터필드 v(s,x_t,t)         │
        │ 증류(Distillation)            │
        ▼                              │
  [1-step 정책 π_o(s, x_0)]           │
  (피드포워드 MLP)                      │
        └──────── Q 최대화 ────────────┘
        ▼ (추론 시)
  단일 forward pass → 행동 출력
```

FQL은 쉽게 구현하고 조정할 수 있다. flow matching의 단순성 덕분에 표준 행동 정규화 actor-critic 프레임워크 위에 몇 줄로 구현할 수 있으며, noise schedule 튜닝 없이 단 하나의 주요 하이퍼파라미터 $\alpha$만 존재한다.

---

## 5. 성능 향상

FQL은 오프라인 RL 및 오프라인-투-온라인 RL에서 73개의 도전적인 상태 및 픽셀 기반 OGBench, D4RL 태스크 전반에 걸쳐 강력한 성능을 달성함을 실험적으로 보인다.

특히, 고도로 멀티모달한 행동 분포를 포함하는 복잡한 태스크에서, FQL은 테스트 타임에 반복적인 flow 단계 없이도 Gaussian 및 diffusion 정책 기반 오프라인 RL 방법보다 훨씬 더 나은 성능을 달성한다.

이전 D4RL 벤치마크에서 FQL은 가장 어려운 태스크 중 하나인 `antmaze-large-play`에서 84%의 최고 성능을 달성한다.

FQL은 8개의 시드 평균으로 73개의 도전적 태스크 대부분에서 10개 방법 중 최고의 오프라인 RL 성능을 달성하며, 전신(whole-body) 휴머노이드 제어, 물체 조작, 픽셀 기반 제어를 포함한다. 특히, 고도로 멀티모달한 행동 분포를 가진 조작 태스크에서 Gaussian 정책 기반 방법보다 실질적으로 더 나은 성능을 달성한다. 또한 FQL의 1-step 가이던스는 이전 정책 추출 기법(FAWAC의 가중 회귀, FBRAC의 재귀적 역전파, IFQL의 거부 샘플링)을 크게 능가한다.

또한, FQL은 온라인 롤아웃(rollout)으로 직접 파인튜닝 가능하며, 기존의 오프라인-투-온라인 RL 방법들을 종종 능가한다.

---

## 6. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 6-1. 멀티모달 분포 대응을 통한 일반화

FQL의 표현력 있는 flow 정책은 임의로 복잡하고 멀티모달한 행동 분포를 가진 데이터를 처리할 수 있어, 다양한 도전적인 로봇 보행 및 조작 태스크에서 강력한 성능을 달성한다.

이는 특정 태스크에 특화된 좁은 분포가 아닌, **임의의 복잡한 분포**를 모델링하는 능력으로, 새로운 환경에 대한 일반화의 토대가 된다.

### 6-2. 오프라인-투-온라인 미세조정을 통한 일반화

FQL은 온라인 롤아웃으로 직접 파인튜닝될 수 있어, 기존의 오프라인-투-온라인 RL 방법들을 종종 능가한다.

이 특성은 오프라인 데이터로 사전학습 후 새로운 환경에서 온라인 미세조정을 가능하게 하여, **도메인 적응(domain adaptation)**과 **전이학습(transfer learning)**에서의 일반화 가능성을 내포한다.

### 6-3. 픽셀 기반 관찰에서의 일반화

FQL은 73개의 도전적인 상태(state) 및 픽셀(pixel) 기반 OGBench와 D4RL 태스크 전반에서 강력한 성능을 보인다.

픽셀 기반 태스크에서도 강력한 성능을 보인다는 것은, **관찰 공간(observation space)의 다양성**에 대한 일반화 가능성을 시사한다.

### 6-4. 일반화의 한계: 인과적 혼동(Confounding) 문제

이러한 알고리즘들은 데이터에 측정되지 않은 교란 변수(unmeasured confounding)가 없다고 가정하는 표준 정책 그래디언트에 기반한다. 그러나 이 조건은 시연자와 학습자의 감각 능력 간에 불일치가 있을 때 픽셀 기반 시연에서 반드시 성립하지는 않으며, 이는 오프라인 데이터에 암묵적 교란 편향을 야기한다.

### 6-5. 일반화의 한계: BC 계수 $\alpha$ 민감도

단, 대부분의 오프라인 RL 방법에서 전형적으로 나타나듯이, BC 계수 $\alpha$를 조정해야 한다.

FQL과 같은 이전 연구들은 필터링 없이 모든 상태-행동 쌍을 무차별적으로 모방한다는 한계가 있다.

이는 데이터셋 품질에 따른 일반화 저하 가능성을 의미한다.

### 6-6. 일반화의 한계: 1-step 정책의 편향(Bias)

flow Q-learning은 표현력 있는 flow 모델과 Q 함수 감소 사이의 적절한 트레이드오프를 달성하는 데 어려움을 겪는데, 이는 Q 함수 추정값에 편향을 도입하는 1-step 최적화 정책에 의존하기 때문이다.

---

## 7. 한계점 요약

| 한계 | 설명 |
|------|------|
| **1-step 편향** | 풀 flow 모델 대비 표현력 손실 및 Q 추정 편향 |
| **$\alpha$ 민감성** | BC 계수 $\alpha$ 태스크별 튜닝 필요 |
| **비인과적 데이터** | 픽셀 기반 혼동 편향에 취약 (Causal FQL에서 지적) |
| **무차별 BC** | 저품질 행동도 모방, 고품질 행동 선택적 활용 미흡 |

---

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8-1. 관련 연구 계보

| 연구 | 연도 | 핵심 방법 | FQL 대비 차이점 |
|------|------|----------|----------------|
| **IQL** (Implicit Q-Learning) | 2021 | Expectile regression | Gaussian 정책, 단순하지만 표현력 제한 |
| **Diffusion-QL** | 2022 | Diffusion policy + Q-learning | 멀티-step 디노이징, BPTT 필요, 느린 추론 |
| **FBRAC** | 2023 | Flow policy + BPTT | BPTT 불안정, FQL의 직접적 선행 연구 |
| **FQL** (본 논문) | 2025 | 1-step policy 증류 + Flow BC | BPTT 없음, 빠른 추론, 높은 표현력 |

### 8-2. FQL 이후 파생 연구

**① One-Step Flow Q-Learning (OFQL)** (arXiv:2508.13904, 2025–2026)

OFQL은 Flow Matching 패러다임 내에서 DQL 정책을 재공식화하되, 정확한 1-step 행동 생성을 직접 지원하는 평균 속도 필드(average velocity field)를 학습함으로써 기존 FM과 차별화된다. 이 설계는 멀티-step 디노이징과 BPTT 업데이트를 제거하여 실질적으로 빠르고 강건한 학습을 가능하게 한다.

D4RL 벤치마크의 광범위한 실험에서 OFQL은 단일 스텝으로 행동을 생성함에도 불구하고 훈련과 추론 모두에서 계산량을 크게 줄이면서 멀티-step DQL을 큰 폭으로 능가하며, 모든 다른 기준선을 능가하여 D4RL에서 최첨단 성능을 달성한다.

**② Causal Flow Q-Learning** (arXiv:2602.02847, 2026)

인과적 관점에서 오프라인 RL의 혼동된 관찰 문제를 조사한다. 교란 편향으로 인해 발생할 수 있는 정책의 최악의 경우 성능을 최적화하는 새로운 인과적 오프라인 RL 목적 함수를 개발한다.

**③ Guided Flow Policy (GFP)** (arXiv:2512.03973, 2025)

GFP는 Value-aware Behavioral Cloning(VaBC) flow 정책과 증류된 1-step 액터를 양방향 가이던스로 통합한다. VaBC는 액터와 크리틱을 활용하여 고가치 데이터셋 행동을 선택적으로 복제하며, 표준 BRAC 방식보다 더 표적화된 정규화를 제공한다. 증류된 액터는 BPTT와 반복 샘플링을 피하면서 크리틱을 최대화한다.

**④ Hierarchical Implicit Flow Q-Learning** (arXiv:2604.08960, 2026)

오프라인 목표 조건부 강화학습(GCRL)은 보상 없는 오프라인 데이터로부터 목표 조건부 정책을 학습하는 것을 목표로 한다. HIQL과 같은 계층적 아키텍처의 최근 발전에도 불구하고, Gaussian 정책의 제한된 표현력과 상위 레벨 정책의 효과적인 서브목표 생성 불능으로 인해 오프라인 GCRL의 장기 제어는 여전히 도전적이다.

**⑤ FlowQ (Energy-Guided)** (arXiv:2505.14139, 2025)

FlowQ는 학습 중 모델로부터의 행동 샘플링 필요성을 최소화하여 생성 스텝 수에 따라 확장 가능하게 한다. 이는 훈련 중 가이던스를 통합하여 $\pi_\theta(a|s) \propto \pi_\beta(a|s) \exp(Q(s,a))$ 형태의 정책을 직접 학습하는 아이디어에 기반한다.

**⑥ MeanFlow 기반 1-step 정책** (arXiv:2511.13035, 2025)

이 방법은 복잡한 다이나믹스와 멀티모달 행동 분포를 필요로 하는 환경에서 강력한 성능을 달성한다. flow 기반 기준선인 FQL, IFQL, FBRAC와 비교하여, 더 단순한 단일 단계 훈련 파이프라인과 1-step 추론 메커니즘을 유지하면서 성능을 맞추거나 능가한다.

---

## 9. 앞으로의 연구에 미치는 영향 및 고려할 점

### 9-1. 연구에 미치는 영향

1. **Flow Matching의 RL 표준 기법화**: FQL은 오프라인 및 오프라인-투-온라인 태스크에서 강건한 성능으로 알려진 최첨단 방법으로, flow Q-learning 프레임워크에 통합되어 활발히 연장·발전되고 있다.

2. **1-step 증류 패러다임의 정착**: FQL은 전문가 시연으로부터의 학습을 actor-critic 구조에 통합하기 위해 도입되었으며, 핵심은 flow 기반 생성 모델을 사용하여 더 표현력 있는 정책을 학습할 수 있도록 Q-함수로 최적화되는 "1-step 정책" 네트워크이다.

3. **수렴성 이론 연구의 촉진**: FQL의 1-step 정책의 수렴 특성과 안정성이 선형 이차 문제에서의 오프라인 설정 하에서 연구되었다. 이 이론적 결과들은 평균 기대 비용 기반의 1-step 정책 손실에 대한 새로운 공식화에 기반하며, 정책 그래디언트 정리의 강력한 이론적 결과를 활용하여 수렴 특성을 연구할 수 있게 한다.

4. **로보틱스 및 자율주행으로의 확장 가능성**: 로보틱스에서 전문가 시연은 로봇이 물체 파지나 환경 내 탐색 등 복잡한 태스크를 수행하는 방법을 가르치는 데 사용될 수 있다. 자율주행에서 전문가 궤적은 자율주행차가 복잡한 교통 시나리오에서 안전하고 효율적으로 주행하도록 훈련하는 데 사용될 수 있다.

### 9-2. 앞으로 연구 시 고려할 점

| 연구 방향 | 세부 내용 |
|-----------|----------|
| **① 고품질 행동 선택적 학습** | 낮은 온도에서 필터링이 매우 선택적이 되어 크리틱에 따른 최고 가치 행동에 거의 독점적으로 집중하지만, 매우 낮은 온도에서의 과도한 집중은 액터가 데이터셋의 행동 분포를 벗어나게 만들어 크리틱 과대추정과 분포 이탈(OOD) 문제로 이어질 수 있다. 따라서 데이터 품질 필터링과 분포 제약 간의 균형이 핵심 연구 과제이다. |
| **② 인과적 강건성 확보** | 교란 편향으로 인한 최악 성능 최적화를 위한 인과적 오프라인 RL 목적 함수를 개발해야 한다. 픽셀 기반 시나리오에서의 인과적 구조 학습이 중요하다. |
| **③ 적응적 에너지 스케일링** | 향후 연구는 이 접근법을 더 넓은 응용으로 확장하고 성능을 더욱 향상시키기 위해 적응적 에너지 스케일링 메커니즘을 조사할 수 있다. |
| **④ 계층적 구조와의 통합** | 목표 조건부 평균 flow 정책은 오프라인 GCRL을 위한 계층적 정책 모델링에 평균 속도 필드를 도입하여, 고수준 및 저수준 정책 모두에서 복잡한 타겟 분포를 학습된 평균 속도 필드를 통해 포착하고 효율적인 행동 생성을 가능하게 한다. |
| **⑤ 이론적 수렴 보장** | 평균 기대 비용 기반의 1-step 정책 손실 공식화는 정책 그래디언트 정리의 강력한 기존 이론적 결과를 활용하여 수렴 특성 연구를 가능하게 한다. 비선형/연속 환경으로의 이론적 확장 연구가 필요하다. |
| **⑥ 온라인 RL로의 확장** | CtrlFlow와 같이 초기 상태에서 고수익 최종 상태까지의 궤적 분포를 직접 모델링하여, 생성된 다양한 궤적 데이터가 정책 학습의 강건성 및 태스크 간 일반화를 크게 향상시킨다. |

---

## 참고 자료 및 출처

| 번호 | 제목 | 출처 |
|------|------|------|
| 1 | **Flow Q-Learning** (FQL 원문) | arXiv:2502.02538, ICML 2025. Park, Li, Levine (UC Berkeley) |
| 2 | **FQL Project Page** | https://seohong.me/projects/fql/ |
| 3 | **OpenReview - Flow Q-Learning** | https://openreview.net/forum?id=KVf2SFL1pi |
| 4 | **One-Step Flow Q-Learning (OFQL)** | arXiv:2508.13904, Nguyen et al. (2025–2026) |
| 5 | **Causal Flow Q-Learning** | arXiv:2602.02847, Li et al. (2026) |
| 6 | **Guided Flow Policy (GFP)** | arXiv:2512.03973 (2025) |
| 7 | **Efficient Hierarchical Implicit Flow Q-Learning** | arXiv:2604.08960, Dong et al. (2026) |
| 8 | **FlowQ: Energy-Guided Flow Policies** | arXiv:2505.14139 (2025) |
| 9 | **One-Step Generative Policies with Q-Learning (MeanFlow)** | arXiv:2511.13035 (2025) |
| 10 | **Convergence of Flow-Policy Gradient Learning** | arXiv:2511.11131, Adib Yaghmaie & Naha (Linköping Univ., 2025) |
| 11 | **Controllable Flow Matching for Online RL** | arXiv:2511.06816 (2025) |
| 12 | **Unleashing Flow Policies with Distributional Critics** | arXiv:2509.23087 (2025) |
| 13 | **Berkeley CS285 Homework 5** | https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5.pdf |

> ⚠️ **정확성 고지**: 본 답변에서 FQL의 수식 일부(특히 손실 함수의 정확한 형태)는 공개된 논문 PDF 및 Berkeley 강의자료, 프로젝트 페이지를 기반으로 재구성하였습니다. 논문 원문(arXiv:2502.02538)의 Section 3~4를 직접 참조하여 세부 수식을 확인하시길 권장합니다.
