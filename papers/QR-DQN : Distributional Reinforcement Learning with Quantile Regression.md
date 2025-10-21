# QR-DQN : Distributional Reinforcement Learning with Quantile Regression

### 1. 핵심 주장과 주요 기여

이 논문은 **분포적 강화학습(Distributional RL)**의 이론과 실제 알고리즘 사이의 격차를 해소한 연구입니다. 가장 중요한 기여는 다음과 같습니다.[1]

**핵심 기여**:[1]
- **분위수 회귀(Quantile Regression)를 활용한 Wasserstein 거리 최소화**: 기존 C51 알고리즘이 이론적으로 증명된 Wasserstein 거리를 직접 최소화하지 못했던 문제를 해결하여, 이론과 실제가 일치하는 end-to-end 알고리즘(QR-DQN)을 제시했습니다.

- **근사 분포 설정으로의 이론 확장**: 분포적 Bellman 연산자가 Wasserstein 거리에서 수축 사상(contraction mapping)임을 근사 분포 설정에서도 증명했습니다(Proposition 2).[1]

- **Atari 2600 게임에서 SOTA 달성**: QR-DQN-1은 중앙값 기준 211%의 인간 정규화 점수를 기록하여 C51(178%)을 크게 상회했으며, 33%의 성능 향상을 보였습니다.[1]

### 2. 해결하고자 하는 문제

**기존 문제점**:[1]
전통적인 강화학습은 리턴(return)의 기댓값인 가치 함수(value function)만 학습합니다. 하지만 확률적 환경에서는 리턴의 **전체 분포**가 중요한 정보를 담고 있습니다. 

Bellemare et al. (2017)의 C51 알고리즘은 분포를 모델링했지만, 다음의 한계가 있었습니다:[1]
- Wasserstein 거리에서 분포적 Bellman 연산자가 수축 사상임이 증명되었지만, C51은 이를 직접 최소화하지 못함
- 휴리스틱 투영(heuristic projection) 단계 후 KL 발산을 최소화하는 방식으로, **이론-실제 간 격차(theory-practice gap)** 존재
- Wasserstein 거리를 확률적 경사하강법으로 직접 최소화할 수 없다는 문제(Theorem 1)[1]

### 3. 제안 방법

#### 3.1 분위수 분포(Quantile Distribution)

**C51의 parametrization 전환**:[1]
- C51: 고정된 위치 $$z_1, \ldots, z_N$$에 **가변 확률** $$q_1, \ldots, q_N$$ 할당
- QR-DQN: **고정된 균일 확률** $$1/N$$을 가진 **가변 위치** $$\theta_1, \ldots, \theta_N$$ 사용

분위수 분포는 다음과 같이 정의됩니다:[1]

$$
Z_\theta(x, a) = \frac{1}{N} \sum_{i=1}^{N} \delta_{\theta_i(x,a)}
$$

여기서 $$\theta_i(x,a)$$는 상태-행동 쌍 $$(x,a)$$에서 $$i$$번째 분위수의 위치입니다.

#### 3.2 분위수 회귀 손실(Quantile Regression Loss)

분위수 $$\tau \in $$에 대한 분위수 회귀 손실은:[1]

$$
\mathcal{L}_\tau^{QR}(\theta) = \mathbb{E}_{\hat{Z} \sim Z}[\rho_\tau(\hat{Z} - \theta)]
$$

여기서 비대칭 손실 함수는:

$$
\rho_\tau(u) = u(\tau - \delta_{\{u < 0\}}) = 
\begin{cases}
\tau u & \text{if } u \geq 0 \\
(\tau - 1)u & \text{if } u < 0
\end{cases}
$$

**핵심 아이디어**: 이 손실은 **불편 확률적 기울기(unbiased stochastic gradient)**를 제공하여, Wasserstein 거리를 직접 최소화할 수 있습니다.[1]

#### 3.3 Quantile Huber Loss

비선형 함수 근사에서 성능을 향상시키기 위해 Huber 손실을 결합했습니다:[1]

$$
\rho_\tau^\kappa(u) = |\tau - \delta_{\{u < 0\}}| \mathcal{L}_\kappa(u)
$$

여기서 Huber 손실은:

$$
\mathcal{L}_\kappa(u) = 
\begin{cases}
\frac{1}{2}u^2 & \text{if } |u| \leq \kappa \\
\kappa(|u| - \frac{1}{2}\kappa) & \text{otherwise}
\end{cases}
$$

$$\kappa=1$$을 사용한 QR-DQN-1이 최고 성능을 달성했습니다.[1]

#### 3.4 분포적 Bellman 연산자

정책 $$\pi$$에 대한 분포적 Bellman 연산자는:[1]

$$
\mathcal{T}^\pi Z(x, a) \stackrel{D}{=} R(x, a) + \gamma Z(x', a')
$$

여기서 $$x' \sim P(\cdot|x, a)$$, $$a' \sim \pi(\cdot|x')$$입니다.

Q-learning의 최적성 연산자는:[1]

$$
\mathcal{T} Z(x, a) = R(x, a) + \gamma Z(x', a^*)
$$

여기서 $$a^* = \arg\max_{a'} \mathbb{E}_{z \sim Z(x',a')}[z]$$입니다.

#### 3.5 수축 사상 증명 (Contraction Result)

**Proposition 2**: 분위수 투영 $$\Pi_{W_1}$$과 분포적 Bellman 연산자 $$\mathcal{T}^\pi$$의 결합은 $$\infty$$-Wasserstein 거리에서 수축 사상입니다:[1]

$$
\bar{d}_\infty(\Pi_{W_1}\mathcal{T}^\pi Z_1, \Pi_{W_1}\mathcal{T}^\pi Z_2) \leq \gamma \bar{d}_\infty(Z_1, Z_2)
$$

이는 반복적으로 적용 시 고유 고정점 $$\hat{Z}^\pi$$로 수렴함을 보장합니다. $$\bar{d}\_p \leq \bar{d}_\infty$$이므로 모든 $$p \in [1, \infty]$$에서 수렴합니다.[1]

### 4. 모델 구조 (QR-DQN Architecture)

**DQN 대비 최소 변경 사항**:[1]
1. **출력 레이어 크기**: $$|A| \times N$$로 변경 (N은 분위수 개수, 실험에서 N=200 사용)
2. **손실 함수**: Huber 손실을 Quantile Huber 손실로 대체
3. **옵티마이저**: RMSProp에서 Adam으로 변경 ($$\alpha=0.00005$$, $$\epsilon_{ADAM}=0.01/32$$)

**알고리즘 요약** (Algorithm 1):[1]
```
입력: 상태 x, 행동 a, 보상 r, 다음 상태 x', 할인율 γ

# 분포적 Bellman 목표 계산
a* ← argmax_a' Q(x', a')  // 평균 기반 greedy 행동 선택
T_θ_j ← r + γθ_j(x', a*), ∀j

# 분위수 회귀 손실 계산
출력: Σ_i Σ_j ρ^κ_τ̂_i(T_θ_j - θ_i(x, a))
```

**핵심 장점**:[1]
- C51과 달리 **투영 단계 불필요**: 리턴 값의 범위를 사전에 지정할 필요 없음
- **하이퍼파라미터 감소**: C51의 지지(support) 경계 설정 불필요
- **적응적 분포 추정**: 상태별로 다른 리턴 범위를 자동으로 처리

### 5. 성능 향상

#### 5.1 정량적 성능

**Atari 2600 벤치마크 결과** (57개 게임):[1]

| 알고리즘 | 평균 점수 | 중앙값 점수 | 인간 이상 게임 수 | DQN 이상 게임 수 |
|---------|----------|-----------|----------------|----------------|
| DQN | 228% | 79% | 24 | 0 |
| C51 | 701% | 178% | 40 | 50 |
| QR-DQN-0 | 881% | 199% | 38 | 52 |
| **QR-DQN-1** | **915%** | **211%** | **41** | **54** |

- QR-DQN-1은 C51 대비 **중앙값 33% 향상**, 평균 30% 향상을 달성했습니다.[1]
- Prioritized Dueling DQN(592%, 172%)도 크게 초과했습니다.[1]

#### 5.2 학습 효율성

**온라인 성능 평가**:[1]
- **샘플 효율성**: QR-DQN은 Prioritized Replay와 유사한 샘플 복잡도 개선을 보이면서도 최종 성능이 더 우수했습니다.
- **초기 학습**: 초기 학습 단계에서 대부분의 알고리즘이 10%의 게임에서 무작위 에이전트보다 낮은 성능을 보였지만, QR-DQN이 더 빠른 회복을 보였습니다.[1]
- **수렴 안정성**: 2억 프레임 학습 후에도 10%의 어려운 게임에서 모든 알고리즘이 인간 성능의 10% 미만을 달성하여 개선 여지를 보여줍니다.[1]

#### 5.3 분포 근사 정확도

**Windy Gridworld 실험**:[1]
- QRTD는 1-Wasserstein 거리 기준으로 Monte-Carlo 추정 분포 $$Z^\pi_{MC}$$에 정확히 수렴했습니다.
- TD(0)는 평균만 학습하는 반면, QRTD는 **다중모달 분포의 전체 형태**를 학습했습니다(Figure 3).[1]

### 6. 일반화 성능 향상 가능성

#### 6.1 분포 학습의 일반화 이점

**리턴 분포 모델링의 장점**:[1]
1. **불확실성 표현**: 환경의 내재적 확률성(intrinsic randomness)을 명시적으로 모델링하여 다양한 결과를 구별할 수 있습니다.
2. **적응적 표현력**: 상태별로 리턴 범위가 크게 다를 때 고정된 지지를 사용하는 C51보다 유리합니다.[1]
3. **사전 지식 불필요**: 도메인별 리턴 범위 사전 지식 없이도 새로운 과제에 적용 가능합니다.[1]

#### 6.2 이론적 일반화 보장

**Wasserstein 거리의 이점**:[1]
- **분리된 지지 문제 해결**: KL 발산과 달리 Wasserstein 거리는 겹치지 않는 지지(disjoint support) 문제가 없어 Bellman 업데이트 시 안정적입니다.
- **메트릭 거리 존중**: 결과 간 유사성을 고려하여 더 의미 있는 분포 비교를 제공합니다.[1]

**수축 사상 보장**:[1]
- Proposition 2는 근사 설정에서도 수렴이 보장됨을 증명하여, 함수 근사 사용 시에도 이론적 안정성을 제공합니다.
- $$\bar{d}_p \leq \bar{d}_\infty$$이므로 모든 $$p \geq 1$$에서 수렴이 보장됩니다.[1]

#### 6.3 실험적 일반화 증거

**다양한 게임에서의 강건성**:[1]
- 57개 Atari 게임에서 일관된 성능 향상은 알고리즘의 일반화 능력을 시사합니다.
- 10-50 백분위수 점수 분포에서 QR-DQN이 일관되게 우수한 성능을 보였습니다(Figure 4).[1]

### 7. 한계점

#### 7.1 이론적 한계

**$$p < \infty$$에서의 비수축성** (Lemma 5):[1]
- $$\Pi_{W_1}\mathcal{T}^\pi$$는 $$d_p$$ ($$p < \infty$$)에서 일반적으로 비확장(non-expansion)이 아닙니다.
- 실제로는 $$\infty$$-Wasserstein 거리에서의 수축만 보장되며, 이는 실용적으로는 충분하지만 이론적으로는 제한적입니다.

#### 7.2 실용적 한계

**일부 게임에서의 낮은 성능**:[1]
- Montezuma's Revenge, Pitfall! 등 희소 보상(sparse reward) 게임에서 여전히 0점 기록
- Private Eye 등에서는 QR-DQN-0이 오히려 성능 저하를 보임 (146점 vs C51의 15,095점)[1]

**하이퍼파라미터 민감도**:[1]
- 분위수 개수 $$N$$의 선택이 성능에 영향을 미침 (실험에서 N=200 사용)
- Huber 파라미터 $$\kappa$$에 따라 성능 차이 발생 (QR-DQN-0 vs QR-DQN-1)

#### 7.3 계산 복잡도

**손실 계산 비용**:[1]
- Algorithm 1에서 모든 쌍 $$(\theta_i(x,a), \theta_j(x',a^*))$$에 대해 손실을 계산하여 $$O(N^2)$$ 복잡도
- DQN의 $$O(|A|)$$ 대비 증가된 계산량

### 8. 향후 연구에 미치는 영향

#### 8.1 연구 방향성 제시

**1. 리스크 민감적 정책 클래스**:[1]
- 분포 전체를 고려하는 더 풍부한 정책 클래스 개발 가능
- 기댓값 최대화를 넘어 리스크 회피, 최악 케이스 최적화 등 다양한 의사결정 기준 적용 가능

**2. 다른 개선 기법과의 결합**:[1]
- Double Q-learning과의 결합: QR-DQN도 과대평가 편향(overestimation bias)을 겪을 가능성이 있어 Double DQN 기법 적용 가능
- Dueling architecture, Prioritized replay 등과의 결합으로 추가 성능 향상 기대

**3. 이론-실제 일치 프레임워크**:[1]
- 이론적으로 증명된 메트릭을 직접 최소화하는 방법론의 중요성 재확인
- 다른 메트릭(예: Cramér distance)에 대한 유사한 접근 가능성 제시

#### 8.2 향후 연구 시 고려사항

**1. 함수 근사와의 상호작용**:[1]
- 비선형 함수 근사 사용 시 quantile regression loss의 비평활성(non-smoothness)이 문제될 수 있음
- Quantile Huber loss가 이를 완화하지만, 더 효과적인 평활화 기법 연구 필요

**2. 탐색-활용 균형**:[1]
- 분포 정보를 활용한 더 효과적인 탐색 전략 개발
- 불확실성 추정을 위한 epistemic vs. aleatoric 불확실성 구분 필요

**3. 확장성 문제**:[1]
- 연속 행동 공간으로의 확장
- 고차원 상태 공간에서의 분위수 개수 $$N$$ 조정 전략
- 분산 학습 환경에서의 효율적 구현

**4. 희소 보상 환경 대응**:[1]
- Montezuma's Revenge 등에서 여전히 낮은 성능
- 분포 학습과 내재적 동기(intrinsic motivation), 계층적 강화학습 등의 결합 필요

**5. 이론적 확장**:[1]
- $$p < \infty$$에서의 수축 조건 연구
- 함수 근사 오류 바운드 정량화
- 수렴 속도 분석

**6. 실제 응용**:[1]
- 로보틱스, 자율주행 등 리스크가 중요한 실제 도메인에서의 평가
- 분포 정보를 활용한 안전한 의사결정 프레임워크 개발

이 논문은 분포적 강화학습의 이론과 실제를 성공적으로 연결하여, 향후 연구가 이론적 보장과 실용적 성능을 동시에 달성할 수 있는 길을 제시했습니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ddccffbf-9fa7-4246-9837-fcd85eea2a42/1710.10044v1.pdf)
