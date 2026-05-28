
# Beyond Mode Collapse: Distribution Matching for Diverse Reasoning (DMPO)

> **논문 정보**
> - **제목:** Beyond Mode Collapse: Distribution Matching for Diverse Reasoning
> - **저자:** Xiaozhe Li et al. (13인 공저)
> - **arXiv ID:** [2605.19461](https://arxiv.org/abs/2605.19461)
> - **발표:** ICML 2026
> - **공식 코드:** [GitHub - OliverLeeXZ/DMPO](https://github.com/OliverLeeXZ/DMPO)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

GRPO와 같은 on-policy 강화학습 방법은 **모드 붕괴(mode collapse)** 문제를 겪는다. 이는 한 번 고보상 궤적을 발견하면 해당 해에 확률 질량을 집중시키고 대안적 전략 탐색을 중단하는 것으로 나타나며, 이 현상은 **역방향 KL 최소화(reverse KL minimization)** 의 mode-seeking 특성에서 기인한다.

### 주요 기여

이를 해결하기 위해 **DMPO(Distribution-Matching Policy Optimization)** 를 제안하며, 이는 **순방향 KL 최소화(forward KL minimization)** 의 원리적 근사를 통해 모드 붕괴를 방지한다. DMPO는 그룹 수준에서 샘플링된 궤적들에 대해 보상에 비례한 타깃 분포를 구성하고, 정책 분포를 이 타깃에 정렬함으로써 전역 타깃 분포를 직접 샘플링하지 않아도 **mode-covering** 동작을 실현하고, 훈련 전반에 걸쳐 지속적인 탐색을 가능하게 한다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

GRPO와 같은 on-policy RL 방법은 **모드 붕괴** 문제를 겪는다. 구체적으로, 하나의 해를 발견하는 순간 해당 해에 확률 질량을 집중하고 다양한 대안적 전략 탐색을 멈춘다.

이는 역방향 KL 최소화의 **mode-seeking** 특성에서 비롯되는데, 처음 발견한 고보상 궤적을 강화하는 방향으로만 업데이트되어 다양한 해의 분포를 유지하지 못한다.

### 2-2. 제안 방법 및 핵심 수식

#### GRPO의 문제점: 역방향 KL

GRPO가 암묵적으로 최소화하는 목표는 **역방향 KL divergence** 이다.

```math
\mathcal{L}_{\text{reverse KL}} = D_{\text{KL}}(\pi_\theta \| p^*) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \log \frac{\pi_\theta(\tau)}{p^*(\tau)} \right]
```

여기서 $p^\*(\tau) \propto r(\tau)$는 보상에 비례한 이상적 분포이다. 이 방향의 KL은 **mode-seeking** 특성을 가지며, $\pi_\theta$가 0에 가까운 곳에서 $p^*$가 크더라도 패널티가 없어 다양한 고보상 모드를 무시한다.

#### DMPO의 핵심: 순방향 KL

DMPO는 **순방향 KL 최소화**를 근사하여 mode collapse를 완화한다. 순방향 KL을 전역적으로 직접 최적화하는 것은 전역 보상 가중 분포에서의 샘플링이 필요하므로 불가능(intractable)하다. DMPO는 **그룹 수준**으로 이 문제를 우회하여, 각 그룹의 샘플링된 궤적에 대해 보상에 비례한 타깃 분포(**Boltzmann 분포**)를 구성하고, 정책 분포를 이 타깃에 명시적으로 정렬한다.

순방향 KL 목표:

```math
\mathcal{L}_{\text{forward KL}} = D_{\text{KL}}(p^* \| \pi_\theta) = \mathbb{E}_{\tau \sim p^*} \left[ \log \frac{p^*(\tau)}{\pi_\theta(\tau)} \right]
```

이 방향은 **mode-covering** 특성을 가져 $p^\*$가 큰 모든 영역을 커버하도록 강제한다.

#### 그룹 수준 Boltzmann 타깃 분포

그룹 내 $G$개의 궤적 $\{\tau_1, \tau_2, \ldots, \tau_G\}$에 대해 타깃 분포를 아래와 같이 정의한다.

$$\tilde{p}(\tau_i) = \frac{\exp(r(\tau_i) / \beta)}{\sum_{j=1}^{G} \exp(r(\tau_j) / \beta)}$$

여기서 $r(\tau_i)$는 궤적 $\tau_i$의 보상, $\beta$는 온도 파라미터이다.

#### DMPO의 최종 목적함수

각 그룹의 타깃 분포를 구성한 뒤, 정책 분포를 이 타깃과 명시적으로 정렬한다. 이 공식화는 GRPO의 그룹 기반 어드밴티지 정규화와 자연스럽게 호환되며, **분포 매칭 정규화 항 하나만을 표준 목적함수에 추가**하는 방식으로 구현된다. 결과적으로, DMPO는 다수의 고보상 모드를 보존하고 훈련 전반에 걸쳐 탐색을 지속한다.

DMPO 목적함수:

$$\mathcal{L}_{\text{DMPO}} = \mathcal{L}_{\text{GRPO}} + \lambda \cdot D_{\text{KL}}(\tilde{p} \| \pi_\theta)$$

여기서 $\lambda$는 분포 매칭 정규화 강도, $\tilde{p}$는 그룹 수준 Boltzmann 타깃 분포이다.

### 2-3. 모델 구조 및 실험 환경

DMPO의 효과를 검증하기 위해, 지수적으로 많은 실현 가능한 해가 존재하지만 소수만이 최적에 가까운 **NP-hard 조합 최적화** 를 테스트베드로 활용한다. 이는 탐색 능력을 평가하기에 이상적인 환경이다.

모델 스케일 실험에서 7B에서 72B 파라미터로 모델 크기를 키우면 QR(Quality Ratio)보다 SR(Success Rate)에서 더 큰 개선(상대적으로 15% vs 8%)이 관찰되며, 이는 큰 모델이 최적화 전략보다 제약 만족을 더 쉽게 학습함을 시사한다.

### 2-4. 성능 향상

NP-hard 조합 최적화에서 DMPO를 검증했으며, 텍스트 기반 NP-Bench에서 Quality Ratio **43.9%** (vs. GRPO의 40.1%), 비전 기반 NP-Bench에서 **43.1%** (vs. 38.4%)를 달성하여 각각 **9%, 12%** 의 상대적 향상을 보인다.

또한 DMPO는 5개의 강력한 베이스라인 대비 최적화 태스크에서 **4.7%–3.8%**, 수학적 추론에서 **2%**, 도메인 외(out-of-domain) 태스크에서 **2.3%** 성능을 상회한다.

### 2-5. 한계

논문 원문에서 명시된 한계에 대한 직접적인 인용 문구를 검색 결과 내에서 확인하기 어렵습니다. 다만, 검색된 내용에 기반하여 추론 가능한 구조적 한계는 다음과 같습니다.

- 순방향 KL의 직접 최적화는 전역 보상 가중 분포에서의 샘플링이 필요하므로 **intractable**하며, DMPO는 이를 그룹 수준에서만 근사한다. 따라서 전역 다양성 보장은 제한적일 수 있다.
- 모델 스케일이 커질수록(7B→72B) Quality Ratio보다 Success Rate의 향상이 두드러져, **대형 모델에서 최적화 전략 다양성 확보에 한계**가 있음이 시사된다.
- 그룹 수준 Boltzmann 분포에서 온도 파라미터 $\beta$의 설정이 성능에 영향을 미칠 수 있으나, 최적 $\beta$ 선택 기준에 대한 명확한 지침은 확인되지 않았습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

DMPO의 이점은 조합 최적화에 국한되지 않는다. DMPO는 수학적 추론에서 **+2.0%**, 도메인 외 태스크에서 **+2.3%** 향상을 달성하며, 이는 **다양성을 보존하는 훈련이 더 강인하고 전이 가능한 추론 능력**을 산출함을 시사한다.

이러한 성과는 수학적 추론과 도메인 외 태스크에도 일반화되며, **다양성을 보존하는 훈련이 모달리티 전반에 걸쳐 일반적인 추론 능력을 향상**시킨다는 것을 보여준다.

즉, 아래와 같은 일반화 메커니즘이 제안된다:

$$\text{훈련 다양성} \uparrow \Rightarrow \text{솔루션 공간 커버리지} \uparrow \Rightarrow \text{도메인 외 일반화} \uparrow$$

또한, DMPO는 텍스트와 비전 등 **모달리티에 무관한(modality-agnostic)** 접근임이 실험적으로 확인된다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 핵심 아이디어 | KL 방향 | 한계 |
|---|---|---|---|
| **GRPO** (DeepSeek, 2025) | 그룹 상대 어드밴티지 | Reverse KL | Mode collapse |
| **DMPO** (본 논문, 2026) | 그룹 Boltzmann 타깃 정렬 | Forward KL 근사 | 전역 근사 한계 |
| **FlowRL** (2025) | Flow balance 기반 보상 분포 매칭 | - | Partition function 학습 필요 |
| **LaDi-RL** (2026) | 연속 잠재공간에서 확산 기반 탐색 | - | 고계산 비용 |
| **DGPO** (2026) | 토큰 수준 어드밴티지 재분배 | Hellinger 거리 | 세밀한 신용 할당 복잡도 |

**FlowRL**은 flow balance를 통해 보상 최대화에서 보상 분포 매칭으로 전환하는 정책 최적화 알고리즘으로, GRPO 대비 수학 벤치마크에서 10.0%, PPO 대비 5.1% 성능 향상을 달성하며 코드 추론 태스크에서도 강한 일반화를 보인다.

**LaDi-RL**은 이산 토큰 공간의 엔트로피 붕괴 문제를 해결하기 위해 연속 잠재공간에서 직접 탐색을 수행하는 프레임워크로, 다단계 디노이징을 통해 확률론적 다양성을 분산시키고 여러 공존 솔루션 모드를 유지한다.

DMPO의 그룹 수준 Boltzmann 타깃은 FlowRL의 학습된 partition function보다 **지역 다양성을 더 잘 유지**하는 것으로 나타난다.

**DGPO**는 거칠은 시퀀스 수준 어드밴티지를 세밀한 토큰 수준 업데이트 신호로 재분배하는 방식을 제안하며, 참조 분포로부터의 편차를 오류가 아닌 탐색을 위한 안내 신호로 재해석한다.

**"Reaching Beyond the Mode"** 연구는 단일 순방향 패스에서 여러 후보 답변을 생성하도록 RL 목표를 수정하여, 질문-응답, 의료 진단, 코딩 벤치마크에서 다양성, 커버리지, 집합 수준 캘리브레이션 점수의 향상을 관찰한다.

---

## 5. 향후 연구에 미치는 영향과 연구 시 고려할 점

### 5-1. 향후 연구에 미치는 영향

1. **RL 목적함수 설계 패러다임의 전환**
   본 연구는 **분포 매칭(distribution matching)** 을 on-policy RL에서 mode collapse를 방지하는 실용적이고 원리적인 접근법으로 확립하며, 지속적 탐색을 통한 일관적 품질 향상을 입증한다.

2. **멀티모달 추론으로의 확장 가능성**
   DMPO가 텍스트 및 비전 양쪽에서 모달리티에 무관한 성능 향상을 보임에 따라, **멀티모달 LLM 훈련**에서의 탐색 다양성 연구로 확장될 수 있다.

3. **조합 최적화와 LLM의 융합 가능성**
   분포 매칭이 조기 수렴을 효과적으로 방지하여 mode-seeking 목표가 발견하지 못하는 고품질 솔루션을 찾아냄을 입증함으로써, LLM 기반 조합 최적화 연구에 새로운 방향을 제시한다.

4. **표준 RL 파이프라인에의 통합 용이성**
   DMPO는 **단일 정규화 항 추가**만으로 구현 가능하므로, 기존 GRPO 기반 파이프라인에 쉽게 통합할 수 있어 후속 연구의 기반 방법론으로 자리잡을 가능성이 높다.

### 5-2. 향후 연구 시 고려할 점

1. **온도 파라미터 $\beta$의 적응적 스케줄링**: 그룹 Boltzmann 분포의 온도 $\beta$는 훈련 단계, 태스크 난이도에 따라 동적으로 조정될 필요가 있다.

2. **대규모 모델에서의 다양성-품질 트레이드오프**:
   모델 크기가 7B에서 72B로 증가하면 SR 향상이 QR보다 크게(15% vs 8%) 나타남에 따라, 대형 모델에서 다양성과 품질 간 균형을 맞추는 새로운 전략이 필요하다.

3. **전역 vs. 그룹 수준 근사 간 간극 해소**: 그룹 내 샘플만으로 전역 Boltzmann 분포를 근사하므로, 그룹 크기 $G$와 샘플 효율성 간의 관계를 심층적으로 분석해야 한다.

4. **오프라인/오프-폴리시 확장 가능성**: 현재 DMPO는 on-policy 방식으로 전역 타깃 분포 샘플링 없이 동작하므로, 오프-폴리시 버퍼와의 결합을 통한 샘플 효율성 향상이 향후 중요한 연구 방향이다.

5. **다양성 지표의 표준화**: Quality Ratio(QR), Success Rate(SR) 등 태스크별 지표 외에 모델의 탐색 다양성을 정량화하는 **범용 지표** 개발이 후속 연구에서 중요하다.

---

## 참고 자료

| 번호 | 출처 |
|---|---|
| 1 | **[주논문]** Xiaozhe Li et al., "Beyond Mode Collapse: Distribution Matching for Diverse Reasoning," arXiv:2605.19461 (ICML 2026). https://arxiv.org/abs/2605.19461 |
| 2 | **[공식 코드]** GitHub - OliverLeeXZ/DMPO. https://github.com/OliverLeeXZ/DMPO |
| 3 | **[관련 연구]** Isha Puri et al., "Reaching Beyond the Mode: RL for Distributional Reasoning in Language Models," arXiv:2603.24844 (2026). https://arxiv.org/abs/2603.24844 |
| 4 | **[관련 연구]** "Beyond Mode Elicitation: Diversity-Preserving Reinforcement Learning via Latent Diffusion Reasoner (LaDi-RL)," arXiv:2602.01705 (2026). https://arxiv.org/html/2602.01705 |
| 5 | **[관련 연구]** "FlowRL: Matching Reward Distributions for LLM Reasoning," arXiv:2509.15207 (2025). https://huggingface.co/papers/2509.15207 |
| 6 | **[관련 연구]** "DGPO: Distribution-Guided Policy Optimization for Fine-Grained Credit Assignment," arXiv:2605.03327 (2026). https://arxiv.org/html/2605.03327 |
| 7 | **[관련 연구]** "Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization," arXiv:2510.08233 (2026). https://arxiv.org/abs/2510.08233 |

> ⚠️ **정확도 주의사항:** 본 논문(arXiv:2605.19461)은 2026년 5월에 공개된 최신 논문으로, 검색을 통해 확인된 정보만을 기반으로 작성하였습니다. 수식 중 DMPO의 최종 목적함수는 논문의 전체 원문에서 직접 확인된 것이 아니라 논문의 핵심 설명(forward KL 근사, 그룹 Boltzmann 타깃, 정규화 항 추가)을 바탕으로 표준적인 방식으로 재구성한 것임을 밝힙니다. 정확한 수식은 원문 PDF를 직접 참조하시기 바랍니다.
