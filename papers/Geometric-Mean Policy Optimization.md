# Geometric-Mean Policy Optimization (GMPO)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

GMPO는 GRPO(Group Relative Policy Optimization)의 **산술 평균(Arithmetic Mean)** 기반 토큰 수준 보상 최적화를 **기하 평균(Geometric Mean)** 기반으로 대체함으로써, 이상치(outlier) 중요도 가중 보상으로 인한 **불안정한 정책 업데이트 문제를 근본적으로 해결**할 수 있다는 것입니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **알고리즘 제안** | 기하 평균 기반 GMPO: 플러그 앤 플레이 방식으로 GRPO를 대체 |
| **이론적 분석** | 좁은 목적함수 범위, 안정적 그래디언트, 낮은 KL 발산 증명 |
| **실험적 검증** | GMPO-7B가 GRPO 대비 평균 Pass@1 최대 **4.1%** 향상 |
| **범용성** | 언어 전용 + 멀티모달 + MoE 모델 모두에서 성능 향상 확인 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

GRPO 훈련 중 각 토큰의 중요도 가중 보상은 다음과 같이 정의됩니다:

$$
\rho_{i,t}(\theta) \hat{A}_i, \quad \text{where} \quad \rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_\text{old}}(o_{i,t}|q, o_{i,<t})}
$$

GRPO는 이 값의 **산술 평균**을 최적화하는데, 산술 평균은 이상치에 민감합니다. 훈련이 진행될수록 $\rho_{i,t}(\theta)$가 극단값에 도달하여:

- **과도하게 공격적인 정책 업데이트** 발생
- **엔트로피 붕괴(Entropy Collapse)** → 탐색 능력 저하
- **KL 발산 급증** → 참조 모델로부터 과도한 이탈
- **성능 정체 또는 붕괴**

### 2.2 제안 방법 및 수식

#### GRPO 목적함수 (기존)

$$
\mathcal{J}^*_\text{GRPO}(\pi_\theta) = \mathbb{E}_{q \sim \mathcal{Q}, \{o_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \rho_{i,t}(\theta) \hat{A}_i \right] \tag{1}
$$

여기서 정규화 이점은 다음과 같습니다:

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, r_2, \cdots, r_G\})}{\text{std}(\{r_1, r_2, \cdots, r_G\})}
$$

#### GMPO 목적함수 (제안)

$$
\mathcal{J}^*_\text{GMPO}(\pi_\theta) = \mathbb{E}_{q \sim \mathcal{Q}, \{o_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \prod_{t=1}^{|o_i|} \left| \rho_{i,t}(\theta) \hat{A}_i \right| \right)^{\frac{1}{|o_i|}} \cdot \text{sgn}(\hat{A}_i) \right] \tag{2}
$$

$\text{sgn}(\hat{A}_i)$는 올바른 최적화 방향을 보장합니다 ($\hat{A}_i > 0$이면 $+1$, 아니면 $-1$).

#### 클리핑을 포함한 완전한 GMPO 목적함수

```math
\mathcal{J}_\text{GMPO}(\pi_\theta) = \mathbb{E}_{q \sim \mathcal{Q}, \{o_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|q)} \frac{1}{G} \sum_{i=1}^{G} \left\{ \prod_{t=1}^{|o_i|} \left| \min\left[ \rho_{i,t}(\theta)\hat{A}_i,\ \text{clip}(\rho_{i,t}(\theta), \epsilon_\text{low}, \epsilon_\text{high})\hat{A}_i \right] \right| \right\}^{\frac{1}{|o_i|}} \cdot \text{sgn}(\hat{A}_i) \tag{3}
```

#### 목적함수 범위 비교 (이론적 안정성 증명)

AM-GM 부등식에 의해:

```math
|\mathcal{J}^*_\text{GMPO}(\pi_\theta)| = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\prod_{t=1}^{|o_i|}|\rho_{i,t}(\theta)\hat{A}_i|\right)^{\frac{1}{|o_i|}}\right] \leq \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}|\rho_{i,t}(\theta)\hat{A}_i|\right] = |\mathcal{J}^*_\text{GRPO}(\pi_\theta)|
```

즉, **GMPO의 목적함수 범위가 항상 GRPO보다 좁거나 같음**이 수학적으로 보장됩니다.

### 2.3 그래디언트 관점 분석

#### GRPO 그래디언트

$$
\nabla_\theta \mathcal{J}^*_\text{GRPO}(\pi_\theta)\Big|_{q, o_i} = \frac{1}{G \cdot |o_i|} \sum_{t=1}^{|o_i|} \rho_{i,t}(\theta) \cdot \hat{A}_i \cdot \nabla_\theta \log(\pi_\theta(o_{i,t}|q, o_{i,<t})) \tag{4}
$$

#### GMPO 그래디언트

$$
\nabla_\theta \mathcal{J}^*_\text{GMPO}(\pi_\theta)\Big|_{q, o_i} = \frac{1}{G \cdot |o_i|} \sum_{t=1}^{|o_i|} \left(\prod_{k=1}^{|o_i|} \rho_{i,k}(\theta)\right)^{\frac{1}{|o_i|}} \cdot \hat{A}_i \cdot \nabla_\theta \log(\pi_\theta(o_{i,t}|q, o_{i,<t})) \tag{5}
$$

**핵심 차이**: GRPO는 각 토큰 $t$의 **개별** 비율 $\rho_{i,t}(\theta)$로 가중치를 부여하는 반면, GMPO는 시퀀스 내 **모든 토큰의 기하 평균** $\left(\prod_{k=1}^{|o_i|}\rho_{i,k}(\theta)\right)^{\frac{1}{|o_i|}}$으로 가중치를 부여하므로 이상치 토큰 하나가 전체 업데이트를 왜곡하는 현상이 방지됩니다.

### 2.4 주요 설계 요소

#### (i) 토큰 수준 클리핑 (vs. 시퀀스 수준)

시퀀스 수준 클리핑은 한 번 트리거되면 해당 시퀀스의 모든 토큰 그래디언트를 0으로 만들어 정보 손실이 큽니다. GMPO는 **토큰 수준 클리핑**을 채택하여 더 세밀하고 안정적인 제어를 구현합니다.

#### (ii) 넓은 클리핑 범위 (Clipping Wider)

| 방법 | 클리핑 범위 |
|------|------------|
| GRPO (표준) | $(0.8, 1.2)$ |
| DAPO | $(0.8, 1.28)$ |
| **GMPO (제안)** | $(e^{-0.4}, e^{0.4}) \approx (0.67, 1.49)$ |

GMPO는 기하 평균의 고유한 이상치 강인성 덕분에 더 넓은 클리핑 범위에서도 안정성을 유지하면서 더 큰 탐색(exploration)을 허용합니다.

#### (iii) 정규화 인수 $\frac{1}{|o_i|}$의 중요성

정규화 항이 없을 경우 응답 길이가 길어질수록 시퀀스 수준 중요도 샘플링 비율이 기하급수적으로 증가하여 불안정성이 심화됩니다. 정규화 항을 제거하면 평균 성능이 52.7% → 52.0%로 저하됩니다.

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 엔트로피 유지를 통한 탐색 능력 향상

GRPO는 훈련이 진행됨에 따라 엔트로피가 급격히 감소하여 **결정론적 정책(Deterministic Policy)**으로 수렴하는 경향이 있습니다. 이는 과적합의 한 형태로 볼 수 있으며, 훈련 분포를 벗어난 새로운 문제에 대한 일반화 능력을 저하시킵니다.

GMPO는 훈련 전반에 걸쳐 **더 높은 토큰 엔트로피**를 유지합니다:

$$
H(\pi_\theta) = -\sum_t \pi_\theta(o_t|\cdot) \log \pi_\theta(o_t|\cdot)
$$

높은 엔트로피는 다음과 같은 일반화 이점을 제공합니다:

- **다양한 추론 경로 탐색** → 보지 못한 문제 유형에 대한 적응력 향상
- **국소 최적해(Local Optima) 회피** → 더 나은 전역 최적해 탐색

### 3.2 낮은 KL 발산을 통한 일반화

$$
D_\text{KL}(\pi_\theta \| \pi_\text{ref}) = \mathbb{E}\left[\log \frac{\pi_\theta(o|q)}{\pi_\text{ref}(o|q)}\right]
$$

GMPO는 GRPO 대비 **사전 훈련 참조 모델로부터의 KL 발산이 더 작습니다**. 이는:

- **사전 훈련에서 습득한 언어 이해 능력을 보존**
- 특정 도메인(수학)에 과도하게 특화되어 다른 능력이 저하되는 **Catastrophic Forgetting 완화**
- 분포 이탈이 적어 **Out-of-Distribution(OOD) 일반화 성능** 향상 가능성

### 3.3 멀티모달 및 다양한 모델 크기에서의 일반화

| 설정 | GRPO | GMPO | 향상 |
|------|------|------|------|
| Qwen2.5-Math-1.5B (5개 벤치마크 평균) | 42.5% | 43.9% | **+1.4%** |
| Qwen2.5-Math-7B (5개 벤치마크 평균) | 51.2% | 52.7% | **+1.5%** |
| DeepSeek-R1-Distill-Qwen-7B (5개 벤치마크 평균) | 59.3% | 63.4% | **+4.1%** |
| Qwen3-32B MoE (MATH500) | 94.6% | 96.7% | **+2.1%** |
| Qwen2.5-VL-Instruct-7B (Geometry3K 멀티모달) | 53.3% | 54.7% | **+1.4%** |

모델 크기, 아키텍처(Dense/MoE), 모달리티(텍스트/멀티모달)에 걸쳐 **일관된 개선**이 관찰됩니다.

### 3.4 MoE 모델에서의 안정성 — 일반화의 전제 조건

CountDown 데이터셋에서 GRPO는 약 250 스텝 이후 성능이 완전히 붕괴하는 반면, GMPO는 안정적으로 성능을 유지합니다. **훈련 안정성은 일반화 능력의 전제 조건**으로, GMPO의 안정적 훈련은 더 나은 수렴점 도달을 가능하게 합니다.

---

## 4. 성능 향상 및 한계

### 4.1 성능 향상 요약

**언어 추론 태스크 (5개 수학 벤치마크)**:

| 벤치마크 | 난이도 | GMPO 향상 (R1-Distill-7B 기준) |
|----------|--------|-------------------------------|
| AIME24 | 최상 | 43.3% vs. 43.3% (동일) |
| AMC | 중상 | 78.3% vs. 67.5% (**+10.8%**) |
| MATH500 | 중간 | 91.4% vs. 89.0% (**+2.4%**) |
| Minerva | 중상 | 37.9% vs. 39.7% (-1.8%) |
| OlympiadBench | 최상 | 62.5% vs. 56.7% (**+5.8%**) |
| **평균** | - | **63.4% vs. 59.3% (+4.1%)** |

### 4.2 한계점

논문에서 명시적으로 언급된 한계와 분석을 통해 도출할 수 있는 한계는 다음과 같습니다:

**1. 평가 도메인 제한성**
- 코드 생성, 일반 QA 등 비수학 영역에서의 검증 부재
- 수학과 멀티모달 추론에 집중되어 있어 **범용 NLP 태스크**로의 적용 가능성 미검증

**2. 보상 함수 설계 의존성**
- 현재 실험은 이진 보상 (정답: 1, 오답: 0)에만 의존
- 밀집 보상(Dense Reward) 또는 부분 보상 환경에서의 성능 미검증

**3. 이론적 수렴 보장 부재**
- 기하 평균이 실제로 더 나은 수렴 특성을 보장하는지에 대한 **형식적 수렴 증명** 미제시

**4. Minerva 벤치마크에서의 소폭 성능 저하**
- R1-Distill-7B 기준 Minerva에서 39.7% → 37.9%로 **1.8% 하락** 관찰
- 이에 대한 분석이 논문에서 충분히 다루어지지 않음

**5. 클리핑 하이퍼파라미터 민감도**
- 최적 클리핑 범위 $(e^{-0.4}, e^{0.4})$의 **도메인 이전 가능성** 미검증
- 새로운 태스크에서 하이퍼파라미터 튜닝이 필요할 수 있음

---

## 5. 최신 관련 연구 비교 분석 (2020년 이후)

### 5.1 GRPO 계열 방법론 비교

| 방법 | 핵심 아이디어 | 안정성 초점 | 탐색 능력 | GMPO와의 차별점 |
|------|--------------|------------|----------|----------------|
| **PPO** (Schulman et al., 2017) | 클리핑 기반 신뢰 영역 | 중간 | 낮음 | 가치 함수 필요, 계산 비용 높음 |
| **GRPO** (Shao et al., 2024) | 그룹 상대 보상, 산술 평균 | 낮음 | 낮음 | GMPO의 베이스라인 |
| **DeepSeek-R1** (Guo et al., 2025) | 시퀀스 수준 클리핑 | 낮음 | 중간 | GMPO는 토큰 수준 클리핑으로 더 세밀한 제어 |
| **DAPO** (Yu et al., 2025) | 동적 샘플링 + clip-higher | 중간 | 중간 | GMPO보다 클리핑 범위가 좁음, 동적 샘플링 필요 |
| **Dr.GRPO** (Liu et al., 2025) | 길이 편향 제거 | 중상 | 중간 | 길이 편향에 특화, 이상치 문제는 미해결 |
| **GPG** (Chu et al., 2025) | 대리 손실/비평가/KL 제약 제거 | 중간 | 높음 | 단순화에 초점, 안정성 분석 부족 |
| **AAPO** (Xiong et al., 2025) | 이점 모멘텀 도입 | 중상 | 중간 | 모멘텀 기반, GMPO와 결합 가능 |
| **BNPO** (Xiao et al., 2025) | 베타 분포 기반 보상 정규화 | 중상 | 중간 | 보상 분포 모델링에 초점 |
| **EMPO** (Zhang et al., 2025b) | 의미론적 엔트로피 활용 | 중간 | 높음 | 엔트로피를 외부에서 주입, GMPO는 자연스럽게 유지 |
| **OPO** (Hao et al., 2025) | 최적 기준선으로 그래디언트 분산 감소 | 중상 | 중간 | 기준선 설계에 초점 |
| **GVPO** (Zhang et al., 2025a) | 분석적 KL 제약 가중치 | 중상 | 중간 | KL 제약에 초점 |
| **PODS** (Xu et al., 2025) | 정보량 높은 롤아웃 선택 | 중간 | 중간 | 데이터 선택에 초점, GMPO와 직교적 |
| **PRIME** (Cui et al., 2025a) | 암묵적 보상으로 프로세스 강화학습 | 중상 | 중간 | 프로세스 수준 보상에 초점 |
| **INTUITOR** (Zhao et al., 2025) | 외부 보상 없이 자기 확신도 활용 | - | 높음 | 보상 없는 설정, 적용 범위 상이 |
| **RAFT** (Xiong et al., 2025b) | 긍정 샘플만으로 학습 | 중간 | 낮음 | 부정 신호 미활용 |
| **GMPO** (제안, 2025) | **기하 평균** 기반 토큰 보상 최적화 | **높음** | **높음** | **알고리즘 수준 수정, 플러그 앤 플레이** |

### 5.2 관련 이론적 발전

**엔트로피와 탐색 관련 연구들**이 GMPO의 설계와 맥을 같이 합니다:

- **80/20 규칙** (Wang et al., 2025): 고엔트로피 소수 토큰이 RL 학습을 주도한다는 발견 → GMPO의 엔트로피 유지 전략과 정합
- **엔트로피 관점 추론** (Cheng et al., 2025): 엔트로피 붕괴를 LLM RL의 핵심 문제로 지목 → GMPO가 이를 자연스럽게 해결함을 실증
- **엔트로피 메커니즘** (Cui et al., 2025b): 단기 성능을 위해 엔트로피를 희생하는 현상 분석 → GMPO는 기하 평균의 이상치 강인성으로 이 트레이드오프를 개선

---

## 6. 향후 연구에 미치는 영향 및 고려 사항

### 6.1 향후 연구에 미치는 영향

**1. RL 훈련 안정성 연구의 새로운 방향 제시**
- GRPO 계열 알고리즘에서 이상치 처리가 중요한 연구 주제로 부상
- 목적함수의 통계적 특성(평균의 종류, 로버스트성)에 대한 이론적 탐구 촉진
- 단순한 수학적 변환($\text{AM} \to \text{GM}$)이 큰 실용적 개선을 가져올 수 있다는 원리 제시

**2. LLM 사후 훈련 패러다임에 대한 시사점**
- **플러그 앤 플레이** 특성으로 인해 기존 GRPO 기반 파이프라인에 즉시 적용 가능
- 멀티모달, MoE 등 다양한 아키텍처로 확장 가능함을 입증
- 향후 DAPO, Dr.GRPO 등과 **결합(Combination)** 연구 가능성

**3. 탐색-활용 트레이드오프 연구**
- 더 넓은 클리핑 범위와 안정성을 동시에 달성하는 방법론 연구 촉진
- 엔트로피 유지를 위한 알고리즘 설계 원리로 활용 가능

**4. 이론적 기반 강화**
- 기하 평균의 수학적 특성(AM-GM 부등식, 로그 오목성)을 RL 이론에 접목하는 연구 촉발
- 중요도 샘플링 비율의 분포 제어를 위한 새로운 이론적 프레임워크 필요성 제기

### 6.2 향후 연구 시 고려할 점

**이론적 측면**

1. **수렴 이론 정립**: 기하 평균 목적함수에 대한 형식적 수렴 증명 필요
   - $\mathcal{J}_\text{GMPO}$의 오목성/볼록성 분석
   - 샘플 복잡도(Sample Complexity) 상한 도출

2. **최적 집계 방식 탐색**: 산술 평균과 기하 평균 사이의 일반화된 거듭제곱 평균(Power Mean) 탐색:

$$
M_p = \left(\frac{1}{|o|}\sum_{t=1}^{|o|} \rho_t(\theta)^p\right)^{1/p}
$$

$p=1$은 산술 평균, $p \to 0$은 기하 평균에 해당하며, 최적 $p$ 값에 대한 연구 가능

**실험적 측면**

3. **다양한 도메인으로 확장**: 코드 생성, 논리 추론, 의료/과학 분야 적용 검증

4. **밀집 보상 환경에서의 검증**: 현재 이진 보상 기반 실험을 과정 보상(Process Reward) 환경으로 확장

5. **클리핑 하이퍼파라미터 자동화**: 태스크/모델별 최적 $(\epsilon_\text{low}, \epsilon_\text{high})$ 자동 탐색 방법론 연구
   - 적응형 클리핑(Adaptive Clipping): 훈련 진행에 따라 동적으로 클리핑 범위 조정

6. **다른 GRPO 변형과의 결합**: Dr.GRPO의 길이 편향 제거 + GMPO의 기하 평균 결합, AAPO의 이점 모멘텀 + GMPO 결합 등

**실용적 측면**

7. **계산 비용 분석**: 로그 공간 연산이 필요한 GMPO의 실제 훈련 시간 오버헤드 정량화

8. **스케일링 법칙(Scaling Law)**: 모델 크기 증가에 따른 GMPO 효과의 스케일링 패턴 분석

9. **강건성 평가**: 다양한 보상 노이즈 수준, 데이터 분포 이동 환경에서의 성능 평가

---

## 참고 자료

**주요 논문 (분석 대상)**
- **Zhao, Y. et al. (2025). "Geometric-Mean Policy Optimization." arXiv:2507.20673v3.**
  - GitHub: https://github.com/callsys/GMPO

**비교 분석에 활용된 참고 문헌**
- Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
- Shao, Z. et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300.
- Guo, D. et al. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948.
- Yu, Q. et al. (2025). "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." arXiv:2503.14476.
- Liu, Z. et al. (2025). "Understanding R1-Zero-like Training: A Critical Perspective (Dr.GRPO)." arXiv:2503.20783.
- Chu, X. et al. (2025). "GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning." arXiv:2504.02546.
- Cui, G. et al. (2025a). "Process Reinforcement through Implicit Rewards (PRIME)." arXiv:2502.01456.
- Cui, G. et al. (2025b). "The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models." arXiv:2505.22617.
- Wang, S. et al. (2025). "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning." arXiv:2506.01939.
- Cheng, D. et al. (2025). "Reasoning with Exploration: An Entropy Perspective." arXiv:2506.14758.
- Hao, Y. et al. (2025). "On-Policy RL with Optimal Reward Baseline (OPO)." arXiv:2505.23585.
- Zhang, K. et al. (2025a). "GVPO: Group Variance Policy Optimization." arXiv:2504.19599.
- Xiao, C. et al. (2025). "BNPO: Beta Normalization Policy Optimization." arXiv:2506.02864.
- Xiong, J. et al. (2025a). "AAPO: Enhance the Reasoning Capabilities of LLMs with Advantage Momentum." arXiv:2505.14264.
- Sutton, R. S. et al. (1999). "Policy Gradient Methods for Reinforcement Learning with Function Approximation." NeurIPS.
