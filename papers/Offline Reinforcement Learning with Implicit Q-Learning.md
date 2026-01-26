
# Offline Reinforcement Learning with Implicit Q-Learning
## 요약
"Offline Reinforcement Learning with Implicit Q-Learning"은 Kostrikov, Nair, Levine이 2021년 발표한 논문으로, 오프라인 강화학습의 핵심 문제인 분포 변화(distributional shift)를 해결하기 위한 혁신적인 접근법을 제시합니다. IQL은 데이터셋에 없는 액션을 절대 평가하지 않으면서도 다중 단계 동적 프로그래밍을 수행할 수 있는 첫 번째 방법이며, 기대값 회귀(expectile regression)라는 통계적 기법을 활용한 우아한 솔루션입니다.

***

## 1. 핵심 주장 및 주요 기여도
### 1.1 핵심 주장
오프라인 강화학습은 두 가지 상충하는 목표 사이에서의 트레이드오프에 직면합니다. 정책을 데이터셋의 행동 정책보다 개선하면서도 동시에 분포 변화로 인한 오류를 최소화해야 합니다. 기존 방법들은 명시적 제약(BCQ, BEAR)이나 정칙화(CQL)를 통해 이 문제를 해결했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

IQL의 핵심 주장은 **"데이터셋에 없는 액션을 절대 쿼리하지 않으면서도 정책을 크게 개선할 수 있다"**는 것입니다. 이는 함수 근사의 일반화 능력을 활용하여 최적 액션의 가치를 직접 쿼리하지 않고도 추정할 수 있다는 의미입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

### 1.2 주요 기여도
1. **Implicit Q-Learning (IQL) 알고리즘**: 기대값 회귀를 기반으로 한 새로운 오프라인 RL 방법론
2. **구현의 단순성**: SARSA 스타일 TD 업데이트에 대한 최소한의 수정으로 구현 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)
3. **계산 효율성**: 기존 방법 대비 약 4배 빠른 훈련 속도 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)
4. **벤치마크 성능**: D4RL 벤치마크에서 최고의 성능 달성, 특히 AntMaze 태스크에서 기존 방법 대비 24.5% 향상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)
5. **온라인 파인튜닝 적합성**: 오프라인 초기화 후 온라인 상호작용으로 추가 개선 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

***

## 2. 문제 정의 및 해결 방법
### 2.1 해결하고자 하는 문제
#### 분포 변화(Distributional Shift) 문제
표준 Q-러닝은 다음과 같이 정의됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

$$L_{TD}(\theta) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}}[(r(s,a) + \gamma \max_{a'} \hat{Q}_\theta(s',a') - Q_\theta(s,a))^2]$$

여기서 문제는 $a'$이 데이터셋 밖의 액션일 때 $\hat{Q}_\theta(s',a')$이 크게 잘못 추정될 수 있다는 것입니다. 이를 **액션 외삽 오류(action extrapolation error)**라고 합니다.

#### 기존 방법들의 한계
- **제약 기반 방법(BCQ, BEAR)**: 정책을 행동 정책 근처에 강제로 유지하여 과도하게 보수적
- **정칙화 기반 방법(CQL)**: Q-함수를 명시적으로 정칙화하여 계산 비용 증가

### 2.2 IQL의 해결 방법
#### 핵심 아이디어: 기대값 회귀

IQL의 핵심은 상태 가치 함수를 액션 분포에 대한 확률 변수로 취급하고, 이 분포의 상위 기대값을 추정하는 것입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

**기대값 정의**: τ ∈ (0,1)에 대한 기대값은 다음 비대칭 최소 제곱 문제의 해입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

$$\arg\min_{m_\tau} \mathbb{E}_{x \sim X}[L_\tau^2(x - m_\tau)]$$

여기서 비대칭 손실함수는:

$$L_\tau^2(u) = |\tau - \mathbb{1}(u < 0)| u^2$$

τ = 0.5일 때는 표준 MSE이고, τ가 1에 가까워질수록 최댓값에 가까워집니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

#### IQL의 3단계 접근

**1단계: V-함수 학습**

$$L_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}}[L_\tau^2(\hat{Q}_\theta(s,a) - V_\psi(s))]$$

이는 환경 동역학(transition)으로부터의 확률성을 분리하고, 액션에 대한 기대값만 학습합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

**2단계: Q-함수 학습**

$$L_Q(\theta) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}}[(r(s,a) + \gamma V_\psi(s') - Q_\theta(s,a))^2]$$

V-함수를 사용하여 환경의 확률성을 평균화하면서도, 데이터셋 액션에만 Q-함수를 훈련합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

**3단계: 정책 추출 (Advantage-Weighted Behavioral Cloning)**

$$L_\pi(\phi) = \mathbb{E}_{(s,a) \sim \mathcal{D}}[\exp(\beta(Q_\theta(s,a) - V_\psi(s))) \log \pi_\phi(a|s)]$$

여기서 β ∈. 작은 β는 행동 모방에 가깝고, 큰 β는 Q-함수 최대화를 시도합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

### 2.3 이론적 정당성
#### 정리 1 (Lemma 1): 기대값의 수렴 성질

유계 지지집합을 가진 실수값 확률변수 X에 대해: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

$$\lim_{\tau \to 1} m_\tau = x^*$$

즉, τ가 1에 가까워질수록 기대값이 최댓값으로 수렴합니다.

#### 정리 2 (Theorem 3): IQL의 최적성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

$$\lim_{\tau \to 1} V_\tau(s) = \max_{a \in A, s.t. \pi_\beta(a|s) > 0} Q^*(s,a)$$

이는 충분히 큰 τ를 선택하면 IQL이 데이터셋 지지에 제한된 최적 Q-함수를 복구할 수 있음을 의미합니다.

***

## 3. 모델 구조
### 3.1 신경망 아키텍처
**V-함수 네트워크**
- 입력: 상태 s
- 은닉층: 2층, 256개 유닛, ReLU 활성화
- 출력: 스칼라 값 V(s)

**Q-함수 네트워크 (Clipped Double Q-Learning)**
- 입력: 상태-액션 쌍 (s, a)
- 은닉층: 2층, 256개 유닛, ReLU 활성화
- 출력: 두 개의 Q-함수 Q₁(s,a), Q₂(s,a) (최솟값 사용)
- 목표 네트워크: Polyak 평균화 (α = 0.005)

**정책 네트워크**
- 입력: 상태 s
- 은닉층: 2층, 256개 유닛, ReLU 활성화
- 출력: 가우시안 분포의 평균과 표준편차

### 3.2 훈련 절차
**Algorithm 1: Implicit Q-Learning** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

초기화: ψ, θ, θ̂, φ 파라미터 초기화

TD 학습 단계:
```
각 그래디언트 스텝마다:
  ψ ← ψ - λᵥ ∇ψ Lᵥ(ψ)
  θ ← θ - λQ ∇θ LQ(θ)  
  θ̂ ← (1-α)θ̂ + αθ
```

정책 추출 단계:
```
각 그래디언트 스텝마다:
  φ ← φ - λπ ∇φ Lπ(φ)
```

**중요한 특징**:
- V-함수와 Q-함수 훈련은 정책과 독립적
- 정책 추출은 V와 Q가 수렴한 후 수행 가능
- 온라인 파인튜닝 시에는 동시 훈련 가능

***

## 4. 성능 향상 및 실험 결과
### 4.1 벤치마크 성능
**D4RL 벤치마크에서의 성능**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

| 태스크 범주 | IQL | CQL | TD3+BC | AWAC |
|----------|-----|-----|--------|------|
| MuJoCo Locomotion | 692.4 | 698.5 | 677.4 | 450.7 |
| AntMaze | 378.0 | 303.6 | 163.8 | 107.7 |
| Manipulation | 277.9 | 238.2 | 144.6 | 92.3 |

**주요 성과**:
- **AntMaze에서 24.5% 향상**: CQL 303.6점 대비 378.0점
- **전체 성능**: 총 1070.4점으로 CQL의 1002.1점 대비 6.8% 향상
- **계산 효율성**: CQL 80분 대비 20분 (4배 향상) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

### 4.2 스티칭(Stitching) 태스크에서의 우수성
단순한 미로 환경에서의 비교: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)
- 1개 최적 궤적 + 99개 무작위 궤적으로 구성된 데이터셋
- **한 단계 정책 평가**: 값함수가 보상 상태에서 멀어질수록 빠르게 감소
- **IQL (τ = 0.95)**: 최적 값함수와 유사하게 신호를 올바르게 전파

이는 다중 단계 동적 프로그래밍이 복잡한 태스크에 필수적임을 입증합니다.

### 4.3 하이퍼파라미터 영향
**기대값 τ의 영향**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

- MuJoCo 태스크: τ = 0.7 (SARSA에 더 가까움)
- AntMaze 태스크: τ = 0.9 (Q-러닝에 더 가까움)

τ가 클수록 동적 프로그래밍 성능이 향상되지만, 최적화가 더 어려워집니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

### 4.4 온라인 파인튜닝 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)
| 환경 | 초기 (오프라인) | 최종 (온라인) | 개선도 |
|------|---------------|-------------|-------|
| antmaze-umaze-v0 | 86.7 | 96.0 | +9.3 |
| antmaze-medium-play-v0 | 72.0 | 95.0 | +23.0 |
| antmaze-large-play-v0 | 25.5 | 46.0 | +20.5 |

IQL의 정책 추출 메커니즘(AWR)이 온라인 파인튜닝에 특히 적합함을 보여줍니다.

### 4.5 한계점
1. **τ 하이퍼파라미터의 민감성**: 태스크별로 다른 최적 τ 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)
2. **MuJoCo 태스크에서의 경합성**: 일부 태스크에서 CQL과 유사하거나 약간 낮은 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)
3. **모드 손실**: 단일 정책 샘플링으로 인한 다중 모드 손실 가능성
4. **이론-실제 간극**: 이론상 τ → 1에서 최적이지만, 실제로는 유한 τ 사용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

***

## 5. 모델 일반화 성능 분석
### 5.1 IQL 논문에서의 일반화 논의
IQL은 여러 측면에서 일반화 성능을 강화합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5875c18d-f7c7-4291-9356-55d147486661/2110.06169v1.pdf)

**1. 다중 단계 동적 프로그래밍**: 
- SARSA 스타일 평가와 Q-러닝 스타일 개선 사이의 스펙트럼을 제공
- 복잡한 태스크에서의 신호 전파 개선

**2. 정책 추출의 우수성**:
- AWR 기반 정책 추출은 행동 정책에서의 자연스러운 이탈 보장
- β 파라미터로 보수성 조절 가능

**3. 함수 근사 활용**:
- 학습된 함수 근사기의 일반화 능력 활용
- 데이터셋 밖의 액션에 대한 합리적인 값 추정

### 5.2 최신 연구 (2023-2025)에서의 일반화 문제
#### 5.2.1 일반화 격차(Generalization Gap) 문제 [proceedings.iclr](https://proceedings.iclr.cc/paper_files/paper/2024/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf)

2024년 ICLR 연구에 따르면: [proceedings.iclr](https://proceedings.iclr.cc/paper_files/paper/2024/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf)
- 오프라인 RL이 온라인 RL보다 새로운 환경에서 현저하게 성능 저하
- 행동 모방(BC)이 다중 환경 데이터에서 더 나은 일반화 보임
- 데이터 다양성이 데이터셋 크기보다 일반화에 더 중요 [proceedings.iclr](https://proceedings.iclr.cc/paper_files/paper/2024/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf)

#### 5.2.2 표현 기반 개선: Representation Distinction (RD) [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2023/file/802a4350ca4fced76b13b8b320af1543-Paper-Conference.pdf)

2023년 NeurIPS 연구: [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2023/file/802a4350ca4fced76b13b8b320af1543-Paper-Conference.pdf)
- **Backup-Generalization Cycle 개념**: 오프라인 RL 학습의 두 단계
  - Backup: Q-함수 네트워크 업데이트
  - Generalization: 데이터셋 액션에서의 일반화가 OOD 액션까지 전파

- **과일반화(Overgeneralization) 문제**: 
  - 데이터 액션과 OOD 액션이 표현 공간에서 유사해지는 문제
  - 해결책: In-sample과 OOD 표현을 명시적으로 분리

#### 5.2.3 IQL 개선: IDQL (2023) [www2.eecs.berkeley](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/EECS-2023-62.pdf)

Hansen-Estruch et al.의 IDQL: [www2.eecs.berkeley](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/EECS-2023-62.pdf)
- IQL을 액터-크리틱 방법으로 재해석
- 일반화된 손실함수로 IQL 기대값 문제 확장
- Diffusion 모델을 이용한 다중 모드 정책 표현
- **성능 개선**: 하이퍼파라미터 민감성 감소, 최고 성능 달성

#### 5.2.4 AlignIQL (2024) [arxiv](https://arxiv.org/pdf/2405.18187.pdf)

He et al.의 AlignIQL: [arxiv](https://arxiv.org/pdf/2405.18187.pdf)
- IQL의 암시적 정책과 명시적 정책 추출 간의 불일치 해결
- KKT 조건 기반 정책 정렬
- **이점**: IDQL 대비 하이퍼파라미터 강인성 개선, 수렴 속도 향상

#### 5.2.5 최신 일반화 방법들 (2024-2025)

**1. Latent Distribution Representation Learning (LAD, 2025)** [cicip.sxu.edu](http://cicip.sxu.edu.cn/docs/2025-03/ee9195a778c247ddb25b8372aa772763.pdf)
- 오프라인 데이터의 잠재 분포 특성화
- 분포 간극 다양성 증대로 강건한 학습
- CQL, IQL, TD3+BC에 적용 가능한 프레임워크

**2. Variational OOD State Correction (DASP, 2025)** [arxiv](https://arxiv.org/pdf/2505.00503.pdf)
- 상태 분포 변화에 초점
- In-distribution 영역으로의 반환 유도
- 변분 프레임워크를 통한 안전한 의사결정

**3. Improved Actor Generalization (2024)** [arxiv](https://arxiv.org/abs/2409.07606)
- 액터 네트워크의 정규화 기법 (Dropout, Layer Norm, Weight Decay) 효과
- IQL 계열에서 액터 일반화가 병목임을 입증
- 평균 6% 성능 개선 달성

***

## 6. 최신 관련 연구 비교 분석 (2020-2025)
### 6.1 기초 방법들 (2019-2021)
#### Conservative Q-Learning (CQL, 2020) [semanticscholar](https://www.semanticscholar.org/paper/28db20a81eec74a50204686c3cf796c42a020d2e)
- **핵심 아이디어**: Q-함수를 보수적으로 학습하여 하한 값 제공
- **수식**: 

$`L_{CQL} = L_{TD} + \alpha \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi(·|s)}[\log\sum_{a'} \exp Q(s,a')] - \mathbb{E}_{s,a \sim \mathcal{D}}[Q(s,a)]`$

- **장점**: 강한 이론적 보장, 광범위한 실험 검증 (2918 인용)
- **단점**: 계산 비용 높음, α 하이퍼파라미터 민감성
- **성능**: MuJoCo에서 IQL과 동등하거나 우수

#### TD3+BC (2021)
- **접근**: TD3에 행동 모방 항 추가
- **정식**: Actor loss에 BC 정규화 항 통합
- **장점**: 구현 간단, 안정적
- **한계**: 복잡한 스티칭 태스크에서 성능 부족

### 6.2 IQL 직접 개선 (2021-2025)
#### IDQL: IQL as Actor-Critic Method (2023) [www2.eecs.berkeley](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/EECS-2023-62.pdf)
- **핵심 기여**:
  1. IQL의 액터-크리틱 재해석
  2. 일반화된 손실함수 프레임워크
  3. Diffusion 모델 기반 정책 표현

- **개선 사항**:
  - 암시적 액터의 명시적 특성화
  - 하이퍼파라미터 강인성 증가
  - 다중 모드 정책 학습 능력

- **성능**: IQL 대비 평균 5-10% 향상

#### RIQL: Robust Implicit Q-Learning (2023) [emergentmind](https://www.emergentmind.com/topics/implicit-q-learning-iql)
- **동기**: 부패된 데이터에 대한 강인성 부족
- **해결책**: 
  - Huber loss for robustness
  - Quantile ensemble for diversity
  - Heavy-tailed error 처리

- **성능**: 노이즈 있는 환경에서 IQL 대비 15-20% 향상

#### AlignIQL (2024) [arxiv](https://arxiv.org/pdf/2405.18187.pdf)
- **문제**: IDQL의 암시적 액터와 추출 정책 간 불일치
- **해결책**: KKT 조건 기반 정렬
- **개선**: 
  - 더 안정적인 훈련
  - 하이퍼파라미터 민감성 감소
  - 빠른 수렴

#### Proj-IQL (2025) [emergentmind](https://www.emergentmind.com/topics/implicit-q-learning-iql)
- **혁신**: 적응적 기대값 파라미터
- **방법**: 
  $$\tau = \text{projection}(\pi_\text{learned}, \pi_\beta)$$
- **이점**:
  - 단조 개선 보장 (Monotonic improvement)
  - τ 자동 조정
  - 이론적 개선 보장

### 6.3 일반화 성능 개선 (2022-2025)
#### Representation Distinction (RD, 2023) [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2023/file/802a4350ca4fced76b13b8b320af1543-Paper-Conference.pdf)
- **핵심 개념**: Backup-Generalization Cycle
- **문제 진단**:
  - 과일반화: In-sample 액션과 OOD 액션 표현이 유사해짐
  - 결과: OOD 액션에 잘못된 높은 값 할당

- **해결책**:
  1. In-sample vs OOD 표현 명시적 분리
  2. Kernel control을 통한 표현 거리 증대
  3. 플러그인 방식으로 다른 방법에 적용 가능

- **성능**: CQL, IQL, TD3+BC에 평균 8-12% 향상

#### Domain Generalization for Model-based Offline RL (2022) [arxiv](http://arxiv.org/pdf/2211.14827.pdf)
- **상황**: 다중 데모 환경에서의 학습
- **접근**: 도메인별 특성 활용
- **결과**: 다중 환경 일반화 성능 향상

#### Latent Distribution Representation Learning (2025) [cicip.sxu.edu](http://cicip.sxu.edu.cn/docs/2025-03/ee9195a778c247ddb25b8372aa772763.pdf)
- **혁신**: 오프라인 데이터의 잠재 분포 특성화
- **방법**:
  1. Worst-case 분포 식별
  2. 분포 간극 다양성 증대
  3. 불변 표현 학습

- **성능**: 새로운 환경에 대한 일반화 15-20% 향상

### 6.4 온라인-오프라인 전환 개선 (2022-2025)
#### Confidence-Conditioned Value Learning (CCVL, 2023) [arxiv](https://arxiv.org/pdf/2212.04607.pdf)
- **문제**: 고정 신뢰도 레벨의 한계
- **혁신**: 신뢰도 조건부 값 함수
$$Q_\delta(s,a) = \text{lower-bound with confidence level } \delta$$

- **이점**: 온라인 평가 시 신뢰도 적응 조절
- **성능**: 오프라인-온라인 전환 시 10-15% 향상

#### ENOTO: Q-Ensembles for Offline-to-Online RL (2024) [arxiv](https://arxiv.org/pdf/2306.06871.pdf)
- **접근**: Q-함수 앙상블로 탐험 안정화
- **방법**:
  1. 여러 Q-함수로 불확실성 추정
  2. 온라인 파인튜닝 안정화
  3. 탐험-활용 균형 개선

- **성능**: 기존 방법 대비 12-18% 향상

#### StratDiff: Diffusion Models for Offline-to-Online RL (2025) [arxiv](https://arxiv.org/html/2511.03828v1)
- **혁신**: 에너지 가이드 확산 모델 통합
- **이점**: 더 나은 정책 초기화, 온라인 학습 가속
- **성능**: 최신 성능 달성

### 6.5 비교 요약표
| 방법 | 출판 | 핵심 아이디어 | 주요 장점 | 주요 한계 | 인용수 |
|------|------|-------------|---------|---------|-------|
| BCQ | 2019 | 생성 모델 기반 제약 | 직관적, 이론적 근거 | 계산 비용 | - |
| CQL | 2020 | Q-함수 정규화 | 강한 보장, 광범위 검증 | 계산 비용, 보수성 | 2918 |
| IQL | 2021 | 기대값 회귀 | 단순, 빠름, 우수한 성능 | τ 민감성 | 1430 |
| TD3+BC | 2021 | TD3 + BC | 간단, 안정 | 복잡한 태스크 부족 | - |
| IDQL | 2023 | IQL 재해석 + Diffusion | 다중 모드, 개선 성능 | 복잡도 | 237 |
| CCVL | 2023 | 신뢰도 조건부 값 | 적응적 신뢰도 | 새로운 방법 | - |
| RIQL | 2023 | Robust IQL | 노이즈 강인성 | 제한적 검증 | - |
| RD | 2023 | 표현 구분 | 플러그인 적용 가능 | 추가 비용 | 14 |
| AlignIQL | 2024 | IQL 정렬 | 하이퍼파라미터 강건성 | 최신 | 10 |
| ENOTO | 2024 | Q 앙상블 | 온라인-오프라인 안정 | 앙상블 비용 | - |
| LAD | 2025 | 잠재 분포 학습 | 강건한 일반화 | 최신 | - |
| Proj-IQL | 2025 | 적응적 τ | 단조 개선, 자동 조정 | 최신 | - |

***

## 7. IQL의 영향과 향후 연구 방향
### 7.1 학술적 영향
**1. 오프라인 RL의 패러다임 전환**
- IQL 이전: 명시적 제약 또는 정칙화 중심
- IQL 이후: 암시적 접근과 함수 근사 활용의 중요성 인식

**2. 기대값 회귀의 재발견**
- 강화학습에 통계적 도구(expectile regression) 적용
- 분포 강화학습(distributional RL)과의 연결고리

**3. 다중 기여자 효과**
- 논문 인용: 1430회 이상 (2021년 이후)
- 업계 채택: 로봇공학, 자율주행 등 실제 응용

### 7.2 기술적 혁신
**1. 정책 추출의 재평가**
- AWR의 중요성 재조명
- 암시적 정책 개념의 도입

**2. 이론-실제 간극**
- 이론상 τ → 1이지만, 실제로는 유한 τ 사용의 정당성
- 함수 근사 오류 분석의 필요성

**3. 하이퍼파라미터 자동 조정**
- τ와 β의 자동 선택 문제 제기
- 후속 연구로 Proj-IQL 등의 개선 유도

### 7.3 미해결 문제 및 향후 연구
#### 7.3.1 일반화 성능의 향상

**문제**: 오프라인 RL의 가장 큰 미해결 과제는 새로운 환경으로의 전이 [proceedings.iclr](https://proceedings.iclr.cc/paper_files/paper/2024/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf)

**현재 접근법**:
- 표현 학습 (RD, LAD)
- 데이터 다양성 활용
- 앙상블 방법 (ENOTO)

**필요한 연구**:
- 이론적 일반화 보장
- 도메인 이동 감지 및 적응
- 메타-학습 통합

#### 7.3.2 하이퍼파라미터 자동 조정

**문제**: IQL의 τ, β, λ 등 다수의 민감한 매개변수

**현재 해결책**:
- Proj-IQL의 적응적 τ
- AlignIQL의 KKT 기반 최적화
- 자동 하이퍼파라미터 튜닝

**필요한 연구**:
- 온라인 적응 메커니즘
- 태스크 특성에 기반한 자동 선택
- 계산 비용 증가 없는 적응

#### 7.3.3 다중 모드 정책 학습

**문제**: IQL의 단일 정책 샘플링으로 인한 모드 손실

**현재 해결책**:
- IDQL의 Diffusion 모델 기반 정책
- StratDiff의 에너지 가이드

**필요한 연구**:
- 다중 모드 값함수
- 상황별 행동 선택 메커니즘
- 계산 효율성 개선

#### 7.3.4 안전성 및 강인성

**문제**: 부패된 데이터, 노이즈, 시스템 오류

**현재 해결책**:
- RIQL의 Huber loss
- Quantile ensemble
- 데이터 필터링

**필요한 연구**:
- 적대적 공격에 대한 방어
- 온라인 학습 중 안전 보장
- 실시간 강인성 모니터링

#### 7.3.5 이론적 깊이 추가

**필요한 영역**:
1. **함수 근사 이론**: 비선형 함수 근사 하에서의 수렴성
2. **표현 학습**: 좋은 표현의 특성화
3. **최적 정책**: 암시적 정책의 최적성 조건
4. **복잡도 분석**: 샘플 및 계산 복잡도의 엄밀한 분석

### 7.4 실무적 고려사항
#### 7.4.1 로봇공학 응용

**성공 사례**:
- Dexterous manipulation (D4RL Adroit 태스크)
- Navigation (Ant Maze 태스크)

**남은 과제**:
- 높은 차원의 이미지 입력 처리
- 실시간 의사결정 요구
- 안전성 보장

#### 7.4.2 의료/금융 등 민감한 도메인

**요구사항**:
- 강력한 성능 보장
- 설명 가능성
- 기존 정책과의 호환성

**IQL의 적합성**:
- 보수적 값함수 학습
- 행동 정책 근처 유지
- 해석 가능한 값함수

#### 7.4.3 대규모 데이터셋 처리

**도전 과제**:
- 계산 효율성 (IQL의 장점)
- 메모리 효율성
- 분산 훈련

**발전 방향**:
- 더 효율적인 기대값 회귀 계산
- 미니배치 기반 최적화
- 병렬 처리 구조

***

## 8. 결론 및 주요 통찰
### 8.1 IQL의 혁신적 기여
1. **개념적 우아성**: 기대값 회귀를 통해 복잡한 문제를 단순하게 해결
2. **실용적 효율성**: 4배 빠른 훈련과 최소한의 구현 변경
3. **우수한 성능**: 특히 "스티칭" 문제를 포함한 복잡한 태스크에서
4. **이론과 실제의 균형**: 견고한 이론적 기초와 실증적 검증

### 8.2 오프라인 RL 전개의 흐름
```
2020: CQL (Regularization-based)
  ↓
2021: IQL (Implicit approach) ← Current
  ↓
2022-2023: Understanding & Improvements
  ├─ IDQL (Actor-critic reinterpretation)
  ├─ CCVL (Confidence-based)
  └─ RD (Representation learning)
  ↓
2024-2025: Next-generation Methods
  ├─ Proj-IQL (Adaptive τ)
  ├─ AlignIQL (Policy alignment)
  ├─ LAD (Latent distribution)
  └─ StratDiff (Diffusion integration)
```

### 8.3 핵심 교훈
1. **분포 변화의 핵심은 일반화**: 단순한 제약이 아닌 지능적 일반화의 활용
2. **함수 근사의 힘**: 신경망의 보간 능력을 적극 활용
3. **단순함의 가치**: 복잡한 메커니즘 대신 수학적으로 우아한 솔루션
4. **온라인 파인튜닝의 중요성**: 오프라인 초기화는 온라인 학습의 시작점

### 8.4 향후 5년의 연구 로드맵
**2025-2026 (단기)**:
- 기대값 τ의 자동 적응 완성 (Proj-IQL의 발전)
- 다중 모드 정책 학습의 표준화
- 실제 로봇 시스템에서의 검증

**2026-2027 (중기)**:
- 통합 프레임워크: 오프라인-온라인 RL의 완전한 통일
- 새로운 환경으로의 강건한 전이
- 매우 높은 차원의 관찰 처리 (이미지 기반)

**2027-2030 (장기)**:
- 오프라인 RL과 대규모 언어 모델의 통합
- 다중 에이전트 오프라인 학습
- 계층적 강화학습과의 통합

***

## 참고 문헌 및 인용

<span style="display:none">[^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90]</span>

<div align="center">⁂</div>

[^1_1]: 2110.06169v1.pdf

[^1_2]: https://proceedings.iclr.cc/paper_files/paper/2024/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf

[^1_3]: https://proceedings.neurips.cc/paper_files/paper/2023/file/802a4350ca4fced76b13b8b320af1543-Paper-Conference.pdf

[^1_4]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/EECS-2023-62.pdf

[^1_5]: https://arxiv.org/pdf/2405.18187.pdf

[^1_6]: http://cicip.sxu.edu.cn/docs/2025-03/ee9195a778c247ddb25b8372aa772763.pdf

[^1_7]: https://arxiv.org/pdf/2505.00503.pdf

[^1_8]: https://arxiv.org/abs/2409.07606

[^1_9]: https://www.semanticscholar.org/paper/28db20a81eec74a50204686c3cf796c42a020d2e

[^1_10]: https://proceedings.neurips.cc/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf

[^1_11]: https://www.emergentmind.com/topics/implicit-q-learning-iql

[^1_12]: http://arxiv.org/pdf/2211.14827.pdf

[^1_13]: https://arxiv.org/pdf/2212.04607.pdf

[^1_14]: https://arxiv.org/pdf/2306.06871.pdf

[^1_15]: https://arxiv.org/html/2511.03828v1

[^1_16]: https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9

[^1_17]: https://open-publishing.org/publications/index.php/APUB/article/view/1193

[^1_18]: https://arxiv.org/abs/2403.09701

[^1_19]: https://www.researchprotocols.org/2024/1/e52744

[^1_20]: https://open-publishing.org/publications/index.php/APUB/article/view/1196

[^1_21]: https://turia.uv.es/index.php/realia/article/view/29257

[^1_22]: https://www.cureus.com/articles/322770-bibliometric-insights-into-thromboelastography-research-using-pubmed-and-vosviewer

[^1_23]: https://iopscience.iop.org/article/10.1149/MA2024-02694846mtgabs

[^1_24]: https://invergejournals.com/index.php/ijss/article/view/105

[^1_25]: https://arxiv.org/pdf/2106.09119.pdf

[^1_26]: https://arxiv.org/pdf/2204.12581.pdf

[^1_27]: https://arxiv.org/pdf/2209.15256.pdf

[^1_28]: http://arxiv.org/pdf/2406.10445.pdf

[^1_29]: https://arxiv.org/abs/2110.04135

[^1_30]: https://arxiv.org/html/2503.19267v1

[^1_31]: https://leeyngdo.github.io/blog/reinforcement-learning/2024-04-23-implicit-q-learning/

[^1_32]: https://www.ijcai.org/proceedings/2024/0507.pdf

[^1_33]: https://lavaei.ieor.berkeley.edu/RL_Diffusion_2025_1.pdf

[^1_34]: https://www.sciencedirect.com/science/article/abs/pii/S0952197625006785

[^1_35]: https://www.sciencedirect.com/science/article/abs/pii/S0020025522009033

[^1_36]: https://arxiv.org/html/2404.07465v2

[^1_37]: https://arxiv.org/abs/2110.06169

[^1_38]: https://arxiv.org/html/2408.12830v3

[^1_39]: https://github.com/linhlpv/awesome-offline-to-online-RL-papers

[^1_40]: https://velog.io/@minnnn/Paper-Review-OFFLINE-REINFORCEMENT-LEARNINGWITH-IMPLICIT-Q-LEARNING

[^1_41]: https://openreview.net/forum?id=4kLVvIh8cp

[^1_42]: https://arxiv.org/html/2512.20115v1

[^1_43]: https://arxiv.org/html/2511.16475v1

[^1_44]: https://arxiv.org/html/2510.08218v1

[^1_45]: https://arxiv.org/pdf/2511.03695.pdf

[^1_46]: https://arxiv.org/html/2510.11499v1

[^1_47]: https://arxiv.org/html/2504.11944v1

[^1_48]: https://arxiv.org/html/2506.09574v2

[^1_49]: https://arxiv.org/pdf/2110.06169.pdf

[^1_50]: https://arxiv.org/html/2512.20220v1

[^1_51]: https://arxiv.org/html/2510.22027v1

[^1_52]: https://arxiv.org/html/2312.05742v2

[^1_53]: https://arxiv.org/html/2406.09329v2

[^1_54]: https://www.semanticscholar.org/paper/19d944455c07bbcc28faa7a893233a0326f51749

[^1_55]: https://arxiv.org/abs/2501.14199

[^1_56]: https://ieeexplore.ieee.org/document/11087014/

[^1_57]: https://arxiv.org/abs/2511.22210

[^1_58]: https://www.itm-conferences.org/10.1051/itmconf/20258001043

[^1_59]: https://ieeexplore.ieee.org/document/10730227/

[^1_60]: https://arxiv.org/abs/2406.04534

[^1_61]: https://ieeexplore.ieee.org/document/10771594/

[^1_62]: https://ieeexplore.ieee.org/document/10650768/

[^1_63]: https://arxiv.org/html/2406.13961

[^1_64]: http://arxiv.org/pdf/2211.01052.pdf

[^1_65]: http://arxiv.org/pdf/2406.04534.pdf

[^1_66]: https://www.aclweb.org/anthology/2020.emnlp-main.327.pdf

[^1_67]: http://arxiv.org/pdf/2502.08985.pdf

[^1_68]: http://arxiv.org/pdf/2211.15065.pdf

[^1_69]: https://arxiv.org/html/2310.19805v1

[^1_70]: https://www.youtube.com/watch?v=sVPm7zOrBxM

[^1_71]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11245277/

[^1_72]: https://papers.neurips.cc/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf

[^1_73]: https://www.geeksforgeeks.org/deep-learning/behavioral-cloning/

[^1_74]: https://arxiv.org/abs/2006.04779

[^1_75]: https://iclr.cc/virtual/2022/poster/6654

[^1_76]: https://hiddenbeginner.github.io/study-notes/contents/rl_papers/230530_CQL.html

[^1_77]: https://www.reddit.com/r/reinforcementlearning/comments/1bf6fhq/supervised_learning_vs_offline_reinforcement/

[^1_78]: https://ropiens.tistory.com/144

[^1_79]: https://openreview.net/pdf?id=AP1MKT37rJ

[^1_80]: https://raw.githubusercontent.com/mlresearch/v244/main/assets/yu24a/yu24a.pdf

[^1_81]: https://arxiv.org/pdf/2006.04779.pdf

[^1_82]: https://arxiv.org/pdf/2210.05158.pdf

[^1_83]: https://arxiv.org/html/2301.11734v2

[^1_84]: https://arxiv.org/html/2206.04745v3

[^1_85]: https://www.arxiv.org/pdf/2408.02165.pdf

[^1_86]: https://arxiv.org/html/2511.03695

[^1_87]: https://www.semanticscholar.org/paper/Conservative-Q-Learning-for-Offline-Reinforcement-Kumar-Zhou/28db20a81eec74a50204686c3cf796c42a020d2e

[^1_88]: https://arxiv.org/pdf/2206.00695.pdf

[^1_89]: https://arxiv.org/html/2406.04534v1

[^1_90]: https://arxiv.org/html/2110.04698v2
