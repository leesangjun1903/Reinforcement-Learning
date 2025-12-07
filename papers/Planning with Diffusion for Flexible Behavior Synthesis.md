# Planning with Diffusion for Flexible Behavior Synthesis
### 1. 논문의 핵심 주장과 주요 기여[1]
본 논문은 모델 기반 강화학습(MBRL)의 근본적인 패러다임 전환을 제안한다. **기존의 문제점**: 전통적인 접근법은 환경 역학(dynamics)을 학습하고 이를 고전적인 궤적 최적화기(trajectory optimizer)에 연결하는 방식을 사용하지만, 학습된 모델은 지나친 이용(exploitation)으로 인해 적대적 사례(adversarial examples)를 찾게 되는 문제가 발생한다.[1]

**핵심 제안**: 저자들은 **Diffuser**라는 확산 기반 궤적 생성 모델을 제안하여 모델링과 계획 과정을 통합한다. 이 접근법의 핵심 아이디어는 "샘플링과 계획이 거의 동일하게 된다"는 것이다. 즉, 궤적 레벨의 생성 모델을 직접 학습하여 장기 계획 능력을 향상시킨다.[1]

**주요 기여**:
- 궤적 데이터를 위한 비자동회귀적(non-autoregressive) 확산 모델 설계
- 조건부 샘플링과 인페인팅(inpainting)을 활용한 강화학습 프레임워크
- 시간적 구성성(temporal compositionality)으로 분포 외 궤적 생성 능력
- 보상 함수 재학습 없이 테스트 시간 유연성 제공[1]

***

### 2. 해결하고자 하는 문제
#### 2.1 기존 모델 기반 강화학습의 문제점[1]

1. **계획-모델링 분리의 한계**: 단일 스텝 동역학 모델은 장기 누적 오차로 인해 장기 계획에 부적합하다
2. **분포 외 적대적 이용**: 강력한 궤적 최적화기가 학습된 모델을 과도하게 이용하여 현실적이지 않은 계획을 생성한다
3. **스파스 보상 처리 어려움**: 단계별 보상이 희박한 환경에서 신용 할당(credit assignment)이 어렵다
4. **테스트 시간 유연성 부족**: 새로운 보상 함수나 제약 조건에 대해 모델을 재학습해야 한다[1]

#### 2.2 궤적 최적화의 공식화[1]

표준 궤적 최적화 문제:

$$ a_{0:T}^* = \arg\max_{a_{0:T}} J(s_0, a_{0:T}) = \arg\max_{a_{0:T}} \sum_{t=0}^{T} r(s_t, a_t) $$

여기서 $T$는 계획 지평선이고, 상태는 동역학 $s_{t+1} = f(s_t, a_t)$를 따른다.

***

### 3. 제안하는 방법: Diffuser 모델
#### 3.1 확산 모델의 수학적 기초[1]

확산 모델은 데이터 생성 과정을 반복적 노이즈 제거 절차로 모델링한다:

**정방향 프로세스** (Forward Process):

$$ q(\epsilon^i | \epsilon^{i-1}) = N(\epsilon^i; \sqrt{1-\beta_i}\epsilon^{i-1}, \beta_i I) $$

여기서 $\epsilon^0$는 노이즈 없는 궤적이고, $\epsilon^N \sim N(0, I)$는 표준 가우시안이다.

**역방향 프로세스** (Reverse Process):

$$ p_\theta(\epsilon^{i-1} | \epsilon^i) = N(\epsilon^{i-1}; \mu_\theta(\epsilon^i, i), \Sigma_i) $$

학습 목표:

$$ L = \mathbb{E}_{i, \epsilon, \epsilon^0} \left\| \epsilon - \epsilon_\theta(\epsilon^i, i) \right\|^2 $$

여기서 $\epsilon_\theta$는 점수(score) 함수이고 $\epsilon_\theta(.)$는 노이즈 예측이다.[1]

#### 3.2 궤적 표현 및 아키텍처[1]

**비자동회귀적 예측**: 전통적인 시간 순서 자동회귀 예측을 거부한다. 대신 모든 타임스텝을 동시에 예측한다:

$$ \text{Trajectory} = \begin{bmatrix} s_0 & s_1 & \cdots & s_T \\ a_0 & a_1 & \cdots & a_T \end{bmatrix} $$

이는 제어 문제의 반인과성(anti-causality)을 반영한다. 예를 들어, 목표 조건부 추론에서 $p(s_1 | s_0, s_T)$는 미래 상태에 의존한다[1].

**시간적 국소성** (Temporal Locality): Diffuser는 제한된 수용장(receptive field)을 가진 시간 합성곱을 사용한다. 각 노이즈 제거 단계에서 국소 일관성만 강제되고, 반복적으로 합성된 이 국소 제약들이 전역 일관성을 유도한다.[1]

**모델 아키텍처**: U-Net 구조에 기반한 시간 합성곱 잔여 블록:
- 6개의 반복된 잔여 블록
- 각 블록: 2개의 시간 합성곱 + 그룹 정규화 + Mish 활성화
- 완전 합성곱 구조로 동적 계획 지평선 지원[1]

#### 3.3 강화학습 응용: 조건부 샘플링[1]

RL 문제를 조건부 샘플링으로 재구성한다. 최적성 변수 $O_t$를 정의:

$$ p(O_t = 1) = \exp(r(s_t, a_t)) $$

조건부 분포:

$$ p(\tau | O_{1:T}) \propto p(\tau) \prod_{t=0}^{T} \exp(r(s_t, a_t)) $$

**Classifier-guided Sampling**: 역방향 프로세스 조정:

$$ p(\epsilon^{i-1} | \epsilon^i, O_{1:T}) = N\left(\epsilon^{i-1}; \mu(\epsilon^i, i) + \Sigma_i \nabla_{\epsilon^i} \log p(O_{1:T} | \epsilon^i), \Sigma_i\right) $$

여기서 기울기는:

$$ \nabla_{\epsilon^i} \log p(O_{1:T} | \epsilon^i) = \nabla_{\epsilon^i} \sum_{t=0}^{T} r(s_t, a_t) = \nabla_{\epsilon^i} J(\tau) $$

별도 보상 예측 네트워크 $J_\phi(\epsilon^i)$가 학습되어 이 기울기 계산에 사용된다.[1]

#### 3.4 목표 조건부 RL: 인페인팅[1]

제약 만족 문제로 재구성:

$$ h(c_t, s_0, a_0, \ldots, s_T, a_T) = \begin{cases} 1 & \text{if } c_t = s_t \\ 0 & \text{otherwise} \end{cases} $$

인페인팅은 제약된 상태를 고정하고 나머지를 샘플링하는 방식으로 구현:

$$ \epsilon^{i-1}_{t} = \begin{cases} \text{sample from } p(\epsilon^{i-1} | \epsilon^i) & \text{if } t \text{ unconstrained} \\ c_t & \text{if } t \text{ constrained} \end{cases} $$

모든 확산 스텝 후에 제약 값으로 대체된다.[1]

#### 3.5 알고리즘: 안내된 확산 계획[1]

**Algorithm 1: Guided Diffusion Planning**

```
Require: Diffuser θ, guide J, scale λ, covariances Σ
while not done do
    Observe state s
    Initialize plan τ ~ N(0, I)                    // Step 3
    for i = N, ..., 1 do
        Compute μθ(τ^i, i)                         // Step 6
        Compute gradient ∇_τ J(τ^i)                // Step 7
        τ^{i-1} ~ N(μθ(τ^i, i) + λΣ_i∇_τ J, Σ_i) // Step 8
        Constrain τ^{i-1}_{s0} = s                 // Step 10
    Execute first action a_0 and repeat            // Step 11
```

가이드 스케일 $\lambda$는 보상 기울기의 영향을 조절한다.[1]

***

### 4. Diffuser의 주요 성질
#### 4.1 학습된 장기 계획[1]

Diffuser는 장기 정확도를 위해 학습되므로 단일 스텝 오차의 누적을 겪지 않는다. 이를 수학적으로:

- **전통적 모델**: 오차 누적 = $O(T \cdot \epsilon_{\text{single-step}})$
- **Diffuser**: 오차 = $O(\epsilon_{\text{trajectory-level}})$

결과적으로 스파스 보상 환경에서 매우 효과적이다.[1]

#### 4.2 시간적 구성성[1]

Diffuser는 국소 일관성을 반복적으로 개선하여 전역 일관성을 달성한다. 이를 통해 분포 내 부분 수열을 새로운 방식으로 조합하여 분포 외 궤적을 생성할 수 있다:

- **예**: 직선 궤적으로만 학습 → V자 궤적 생성 가능
- 이는 마르코프 모델의 구성성을 비마르코프 설정에서 재현한다[1]

#### 4.3 가변 길이 계획[1]

모델이 완전히 합성곱이므로, 계획 지평선은 입력 노이즈의 차원으로 결정된다. 따라서 아키텍처 변경 없이 다양한 지평선에 대응 가능하다.[1]

#### 4.4 작업 구성성[1]

Diffuser는 보상 함수와 독립적이다. 테스트 시간에 새로운 보상으로 가중치를 재학습하지 않고도 다양한 작업에 적응할 수 있다:

$$ J_{\text{multi-task}} = \lambda_1 J_1 + \lambda_2 J_2 + \cdots $$

***

### 5. 성능 향상 및 실험 결과
#### 5.1 장기 계획: Maze2D 환경[1]
Maze2D 환경에서 Diffuser는 기저선을 현저히 능가한다:

| 환경 | MPPI | CQL | IQL | Diffuser | 개선율(%) |
|------|------|-----|-----|----------|-----------|
| **단일 작업** | | | | | |
| U-Maze | 33.2 | 5.7 | 47.4 | **113.9** | +144% (vs IQL) |
| Medium | 10.2 | 5.0 | 34.9 | **121.5** | +248% |
| Large | 5.1 | 12.5 | 58.6 | **123.0** | +110% |
| **다중 작업** | | | | | |
| U-Maze | 41.2 | - | 24.8 | **128.9** | +420% (vs IQL) |
| Medium | 15.4 | - | 12.1 | **127.2** | +950% |
| Large | 8.0 | - | 13.9 | **132.1** | +850% |

**핵심 관찰**:
- Diffuser는 다중 작업 설정에서 기저선(IQL)보다 4.2배 이상 성능 향상
- MPPI (지면 진실 동역학 사용)도 장기 계획에서 어려움을 보임, Diffuser의 우월성 강조[1]

#### 5.2 테스트 시간 유연성: 블록 쌓기[1]

| 작업 | BCQ | CQL | Diffuser | 개선율 |
|------|-----|-----|----------|--------|
| 무조건 쌓기 | 0.0 | 24.4 | **58.7** | +141% |
| 조건부 쌓기 | 0.0 | 0.0 | **45.6** | ∞ |
| 재배치 | 0.0 | 0.0 | **58.9** | ∞ |

이 작업들은 학습 중에는 보지 못한 목표 상태를 요구한다. Diffuser는 한 번의 모델 학습으로 세 가지 모두 처리하며, 두 기준선은 조건부 설정에서 완전히 실패한다.[1]

#### 5.3 오프라인 강화학습: D4RL 벤치마크[1]
D4RL 벤치마크에서 Diffuser는 경쟁력 있는 성능을 보인다:

**평균 성능 (데이터셋별)**:
- Medium-Expert: CQL(101.4), TT(102.6), Diffuser(99.7) - 경쟁력 있음
- Medium: MOReL(84.8), Diffuser(65.2) - 중간 수준
- Medium-Replay: Diffuser(67.3) - 경쟁력 있음

**중요한 발견**:
- Diffuser를 MPPI 같은 고전적 궤적 최적화기와 함께 사용하면 무작위 수준 성능 (최악)
- 이는 Diffuser의 효과가 **결합된 모델-계획 설계**에서 비롯됨을 증명[1]

#### 5.4 계산 효율성: 워밍 스타트[1]

계획 속도는 제한 요소이다. 이전에 생성된 계획으로부터 워밍 스타트:

- **100 확산 단계** (기본): 기준선 성능
- **20 단계**: 작은 성능 저하, 5배 빠름
- **2 단계**: 중간 성능 저하, 50배 빠름

이는 실제 배포에서 실시간 성능과 정확성 간 트레이드오프를 가능하게 한다.[1]

***

### 6. 모델 일반화 성능 향상 가능성
#### 6.1 현재 일반화 메커니즘[1]

**1. 시간적 구성성**: 분포 내 부분 수열을 조합하여 분포 외 궤적 생성:
- 학습: 직선 경로만
- 생성: V자 모양 경로 (새로운 조합)
- 메커니즘: 국소 일관성의 반복적 개선이 전역 일관성으로 유도

**2. 작업 구성성**: 보상 함수 독립성:
- 학습 중 보지 못한 보상 함수에 적응
- 추가 학습 없이 테스트 시간 유연성

**3. 궤적 레벨 학습**: 
- 단일 스텝 오차 누적 회피
- 장기 정확도가 자동 최적화됨

#### 6.2 일반화 향상을 위한 가능한 방향[2][3][4]

2020년 이후의 관련 연구들은 다음과 같은 방향을 제시한다:

**1. AdaptDiffuser (2023)**:[4]
- 보상 기울도로 합성 전문가 데이터 생성
- 판별기로 고품질 데이터 선택
- 모델 재학습으로 미학습 작업에 적응
- 결과: Maze2D에서 20.8%, 로봇 조작에서 27.9% 향상

**2. 상태 재구성 학습 (2023)**:[5]
- 분포 외 상태 일반화 강화
- 분포 외-분포 내 상태 간 이동 가능
- 희박 작업에서 167% 향상

**3. 온라인 재계획 (2023)**:[6]
- 동적 환경에서 선택적 재계획
- 모델 우도 추정으로 재계획 시기 결정
- 이전 계획으로 부트스트래핑
- Maze2D에서 38% 향상

**4. 보상 기울도 안내의 개선**:
- 더 정교한 가이드 스케일 학습
- 다중 보상 조합의 최적화
- 동적 기울도 스케일링

**5. 제약 조건 정렬 (2025)**:[7]
- 동역학 타당성 명시적 강제
- 하이브리드 손실 함수로 제약 위반 페널티
- 궤적 최적화 문제에서 실행 가능성 보장

#### 6.3 분포 외 일반화의 도전 과제[8][9][10][11]

**분포 편이 문제**:
- 오프라인 RL의 근본적 문제: 학습 데이터와 테스트 데이터의 분포 차이
- Diffuser도 극단적 분포 외 상태에서는 성능 저하 가능

**현재 접근법**:
- **보수적 추정**: 분포 외 행동의 과도한 낙관적 평가 방지
- **표현 학습**: 불변 표현으로 분포 간 격차 감소
- **데이터 증강**: 동역학 제약을 유지하며 분포 확장

***

### 7. 한계점
#### 7.1 논문에 명시된 한계[1]

1. **계산 비용**: 개별 계획 생성이 반복적 노이즈 제거로 인해 느림
   - 솔루션: 워밍 스타트로 5-10배 속도 향상 가능
   - 트레이드오프: 계획 정확도 약간 감소

2. **D4RL 벤치마크에서의 혼합 성능**: 
   - Medium-Expert에서 경쟁력 있음 (99.7 vs TT 102.6)
   - Medium에서는 중간 수준 (65.2 vs MOReL 84.8)
   - 이유: 이 환경은 단일 스텝 정확도를 더 많이 요구[1]

3. **모델-계획 결합의 필요성**:
   - Diffuser + MPPI = 무작위 수준 성능
   - 모델이 계획 알고리즘과 함께 설계되어야 함
   - 전이 가능성 제한[1]

#### 7.2 제기 가능한 추가 한계

1. **매우 높은 차원 문제**: 수백 개 상태/행동 차원 문제로의 확장성
2. **동역학 변화가 큰 환경**: 시간에 따라 동역학이 크게 변하는 비정상 환경
3. **다양한 궤적 품질**: 학습 데이터가 매우 이질적일 때 성능 저하[1]

***

### 8. 앞으로의 연구에 미치는 영향 및 고려사항
#### 8.1 패러다임적 영향[1][3][4]

**1. 모델-계획 통합의 가치 입증**:
- 분리된 모델과 계획 절차의 한계 명확화
- 생성 모델의 계획 문제 특화 설계 중요성 강조
- 강화학습에서 생성 모델의 활용도 증대

**2. 확산 모델의 강화학습 응용 개척**:
- 이전: 이미지/음성 생성 중심
- 이후: 의사결정 및 제어 문제의 자연스러운 프레임워크
- 다양한 확산 기반 강화학습 연구 촉발[2][12][3][4][6]

**3. 비자동회귀적 궤적 모델링**:
- 기존: 자동회귀 정책/모델 (순차 예측)
- 혁신: 동시 다중 타임스텝 예측의 이점
- 재귀적 오류 누적 회피[1]

#### 8.2 2020년 이후 관련 연구 동향[12][3][4][6][13][14][15][2]

**A. Diffusion 기반 계획의 직접적 확장**:

| 연구 | 주요 기여 | 성능 |
|------|---------|------|
| **MTDiff (2023)[3]** | 다중 작업 확산 계획, 프롬프트 학습 | Meta-World 50개 작업, Maze2D 우월 |
| **AdaptDiffuser (2023)[4]** | 자기 진화 계획, 합성 데이터 생성 | Maze2D +20.8%, 로봇 +27.9% |
| **Adaptive Replanning (2023)[6]** | 동적 환경 재계획, 모델 우도 기반 | Maze2D +38% |
| **Trajectory Diffuser (2024)[16]** | 2단계 생성-최적화 분해 | 3-10배 빠른 추론 |
| **Decision Diffuser (2023)[17]** | 리턴 조건부 확산, 분류 자유 기울도 | D4RL에서 경쟁력 있는 성능 |

**B. 확산 모델의 이론적 진전**:

1. **Consistency Models (2023)**:[18]
   - 1스텝 생성으로 이론적 기초 제공
   - 계획의 계산 효율성 50-100배 향상
   - Diffuser 워밍 스타트와 상호보완

2. **Score-based Models (2024)**:[19]
   - 점수 함수의 근사 이론
   - 차원 독립적 수렴율 증명
   - 궤적 생성의 이론적 보증

**C. 오프라인 강화학습의 진화**:

1. **Diffusion Policies for OOD Generalization (2023)**:[5][20]
   - 상태 재구성으로 분포 외 일반화
   - D4RL에서 최신 수준 달성
   - Diffuser의 약점(제한된 OOD 일반화) 보완

2. **Policy-Guided Diffusion (2024)**:[13]
   - 목표 정책과 행동 정책의 균형
   - 합성 데이터의 역학 오류 감소
   - 기존 오프라인 알고리즘과 호환

**D. 강화학습 이외 응용**:

1. **로봇 조작**: 블록 쌓기, 픽-앤-플레이스 (Diffuser의 강점)
2. **자율 주행**: Gen-Drive (보상 모델 + RL 미세조정)[21]
3. **비디오 예측**: 궤적-조건 생성[22][23]
4. **텍스트-음성 합성**: 음성 품질 우도 기반 강화학습[12]

#### 8.3 미래 연구 시 고려사항

**1. 계산 효율성 개선**:
- Consistency Models 또는 Flow Matching 통합
- 조기 정지 기준 개발
- 캐시된 확산 스텝 활용[24][25]

**2. 이론적 보증 강화**:
- 분포 외 일반화의 이론적 한계 도출
- 조건부 샘플링의 정렬 오차 분석
- 확산 단계 수와 계획 성능 관계 수립[19][18]

**3. 확장성 개선**:
- 초고차원 문제(이미지 관찰)에 대한 효율적 확산 모델
- 계층적 계획 (고수준 추상화 + 저수준 세부)
- 모달리티 통합 (텍스트 지시 + 시각 관찰 + 보상)[23][26]

**4. 실용적 배포**:
- 강화된 안정성: 제약 조건 명시적 강제[7]
- 샘플 효율: 오프라인 데이터 + 온라인 상호작용 혼합
- 전이 학습: 다양한 환경 간 정책 전이[27]

**5. 하이브리드 접근법**:
- Diffusion + Transformer: 더 나은 시퀀스 모델링[15][28]
- Diffusion + 가치 함수: 값 기반 유도[14][10]
- Diffusion + 제약: 동역학 및 안전 제약 동시 만족[29][7]

**6. 멀티태스크 및 일반화**:
- 프롬프트 기반 조건화로 언어 지시 지원[30][31]
- 메타 러닝: 새로운 환경에 신속히 적응
- 커리큘럼 학습: Diffusion 기반 과제 생성[32]

***

### 9. 결론
"Planning with Diffusion for Flexible Behavior Synthesis"는 **모델 기반 강화학습의 기본 패러다임을 재검토**하는 획기적 연구이다. Diffuser의 핵심 창의성은 다음에 있다:

1. **모델과 계획의 통합**: 분리된 추상화를 거부하고 상호 영향적 설계 추구
2. **비자동회귀적 생성**: 궤적의 모든 타임스텝을 동시에 예측하여 재귀적 오류 누적 방지
3. **테스트 시간 유연성**: 사전학습 후 보상 함수 재학습 없이 다양한 작업에 적응

**성능 측면에서**:
- Maze2D의 장기 계획에서 기저선 대비 4배 이상 성능 향상
- 블록 쌓기 같은 복잡한 조작 작업에서 유일한 경쟁력 있는 방법
- D4RL에서는 경쟁력 있지만 일부 환경에서는 기존 방법에 미치지 못함

**일반화 성능**의 관점에서, Diffuser는 고유한 장점과 한계를 모두 보유한다:
- **장점**: 시간적 구성성으로 새로운 궤적 조합 생성, 단일 모델로 다중 작업 처리
- **한계**: 극단적 분포 외 상태에서는 성능 저하 가능, D4RL의 일부 설정에서 혼합된 결과

2020년 이후의 후속 연구들은 **Consistency Models, Adaptive Replanning, State Reconstruction** 등으로 Diffuser의 한계를 보완하고 있으며, 강화학습, 자율 주행, 로봇 조작 등 다양한 분야에서 확산 기반 계획의 실용성을 입증하고 있다. 앞으로의 연구는 **계산 효율성, 이론적 보증, 확장성, 그리고 제약 조건 통합**에 초점을 맞춰 나갈 것으로 전망된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3f75f04d-68d6-403a-98e1-c52fe8fbc592/2205.09991v2.pdf)
[2](https://arxiv.org/abs/2404.04356)
[3](https://arxiv.org/abs/2305.18459)
[4](https://arxiv.org/abs/2302.01877)
[5](https://ieeexplore.ieee.org/document/10423845/)
[6](https://arxiv.org/abs/2310.09629)
[7](https://arxiv.org/abs/2504.00342)
[8](https://www.semanticscholar.org/paper/348a855fe01f3f4273bf0ecf851ca688686dbfcc)
[9](https://ojs.aaai.org/index.php/AAAI/article/view/35402)
[10](https://arxiv.org/abs/2411.07934)
[11](https://arxiv.org/pdf/2205.11027.pdf)
[12](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[13](https://arxiv.org/pdf/2404.06356.pdf)
[14](https://arxiv.org/pdf/2502.02316.pdf)
[15](https://arxiv.org/abs/2401.08478)
[16](https://arxiv.org/abs/2407.16142)
[17](https://arxiv.org/pdf/2211.15657.pdf)
[18](https://arxiv.org/abs/2310.02279)
[19](https://proceedings.iclr.cc/paper_files/paper/2024/file/334da4cbb76302f37bd2e9d86f558869-Paper-Conference.pdf)
[20](https://arxiv.org/pdf/2307.04726.pdf)
[21](https://ieeexplore.ieee.org/document/11127286/)
[22](https://arxiv.org/pdf/2409.16950.pdf)
[23](https://dl.acm.org/doi/10.1145/3641519.3657513)
[24](https://www.semanticscholar.org/paper/38c3acbe4531a123acde40c9d93abc63d804c3f9)
[25](https://arxiv.org/abs/2404.02241)
[26](https://arxiv.org/pdf/2411.17376.pdf)
[27](https://arxiv.org/html/2502.14998v1)
[28](https://www.semanticscholar.org/paper/Decision-Transformer:-Reinforcement-Learning-via-Chen-Lu/c1ad5f9b32d80f1c65d67894e5b8c2fdf0ae4500)
[29](https://arxiv.org/abs/2510.04436)
[30](https://wnzhang.net/teaching/sjtu-rl-2024/slides/15-diffusion-rl.pdf)
[31](https://pure.kaist.ac.kr/en/publications/guided-trajectory-generation-with-diffusion-models-for-offline-mo/)
[32](https://proceedings.neurips.cc/paper_files/paper/2024/file/b0e89a49af1fb2ebea69bfc39df0be4a-Paper-Conference.pdf)
[33](https://www.semanticscholar.org/paper/b015f66f73db514b5b24b1e7a9d2a26607ca19c8)
[34](https://arxiv.org/abs/2403.10794)
[35](https://arxiv.org/abs/2306.00603)
[36](https://arxiv.org/pdf/2402.03570.pdf)
[37](http://arxiv.org/pdf/2503.00535.pdf)
[38](https://arxiv.org/html/2310.02505v3)
[39](https://arxiv.org/abs/2305.13122)
[40](https://arxiv.org/pdf/2502.20476.pdf)
[41](https://proceedings.mlr.press/v238/regol24a/regol24a.pdf)
[42](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06735.pdf)
[43](https://pubs.acs.org/doi/10.1021/jacs.2c13467)
[44](https://neurips.cc/virtual/2024/poster/96200)
[45](https://arxiv.org/abs/2311.01223)
[46](https://www.semanticscholar.org/paper/9560080a2c32682bd1c1a9850a54ca6163f1956e)
[47](https://www.semanticscholar.org/paper/23abe79046d7f7b430d5d21b6a93598d0aa1b9c2)
[48](https://arxiv.org/abs/2206.04745)
[49](https://www.semanticscholar.org/paper/e8b72714171367b9d330e51b460a9e6b1f136b80)
[50](https://ieeexplore.ieee.org/document/10446781/)
[51](http://arxiv.org/pdf/2411.07934.pdf)
[52](https://arxiv.org/pdf/2309.07578.pdf)
[53](https://arxiv.org/pdf/2209.15256.pdf)
[54](http://arxiv.org/pdf/2211.14827.pdf)
[55](https://arxiv.org/html/2309.08925v3)
[56](http://arxiv.org/pdf/2209.13132.pdf)
[57](https://arxiv.org/abs/2307.04726)
[58](https://www.nature.com/articles/s41467-023-41379-3)
[59](https://liner.com/review/doubly-mild-generalization-for-offline-reinforcement-learning)
[60](https://www.sciencedirect.com/science/article/pii/S147466701530690X)
[61](https://ebbnflow.tistory.com/378)
[62](https://proceedings.mlr.press/v235/wang24aj.html)
[63](https://arxiv.org/html/2406.08855v1)
[64](https://openreview.net/pdf?id=a7APmM4B9d)
[65](https://ieeexplore.ieee.org/document/10423845)
[66](https://arxiv.org/abs/2406.00356)
[67](https://www.semanticscholar.org/paper/ade78b8c55d8e241ad2ba7e7adcd8c5cfecd0c7d)
[68](https://arxiv.org/abs/2403.12510)
[69](https://arxiv.org/abs/2403.00835)
[70](https://arxiv.org/abs/2405.11252)
[71](https://ieeexplore.ieee.org/document/11094017/)
[72](https://arxiv.org/abs/2401.10150)
[73](http://arxiv.org/pdf/2404.19330.pdf)
[74](https://arxiv.org/pdf/2408.13918.pdf)
[75](https://arxiv.org/pdf/2503.10434.pdf)
[76](https://arxiv.org/html/2404.15380v1)
[77](http://arxiv.org/pdf/2402.08698.pdf)
[78](https://arxiv.org/html/2501.00184v1)
[79](http://arxiv.org/pdf/2404.06351.pdf)
[80](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/JoaoCarvalho/thesi_mark_SBMMotionPlanning.pdf)
[81](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_210.pdf)
[82](https://proceedings.iclr.cc/paper_files/paper/2024/file/852f50969a9e523ec41d26f2f68bd456-Paper-Conference.pdf)
[83](https://openreview.net/forum?id=ymjI8feDTD)
[84](https://www.corelinesoft.com/blog/CoreWiki/Scorebased-generative-modeling-SBGM%EC%9D%B4%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C-4)
[85](https://arxiv.org/abs/2305.14550)
