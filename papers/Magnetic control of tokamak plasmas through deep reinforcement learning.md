# Magnetic control of tokamak plasmas through deep reinforcement learning

### 1. 핵심 주장과 주요 기여[1]

본 논문은 토카막의 플라즈마 자기장 제어를 위해 심층강화학습(Deep Reinforcement Learning, DRL)을 처음으로 적용한 연구입니다. 핵심 주장은 **기존의 공학적 제어 설계 패러다임을 인공지능 기반의 목적지향적 최적화로 전환**할 수 있다는 점입니다.[1]

주요 기여는 다음과 같습니다:

- **단일 신경망 컨트롤러**: 기존의 다층 제어 아키텍처(다중 독립적 PID 제어기)를 하나의 신경망으로 통합하여 시스템 복잡도를 획기적으로 감소[1]
- **시뮬레이션-하드웨어 전이(Sim-to-Real Transfer)**: 토카막 시뮬레이터에서 학습한 정책이 직접 TCV 실기기에서 제로샷(Zero-Shot) 전이 성공[1]
- **다양한 플라즈마 구성 제어**: 종래적 형태, 길쭉한(Elongated) 형태, 음성삼각형(Negative Triangularity), 눈송이(Snowflake) 구성, 그리고 동시에 존재하는 두 개의 플라즈마 방울(Droplets) 제어 달성[1]

***

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능[1]

#### 2.1 해결하고자 하는 문제

토카막 자기장 제어는 다음과 같은 도전과제를 포함합니다:

- **고차원 측정 및 제어**: 92개의 입력 센서와 19개의 제어 코일에 대한 실시간 피드백 제어
- **불안정성 성장**: 수직 불안정성 성장률이 1.4 kHz에 도달하는 빠른 역학
- **비선형 동역학**: 플라즈마 압력, 전류밀도 프로필, 저항성 변화에 따른 복잡한 시스템 거동
- **간접 측정**: 직접 관찰 불가능한 플라즈마 형태를 자기장 측정으로부터 추론

#### 2.2 제안하는 방법[1]

**삼단계 아키텍처**:

**1단계: 목적 함수 설정**

목표를 다양한 물리량에 대한 개별 보상 성분들로 정의하고, 이들을 다음과 같이 결합:

$$R_{\text{total}} = w_1 \cdot q_{\text{Ip}} + w_2 \cdot q_{\text{shape}} + w_3 \cdot q_{\text{position}} + \ldots$$

여기서 $q_i$는 각 목표에 대한 품질 측도(0~1), $w_i$는 가중치입니다.

**2단계: 강화학습 알고리즘**

**MPO(Maximum A Posteriori Policy Optimization)** 알고리즘을 사용. Actor-Critic 구조:

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}[Q(s, a) \log \pi(a|s)]$$

$$\mathcal{L}_{\text{critic}} = \mathbb{E}[(Q(s,a) - r(s,a) - \gamma Q(s', a'))^2]$$

여기서 $s$는 상태, $a$는 행동, $r$은 보상, $\gamma = 0.99$는 할인 계수입니다.

**3단계: 실시간 배포**

학습된 정책을 실시간 제어 시스템에 컴파일하여 배포 (10 kHz 제어율).

#### 2.3 모델 구조[1]

**동역학 시뮬레이터 (FGE)**:

자유 경계 플라즈마 진화 모델:

$$\frac{\partial \Psi}{\partial t} = R_p \nabla^2 \Psi + \text{Source terms}$$

여기서 $\Psi$는 자기선속(Magnetic flux)이며, Grad-Shafranov 방정식으로 평형 조건 만족:

$$\frac{\partial \pi}{\partial r} = J_\phi B_r - J_r B_\phi$$

플라즈마 전류 진화:

$$\frac{d I_p}{dt} = \frac{V_\text{loop} - I_p R_p}{L_p}$$

**신경망 아키텍처**:

- **액터(정책) 네트워크**: 빠른 실시간 실행을 위해 경량화
  - 입력층: 132개 (센서 측정) + 23개 (시간변화 목표) = 총 155차원
  - 숨겨진층: LayerNorm → Tanh 활성화 → 3층 MLP (각 256개 단위)
  - 출력층: 19개 (코일 전압)
  - 총 파라미터: 266,280개

$$\pi(a|s) = \mathcal{N}(\mu(s), \sigma(s)^2)$$

- **비평자(가치) 네트워크**: 학습 중에만 사용되는 표현력 높은 신경망
  - LSTM층 (256단위) + 2층 MLP (각 256단위)
  - 총 파라미터: 718,337개 (비대칭 설계)

#### 2.4 센서 및 작동기 모델링

시뮬레이션-실제 간극 극복을 위해 다음을 모델링:

$$\text{Measured}_t = \text{LowPass}(\text{True}_t) + \delta t_{\text{delay}} + \mathcal{N}(0, \sigma_{\text{noise}})$$

$$V_{\text{command}} = V_{\text{ideal}} + V_{\text{bias}} + \Delta V_{\text{random}}$$

#### 2.5 도메인 랜더마이제이션(Domain Randomization)[1]

강건성을 위해 훈련 중 플라즈마 파라미터를 변화:

- 플라즈마 저항성 $R_p$: 로그 균등 분포
- 정규화 압력 $\beta_p$: 로그 균등 분포  
- 축 안전 계수 $q_A$ (전류밀도 형태): 로그 균등 분포

$$\log R_p \sim \text{LogUniform}(R_{p,\min}, R_{p,\max})$$

***

### 3. 성능 향상[1]

#### 3.1 기본 능력 시연 (Figure 2)[1]

기본적인 플라즈마 제어 작업 달성:

| 작업 | RMSE | 목표 | 달성도 |
|-----|------|------|-------|
| 플라즈마 전류 ($I_p$) | 0.62 kA | ±5 kA (3%) | ✓ |
| 형태 (Shape) | 0.75 cm | ±2 cm (8%) | ✓ |
| 수직 위치 안정화 | 성장률 150 Hz 안정화 | - | ✓ |

#### 3.2 고급 제어 시연[1]

**①길쭉한 플라즈마 (Elongation 1.9)**
- 수직 불안정성 성장률: 1.4 kHz
- 형태 추적 RMSE: 0.018
- 전류 RMSE: 1.2 kA

**②중성빔 가열(H-모드) 조건**
- $\beta_p = 1.12$에서 플라즈마 성질 변화 추적
- 전류 RMSE: 2.6 kA
- 외부 가열 변화에 강건한 응답

**③음성삼각형 구성**
- 삼각형 $\delta = -0.8$ 정확히 달성
- 삼각형 RMSE: 0.070
- 전류 RMSE: 3.5 kA

**④눈송이 구성**
- 시간변화 X점 거리 제어
- 초기 분리: 34 cm → 최종: 6.6 cm
- 결합 RMSE: 3.7 cm

#### 3.3 다중 도메인 플라즈마 (Droplets)[1]

**역사적 성과**: 동시에 두 개의 독립적인 플라즈마 방울 유지 (200 ms)
- 각 도메인의 전류 증가 동시 제어
- 단순한 보상 함수로도 자동으로 최적 형태 발견

***

### 4. 모델 일반화 성능 분석[1]

#### 4.1 일반화 메커니즘[1]

**①도메인 랜더마이제이션**[1]

훈련 중 물리적 파라미터 변화를 통한 강건한 정책 학습:

- 플라즈마 저항성, 압력, 전류밀도 프로필 변화
- 센서 노이즈, 시간 지연, 전압 오프셋 포함
- 각 에피소드마다 새로운 파라미터 샘플링

**학습된 파라미터는 직접 노출되지 않음** → 에이전트가 센서 측정으로부터만 추론

**②비대칭 Actor-Critic 설계**[1]

**핵심 혁신**: 비평자는 표현력 높은 LSTM 사용, 액터는 실시간 제약

$$Q_{\text{critic}} = f_{\text{large}}(s_t, a_t, s_{t-1}, a_{t-1}, \ldots) \text{ (LSTM)} $$

$$\pi_{\text{actor}} = f_{\text{small}}(s_t) \text{ (MLP)}$$

비평자가 **복잡한 상태-행동 역학과 지연을 처리** → 액터에 지식 증류

Extended Data Figure 5 분석:
- 비대칭 설계: 최고 성능
- 대칭 설계 (작은 비평자): 성능 40% 저하
- 표현력만 증가 (피드포워드 구조 유지): 개선 미미

**③학습된 상태 재구성**[1]

기존 접근: 실시간 평형 재구성 필요
본 논문: 신경망이 자동으로 센서로부터 상태 추론

$$\hat{s}_t = f_{\text{LSTM}}([m_t, m_{t-1}, \ldots, a_{t-1}, a_{t-2}, \ldots])$$

#### 4.2 다른 토카막으로의 전이 가능성[1]

**적용 조건**:

- 자유 경계 시뮬레이터 기반 모델 일반성
- 기본 수정으로 다른 기기 적용 가능:
  - 머신 기하학적 파라미터
  - 센서/코일 특성
  - 작동 조건 범위

**일반성**: 입출력 차원 자동 조정, 알고리즘 수정 불필요

#### 4.3 반복 가능성 검증[1]

같은 정책을 동일한 목표로 두 번 실행:
- 플라즈마 형태 유사도 (RMSE): 1.5 cm 이하
- 길쭉함, 삼각형 등 매개변수: 높은 일관성
- 센서 노이즈와 미세한 변화에도 안정적

***

### 5. 논문의 한계[1]

#### 5.1 기술적 한계

- **정상상태 오차**: 일부 목표값에서 작은 오차 존재 (Extended Data Fig. 3)
  - 해결책: 순환 신경망(RNN) 정책 사용 (미래 연구)
  - 주의: 시뮬레이터 동역학에 과적합 위험

- **붕괴 보장 부재**: 학습된 정책이 플라즈마 붕괴(Disruption)를 완전히 방지하지 못함
  - 기계 보호 계층(Fallback controller, Interlock) 필요

- **계산 복잡도**: 도메인 랜더마이제이션으로 인한 높은 훈련 시간 (1~3일, 5,000 병렬 액터)

#### 5.2 물리적 한계

- **시뮬레이터 신뢰도**: 자유 경계 모델이 특정 영역에서 부정확
  - 학습된 영역 회피(Learned-Region Avoidance) 도입으로 제어

- **미지의 토카막 구성**: 이전 데이터 부족한 새로운 형태 제어 시 시뮬레이터만 의존

***

### 6. 최신 연구 기반 미래 연구 방향 및 고려사항[2][3][4][5][6][7]

#### 6.1 논문의 후속 연구 성과[2]

**2024년 연구 진전**:

**①플라즈마 붕괴 회피 (Seo et al., 2024)**[2]

토카막 플라즈마의 **찢김 불안정성(Tearing Instability)** 회피를 심층강화학습으로 해결:
- 장애물 회피 문제로 재정의
- 실험 데이터 검증
- 논문의 기본 프레임워크를 붕괴 방지로 확장

**②고충실도 데이터 기반 동역학 모델 (HL-3 Tokamak)**[3]

- 자유원리 시뮬레이터 대신 신경망 기반 대리 모델 개발
- 계산 속도 대폭 향상으로 RL 훈련 효율화

**③신경 상미분방정식(Neural ODE) 적용**[5]

- Grad-Shafranov 방정식 제약이 있는 GS-DeepNet 개발
- 물리 제약을 자동으로 학습하여 일반화 성능 향상

**④Fourier Neural Operators를 이용한 플라즈마 시뮬레이션**[6]

- 다중 시간 규모 플라즈마 거동 모델링
- 기존 수치 해석보다 수백 배 빠른 예측

#### 6.2 최신 산업 협력: Google DeepMind - Commonwealth Fusion Systems[8][9]

**2025년 가장 획기적 진전**:

DeepMind와 CFS(Commonwealth Fusion Systems)의 SPARC 토카막 협력:

**①TORAX 소프트웨어 개발**[9][8]
- 오픈소스 미분 가능한(Differentiable) 플라즈마 시뮬레이터
- 여러 AI 모델 결합 가능
- 다중 목적 최적화 가능

**②다중 목적 최적화 확장**[8]
- 기존: 플라즈마 형태 제어만 다룸
- 미래: 동시에 여러 성능 지표 최적화
  - 융합 전력 최대화
  - 열 부하 관리
  - 기기 안전 마진 증대

**③진화 알고리즘 통합**[8]
- AlphaEvolve와 같은 진화 알고리즘 결합
- 광범위한 운영 시나리오 탐색
- 최적 전략 신속 도출

#### 6.3 일반화 성능 향상을 위한 향후 연구 방향

**①전이학습(Transfer Learning) 강화**[3][1]

- 기존: 단일 토카막(TCV) 검증
- 미래: 여러 토카막 데이터로 사전학습
- 새 기기에 빠른 미세조정

**②메타강화학습(Meta-RL)**

- 다양한 플라즈마 조건 빠르게 적응
- 분포 변화에 강건한 정책

**③물리 제약 신경망(Physics-Informed Neural Networks, PINNs)**[5]

Grad-Shafranov 방정식 등 물리 법칙을 손실함수에 직접 포함:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{physics}} \mathcal{L}_{\text{GS}}$$

여기서:

$$\mathcal{L}_{\text{GS}} = \left\| \frac{\partial \pi}{\partial r} - \frac{1}{B^2}\left(J_\phi B_r - J_r B_\phi\right) \right\|^2$$

**장점**: 물리적 타당성 자동 보장 → 일반화 개선

**④불확실성 정량화**

- Bayesian Neural Networks, Ensembles 활용
- 신뢰 영역 외 경고 시스템
- 안전 여유 자동 설정

**⑤다중 물리 통합 시뮬레이터**[5]

현재: Grad-Shafranov 평형만 모델링
미래: 다음 포함 가능
- 에너지/전류 수송 동역학
- 난류 모델링
- 자기유체역학(MHD) 불안정성

#### 6.4 실제 배포 시 고려사항

**①Sim-to-Real 격차 지속 해소**[8][1]

| 측면 | 현재 (TCV) | SPARC (미래) | 고려사항 |
|------|-----------|------------|---------|
| 크기 | 중규모 | 대형 | 스케일 효과, 자기장 분포 차이 |
| 조건 | 상온 | 극저온 | 전도도, 저항 변화 |
| 환경 | 기존 기기 | 새 기기 | 미지의 비선형성 |
| 가열 | ECRH, NBI | RF, ICRH | 새로운 물리 현상 |

**대응 전략**:
- 다중 기기 시뮬레이터 훈련
- 온라인 정책 개선(Online Policy Improvement)

**②안전 검증 프레임워크**[1]

$$\mathcal{L}_{\text{safety}} = \text{penalty}(\text{constraint violation}) + \text{penalty}(\text{disruption risk})$$

- 비선형 동역학 안정성 분석
- 확률적 모델 예측 제어(Probabilistic MPC)
- Machine protection layer 통합

**③계산 효율성**[1]

현재: 10 kHz 실시간 실행 달성 (TCV)
미래 확장:
- 더 큰 신경망 (Larger SPARC, ITER 준비)
- 추론 최적화: 양자화, 가지치기, 증류
- 하드웨어 가속화 (GPU/NPU 활용)

#### 6.5 EUROfusion의 15개 AI/ML 프로젝트[10]

2024년 선정된 최신 주제:
- 고차 물리 통합 모델
- 다중 토카막 데이터 활용
- 실시간 성능 최적화
- 위험 예측 모델

***

### 7. 결론: 논문의 과학적 의의 및 실용적 함의

**학술적 기여**:
- **RL의 실세계 응용**: 복잡한 고차원, 고속 역학 시스템의 첫 성공적 RL 제어 사례
- **Sim-to-Real 갭 극복**: 시뮬레이터만 학습해도 실기계 제어 가능 증명
- **비대칭 Actor-Critic**: 제약 조건 하에서의 새로운 아키텍처 패턴 제시

**실용적 영향**:
- **개발 시간 단축**: 기존 수년 → 수주 수준으로 새 구성 탐색
- **제어 시스템 단순화**: 다층 구조 → 단일 신경망
- **융합 에너지 연구 가속화**: ITER, SPARC 등 미래 기기 개발 촉진

**장기 전망**: 
본 논문은 단순히 토카막 제어를 넘어, **AI가 미래 핵융합 발전소의 핵심 제어 시스템**이 될 가능성을 입증하였습니다. 2025년 DeepMind-CFS 협력 등 최신 진전은 이 가능성이 상용 실현 단계로 진입 중임을 시사합니다.

***

### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/215eab32-2dc9-491f-a15f-3933b398ef3b/s41586-021-04301-9.pdf)
[2](https://pmc.ncbi.nlm.nih.gov/articles/PMC10881383/)
[3](https://arxiv.org/abs/2409.09238)
[4](https://arxiv.org/pdf/2402.09387.pdf)
[5](http://arxiv.org/pdf/2403.01635.pdf)
[6](http://arxiv.org/pdf/2311.05967.pdf)
[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC10516960/)
[8](https://deepmind.google/blog/bringing-ai-to-the-next-generation-of-fusion-energy/)
[9](https://blog.cfs.energy/with-ai-alliance-google-deepmind-and-cfs-take-fusion-to-the-next-level/)
[10](https://euro-fusion.org/eurofusion-news/eurofusion-spearheads-advances-in-artificial-intelligence-and-machine-learning-to-unlock-fusion-energy/)
[11](https://pmc.ncbi.nlm.nih.gov/articles/PMC8850200/)
[12](https://arxiv.org/html/2403.18912v1)
[13](https://www.nature.com/articles/s41586-021-04301-9)
[14](https://www.sciencedirect.com/science/article/abs/pii/S0042207X03000757)
[15](https://www.sciencedirect.com/science/article/abs/pii/S0920379623005902)
[16](https://www.egr.msu.edu/annweb/papers/intelligent_control/icnn96_plasma.pdf)
[17](https://arxiv.org/html/2506.20096v1)
[18](https://pubs.aip.org/avs/jvb/article-pdf/14/1/504/11973263/504_1_online.pdf)
