# Quantifying Generalization in Reinforcement Learning | Overfitting in RL, ε-greedy, Entropy Bonus

## 1. 핵심 주장과 주요 기여, 주요 키워드  
**핵심 주장**  
- 심층학습(RL) 에이전트는 훈련 환경에 과도하게 특화(overfitting)되어, 새로운(test) 환경에서 성능이 급격히 저하된다.  
- 절차적 생성(proc. generation) 기법을 이용해 훈련/테스트 환경을 명확히 분리하면 에이전트의 일반화 능력을 계량화·개선할 수 있다.  

**주요 기여**  
- **CoinRun 벤치마크 제안**: 훈련용과 테스트용 레벨을 분리한 플랫폼 게임 환경을 설계.  
- **일반화 곡선(generalization curve) 도입**: 훈련 레벨 수에 따른 훈련·테스트 성능 차이를 정량적으로 측정.  
- **구조·정규화 기법 평가**:  
  - **IMPALA-CNN** 구조가 기존 Nature-CNN 대비 일반화 성능↑  
  - L2 Regularization, Dropout, Data Augmentation, Batch Normalization, 환경·정책의 확률적 변형(stochasticity) 도입 시 일반화 성능↑  

**주요 키워드**  
- Overfitting in RL  
- Procedural Generation  
- CoinRun Benchmark  
- IMPALA-CNN  
- Regularization (L2, Dropout, BatchNorm, Data Augmentation)  
- Stochasticity (ε-greedy, Entropy Bonus)  
- Generalization Curve  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
- 기존 RL 벤치마크 대부분 훈련·테스트 환경을 동일하게 사용 → 에이전트의 진정한 **일반화 능력**(new level zero-shot 성능)을 측정 불가  
- 훈련 레벨에 과도하게 적합된 정책이 새로운 레벨에서 실패  

### 2.2 제안 방법  
- **절차적 레벨 생성**(procedural generation)으로 훈련·테스트 레벨 분리  
- **일반화 곡선**: 훈련 레벨 수 $$N$$ 에 따른 훈련 성능 $$A_{\text{train}}(N)$$ 과 테스트 성능 $$A_{\text{test}}(N)$$ 간의 격차 측정  
- **정규화·아키텍처 실험**:  
  1) **L2 Regularization**: 손실 함수에 $$\ell_2$$ 벌점 추가  

$$
       \mathcal{L} = \mathcal{L}_{\text{PPO}} + w \sum_i \theta_i^2
$$  
  
  2) **Dropout**: 확률 $$p$$ 로 뉴런 출력 무작위 제거  
  3) **Data Augmentation**: Cutout 변형, 관측(observation) 이미지에서 임의 영역 마스킹  
  4) **Batch Normalization**: 각 합성곱 층 뒤에 배치 정규화 삽입  
  5) **Stochasticity**:  
     - 환경에 ε-greedy 행동 선택: 확률 ε 로 무작위 행동  
     - 정책 엔트로피 보너스 $$k_H$$ 조정  

### 2.3 모델 구조  
| 모델           | 특징                                                                                   |
|---------------|-----------------------------------------------------------------------------------------|
| Nature-CNN    | Mnih et al. (2015) 3-layer CNN (Nature DQN 아키텍처)                                         |
| IMPALA-CNN    | Espeholt et al. (2018) Residual block 기반, 깊이(3 blocks)↑ 채널 수↑                               |
| IMPALA-Large  | Residual block 5개, 채널 수 두 배                                                                  |
| LSTM 확장      | CoinRun-Platforms, RandomMazes에서 메모리 필요해 CNN 뒤 LSTM 결합                                    |

### 2.4 성능 향상  
- **일반화 곡선** (CoinRun, 256M timesteps):  
  - Nature-CNN: 훈련 100ℓ→테스트 약 67% 성공률, 16,000ℓ→≈87.6%  → 여전히 overfit  
  - IMPALA-CNN: 같은 조건에서 테스트 ≈97.8%  → **+10%p** 이상(표준편차 고려)  
- **정규화 기법** (500ℓ 고정) 테스트 성능 향상:  
  - L2 ($$w=10^{-4}$$) → 약 **+3%p**  
  - Dropout ($$p=0.1$$) → **+1.5%p**  
  - Data Augmentation → **+4%p**  
  - BatchNorm → **+6%p**  
- **Stochasticity** (512M timesteps):  
  - ε-greedy($$\varepsilon=0.1$$) / Entropy Bonus($$k_H=0.075$$) → 각각 **+8–10%p**  
- **복합 기법** 결합 시 시너지 작으나 소폭 추가 개선  

### 2.5 한계  
- **환경 단순성**: CoinRun 계열이 실제 복잡한 RL 과제(로봇 제어, 전략 게임 등)와 차이  
- **절차적 생성 편향**: 생성기 설계에 따라 일반화 경로가 달라질 수 있음  
- **메모리 의존 과제**: LSTM 한계로 RandomMazes 등 순환구조 일반화 어려움  
- **하이퍼파라미터 민감도**: 정규화·확률 기법 효과가 환경별로 크게 변동  

## 3. 모델 일반화 성능 향상 관련 주요 내용 집중 분석  
- **데이터 다양성 확보**: 레벨 수 $$N$$ → 무한(매 에피소드마다 새 레벨) 시 overfitting 해소  
- **아키텍처 개선**:  
  - **Residual 블록** 기반 IMPALA 구조가 CNN 깊이·채널 증가에도 안정적 학습 및 일반화 제공  
  - 대규모 네트워크(IMPALA-Large) → 추가 성능↑, 그러나 수익 체감  
- **정규화 기법 적용**:  
  - *Batch Normalization* 이 가장 큰 폭의 일반화 격차 축소  
  - *Data Augmentation*은 실제 환경 시뮬레이터 간 도메인 포용에도 활용 가능  
- **확률성 도입**:  
  - 환경·정책 양쪽에서 무작위성 부여가 에이전트의 특화 학습 억제  
  - 특히 엔트로피 보너스와 ε-greedy 조합은 탐색과 일반화 균형에 효과적  

## 4. 향후 연구에 미치는 영향 및 고려 사항  
**영향**  
- **일반화 평가 프로토콜 표준화**: RL 연구에서 훈련/테스트 분리 벤치마크 필수화  
- **절차적 생성 활용**: 복잡한 현실 과제에도 proc. generation 기반 일반화 측정 확산  
- **정규화·확률성 기법**: SL 관행을 RL에 적극 도입, 범용성 높은 regularizer 개발 촉진  

**고려 사항**  
- **생성기 설계 편향** 최소화: 인간 설계 레벨 vs. 절차적 레벨 분산 고려  
- **순환 모델 재검토**: Transformer 기반 메모리 구조 등 새로운 순환 대안 평가  
- **환경 복잡도 확장**: MuJoCo, StarCraft II 등 고차원 과제에 일반화 기법 이식  
- **하이퍼파라미터 자동화**: AutoML 기반 최적 정규화·확률성 설정 탐색  

---  

Quantifying Generalization in Reinforcement Learning은 RL에서 진정한 일반화 능력을 계량화·개선하는 방향을 제시하며, 앞으로의 RL 연구에 **훈련/테스트 분리**, **절차적 데이터 다양화**, **정규화·확률성 기법 통합** 등의 관행을 확산시킬 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/6b8953ed-3ed5-46f0-88f9-17773e2e98c1/1812.02341v3.pdf
