# DreamerV3: Mastering Diverse Domains through World Models

## 1. 핵심 주장과 주요 기여

DreamerV3는 강화학습 분야의 근본적인 도전 과제를 해결하는 범용 알고리즘입니다. 이 논문의 핵심 주장은 **단일 고정 하이퍼파라미터 세트로 150개 이상의 다양한 작업에서 전문화된 알고리즘을 능가**할 수 있다는 것입니다.[1]

주요 기여는 다음과 같습니다:

**범용성(Generality)**: DreamerV3는 연속 및 이산 행동, 시각적 및 저차원 입력, 밀집 및 희소 보상, 2D 및 3D 환경 등 8개 도메인의 150개 이상 작업에서 작동합니다. 특히 Atari, ProcGen, DMLab, Minecraft, Control Suite 등 다양한 벤치마크에서 뛰어난 성능을 보였습니다.[1]

**로버스트니스 기법**: 정규화(normalization), 균형화(balancing), 변환(transformation)에 기반한 일련의 기술들이 도메인 간 안정적 학습을 가능하게 합니다. 이는 하이퍼파라미터 튜닝 없이도 새로운 문제에 적용할 수 있게 합니다.[1]

**Minecraft 다이아몬드 수집**: 인간 데이터나 커리큘럼 없이 처음부터 Minecraft에서 다이아몬드를 수집한 최초의 알고리즘입니다. 이는 희소 보상, 긴 시간 지평, 복잡한 탐색이 요구되는 AI의 중요한 이정표입니다.[1]

**확장성(Scalability)**: 모델 크기를 12M에서 400M 파라미터로 증가시킬 때 성능이 단조롭게 향상되며, 더 큰 모델은 더 높은 성능뿐만 아니라 더 적은 환경 상호작용을 요구합니다.[1]

## 2. 문제, 방법론, 구조 및 성능

### 해결하고자 하는 문제

기존 강화학습 알고리즘은 새로운 도메인에 적용할 때 상당한 전문 지식과 하이퍼파라미터 튜닝이 필요합니다. 예를 들어, PPO는 널리 사용되지만 많은 경험이 필요하고, SAC은 연속 제어에 효과적이지만 엔트로피 스케일 튜닝이 필요하며 고차원 입력에서 어려움을 겪습니다. MuZero는 강력하지만 복잡한 구성 요소로 인해 재현이 어렵습니다.[1]

### 제안하는 방법

DreamerV3는 세 가지 신경망으로 구성됩니다:[1]

**World Model (세계 모델)**: RSSM(Recurrent State-Space Model) 구조를 사용하여 환경을 학습합니다.[1]

World Model은 다음 구성 요소로 이루어집니다:

$$
\begin{aligned}
&\text{Sequence model: } h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \\
&\text{Encoder: } z_t \sim q_\phi(z_t | h_t, x_t) \\
&\text{Dynamics predictor: } \hat{z}_t \sim p_\phi(\hat{z}_t | h_t) \\
&\text{Reward predictor: } \hat{r}_t \sim p_\phi(\hat{r}_t | h_t, z_t) \\
&\text{Continue predictor: } \hat{c}_t \sim p_\phi(\hat{c}_t | h_t, z_t) \\
&\text{Decoder: } \hat{x}_t \sim p_\phi(\hat{x}_t | h_t, z_t)
\end{aligned}
$$

World Model은 세 가지 손실 함수로 최적화됩니다:

$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}\left[\sum_{t=1}^{T}\left(\beta_{\text{pred}}\mathcal{L}_{\text{pred}}(\phi) + \beta_{\text{dyn}}\mathcal{L}_{\text{dyn}}(\phi) + \beta_{\text{rep}}\mathcal{L}_{\text{rep}}(\phi)\right)\right]
$$

여기서:

$$
\begin{aligned}
\mathcal{L}_{\text{pred}}(\phi) &= -\ln p_\phi(x_t | z_t, h_t) - \ln p_\phi(r_t | z_t, h_t) - \ln p_\phi(c_t | z_t, h_t) \\
\mathcal{L}_{\text{dyn}}(\phi) &= \max\left(1, \text{KL}\left(\text{sg}(q_\phi(z_t | h_t, x_t)) \| p_\phi(z_t | h_t)\right)\right) \\
\mathcal{L}_{\text{rep}}(\phi) &= \max\left(1, \text{KL}\left(q_\phi(z_t | h_t, x_t) \| \text{sg}(p_\phi(z_t | h_t))\right)\right)
\end{aligned}
$$

**Critic (평가자)**: 각 상태에서의 리턴 분포를 예측합니다. λ-return을 사용하여 학습합니다:[1]

$$
\begin{aligned}
\mathcal{L}(\psi) &= -\sum_{t=1}^{T} \ln p_\psi(R_t^\lambda | s_t) \\
R_t^\lambda &= r_t + \gamma c_t\left((1-\lambda)v_t + \lambda R_{t+1}^\lambda\right) \\
R_T^\lambda &= v_T
\end{aligned}
$$

Critic은 다양한 환경에서 작동하도록 지수적으로 분포된 빈(bin)을 가진 범주형 분포로 매개변수화됩니다.[1]

**Actor (행위자)**: 가장 가치 있는 결과에 도달하기 위한 행동을 선택합니다. 리턴 정규화를 통해 엔트로피 정규화와 균형을 맞춥니다:[1]

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{T} \text{sg}\left(\frac{R_t^\lambda - v_\psi(s_t)}{\max(1, S)}\right) \log \pi_\theta(a_t | s_t) + \eta H(\pi_\theta(a_t|s_t))
$$

여기서 리턴 범위 $$S $$는 이상치에 강건하게 계산됩니다:

$$
S = \text{EMA}\left(\text{Per}(R_t^\lambda, 95) - \text{Per}(R_t^\lambda, 5), 0.99\right)
$$

### 로버스트 예측 기법

**Symlog 변환**: 입력 재구성과 보상/리턴 예측의 스케일 문제를 해결하기 위해 bi-symmetric logarithm 변환을 사용합니다:[1]

$$
\begin{aligned}
\text{symlog}(x) &= \text{sign}(x) \ln(|x| + 1) \\
\text{symexp}(x) &= \text{sign}(x)\left(\exp(|x|) - 1\right)
\end{aligned}
$$

이를 사용한 손실 함수:

$$
\mathcal{L}(\theta) = \frac{1}{2}\left(f(x, \theta) - \text{symlog}(y)\right)^2
$$

**Symexp Twohot Loss**: 확률적 타겟(보상, 리턴)에 대해 지수적으로 분포된 빈에 대한 소프트맥스 분포를 사용합니다:[1]

$$
\hat{y} = \text{softmax}(f(x))^T B, \quad B = \text{symexp}(\{-20, \ldots, +20\})
$$

Twohot 인코딩을 사용한 크로스 엔트로피 손실:

$$
\mathcal{L}(\theta) = -\text{twohot}(y)^T \log \text{softmax}(f(x, \theta))
$$

### 모델 구조

- **Encoder/Decoder**: 이미지 입력에 대해 CNN, 벡터 입력에 대해 MLP를 사용합니다[1]
- **Sequence Model**: 8개 블록으로 구성된 블록 대각 GRU를 사용하여 메모리 단위를 확장하면서도 파라미터와 FLOP의 이차적 증가를 방지합니다[1]
- **Dynamics/Reward/Continue Predictors**: MLP 구조를 사용합니다[1]
- **표현(Representations)**: 소프트맥스 분포 벡터에서 샘플링되며, 샘플링 단계를 통해 straight-through gradient를 사용합니다[1]

### 성능 향상

DreamerV3는 8개 벤치마크에서 뛰어난 성능을 달성했습니다:[1]

- **Atari (57 tasks, 200M steps)**: MuZero를 능가하며 계산 자원의 일부만 사용[1]
- **ProcGen (16 tasks, 50M steps)**: 튜닝된 PPG와 동등한 성능[1]
- **DMLab (30 tasks, 100M steps)**: 1B 스텝의 IMPALA/R2D2+를 100M 스텝에서 능가하여 1000% 이상의 데이터 효율성 향상[1]
- **Atari100k (26 tasks, 400K steps)**: IRIS, TWM 등 최고 방법들을 능가[1]
- **Proprio/Visual Control**: 각 벤치마크에서 새로운 state-of-the-art 달성[1]
- **BSuite (23 environments)**: 특히 스케일 로버스트니스 카테고리에서 개선[1]
- **Minecraft Diamond**: 100M 스텝에서 모든 학습된 에이전트가 다이아몬드 발견(평균 리턴 9.1)[1]

**Ablation Study 결과**:[1]
- 모든 로버스트니스 기법이 전체 성능에 기여하며, 특히 KL 목적 함수, 리턴 정규화, symexp twohot 회귀가 중요합니다[1]
- Dreamer는 주로 비지도 재구성 손실에 의존하며, 이는 대부분의 이전 알고리즘이 보상/가치 예측 그래디언트에 주로 의존하는 것과 대조적입니다[1]

### 한계

논문에서 명시적으로 언급된 한계는 제한적이지만, 다음과 같은 점들을 추론할 수 있습니다:

- **탐색 한계**: Deep Sea 및 Deep Sea Stochastic 태스크에서 0.0 점수를 기록하여 특정 탐색 문제에서 어려움을 겪습니다[1]
- **복잡한 탐색 도메인**: Minecraft에서 에피소드의 0.4%만 다이아몬드를 획득하여 개선의 여지가 있습니다[1]
- **Lasertag 태스크 성능**: 일부 Lasertag 태스크에서 낮은 또는 음의 점수를 기록했습니다[1]
- **균일 재생 버퍼**: 우선순위 재생이 성능을 개선할 수 있지만 구현의 용이성을 위해 사용하지 않았습니다[1]

## 3. 일반화 성능 향상

DreamerV3의 일반화 성능은 여러 측면에서 뛰어납니다:

### 도메인 간 일반화

**고정 하이퍼파라미터**: 모든 도메인에 걸쳐 동일한 하이퍼파라미터를 사용하여 새로운 문제에 즉시 적용 가능합니다. 이는 학습률, 배치 크기, 엔트로피 스케일 등 모든 설정이 고정되어 있음을 의미합니다.[1]

**다양한 입력 모달리티**: 64×64×3 이미지부터 400개 이상의 아이템을 포함하는 벡터 관찰까지 처리합니다. Symlog 변환은 대규모 입력과 재구성 그래디언트를 방지하여 표현 손실과의 균형을 안정화합니다.[1]

**보상 스케일 로버스트니스**: BSuite의 스케일 로버스트니스 카테고리에서 이전 알고리즘(0.60) 대비 크게 개선(0.82)되었습니다. Symlog/symexp 변환과 리턴 정규화가 핵심 역할을 합니다.[1]

### 절차적 생성 환경에서의 일반화

**ProcGen**: 무작위 레벨과 시각적 산만함이 있는 16개 게임에서 정규화 평균 66.01을 달성하여 PPG(64.89)를 능가했습니다. 이는 에이전트의 강건성과 새로운 레벨에 대한 일반화 능력을 입증합니다.[1]

**DMLab**: 공간적 및 시간적 추론이 필요한 3D 환경 30개 태스크에서 우수한 성능을 보였습니다. 언어 이해, 시각 검색, 메모리 태스크 등 다양한 인지 능력이 요구되는 환경에서 일반화됩니다.[1]

### 스케일링과 일반화

**모델 크기**: 12M에서 400M 파라미터로 증가 시 성능이 단조롭게 향상되며, 큰 모델은 더 높은 작업 성능과 함께 더 적은 환경 상호작용을 필요로 합니다. 이는 모델 용량이 일반화 능력을 향상시킴을 시사합니다.[1]

**재생 비율**: 높은 재생 비율은 예측 가능하게 성능을 향상시킵니다. 이는 계산 자원과 데이터 효율성 간의 명확한 trade-off를 제공합니다.[1]

### 비지도 학습과 일반화

**재구성 손실의 중요성**: 학습 신호 절제 연구에서 보상/가치 그래디언트 없이도 재구성 손실만으로 상당한 성능을 달성할 수 있음을 보였습니다. 이는 비지도 학습이 작업 불가지론적 표현을 학습하여 일반화를 촉진함을 시사합니다.[1]

**사전 학습 가능성**: 비지도 목적에 주로 의존하는 특성은 비지도 데이터에 대한 사전 학습을 활용할 수 있는 미래 알고리즘 변형의 가능성을 열어줍니다.[1]

### 메모리와 신용 할당

**BSuite 메모리 카테고리**: 이전 방법(0.02) 대비 0.62로 크게 개선되어 장기 의존성 처리 능력이 향상되었습니다. RSSM의 순환 구조가 시간적 추상화와 Markovian 표현을 가능하게 합니다.[1]

## 4. 향후 연구에 미치는 영향 및 고려사항

### 향후 연구 방향 (논문 제안)[1]

**인터넷 비디오로부터 세계 지식 학습**: 비지도 재구성 손실에 주로 의존하는 특성은 대규모 비디오 데이터셋에서 사전 학습할 수 있는 가능성을 열어줍니다.[1]

**도메인 간 단일 세계 모델**: 여러 도메인에 걸쳐 하나의 세계 모델을 학습하여 인공 에이전트가 점진적으로 일반적인 지식과 역량을 구축할 수 있도록 합니다.[1]

### 최신 연구 동향 (2024-2025)

**Transformer 기반 세계 모델**: STORM 등의 연구는 RNN 기반 세계 모델을 Transformer로 대체하여 더 나은 학습 효율성과 스케일링 특성을 제공하고 있습니다.[2]

**대조 학습 통합**: Curled-Dreamer는 CURL의 대조 손실을 DreamerV3에 통합하여 시각적 강화학습 작업에서 성능을 향상시켰습니다. 이는 표현 학습의 개선 방향을 제시합니다.[3]

**재구성 없는 학습**: MuDreamer는 픽셀 재구성 손실 없이 예측 세계 모델을 학습하여 시각적 산만함이 있을 때 더 강건한 성능을 보입니다.[4]

**전이 학습 및 맥락 RL**: Model-Based Transfer Learning (MBTL)은 일반화 성능을 명시적으로 모델링하여 43배 개선된 샘플 효율성을 달성했습니다. 이는 작업 선택 전략의 중요성을 강조합니다.[5][6]

**탐색 개선**: DreamerV3-XP는 우선순위 재생 버퍼와 앙상블 기반 내재적 보상을 추가하여 희소 보상 환경에서 더 빠른 학습과 낮은 동역학 모델 손실을 달성했습니다.[7]

**실제 응용**: DreamerNav는 DreamerV3를 동적 실내 환경에서의 자율 내비게이션에 확장하여 멀티모달 공간 인식과 하이브리드 전역-지역 계획을 통합했습니다. Dream to Fly는 시각 기반 드론 비행에 적용했습니다.[8][9]

### 고려할 사항

**산만함 처리**: 배경 산만함이 복잡한 환경에서 DreamerV3가 여전히 영향을 받으며, 정책 형태 예측 및 작업 인식 재구성 손실 등의 방법이 필요합니다.[10]

**메모리 능력 강화**: R2I와 같은 state space model을 통합하여 장기 메모리와 신용 할당 능력을 개선할 수 있습니다.[11]

**계산 효율성**: 기본 200M 모델은 상당한 계산 자원(A100 GPU)이 필요하지만, 12M 모델이 많은 작업에서 유사한 성능을 제공합니다. 연구자들은 가용 자원에 따라 모델 크기를 선택해야 합니다.[1]

**탐색 전략**: Deep Sea 및 유사한 하드 탐색 문제에서의 한계는 보다 정교한 탐색 메커니즘의 필요성을 시사합니다.[7][1]

**안전성과 신뢰성**: 실제 로봇 시스템에 적용할 때 안전 보장과 예측 가능한 행동이 중요합니다. 불확실성 추정과 안전 제약의 통합이 필요합니다.[9][7]

**지속 학습**: 환경 변화에 대한 적응과 재앙적 망각 방지는 여전히 열린 문제입니다. Meta-RL 접근법이 동적 워크로드에 대한 빠른 적응을 가능하게 합니다.[12][13]

**자기 지도 학습**: 라벨이 없는 대규모 데이터 활용은 미래 AI의 핵심이 될 것이며, 세계 모델은 이러한 패러다임에 자연스럽게 부합합니다.[13][14]

DreamerV3는 강화학습의 실용적 적용을 위한 중요한 이정표를 제시하며, 향후 연구는 탐색, 메모리, 전이 학습, 그리고 실제 응용에서의 강건성 개선에 초점을 맞춰야 합니다.[3][4][2][8][10][5][9][7][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/639d1e3a-fa5a-4948-932f-6dce2962a067/2301.04104v2.pdf)
[2](http://arxiv.org/pdf/2503.04416.pdf)
[3](https://arxiv.org/html/2408.05781)
[4](http://arxiv.org/pdf/2405.15083.pdf)
[5](https://arxiv.org/abs/2408.04498)
[6](https://proceedings.neurips.cc/paper_files/paper/2024/file/a10c3d85879c29331ba4d73ede56c7d3-Paper-Conference.pdf)
[7](https://arxiv.org/html/2510.21418v1)
[8](https://arxiv.org/html/2501.14377v1)
[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC12510832/)
[10](https://arxiv.org/html/2412.05766)
[11](https://openreview.net/forum?id=1vDArHJ68h)
[12](http://arxiv.org/pdf/2503.08872.pdf)
[13](https://deepscienceresearch.com/dsr/catalog/book/6/chapter/67)
[14](https://www.sciencedirect.com/science/article/pii/S0893608022001150)
[15](http://arxiv.org/pdf/1912.01603.pdf)
[16](https://pmc.ncbi.nlm.nih.gov/articles/PMC12003158/)
[17](https://github.com/danijar/dreamerv3)
[18](https://openaccess.thecvf.com/content/CVPR2025/papers/Hao_Neural_Motion_Simulator_Pushing_the_Limit_of_World_Models_in_CVPR_2025_paper.pdf)
[19](https://danijar.com/project/dreamerv3/)
[20](https://proceedings.mlr.press/v202/walker23a/walker23a.pdf)
[21](https://thesai.org/Downloads/Volume16No1/Paper_49-A_Review_of_Reinforcement_Learning_Evolution.pdf)
[22](https://arxiv.org/abs/2301.04104)
[23](https://dl.acm.org/doi/10.5555/3737916.3740717)
[24](https://arxiv.org/html/2411.14499v1)
[25](https://github.com/opendilab/awesome-model-based-RL)
[26](https://arxiv.org/abs/2408.04498v1/)
[27](https://worldmodels.github.io)
[28](https://www.nature.com/articles/s41586-025-08744-2)
[29](https://ieeexplore.ieee.org/ielaam/34/10269680/10172347-aam.pdf)
