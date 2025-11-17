# RL Unplugged: A Suite of Benchmarks for Offline Reinforcement Learning

### 1. 핵심 주장 및 기여도

**RL Unplugged**는 오프라인 강화학습의 재현성 및 접근성을 개선하기 위해 제안된 통합 벤치마크 스위트입니다. 이 논문의 핵심 주장은 다음과 같습니다.[1][2]

**주요 기여도는 네 가지**입니다: (i) 데이터셋을 위한 통일된 API, (ii) 다양한 환경의 통합, (iii) 명확한 평가 프로토콜, (iv) 참조 성능 기준선. 특히 이 논문은 오프라인 RL 연구의 **재현성 위기**(reproducibility crisis)를 해결하는 것을 목표로 합니다. BCQ, BRAC, SAC 등 주요 알고리즘들의 구현이 어렵고, 논문들이 일관되지 않은 데이터셋과 평가 프로토콜을 사용하여 알고리즘 비교가 어렵다는 문제를 지적합니다.[1]

---

### 2. 해결하고자 하는 문제와 제안 방법

#### 2.1 문제 정의

오프라인 RL 방법들이 성공적인 결과를 보이고 있음에도 불구하고, 다음과 같은 **근본적인 문제**들이 존재합니다:[1]

1. **표준화 부재**: 비표준화된 평가 프로토콜, 서로 다른 데이터셋, 기준선 부재로 인한 알고리즘 비교의 어려움
2. **현실 세계 특성 미표현**: 부분 관측성(partial observability), 고차원 센서(이미지), 다양한 액션 스페이스, 탐험 문제, 비정상성, 확률론적 역학
3. **제한된 벤치마크**: 기존 벤치마크들이 단순한 MDP나 완전 관측 환경에 집중

#### 2.2 제안 방법 및 평가 프로토콜

논문은 두 가지 **평가 프로토콜**을 제시합니다:[1]

**온라인 정책 선택(Online Policy Selection)**:

$$
\text{Score}_{\text{online}} = \arg\max_{\theta} \mathbb{E}[R(\pi_{\theta})]
$$

여기서 환경과의 상호작용을 통해 하이퍼파라미터를 검증합니다. 이는 오프라인 RL 방법을 고립되어 평가하지만, 현실에서는 불가능한 가정입니다.

**오프라인 정책 선택(Offline Policy Selection)**:

$$
\text{Score}_{\text{offline}} = \arg\max_{\theta} \widehat{V}_{\text{OPE}}(\pi_{\theta})
$$

여기서 OPE(Offline Policy Evaluation) 방법을 사용하여 환경 상호작용 없이 정책을 평가합니다. 현실의 오프라인 RL 문제를 더 잘 반영합니다.

#### 2.3 모델 구조 및 알고리즘

논문은 **5가지 기준 알고리즘**을 구현합니다:[1]

| 알고리즘 | 특징 | 수식 |
|---------|------|------|
| **BC** (Behavior Cloning) | 감독 학습 기반 | $$\min_{\phi} \mathbb{E}\_{(s,a) \sim D} $$ $$-\log \pi_{\phi}(a \mid s)$$ |
| **D4PG** | 온라인 RL을 오프라인에 적용 | 분배 비평가(distributional critic) 사용 |
| **BCQ** | VAE 기반 행동 제약 | $$a = \pi_{\phi}(s) + \xi \odot \text{VAE}(s)$$ |
| **BRAC** | 행동 정규화 | $$\min_{\pi} \mathbb{E}[KL(\pi, \hat{\beta})]$$ 제약 |
| **RABM** | 분배 이점 가중 회귀 | MPO와 이점 가중 회귀 결합 |

특히 **DM Control Suite** 작업에서는 다음 구조를 사용합니다:
- 8층 MLP (크기 1024), 잔여 연결 및 인스턴스 정규화
- 로코모션 작업에서는 ResNet으로 이미지 처리 후 MLP와 결합
- 순차 작업에서는 2개의 LSTM (숨겨진 크기 1024) 추가

***

### 3. 성능 향상 및 결과

#### 3.1 DM Control Suite 결과

**온라인 정책 선택**에서는 간단한 작업(Cartpole swingup, Walker stand)에서 모든 알고리즘이 유사한 성능을 보입니다. 하지만 **어려운 작업**(Humanoid run, Manipulator insert ball)에서는:[1]

- BC와 RABM이 최고 성능을 발휘
- D4PG, BRAC는 현저히 낮은 성능
- 어느 알고리즘도 온라인 방법 수준에 도달하지 못함

**오프라인 정책 선택**에서는 성능이 평가되지만, 일관되게 낮은 성적을 기록합니다.

#### 3.2 DM Locomotion 결과

이 도메인은 **고 차원 액션 공간**과 **어려운 탐험** 특성으로 인해 가장 도전적입니다:[1]

- BC와 RABM이 우수한 성능을 보임
- D4PG는 극히 낮은 성능
- BCQ와 BRAC는 구현의 어려움으로 결과 미포함 (재현성 문제 재확인)

#### 3.3 Atari 2600 결과

정규화된 점수 기준으로:[1]

| 알고리즘 | 온라인 선택 | 오프라인 선택 |
|---------|-----------|-------------|
| BC | ~50 | ~75 |
| DQN | ~100 | ~40-60 |
| BCQ | ~100 | ~100-120 |
| REM | ~100 | ~100-120 |
| IQN | ~100 | ~100-120 |

REM과 IQN이 오프라인 설정에서도 견고한 성능을 유지합니다.

***

### 4. 일반화 성능 향상 가능성

#### 4.1 논문의 일반화 관련 발견

논문은 **부분 관측성이 일반화의 주요 도전과제**임을 드러냅니다. DM Locomotion의 에고센트릭 카메라 관측(64×64×3 이미지)을 포함한 작업에서:[1]

- 모든 오프라인 RL 방법이 현저히 낮은 성능을 기록
- 메모리 기반 구조(LSTM)의 필요성이 일반화 성능에 미치는 영향 증명

#### 4.2 최신 연구의 일반화 개선 방법

**2024-2025년 최신 연구**에서는 일반화 성능을 향상시키기 위한 새로운 접근법들이 제시됩니다:[3][4][5][6][7]

**1) 영점 일반화(Zero-Shot Generalization, ZSG)**

최신 연구는 다양한 환경에서 학습한 오프라인 데이터셋으로 본 적 없는 환경에서도 잘 수행하는 방법을 제시합니다:[3]

- **PERM(Pessimistic Empirical Risk Minimization)**: 

$$
\min_{\pi} \mathbb{E}_{e \sim \mathcal{E}_{\text{train}}} \left[ -\hat{V}^{\pi}(s_0^e) + \lambda \|\pi - \hat{\beta}^e\|^2 \right]
$$

- **PPPO(Pessimistic Proximal Policy Optimization)**: 비관적 정책 평가를 통한 일반화 개선

**2) 데이터 증강 기반 방법**

- **Equivariant Data Augmentation**: 등변 변환을 인식하는 동적 모델 학습으로 분포 외 목표에 대한 일반화 개선[4]
- **S2P(State-conditioned Image Synthesis)**: 상태 공간에서 이미지 합성으로 미관측 이미지 분포 탐색[8]

**3) 온화한 일반화(Mild Generalization)**

2024년 최신 논문인 Doubly Mild Generalization(DMG)은 기존의 극단적 접근법(완전 제약 vs. 완전 일반화)의 중간 지점을 제시합니다:[7]

$$
a^* = \arg\max_{a \in \mathcal{N}(a_b)} Q(s, a)
$$

여기서 $$\mathcal{N}(a_b)$$는 행동 데이터셋 근처의 이웃입니다.

**4) 궤적 생성을 통한 일반화**

**OTTO(Offline Trajectory Generalization through world Transformers)**는 월드 트랜스포머로 상태 역학을 학습하여 고보상 궤적 시뮬레이션 생성합니다.[9]

***

### 5. 논문의 한계

#### 5.1 본질적 한계

1. **모델 성능 격차**: 모든 오프라인 방법이 온라인 방법에 미달[1]
2. **부분 관측성 문제**: 로코모션 작업에서 오프라인 RL 방법이 특히 취약[1]
3. **정책 선택의 어려움**: 오프라인 정책 선택 설정에서 하이퍼파라미터 튜닝이 여전히 도전적[1]
4. **분포 이동(Distributional Shift)**: 행동 정책과 학습 정책 간의 불일치로 인한 외삽 오차[10][11][12]

#### 5.2 구현 관련 한계

논문은 **재현성의 심각한 문제**를 지적합니다:[1]

- BCQ와 BRAC를 로코모션 작업에서 재현 실패
- 알고리즘 저자들도 구현 어려움 확인 (Peng et al. 2019, Fujimoto et al. 2019 참조)
- SAC 구현의 어려움

***

### 6. 향후 연구에 미치는 영향 및 고려사항

#### 6.1 벤치마크로서의 영향

RL Unplugged는 오프라인 RL 연구에 **중추적 역할**을 수행합니다:[13][1]

- **재현성 확보**: 통일된 API와 평가 프로토콜로 일관된 비교 가능
- **접근성 향상**: 제한된 계산 자원으로 도전적 작업 연구 가능
- **커뮤니티 협력**: "living benchmark"로서 지속적 확대

D4RL과 함께 현재 오프라인 RL 연구의 표준 벤치마크로 인정받고 있습니다.[14]

#### 6.2 최신 연구의 경향

**2024-2025년 오프라인 RL 연구**는 다음 방향으로 진화하고 있습니다:[5][6][15][7][9][3]

1. **일반화 능력 강화**: 영점 일반화와 도메인 일반화에 집중
2. **정책 선택 방법 개선**: 불확실성 기반 정책 전환 메커니즘 등장[16]
3. **멀티태스크 학습**: 제한된 데이터로 여러 작업 간 지식 공유[6]
4. **LLM 통합**: 작업 기술을 활용한 정책 학습 개선[15]

#### 6.3 앞으로의 연구 시 고려사항

| 고려사항 | 설명 |
|---------|------|
| **평가 프로토콜 선택** | 온라인 vs. 오프라인 정책 선택의 트레이드오프 이해 필수 |
| **분포 이동 대응** | 외삽 오차와 값 과대평가 완화가 핵심 |
| **부분 관측성** | 순환 모델과 메모리 메커니즘 필수 |
| **데이터 질 고려** | 행동 정책의 다양성과 데이터 커버리지가 중요 |
| **실세계 적용** | 행동 지연, 확률론적 역학, 제약 조건 포함 |
| **하이퍼파라미터 민감성** | 오프라인 설정에서의 견고한 하이퍼파라미터 선택 |

#### 6.4 새로운 연구 방향

최근 논문들이 제시하는 **유망한 방향**:[7][9][16][3]

1. **불확실성 기반 적응**: 에피스템 불확실성을 활용한 실시간 정책 전환[16]
2. **비관적-낙관적 균형**: DMG처럼 보수성과 일반화의 균형 추구[7]
3. **멀티모달 정책**: 혼합 가우시안 분포 출력으로 데이터 다양성 포착[1]
4. **오프라인-온라인 연속성**: 오프라인 사전학습에서 온라인 미세조정으로의 매끄러운 전환[16]

***

### 결론

RL Unplugged는 오프라인 강화학습의 재현성 위기를 해결하기 위한 **획기적인 벤치마크**입니다. 통합된 평가 프로토콜과 다양한 도메인을 통해 알고리즘 비교를 표준화했으며, 부분 관측성과 고차원 관찰 같은 현실적 도전과제를 포함했습니다. 

그러나 일반화 성능에 있어서는 여전히 상당한 개선 여지가 있으며, 최신 연구(2024-2025)는 **영점 일반화, 데이터 증강, 온화한 일반화, 불확실성 기반 적응** 등의 방법으로 이 문제를 해결하고 있습니다. 향후 연구는 분포 이동에 더욱 견고한 알고리즘 개발과, 실세계 응용을 위한 제약 조건 포함에 집중할 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/57e2d676-cf44-444b-9561-6455d3e4a0a2/2006.13888v4.pdf)
[2](http://arxiv.org/pdf/2310.20025.pdf)
[3](http://arxiv.org/pdf/2503.07988.pdf)
[4](https://arxiv.org/pdf/2309.07578.pdf)
[5](http://arxiv.org/pdf/2211.14827.pdf)
[6](http://arxiv.org/pdf/2312.15909.pdf)
[7](http://arxiv.org/pdf/2411.07934.pdf)
[8](https://arxiv.org/pdf/2209.15256.pdf)
[9](https://arxiv.org/html/2404.10393)
[10](http://arxiv.org/pdf/2406.09089.pdf)
[11](http://arxiv.org/pdf/2106.10783.pdf)
[12](https://arxiv.org/pdf/2303.15810.pdf)
[13](https://deepmind.google/blog/rl-unplugged-benchmarks-for-offline-reinforcement-learning/)
[14](https://arxiv.org/html/2503.19267v1)
[15](https://arxiv.org/html/2509.00347v1)
[16](https://arxiv.org/abs/2503.12222)
[17](https://proceedings.neurips.cc/paper/2021/hash/274a10ffa06e434f2a94df765cac6bf4-Abstract.html)
[18](https://icml.cc/virtual/2025/poster/46618)
[19](https://proceedings.neurips.cc/paper/2020/file/51200d29d1fc15f5a71c1dab4bb54f7c-Paper.pdf)
[20](https://neurips.cc/virtual/2024/poster/96454)
[21](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1520.pdf)
[22](https://openreview.net/forum?id=EBT0oymkZb)
[23](https://arxiv.org/pdf/2302.04782.pdf)
[24](http://arxiv.org/pdf/2310.06268.pdf)
[25](https://arxiv.org/pdf/2206.04745.pdf)
[26](https://arxiv.org/pdf/2309.06599.pdf)
[27](https://openreview.net/pdf?id=AuwUAbXohW)
[28](https://proceedings.mlr.press/v157/zhang21a/zhang21a.pdf)
[29](http://bair.berkeley.edu/blog/2020/06/25/D4RL/)
[30](https://proceedings.neurips.cc/paper_files/paper/2024/file/5c14b3ee78d09e8b3240ffb1fb6cc819-Paper-Conference.pdf)
[31](http://bair.berkeley.edu/blog/2022/04/25/rl-or-bc/)
[32](https://www.emergentmind.com/topics/d4rl-benchmarks)
[33](https://arxiv.org/abs/2407.13006)
[34](https://proceedings.neurips.cc/paper/2021/file/f5e647292cc4e1064968ca62bebe7e47-Paper.pdf)
[35](https://arxiv.org/abs/2004.07219)
[36](https://nips.cc/virtual/2024/poster/96808)
