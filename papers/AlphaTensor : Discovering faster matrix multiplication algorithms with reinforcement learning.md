# AlphaTensor : Discovering faster matrix multiplication algorithms with reinforcement learning

### **1. 핵심 주장과 주요 기여**

본 논문의 핵심 주장은 **심층 강화학습(Deep Reinforcement Learning, DRL)을 활용하여 행렬 곱셈 알고리즘을 자동으로 발견할 수 있다**는 것입니다. 구체적으로:[1]

**주요 기여:**

- **알고리즘 자동 발견**: AlphaZero 기반의 강화학습 에이전트 AlphaTensor를 통해 인간이 설계한 알고리즘보다 우수한 행렬 곱셈 알고리즘 발견[1]
- **역사적 돌파**: 4×4 행렬의 유한체에서 50년 전 Strassen의 알고리즘을 처음으로 개선 (49개에서 47개의 곱셈으로 감소)[1]
- **다양한 응용**: 구조화된 행렬 곱셈, 하드웨어별 최적화, 임의의 쌍선형 연산에 대한 알고리즘 발견[1]

***

### **2. 문제 정의 및 제안 방법**

#### **2.1 해결하고자 하는 문제**

행렬 곱셈은 신경망부터 과학 계산까지 광범위한 시스템의 핵심 연산입니다. 하지만:

- 3×3 행렬 곱셈의 최적 알고리즘도 여전히 미해결[1]
- 기존 방법들(인간의 탐색, 연속 최적화, 조합 탐색)은 휴리스틱에 의존하여 효율성이 제한적[1]
- 테이서 분해는 NP-hard 문제로 거대한 탐색 공간을 가짐[1]

#### **2.2 핵심 아이디어: 행렬 곱셈을 테이서 분해 문제로 변환**

행렬 곱셈 연산은 3D 테이서로 표현될 수 있습니다. 예를 들어, 2×2 행렬 곱셈은 크기 4×4×4인 테이서 $$T_2$$로 표현됩니다.[1]

**테이서 분해 정의:**

$$T_n = \sum_{r=1}^{R} u^{(r)} \otimes v^{(r)} \otimes w^{(r)} \quad (1)$$

여기서 $$\otimes$$는 외적(outer product)이고, 분해의 계수 R은 행렬 곱셈에 필요한 스칼라 곱셈 개수를 나타냅니다.[1]

**메타-알고리즘 (Algorithm 1):**

$$n \times n$$ 행렬 C=AB를 계산하기 위해:

$$(1) \text{ for } r=1,...,R \text{ do}$$
$$m_r \leftarrow (u_1^{(r)}a_1 + ... + u_{n^2}^{(r)}a_{n^2})(v_1^{(r)}b_1 + ... + v_{n^2}^{(r)}b_{n^2})$$
$$(2) \text{ for } i=1,...,n^2 \text{ do}$$
$$c_i \leftarrow w_1^{(1)}m_1 + ... + w_1^{(R)}m_R$$

이 알고리즘은 재귀적으로 적용될 수 있으며, 점근 복잡도는 $$O(N^{\log_n(R)})$$입니다.[1]

#### **2.3 TensorGame: 강화학습 환경**

알고리즘 발견을 단일-플레이어 게임 TensorGame으로 모델화합니다:[1]

- **초기 상태**: $$S_0 = T_n$$
- **각 스텝**: 플레이어가 triplet $$(u^{(t)}, v^{(t)}, w^{(t)})$$를 선택하여 상태 업데이트:

$$S_t \leftarrow S_{t-1} - u^{(t)} \otimes v^{(t)} \otimes w^{(t)} \quad (2)$$

- **목표**: 최소 스텝으로 영 테이서($$S_R = 0$$)에 도달

**보상 함수:**

$$r_t = -1 \text{ (매 스텝마다)}$$
$$r_{\text{terminal}} = -\gamma(S_{R_{\text{limit}}})$$

여기서 $$\gamma$$는 최종 테이서 계수의 상한입니다.[1]

***

### **3. 모델 구조 (AlphaTensor)**

AlphaTensor는 AlphaZero를 기반으로 다음과 같이 구성됩니다:[1]

#### **3.1 신경망 아키텍처**

**Transformer 기반 설계:**

- **입력 처리**: $$S \times S \times S$$ 테이서를 3개의 $$S \times S$$ 그리드로 변환 (선형 계층 사용)
- **주요 모듈**: 축 주의(Axial Attention)를 기반으로 3개의 그리드 쌍에 대해 어텐션 연산 수행[1]

**정책 헤드 (Policy Head):**
- 자기회귀(Autoregressive) Transformer를 사용하여 $$(u, v, w)$$ 분해
- 교사 강제(Teacher Forcing) 사용

**값 헤드 (Value Head):**
- 다층 퍼셉트론(MLP)이 분위수 회귀(Quantile Regression)로 리턴 분포 추정[1]

#### **3.2 Monte Carlo Tree Search (MCTS)**

상태 $$s$$에서 최적 행동 선택:

$$\arg\max_a Q(s,a) + c(s) \cdot \hat{\pi}(s,a) \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)} \quad (3)$$

여기서 $$N(s,a)$$는 방문 횟수, $$Q(s,a)$$는 행동 가치, $$\hat{\pi}(s,a)$$는 정책입니다.[1]

#### **3.3 핵심 혁신 기법**

**1) 합성 시연(Synthetic Demonstrations):**
- 무작위로 생성된 인수분해 쌍으로 학습 (역문제는 다항식 시간에 해결 가능)
- 감독 손실과 RL 손실의 혼합 학습[1]

**2) 기저 변환(Change of Basis):**
행렬 곱셈 테이서의 계수는 기저 선택에 무관하므로:

$$T^{(A,B,C)} = \sum_{i,a}\sum_{j,b}\sum_{k,c} A_{ia}B_{jb}C_{kc}T_{abc} \quad (3)$$

100,000개의 무작위 기저에서 병렬로 게임 진행 → 다양성 증가[1]

**3) 데이터 증강 (Data Augmentation):**
- 게임에서 임의의 행동과 마지막 행동을 교환하여 학습 쌍 생성
- 부호 있는 순열(Signed Permutations)을 활용한 입력 변환[1]

**4) 행동 정규화(Action Canonicalization):**
동등한 행동 $$(u, v, w)$$와 $$(\lambda_1 u, \lambda_2 v, \lambda_3 w)$$ (단, $$\lambda_1\lambda_2\lambda_3=1$$) 구분 제거[1]

***

### **4. 성능 향상 결과**

#### **4.1 작은 행렬 크기에서의 개선**

| 행렬 크기 (n,m,p) | 기존 최선 | AlphaTensor (유한체 Z₂) | 개선도 |
|:---:|:---:|:---:|:---:|
| (2,2,2) | 7 | 7 | - |
| (3,3,3) | 23 | 23 | - |
| **(4,4,4)** | **49** | **47** | **-2** ✓ |
| (4,5,5) | 80 | 76 | -4 ✓ |

**특히 (4,4,4) 경우**: Strassen 알고리즘 개선 이후 50년 만의 첫 번째 개선[1]

#### **4.2 큰 행렬 크기에서의 개선**

70개 이상의 행렬 곱셈 테이서 ($$n, m, p \leq 12$$)에서 최신 기술을 능가[1]

예:
- $$(9,9,9)$$: 511 (기존) → 498 (AlphaTensor)
- $$(11,12,12)$$: 1,022 (기존) → 990 (AlphaTensor)

#### **4.3 실제 하드웨어 성능**

AlphaTensor는 하드웨어별 최적화된 알고리즘 발견:[1]

- **NVIDIA V100 GPU**: 4.3% ~ 19.6% 속도 향상
- **Google TPU v2**: 6.6% ~ 13.9% 속도 향상
- Strassen-square 알고리즘 초과[1]

#### **4.4 구조화된 행렬 곱셈**

비대칭 행렬-벡터 곱셈 (Skew-symmetric matrix-by-vector)의 경우:

$$\text{곱셈 수} \approx \frac{(n-1)(n+2)}{2} \sim \frac{1}{2}n^2 \quad (4)$$

기존 $$n^2$$에서 개선하고 **점근 최적**임을 증명[1]

***

### **5. 일반화 성능 향상**

#### **5.1 전이 학습 (Transfer Learning)**

**단일 에이전트로 다중 테이서 처리:**
- 다양한 크기의 테이서 $$(n, m, p)$$에 대해 단일 에이전트 학습
- 서로 다른 테이서 간 패턴 인식과 전략 공유[1]

**성능 향상:**
혼합 학습 전략 (합성 데이터 + 실제 데이터)이 각각 단독 학습보다 우수[1]

#### **5.2 알고리즘 다양성**

**발견된 알고리즘의 풍부함:**
- 각 크기당 최대 수천 개의 비동등 알고리즘 발견
- 예: $$(2,2,2) \otimes (2,2,2)$$의 경우 기존 "Strassen-square" 외 **14,000개 이상의 새로운 분해**[1]

이는 행렬 곱셈 알고리즘 공간이 이전에 알려진 것보다 훨씬 더 풍부함을 시사합니다.[1]

#### **5.3 알고리즘 재귀 조합**

작은 테이서의 분해들을 **재귀적으로 조합**하여 큰 테이서 분해:
- 예: $$(9,9,9)$$ = 6×(3,3,3) + 9×(6,3,3)[1]
- $$T_4$$는 $$10^{10}$$배 더 큰 행동 공간을 가지나 AlphaTensor는 이를 초과[1]

***

### **6. 한계 (Limitations)**

논문은 다음 한계를 인정합니다:[1]

**1) 사전 정의된 계수 집합 의존성:**
- 인수 항목이 특정 집합 $$F = \{-2, -1, 0, 1, 2\}$$에 제한됨
- 더 큰 계수를 가진 효율적 알고리즘 누락 가능성

**2) 계산 복잡도:**
- 해석 가능한 개선이지만, 매우 큰 행렬 크기에서 실제 이득은 제한적

**3) 정확도와 안정성:**
- 수치 안정성이나 에너지 효율 같은 다른 기준에 대해서는 추가 연구 필요

***

### **7. 이후 연구에 미치는 영향 및 고려사항**

#### **7.1 최신 연구 동향 (2023-2025)**

**OpenTensor (2024):**
- AlphaTensor의 재현 및 개선 시도
- 기술 세부 사항 명확화 및 학습 프로세스 최적화[2]

**mallocMuZero (2023):**
- AlphaTensor 원칙을 메모리 매핑 최적화에 적용
- AlphaTensor 행렬 곱셈 모델 성능 추가 향상[3]

**New Bounds for Matrix Multiplication (2023):**
- 레이저 방법의 개선으로 행렬 곱셈 지수 $$\omega$$의 새로운 상한 제시 ($$\omega \leq 2.371339$$)[4]

#### **7.2 앞으로의 연구 방향**

**1) 계수 집합 적응:**
- 최적의 $$F$$ 자동 탐색 메커니즘 개발[1]

**2) 다른 NP-hard 문제로 확장:**
- 비음(Non-negative) 인수분해
- 경계 계수(Border Rank) 계산
- 다항식 곱셈, 그래프 알고리즘 등[1]

**3) 하이브리드 접근:**
- 수학적 이론 (연속 최적화, 조합론)과 DRL의 결합
- 증명 가능한 복잡도 경계 달성[1]

**4) 실무 적용:**
- 정확한 알고리즘 구현의 병렬화 및 최적화
- 신경망 및 과학 계산 라이브러리에 통합[1]

**5) 일반화 성능 강화:**
- 분포 외(Out-of-Distribution) 문제에 대한 견고성 개선
- 더 큰 문제 크기에 대한 스케일링[2]

#### **7.3 핵심 고려사항**

| 측면 | 고려사항 |
|:---:|:---|
| **이론적** | 발견된 알고리즘의 수학적 성질 분석 및 일반화 가능성 |
| **실제적** | 하드웨어별 최적화 알고리즘의 구현 효율성 |
| **확장성** | 더 큰 문제로의 전이 학습 및 메모리 효율성 |
| **일반성** | 다양한 환경 (유한체, 실수, 커스텀 연산) 지원 |
| **신뢰성** | 수치 안정성 및 증명 가능한 정확성 보장 |

***

### **결론**

AlphaTensor는 **기계학습이 기본적인 산술 문제를 자동으로 해결할 수 있음**을 보여줍니다. 단순히 성능 개선을 넘어, 이는 **인간 직관을 초월한 알고리즘 발견의 가능성**을 제시하며, 수학 및 컴퓨터과학의 미해결 문제 해결에 새로운 길을 열었습니다. 향후 연구는 이 접근법을 더 광범위한 조합 최적화 문제로 확장하고, 이론적 깊이와 실제 적용성 사이의 균형을 유지하는 것이 중요합니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/55676e95-2ecf-471a-a946-287bd16b1914/s41586-022-05172-4.pdf)
[2](http://arxiv.org/pdf/2405.20748.pdf)
[3](https://arxiv.org/pdf/2305.07440.pdf)
[4](https://arxiv.org/pdf/2307.07970.pdf)
[5](http://arxiv.org/pdf/2404.00639.pdf)
[6](http://arxiv.org/pdf/1805.08166.pdf)
[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC9534758/)
[8](https://arxiv.org/pdf/2207.05808.pdf)
[9](http://arxiv.org/pdf/2411.06360.pdf)
[10](https://community.deeplearning.ai/t/optimizing-matrix-multiplication-alphatensor-for-faster-matrix-multiplication-explained/357762)
[11](https://www.esann.org/sites/default/files/proceedings/2020/ES2020-3.pdf)
[12](https://arxiv.org/html/2411.06360v3)
[13](https://www.nature.com/articles/s41586-022-05172-4)
[14](https://www.cogitatiopress.com/mediaandcommunication/article/viewFile/9623/4338)
[15](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm)
[16](https://aiforgood.itu.int/event/alphatensor-discovering-mathematical-algorithms-with-reinforcement-learning/)
[17](https://arxiv.org/abs/1606.03212)
[18](https://www.sciencedirect.com/science/article/pii/S0925231224005587)
[19](https://arxiv.org/abs/2405.20748)
