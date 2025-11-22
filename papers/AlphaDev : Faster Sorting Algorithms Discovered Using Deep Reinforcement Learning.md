# Faster Sorting Algorithms Discovered Using Deep Reinforcement Learning

### 1. 핵심 주장 및 주요 기여 요약

본 논문은 **Deep Reinforcement Learning(DRL)을 활용하여 수십 년간 인류가 최적화한 정렬 알고리즘을 능가하는 새로운 알고리즘을 자동으로 발견**했다는 것을 보여줍니다. AlphaDev라는 DRL 에이전트는 CPU 어셈블리 명령어 수준에서 작동하여 기존 벤치마크보다 빠른 정렬 알고리즘을 생성했으며, 이는 LLVM 표준 C++ 라이브러리에 성공적으로 통합되었습니다. 이는 프로그램 합성 분야에서 강화학습이 실제 산업 환경에 영향을 미친 첫 사례로 평가됩니다.[1]

### 2. 해결하고자 하는 문제 및 제안 방법

#### 문제 정의

정렬 알고리즘은 **매일 수조 번 실행되는 기본 알고리즘**입니다. 계산 수요가 계속 증가하는 현 상황에서 인간 전문가도 이들 알고리즘을 더 이상 개선하지 못해 심각한 계산 병목 현상이 발생했습니다. 특히 divide-and-conquer 접근법에서 반복적으로 호출되는 **고정 길이 및 가변 길이 정렬 알고리즘의 소규모 구현**이 중요한 최적화 대상입니다.[1]

#### AssemblyGame: 문제 공식화

논문은 알고리즘 발견 문제를 **단일 플레이어 게임으로 공식화**합니다. 각 상태에서 플레이어가 어셈블리 명령어를 선택하여 새로운 정렬 알고리즘을 구성합니다. 이 게임의 상태는 다음과 같이 정의됩니다:[1]

$$S_t = \langle P_t, Z_t \rangle$$

여기서:
- $P_t$: 현재까지 생성된 알고리즘의 표현
- $Z_t$: 미리 정의된 입력에 대해 현재 알고리즘을 실행한 후 메모리와 레지스터의 상태

#### 보상 함수

보상 $r_t$는 두 가지 요소로 구성됩니다:

1. **정확성 보상 ($r_c$)**: 생성된 알고리즘이 $N$개의 테스트 시퀀스를 올바르게 정렬하는지 확인
2. **성능 보상**: 
   - 알고리즘 길이 보상 (초기 단계)
   - 실제 지연 시간 측정 (최종 단계)

총 보상은:

$$r_t = r_c + \lambda \cdot r_{latency}$$

### 3. 모델 구조: AlphaDev

#### 핵심 아키텍처

AlphaDev는 AlphaZero의 확장으로, 두 가지 주요 구성 요소를 갖습니다:

1. **표현 함수(Representation Function)**
2. **학습 알고리즘** (AlphaZero + MCTS)

#### 표현 네트워크

AlphaDev의 표현 네트워크는 특히 혁신적입니다:[1]

$$h_t = f_{rep}(S_t)$$

이는 두 개의 서브네트워크로 구성됩니다:

**a) 트랜스포머 인코더:**
- 각 어셈블리 명령어의 Opcode와 Operands를 원핫(one-hot) 인코딩으로 변환
- 다중 쿼리 트랜스포머(MultiQuery Transformer)를 사용하여 어셈블리 명령 시퀀스를 임베딩으로 매핑

**b) CPU 상태 인코더:**
- 다층 퍼셉트론(MLP)으로 각 레지스터 및 메모리 위치의 상태를 받아들임
- 알고리즘이 메모리와 레지스터 역학에 미치는 영향을 예측

두 임베딩의 결합으로 최종 AlphaDev 상태 표현을 얻습니다.

#### 예측 네트워크

$$\hat{v}_t, \hat{\pi}_t = f_{pred}(h_t)$$

여기서:
- $\hat{v}_t$: 가치 함수 (정확성 헤드와 지연 시간 헤드의 이중 구조)
- $\hat{\pi}_t$: 정책 (행동 공간에 대한 분포)

#### 동적 계획: MCTS

$$h_t^{k+1}, \hat{r}_t^{k+1} = f_{dyn}(h_t^k, a_t^k)$$

AlphaDev는 MCTS를 통해 미래 상태를 계획합니다. 행동 선택은 **예측기 상한 신뢰 구간(Predictor Upper Confidence Bound, PUCB)** 전략을 사용합니다:

$$PUCB(a) = \hat{Q}(a) + c \cdot \hat{\pi}(a) \cdot \sqrt{\frac{\sum_b N(b)}{1 + N(a)}}$$

### 4. 성능 향상

#### 고정 길이 정렬 알고리즘 성능

AlphaDev는 인간 벤치마크(최적 정렬 네트워크)와 비교하여 다음과 같은 개선을 달성했습니다:[1]

| 알고리즘 | AlphaDev | 인간 벤치마크 | 개선 |
|---------|----------|-----------|------|
| Sort 3 | 17 명령어 | 18 명령어 | 1 명령어 감소 |
| Sort 4 | 28 명령어 | 28 명령어 | 동등 |
| Sort 5 | 42 명령어 | 46 명령어 | 4 명령어 감소 |

#### 가변 길이 정렬 알고리즘 성능 (지연 시간 최적화)

가변 길이 정렬에서 지연 시간을 직접 최적화할 경우:[1]

| 알고리즘 | AlphaDev (마이크로초) | 인간 벤치마크 (마이크로초) | 개선율 |
|---------|---------------------|----------------------|--------|
| VarSort3 | 236,498 | 246,040 | 3.8% |
| VarSort4 | 279,339 | 294,963 | 5.3% |
| VarSort5 | 312,079 | 331,198 | 5.8% |

#### 실제 영향: LLVM libc++ 라이브러리

AlphaDev가 발견한 정렬 알고리즘은 LLVM 표준 라이브러리에 통합되어:[1]
- **5개 요소 이하의 시퀀스**: 최대 70% 성능 개선
- **250,000개 이상의 요소**: 약 1.7% 성능 개선
- **지원 아키텍처**: ARMv8, Intel Skylake, AMD Zen 2
- **데이터 타입**: uint32, uint64, float

이는 **매일 수조 번 호출되는 알고리즘**이 포함되어 있습니다.[1]

### 5. 새로운 알고리즘 발견

#### AlphaDev Swap Move

AlphaDev는 기존 정렬 네트워크에서 한 명령어를 절약하는 새로운 최적화 기법을 발견했습니다.[1]

**원본 연산**: 입력 $\langle A, B, C \rangle$에 대해:
- $A \rightarrow \min(A, B, C)$
- $B \rightarrow \max(\min(A, C), B)$
- $C \rightarrow \max(A, C)$

**AlphaDev Swap Move** (B ≤ C 보장 시):
- $A \rightarrow \min(A, B)$ (1 명령어 감소)
- $B \rightarrow \max(\min(A, C), B)$
- $C \rightarrow \max(A, C)$

#### AlphaDev Copy Move

Sort 8 정렬 네트워크에서 발견된 또 다른 최적화:

| 입력 | 원본 출력 | AlphaDev Copy Move |
|------|----------|-------------------|
| A | min(A,B,C,D) | min(A,B,C,D) |
| B | max(B, min(A,C,D)) | max(B, min(A,C)) |
| C | max(C, min(A,D)) | max(C, min(A,D)) |
| D | max(A,D) | max(A,D) |

**핵심**: D ≥ min(A,C) 조건에서 세 번째 입력을 제거하여 **1 명령어 절약**[1]

#### VarSort4의 본질적으로 다른 접근법

흥미로운 발견은 VarSort4 알고리즘입니다:[1]
- **인간 벤치마크**: 입력 길이를 확인한 후 해당 정렬 네트워크 호출
- **AlphaDev 솔루션**: 먼저 처음 3개 요소를 Sort 3으로 정렬한 후, 나머지 요소를 간단한 Sort 4로 정렬

이는 분기(branching) 구조로 인한 복잡성을 효율적으로 관리하는 완전히 새로운 알고리즘적 접근법입니다.

### 6. 일반화 성능 및 한계

#### 일반화 능력

**성공적인 일반화:**

1. **VarInt 역직렬화** (Protocol Buffer): 단일 값 입력에 대해 **약 3배 빠른 성능** 달성[1]
   - 브랜치리스 솔루션으로 기존 대비 상당한 성능 개선
   - 새로운 **VarInt 할당 이동**을 발견하여 두 연산을 단일 명령어로 결합

2. **경쟁 프로그래밍 문제**: 추가 도메인에서도 우수한 성능 달성[1]

#### 한계 분석

**AlphaDev의 현재 제한 사항:**

1. **확장성 문제**: 알고리즘이 커질수록 탐색 공간이 기하급수적으로 증가
   - Sort 6-8: 각각 3, 2, 1개 명령어 절약 (증분 감소)
   - 더 큰 알고리즘으로의 확장은 계산적으로 제한적

2. **저수준 어셈블리 최적화**: 
   - 현재는 x86 아키텍처의 하위 집합만 지원
   - CPU 명령어 수준 최적화로 인한 이식성 제한

3. **높은 수준 언어 최적화 부재**: 
   - 논문에서 인정한 주요 한계 사항
   - **향후 방향**: C++ 등 고수준 언어에서 직접 최적화 추구[2]

4. **검증 집약성**: 
   - 모든 테스트 시퀀스에 대한 정확성 검증이 필수
   - 이론적 정확성 증명 불가능

### 7. 일반화 성능 향상에 관련된 상세 분석

#### Stochastic Search와의 비교

논문은 최신 **Stochastic Superoptimization (AlphaDev-S)**과 비교하여 AlphaDev의 우월성을 입증했습니다:[1]

**Cold Start 설정 (사전 지식 없음):**

| 설정 | Sort 3 | Sort 5 | VarSort5 |
|-----|--------|--------|----------|
| AlphaDev | 최적 | 최적 | 최적 |
| AlphaDev-S-CS | 실패 | 실패 | 실패 (31조 개 프로그램 탐색) |

**탐색 효율성:**

AlphaDev는 VarSort5에서:
- **최대 12백만 개 프로그램 탐색**
- AlphaDev-S는 **최대 31조 개 프로그램 탐색**[1]
- 효율성: **2,500배 이상 우월**

#### 탐색 가설 (Exploration Hypothesis)

t-SNE 시각화 분석 결과, AlphaDev-S의 탐색 문제를 다음과 같이 설명할 수 있습니다:[1]

**AlphaDev-S 특성:**
- 초기 시드 주변에 밀집된 원형 영역만 탐색
- 부분 최적 해에 갇혀 탈출 불가

**AlphaDev 특성:**
- 장기 가치 함수가 알고리즘 공간의 새로운 영역 탐색을 유도
- 부정확한 알고리즘 공간에서 정확한 알고리즘 공간으로 성공적으로 전이

이는 **DRL의 value function 기반 탐색이 stochastic search의 breadth-first 탐색보다 훨씬 더 효과적**임을 보여줍니다.

#### 지연 시간 가치 함수의 역할

AlphaDev는 이중 가치 함수 설정을 도입했습니다:[1]

$$\text{Loss} = L_{correctness} + \alpha \cdot L_{latency}$$

여기서:
- 정확성 헤드: 알고리즘 정확성 예측
- 지연 시간 헤드: 프로그램의 실제 측정된 지연 시간 직접 예측

이 설정으로 **vanilla 단일 헤드 설정 대비 현저한 성능 개선** 달성[1]

### 8. 최신 연구 기반 영향과 향후 고려사항

#### AlphaDev의 현재 영향 (2023-2025)

**1. 산업 채택의 가속화**

DeepMind는 AlphaDev의 성공에 이어 **새로운 해싱 알고리즘 발견**(2024)도 공개했습니다:[2]
- 9-16 바이트 범위에서 **30% 성능 개선**
- Abseil 오픈소스 라이브러리에 통합
- **매일 수조 번 호출되는 수준**으로 배포[2]

**2. 프로그램 합성 분야의 패러다임 변화**

최신 연구들이 AlphaDev의 접근 방식을 확장하고 있습니다:[3]
- Merge Sort와 Quick Sort의 기본 케이스로 AlphaDev의 정렬 네트워크 활용
- 큰 배열 정렬에서 종단간 성능 개선 달성

**3. 교육학적 영향**

AlphaDev가 발견한 새로운 구조(Swap Move, Copy Move)는:[1]
- 정렬 이론 교육에 새로운 최적화 기법 제공
- 컴퓨터 과학 학생들에게 AI 기반 알고리즘 설계의 가능성 제시

#### 향후 연구 방향 및 고려사항

**1. 확장성 문제 해결**

현재 한계:
- Sort 6-8로 갈수록 성능 이득 감소 (각 1-3 명령어)
- 더 큰 알고리즘으로의 확장은 계산상 어려움

향후 개선:
- **계층적 생성**: 작은 블록으로부터 큰 알고리즘 조립
- **전이 학습(Transfer Learning)**: 작은 정렬에서 학습한 지식을 큰 정렬에 활용
- **구성성(Compositionality) 향상**: 알고리즘 부분의 재사용 가능성 증대[1]

**2. 고수준 언어 지원**

논문에서 명시한 주요 미래 방향:[2]
- C++, Python, Rust 등 고수준 언어에서 직접 최적화
- 현재 어셈블리 최적화의 이식성 문제 해결
- 개발자 접근성 향상

**3. 일반화 성능 향상 전략**

**a) 도메인 적응:**
- 다양한 아키텍처(x86, ARM, RISC-V) 지원 확대
- 특정 응용 분야 최적화 (머신러닝, 그래픽스 등)

**b) 메타 학습:**
- 알고리즘 클래스 간 일반화 개선
- 유사한 최적화 문제에 대한 사전 학습 활용[4]

**c) 신경-기호 결합:**
- 명시적 제약 조건과 휴리스틱 통합
- 알고리즘 정확성에 대한 형식 검증 추가[5]

**4. 공동 최적화 전략**

최신 연구 동향:[5]
- **GALOIS 프레임워크**: DRL과 일반화 가능한 논리 합성의 결합
- 계층적 원인-결과 로직으로 해석 가능하고 일반화 가능한 프로그램 생성

**5. 계산 효율성 개선**

현재 훈련 비용:
- TPU v.3 및 v.4 사용 (512 액터)
- 최악의 경우 2일 수렴

향후 개선 방향:
- **동적 시뮬레이션**: 분포 밖 입력에 대한 성능 예측 개선
- **기울기 기반 미세 조정**: MCTS 탐색 결과의 신경망 증류 최적화

#### 연구 시 고려할 중요 사항

**1. 문제 특성 선택**
- AlphaDev는 **정확성이 이진적(완전 정렬 or 오류)**인 문제에 최적화
- 해싱, 암호화, 근사 알고리즘 등으로 확장 시 보상 함수 재설계 필수[1]

**2. 상태 표현의 중요성**
- Transformer 기반 표현과 CPU 상태 인코더의 결합이 핵심
- 새로운 도메인에서도 구조를 캡처하는 강력한 표현 필수

**3. 실제 성능 측정**
- 미세 벤치마킹의 노이즈 처리 중요
- AlphaDev의 **다중 머신 측정 (100개 머신) + 5번째 백분위수** 접근법은 필수[1]

**4. 탐색-착취 균형**
- 제약 조건이 많은 문제(불가능한 명령어 조합)에서는 높은 모정책(high prior policy) 필요
- 행동 가지치기(Action Pruning) 규칙의 신중한 설계 필수

### 결론

"Faster Sorting Algorithms Discovered Using Deep Reinforcement Learning"은 **AI가 기본 알고리즘 최적화에서 인류를 능가할 수 있음**을 처음으로 입증한 획기적인 연구입니다. AlphaDev의 Swap Move와 Copy Move는 단순한 성능 개선을 넘어 **새로운 알고리즘적 통찰**을 제공합니다.[1]

최신 연구 동향은 이 성공을 바탕으로 **확장성, 일반화, 고수준 언어 지원**으로 나아가고 있습니다. 특히 Merge Sort/Quick Sort 기본 케이스 통합, GALOIS 같은 신경-기호 결합 접근법, 그리고 다양한 아키텍처 지원 확대는 AlphaDev 이후의 중요한 연구 방향입니다.[3][2]

향후 연구는 **계산 효율성 개선**, **형식 검증 통합**, **메타 학습을 통한 일반화 능력 강화**에 초점을 맞춰야 하며, 이를 통해 AI 기반 프로그램 합성이 산업 전반의 알고리즘 최적화를 혁신할 가능성이 높습니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/43c7c811-3f7c-4d08-9c0e-2ab71213255f/s41586-023-06004-9.pdf)
[2](https://deepmind.google/discover/blog/alphadev-discovers-faster-sorting-algorithms/)
[3](http://arxiv.org/pdf/2503.05934.pdf)
[4](https://pure.kaist.ac.kr/en/publications/learning-to-synthesize-programs-as-interpretable-and-generalizabl)
[5](https://openreview.net/forum?id=XSV1T9jMuz9)
[6](https://arxiv.org/pdf/2307.14503.pdf)
[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC10247365/)
[8](https://arxiv.org/pdf/2503.10466.pdf)
[9](https://arxiv.org/pdf/2411.18892.pdf)
[10](https://arxiv.org/pdf/2311.00749.pdf)
[11](https://arxiv.org/html/2501.00913v1)
[12](https://www.ijert.org/research/state-of-the-art-reinforcement-learning-algorithms-IJERTV8IS120332.pdf)
[13](https://syncedreview.com/2023/06/12/deepminds-alphadev-leverages-deep-reinforcement-learning-to-discover-faster-sorting-algorithms/)
[14](https://blog.codingconfessions.com/p/exploring-deepminds-alphadev-breakthrough)
[15](https://indiaai.gov.in/article/alphadev-employs-reinforcement-learning-to-find-faster-sorting-algorithms)
[16](https://arxiv.org/abs/2108.13643)
[17](https://github.com/kyegomez/AlphaDev)
[18](https://arxiv.org/pdf/2503.05934.pdf)
[19](https://deepmind.google/blog/alphadev-discovers-faster-sorting-algorithms/)
