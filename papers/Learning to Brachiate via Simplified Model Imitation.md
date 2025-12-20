# Learning to Brachiate via Simplified Model Imitation

### 1. 핵심 주장 및 기여도 요약

이 논문은 **강화학습(Reinforcement Learning)과 단순화된 모델 모방 학습(Simplified Model Imitation Learning)**을 결합하여 물리기반(physics-based) 브래키에이션 제어를 학습하는 혁신적인 접근 방식을 제시합니다. 브래키에이션은 원숭이나 긴팔원숭이가 나뭇가지에서 나뭇가지로 팔만을 사용하여 이동하는 방식으로, 제어 권한이 제한적이고 정밀한 파지 동작이 필요하며 미래 움직임의 사전 계획이 필요하다는 점에서 극도로 도전적인 로봇 제어 문제입니다.

주요 기여는 다음과 같습니다:

- **두 단계 강화학습 접근법**: 단순화된 모델에서 먼저 정책을 학습한 후, 이를 통해 생성된 참조 궤적을 이용하여 복잡한 전체 모델의 정책을 학습
- **자발적 행동 발현(Emergent Behavior)**: 추가 뒤로-앞으로 스윙(pumping behavior) 같은 자발적 행동이 자동으로 학습됨
- **14-링크 평면 모델 성공**: 손가락이 없는 14개 관절의 현실적인 긴팔원숭이 모델에서 도전적인 파지 수열을 통과할 수 있는 역동적 리코셰탈 브래키에이션(ricochetal brachiation) 달성

***

### 2. 해결하고자 하는 문제

#### 2.1 문제 정의

브래키에이션 제어는 다음과 같은 본질적인 어려움을 포함합니다:

1. **제한된 제어 권한(Limited Control Authority)**: 팔만으로 움직여야 하며, 나뭇가지 선택이 이산적(discrete)이어서 연속적인 지형 조정이 불가능
2. **정밀성 요구(Precision Requirements)**: 손이 정확히 다음 파지점(handhold)에 도달해야 함
3. **사전 계획 필요성(Advance Planning)**: 모멘텀을 고려한 미래 움직임의 전략적 계획이 필수
4. **동적 작용(Dynamic Behavior)**: 비행 단계에서의 물리적 역학을 정확히 관리해야 함

#### 2.2 기존 연구의 한계

논문에서 지적하는 기존 방법의 문제점:

- **휴리스틱 기반 제어**: 수동으로 설계된 움직임 단계(motion phases)가 특정 작업에만 적용
- **단순 모델 기반 해결책**: 2-3 링크 모델에서만 효과적
- **제한된 일반화**: 다양한 파지 수열이나 높이 변화에 대응 불가능
- **강화학습의 어려움**: 희소 보상(sparse reward)에서의 학습이 극도로 어려움

***

### 3. 제안하는 방법

#### 3.1 시스템 개요

논문의 핵심 아이디어는 **계층적 모방 학습 파이프라인**입니다:

```
단순화된 모델 학습 → 참조 궤적 생성 → 전체 모델 모방 학습 → 통합 제어
```

#### 3.2 단순화된 모델(Simplified Model)

**구조**: 점 질량(point mass)에 가상 신축 팔(virtual extensible arm)

**동역학**:
- **스윙 단계(Swing Phase)**: 파지된 상태에서 스프링-댐퍼 펜듈럼 시스템

$$\text{Swing dynamics: } m\ddot{x} = -kx - c\dot{x} + F_{\text{control}}$$

여기서 $m$은 질량, $k$는 스프링 상수, $c$는 댐퍼 계수, $F_{\text{control}}$은 제어력입니다.

- **비행 단계(Flight Phase)**: 수동 물리 (무제한 자유도)

$$\text{Flight: } \ddot{x} = g, \quad \ddot{y} = g$$

(중력 상수 $g$에서의 포물선 궤적)

**팔 길이 범위**: $r \in [r_{\min}, r_{\max}] = [0.1, 0.75] \text{ m}$

#### 3.3 전체 모델(Full Model)

**구성**: 13개 힌지 관절 + 1개 허리 관절
- 어깨(shoulders), 팔꿈치(elbows), 손목(wrists), 엉덩이(hips), 무릎(knees), 발목(ankles)
- 신체 질량: 9.8 kg
- 팔 길이: 60 cm
- 파지 메커니즘: 포인트-투-포인트 제약(point-to-point constraints)

**관절 토크 제어**:

$$\tau_i = k_p(\theta_{i,\text{target}} - \theta_i) - k_d\dot{\theta}_i$$

여기서 $k_p = \text{joint range in radians}$, $k_d = k_p/10$

#### 3.4 학습 알고리즘

**강화학습 프레임워크**: Proximal Policy Optimization (PPO)

목적 함수:

$$J_{RL}(\theta) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right]$$

여기서 $\gamma \in [0,1)$는 할인 인자입니다.

**단순화된 모델의 보상**:

$$r_{\text{simple}} = \begin{cases} +1 & \text{if handhold grabbed} \\ 0 & \text{otherwise (sparse reward)} \end{cases}$$

**전체 모델의 보상** (복합 구성):

$$r_{\text{full}} = \exp(r_{\text{aux}} + r_{\text{style}}) + r_{\text{task}}$$

$$r_{\text{aux}} = w_t r_{\text{tracking}} + w_r r_{\text{reaching}}$$

$$r_{\text{style}} = w_u r_{\text{upright}} + w_a r_{\text{arm}} + w_l r_{\text{legs}} + w_e r_{\text{energy}}$$

**보상 항목 설명**:

| 보상 항목 | 정의 | 가중치 |
|---------|------|------|
| 추적(tracking) | $-\lvert\lvert p_{\text{body}} - p_{\text{reference}} \rvert\rvert_2^2$ | $w_t = -4$ |
| 도달(reaching) | $-\lvert\lvert p_{\text{hand}} - p_{\text{target}} \rvert\rvert_2^2$ (비행 단계만) | $w_r = -0.1$ |
| 직립(upright) | $\max(0, \lvert\text{pitch}\rvert - 40°)$ | $w_u = -1$ |
| 팔 회전(arm) | $-\lvert\omega_{\text{arm}}\rvert$ | $w_a = -0.1$ |
| 다리(legs) | $-\lvert\lvert\theta_{\text{knees}} - 110°\rvert\rvert_1$ | $w_l = -0.1$ |
| 에너지(energy) | $-(\lvert\lvert\tau\rvert\rvert_2^2 + \lvert\lvert\omega\rvert\rvert_1)$ | $w_e = -0.01$ |

#### 3.5 상태 및 행동 공간

**단순화된 모델의 상태**:

$$s_{\text{simple}} = [\dot{x}, \dot{y}, t_{\text{elapsed}}] \in \mathbb{R}^3$$

**전체 모델의 상태**:

$$s_{\text{full}} = [\dot{x}, \dot{y}, \text{pitch}, \theta_1, ..., \theta_{13}, \dot{\theta}_1, ..., \dot{\theta}_{13}, h_{\text{grab}}, g_1, g_2] \in \mathbb{R}^{45}$$

**행동 공간** (두 모델 모두):
- 길이 오프셋(length offset): $\in [-1, 1]$ (PD 제어 스케일링)
- 파지/해제 플래그(grab/release flag): $\in \{0, 1\}$

***

### 4. 모델 구조 및 신경망 아키텍처

#### 4.1 단순화된 모델 정책

**컨트롤러 네트워크**:
- 3층 피드-포워드 구조
- 은닉층 크기: 256
- 활성화 함수: ReLU (은닉층), Tanh (출력층)

**가치 함수 네트워크**:
- 동일한 아키텍처 (출력층만 다름)
- 스칼라 출력: $V(s)$

#### 4.2 전체 모델 정책

**더 큰 네트워크 필요성**:
$$\text{Policy: } 5 \text{ layers} \times 256 \text{ units}$$
- 처음 3개 층: Softsign 활성화
- 마지막 2개 층: ReLU 활성화
- 출력: Tanh 정규화

**학습 설정**:
- 러닝 레이트: $3 \times 10^{-4}$ (초기), $3 \times 10^{-5}$ (최종)
- 배치 크기: 2000
- 최대 샘플 수: $2.5 \times 10^7$
- 옵티마이저: Adam

***

### 5. 성능 향상 및 실험 결과

#### 5.1 단순화된 모델 결과

**Look-ahead 파지 수(Number of Look-ahead Handholds) 실험**:

| Look-ahead 개수 | 완료된 파지점 | 평균 에피소드 보상 |
|----------------|------------|-----------------|
| 1 | 17.8 ± 2.1 | 7.3 ± 0.8 (×100) |
| 2 | 14.6 ± 5.0 | 5.9 ± 1.8 (×100) |
| 3 | 4.9 ± 5.7 | 2.0 ± 2.4 (×100) |
| 5 | 7.6 ± 5.8 | 3.1 ± 2.4 (×100) |
| 10 | 0.3 ± 0.6 | 0.1 ± 0.3 (×100) |

**발견**: $N=1$ (즉각적 목표만 고려)이 최적 성능을 보임

**조기 종료(Early Termination) 효과**:

| 종료 전략 제거 | 평균 완료 파지점 |
|--------------|-----------------|
| 없음(기준) | 14.6 ± 5.1 |
| 복구 불가 기준만 제거 | 10.2 ± 7.3 |
| 최소 지속 시간만 제거 | 7.4 ± 9.3 |
| 최대 지속 시간만 제거 | 2.9 ± 2.0 |
| 최소 & 최대 제거 | 3.1 ± 4.1 |

**자발적 행동**: 큰 간격을 넘기 위해 추가 뒤-앞 스윙이 자동으로 학습됨

#### 5.2 전체 모델 결과

**다양한 모방 학습 전략 비교**:

- **A. 추적 보상만**: 기본 모방 학습
- **B. 추적 + 모든 보상**: 스타일 보상 추가
- **C. 보상 + 해제 타이밍**: 단순화 모델의 해제 시간 사용
- **D. 보상 + 상태**: 참조 궤적의 미래 상태 포함
- **E. 보상 + 상태 + 파지 정보**: 미래 파지 플래그 추가

**결론**: 옵션 D와 E가 최고 성능 (학습 곡선에서 약 12 handholds 도달)

**기준(Baseline) 실패**: 모방 보상 없이는 2번째 파지점도 통과 불가

#### 5.3 계획과 통합

**모델 예측 제어(MPC) 스타일 재계획**:

$$J(P_k) = V_{\text{full}}(s_t, k[0:N]) + \sum_{j=1}^{H} R_j$$

여기서:
- $V_{\text{full}}$: 전체 모델의 가치 함수
- $R_j$: 단순화 모델의 보상
- $N$: 정책의 look-ahead 값
- $H$: 계획 범위 ($> N$)

**계획 성능** (40개 지형에서 테스트):

| 계획 방법 | Gap 0개 통과 | Gap 1개 통과 | Gap 1+2 통과 | 모든 Gap 통과 |
|---------|-----------|-----------|-----------|-----------|
| 값 함수만 | 40/40 | 22/40 | 9/40 | 7/40 |
| 단순 보상만 | 40/40 | 36/40 | 26/40 | 15/40 |
| 값 함수 + 단순 보상 | 40/40 | 35/40 | 26/40 | 21/40 |

***

### 6. 일반화 성능 향상 가능성

#### 6.1 현재 일반화 능력

**강점**:
- **다양한 파지 간격 처리**: 균일 분포 $d \sim U(1, 2)$ m에서 학습
- **높이 변화 대응**: 피치 각도 $\phi \in [-15°, 15°]$ 범위에서 동작
- **동적 적응**: 간격 크기에 따라 자동으로 스윙 전략 조정
- **자발적 최적화**: 추가 스윙으로 모멘텀 조절

**제약**:

1. **2D 평면 제한**: 3D 공간에서의 브래키에이션 미지원
2. **고정 물리 매개변수**: 신체 질량, 팔 길이 등 변화에 대응 불가
3. **제한된 형태 변화**: 다양한 신체 비율을 처리하지 못함
4. **훈련 분포에 대한 의존성**: 훈련 범위 밖의 파지 수열에서 성능 저하 가능

#### 6.2 일반화 향상을 위한 제안

**1. 도메인 난이도 커리큘럼**:
$$\text{Task complexity}(t) = \min(d_{\min}, d_{\min} + \alpha t), \quad \alpha = \text{decay rate}$$

**2. 적응형 가치 추정**:
$$V_{\text{adaptive}}(s) = V_{\text{base}}(s) + \Delta V(s; \text{env params})$$

**3. 형태 불변 특성 학습**:
- 신체 비율이나 질량에 무관한 특성 표현 학습
- 메타 강화학습 활용

**4. 전이 학습(Transfer Learning)**:
- 새로운 형태에 대해 마지막 층만 재학습
- 미세 조정(fine-tuning) 최적화

***

### 7. 한계 및 제약

#### 7.1 기술적 한계

1. **계산 비용**:
   - 단순화 모델: ~5분
   - 전체 모델: ~1시간
   - 실시간 재계획: 1초에 10k 궤적 (비디오 전용)

2. **학습 안정성**:
   - Look-ahead 개수 증가 시 성능 저하
   - 네트워크 크기와 성능의 비선형 관계

3. **시뮬레이션 한계**:
   - Sim-to-real 갭 미해결
   - 손가락 모델링 부재

#### 7.2 방법론적 한계

1. **보상 설계 의존성**:
   - 여러 보상 항목과 가중치의 신중한 조정 필요
   - 과도한 엔지니어링 위험

2. **모방 학습의 필요성**:
   - 단순 강화학습만으로는 학습 실패
   - 단순화 모델에 의존적

3. **평가 지표 부족**:
   - 실제 긴팔원숭이와의 정량적 비교 미비
   - 동작 자연성 평가 주관적

***

### 8. 앞으로의 연구에 미치는 영향

#### 8.1 긍정적 기여

**1. 계층적 학습 패러다임 확립**:
- 단순화 모델을 통한 학습 가속화는 매우 효과적
- 다양한 복잡한 제어 문제에 적용 가능한 프레임워크

**2. 강화학습의 실용적 적용성 향상**:
- 희소 보상 환경에서의 학습 가능 입증
- 제약 기반 물리 시뮬레이션의 활용도 증가

**3. 자발적 행동 생성**:
- 보상 함수의 정의만으로 복잡한 행동이 자동 생성됨을 시연
- 인공지능의 창발성 연구에 기여

#### 8.2 후속 연구 방향

**1. 3D 브래키에이션**:
- 현재 2D 평면 시뮬레이션 확대
- 측면 움직임(transverse brachiation) 추가
- 실제 환경의 3D 구조 활용

**2. 형태 일반화**:
```
메타-강화학습: π(a|s; φ) where φ ∈ 형태 매개변수
적응형 컨트롤러: 신체 비율 변화에 즉시 대응
```

**3. 실제 로봇 적용**:
- 시뮬레이션-현실 간극 해소
- 손가락 파지력 모델 정교화
- 실제 아비앙(avianing) 로봇에 탑재

**4. 다중 동작 학습**:
- 하강(descent), 상승(ascent) 추가
- 다리 활용 학습
- 복합 이동 패턴 생성

**5. 강화학습 알고리즘 발전**:
- PPO 대신 정책 공간 탐색 개선
- 모방 학습과 강화학습의 더 효율적인 결합
- 불확실성 기반 탐색

***

### 9. 2020년 이후 관련 최신 연구 비교 분석

#### 9.1 주요 동시대 연구

**1. UniCon (2020) - 범용 신경 컨트롤러**[1]
- 다양한 동작을 하나의 제어기로 처리
- 계층적 아키텍처 (스케줄러 + 실행자)
- **차이점**: 본 논문은 단일 과제 집중, UniCon은 다중 동작 범용성

**2. ALLSTEPS (2020) - 커리큘럼 기반 학습**[1]
- 스텝핑 스톤 학습에 강화학습 적용
- 4가지 커리큘럼 전략 비교
- **유사점**: 단순 환경에서 복잡 환경으로 학습 진행

**3. DeepMimic 계열**
- 모션 캡처 데이터 기반 모방 학습
- **차이점**: 본 논문은 단순화 모델을 참조 생성, DeepMimic은 모션 캡처 사용

#### 9.2 계층적/단순화 모델 접근법 비교

| 연구 | 방법 | 단순화 전략 | 적용 대상 |
|-----|------|----------|---------|
| 본 논문 | RL + 모방 | 점질량 + 가상팔 | 브래키에이션 |
| MetaLoco (2024) | 메타-RL | 절차적 생성 로봇 | 사족 보행 |
| RACon (2024) | 검색 증강 RL | 모션 DB 검색 | 캐릭터 제어 |
| AdaptNet (2023) | 정책 적응 | 기존 정책 수정 | 형태/스타일 변화 |
| ControlVAE (2022) | VAE 기반 제어 | 잠재 공간 학습 | 다중 동작 |

#### 9.3 최신 트렌드와의 관계

**확산 모델(Diffusion Models) 도입**:

PDP (2024)는 확산 정책을 사용하여:
$$p(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**개선 효과**: 더 안정적이고 다양한 동작 생성, 하지만 계산 비용 증가

**메타 강화학습(Meta-RL) 발전**:

MetaLoco (2024)는:
$$\pi_\phi(a|s, \psi), \text{ where } \psi = \text{fast-adapted policy}$$

**개선 효과**: 다양한 로봇 형태에 대한 Zero-shot 일반화

#### 9.4 본 논문의 위치 및 영향

**혁신성 평가**:
1. **아이디어 독창성**: 단순화 모델→전체 모델 파이프라인은 새로운 접근법 ✓
2. **기술적 우수성**: PPO 기반이지만 구조적 혁신 ✓
3. **응용 가능성**: 많은 후속 연구의 영감 제공 ✓

**후속 영향**:
- 계층적 강화학습의 정당성 증명
- 모방 학습과 강화학습의 효과적 결합 시연
- 복잡 제어 문제의 분해 전략 확립

**미충족 과제**:
- 3D 확장 여전히 미해결
- 실제 로봇 응용 제한적
- 계산 효율성 개선 필요

***

### 10. 앞으로 연구 시 고려할 점

#### 10.1 방법론 개선 사항

**1. 보상 함수 학습화**:
```python
# 현재: 수동 설계 가중치
r_total = exp(r_aux + r_style) + r_task

# 제안: 역강화학습으로 보상 자동화
r_learned = IRL_network(demonstrations)
```

**2. 동적 커리큘럼**:
$$\text{difficulty}(t) = f_\text{adaptive}(\text{success rate}, t)$$
- 성공률에 따라 자동 조정
- 최적 학습 속도 유지

**3. 전이 학습 프레임워크**:
- 사전학습된 특성 표현 재사용
- 새로운 과제에서의 수렴 속도 향상

#### 10.2 기술적 고려사항

**1. 시뮬레이션 정확도**:
- 실제 그립 메커니즘 정교화
- 마찰력 및 마찰 모델 추가
- 관절 유연성 모델링

**2. 계산 효율성**:
```
단순화 모델 학습: GPU 병렬화 (✓ 이미 구현)
전체 모델 학습: 분산 학습 고려
추론 속도: ONNX/TensorRT 최적화
```

**3. 안정성 및 강건성**:
- 초기 상태 다양성 증대
- Perturbation 내성 테스트
- Adversarial 환경 학습

#### 10.3 평가 및 검증

**1. 정량적 메트릭 설정**:
| 메트릭 | 수식 | 목표 |
|------|------|------|
| 성공률 | $\frac{\text{Goals reached}}{\text{Total episodes}}$ | > 80% |
| 에너지 효율 | $\frac{\text{Distance traveled}}{\text{Torque²}}$ | 증가 |
| 자연성 | 전문가 평가 | 점수 > 7/10 |
| 일반화 | 훈련 외 환경 성공률 | > 60% |

**2. 공정한 비교**:
- 동일한 컴퓨팅 리소스 제어
- 하이퍼파라미터 최적화 과정 상세 기록
- 여러 무작위 시드로 통계적 유의성 확인

**3. 실제 영상 비교**:
- 원숭이 비디오와의 정성적 비교
- 생물역학 전문가 검증
- 모션 캡처 데이터와의 정량 비교

***

### 11. 결론

"Learning to Brachiate via Simplified Model Imitation"은 **계층적 학습의 강력함을 입증**하는 중요한 연구입니다. 단순화된 모델에서의 효율적인 탐색을 통해 전체 모델의 학습을 가속화하는 아이디어는:

1. **이론적으로** 강화학습의 공간 탐색 문제를 구조화된 접근으로 해결
2. **실무적으로** 복잡한 제어 문제를 해결 가능한 크기로 분해
3. **실용적으로** 자발적이고 자연스러운 동작의 생성을 가능하게 함

2024-2025년의 최신 연구들이 확산 모델, 메타 강화학습, 검색 증강 등으로 진화하고 있지만, **본 논문이 제시한 계층적 분해와 모방 학습의 결합**은 여전히 핵심 원리로 활용되고 있습니다. 특히 **형태 일반화**, **3D 확장**, **실제 로봇 적용**이 다음 세대 연구의 주요 과제가 될 것으로 예상됩니다.

[1](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14115)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bc7fb1a2-d917-4a1c-a988-844b12c0f730/2205.03943v1.pdf)
[3](https://www.semanticscholar.org/paper/8b62d928a7be4a6f408cc7a433c215a749604a95)
[4](https://dl.acm.org/doi/10.1145/3680528.3687683)
[5](https://ieeexplore.ieee.org/document/9564143/)
[6](https://ieeexplore.ieee.org/document/10688280/)
[7](https://www.semanticscholar.org/paper/e0205cc6047f8672da286048b8e4477e7d7a41ac)
[8](https://www.semanticscholar.org/paper/7b542c9fc0873aeb04225663adf31e8dd8d0aba2)
[9](http://www.roboticsproceedings.org/rss20/p103.pdf)
[10](https://dl.acm.org/doi/10.1145/3197517.3201315)
[11](https://fahruddin.org/smart/article/view/585)
[12](https://arxiv.org/html/2406.17795)
[13](https://www.eneuro.org/content/eneuro/early/2024/02/29/ENEURO.0383-23.2024.full.pdf)
[14](https://arxiv.org/html/2502.03122v1)
[15](https://arxiv.org/abs/2203.04735)
[16](https://arxiv.org/html/2407.17502v2)
[17](https://arxiv.org/pdf/2103.14274.pdf)
[18](http://arxiv.org/pdf/2405.15541.pdf)
[19](https://arxiv.org/pdf/2104.06358.pdf)
[20](https://www.diva-portal.org/smash/get/diva2:1355317/FULLTEXT01.pdf)
[21](https://mallada.ece.jhu.edu/pubs/2025-ACC-Tutorial-DNBetal.pdf)
[22](https://arxiv.org/html/2508.17449v1)
[23](https://www.youtube.com/watch?v=8oIQy6fxfCA)
[24](https://fugumt.com/fugumt/paper_check/2310.04582v2_enmode)
[25](https://arxiv.org/html/2506.13498v1)
[26](https://arxiv.org/abs/2406.00960)
[27](https://www.sciencedirect.com/science/article/abs/pii/S0097849325001748)
[28](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.777363/full)
[29](https://pdfs.semanticscholar.org/9c7f/7e7a5138d6bf6745d40bf81aeb38272b2cbb.pdf)
[30](https://arxiv.org/html/2510.18518v1)
[31](https://arxiv.org/pdf/2405.04235.pdf)
[32](https://pdfs.semanticscholar.org/361a/7d5369aec4bbf63f2b3d45b5f1410e9fdb9e.pdf)
[33](https://arxiv.org/html/2506.14831v2)
[34](https://arxiv.org/html/2503.06072v3)
[35](https://arxiv.org/html/2510.15842v1)
[36](https://arxiv.org/pdf/2510.18518.pdf)
[37](https://arxiv.org/html/2509.24948v1)
[38](https://arxiv.org/pdf/2402.02218.pdf)
[39](https://www.youtube.com/watch?v=tDilOjKfBaY)
[40](https://arc.aiaa.org/doi/10.2514/6.2024-0945)
[41](https://www.ijfmr.com/papers/2023/3/29706.pdf)
[42](https://pmc.ncbi.nlm.nih.gov/articles/PMC10946027/)
[43](http://www.diva-portal.org/smash/get/diva2:1355317/FULLTEXT01.pdf)
[44](https://ejournal.ppsdp.org/index.php/pijed/article/view/814)
[45](https://www.mdpi.com/2075-4701/15/9/966)
[46](https://arxiv.org/abs/2507.20445)
[47](https://dl.acm.org/doi/10.1145/3736539.3737502)
[48](https://dl.acm.org/doi/pdf/10.1145/3588432.3591525)
[49](https://dl.acm.org/doi/pdf/10.1145/3618375)
[50](https://arxiv.org/pdf/2302.00883.pdf)
[51](http://arxiv.org/pdf/2406.00960.pdf)
[52](https://arxiv.org/html/2411.06459v1)
[53](https://academic.oup.com/pnasnexus/advance-article-pdf/doi/10.1093/pnasnexus/pgad015/48827915/pgad015.pdf)
[54](https://arxiv.org/abs/2206.03198)
[55](https://arxiv.org/html/2409.14393v1)
[56](https://dl.acm.org/doi/10.1145/3550454.3555434)
[57](https://www.sciencedirect.com/science/article/abs/pii/S0736584525001656)
[58](https://stanfordasl.github.io/wp-content/papercite-data/pdf/Celestini.Gammelli.ea.RAL24.pdf)
[59](https://dl.acm.org/doi/10.1145/3618375)
[60](https://arxiv.org/pdf/2506.01563.pdf)
[61](https://www.sciencedirect.com/science/article/pii/S2405896325016441)
[62](https://arxiv.org/pdf/2405.16236.pdf)
[63](https://www.sciencedirect.com/science/article/pii/S073658452500225X)
[64](https://ieeexplore.ieee.org/document/9303903)
[65](https://openreview.net/pdf?id=pZISppZSTv)
[66](https://pdfs.semanticscholar.org/c3e7/c81d289f8beeb062ed5b1aec0f84e8675a99.pdf)
[67](https://arxiv.org/pdf/2506.24044.pdf)
[68](https://arxiv.org/html/2509.02547v1)
[69](https://arxiv.org/pdf/2410.02389.pdf)
[70](https://arxiv.org/pdf/2511.03684.pdf)
[71](https://arxiv.org/pdf/2509.12406.pdf)
[72](https://arxiv.org/pdf/2508.21101.pdf)
[73](https://arxiv.org/pdf/2509.09176.pdf)
[74](https://arxiv.org/html/2509.16457v1)
