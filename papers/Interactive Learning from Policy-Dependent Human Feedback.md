# Interactive Learning from Policy-Dependent Human Feedback

## 1. 핵심 주장 및 주요 기여  
이 논문은 **인간 트레이너의 피드백이 에이전트의 현재 정책(behavior policy)에 따라 달라진다**는 실험적 증거를 제시하고, 이를 반영한 새로운 배우-비평가(actor-critic) 알고리즘인 **COACH(Convergent Actor–Critic by Humans)**를 제안한다. 주요 기여는 다음과 같다:[1]
- 인간 피드백이 정책-의존적(policy-dependent)임을 입증하는 사용자 연구  
- Advantage 함수 $$A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s)$$를 **인간 피드백 모델**로 활용  
- COACH 알고리즘 이론적 수렴 보장  
- TurtleBot 로봇 실험을 통해 실시간 학습 가능성과 복합 행동 학습 능력 입증  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 문제 정의  
- **Human-Centered Reinforcement Learning (HCRL)**: 보상 신호가 고정된 보상 함수 대신 인간 트레이너의 피드백으로 주어지는 상황.[1]
- 기존 HCRL은 피드백을 정책-독립적(policy-independent)이라고 가정하였으나, 이는 실제 인간 피드백과 불일치하여 비효율적 학습 및 의도치 않은 행동을 초래할 수 있음.

### 2.2 제안 방법: COACH  
1. **Advantage 함수 기반 피드백 모델**  

$$
     A^{\pi}(s,a) = Q^{\pi}(s,a)\;-\;V^{\pi}(s)
   $$  
   
- 행동이 현재 정책 대비 얼마나 ‘개선’되었는지 정량화  
2. **정책 업데이트 규칙**  

$$
     \Delta \theta_t = \alpha\, f_{t+1}\, \nabla_\theta \ln \pi_\theta(s_t,a_t)
   $$  
   
여기서 $$f_{t+1}$$은 휴먼 피드백(≈Advantage), $$\pi_\theta$$는 softmax 정책.[1]
3. **Real-time COACH**  
   - **보상 집계**: 이산 크기의 피드백값 + 다중 피드백 누적  
   - **Eligibility Trace**: 과거 행동에 지연-가중치 적용($$\lambda$$)-decay  
   - **피드백 지연 보정**: 인간 반응 지연 $$d$$ 스텝 보정  

### 2.3 모델 구조  
- **정책 파라미터** $$\theta(s,a)$$를 선형 함수로 표현  
- TurtleBot 실험: RGB 영상 특징 → 컬러별 sum-pooling → tanh 활성화 + bias 파라미터 구조  

### 2.4 성능 향상  
- **정책-의존적 피드백**을 활용한 COACH가 다양한 피드백 전략(task, action, improvement)에서 기존 Q-learning 및 TAMER 대비 빠르고 안정적 수렴.[1]
- **TurtleBot 로봇 실험**: 5개 행동(push–pull, hide, ball following, alternate, cylinder navigation)을 2분 이내에 학습, 복합·합성 행동에도 성공.[1]

### 2.5 한계  
- 고차원 관측(원시 영상)에서 **특징 추출(feature engineering)**이 수작업에 의존  
- 트레이너의 정책 모델 추정 오차 및 **피드백 지연**, 과도한 자격 추적(trace) 관리 필요  
- 대규모 상태·행동 공간에서 **계산·샘플 효율성** 검증 부족  

## 3. 일반화 성능 향상 관점  
- Advantage 기반 피드백은 **Diminishing Returns**, **Differential Feedback**, **Policy Shaping** 등 행태 분석 원칙을 자연스럽게 반영하여 과적합 위험을 감소시키고 **일반화**를 촉진.[1]
- **Eligibility trace**로 과거 행동 영향력을 조절, 지연 피드백에서도 효과적으로 정책을 개선하여 다양한 환경 변화에 적응 가능  
- 향후 **딥러닝 기반 표현학습**과 결합 시, 정책-의존적 피드백이 심층 신경망에서도 일반화 성능을 획기적으로 개선할 잠재력 보유  

## 4. 미래 연구에의 영향 및 고려사항  
COACH는 인간-강화학습 인터페이스 설계에 새로운 패러다임을 제공하며, 다음 연구 시 고려할 점은 다음과 같다:
- **딥 RL 결합**: 대규모 신경망으로 확장해 인간 피드백 신호 활용  
- **자동 특징 학습**: End-to-end 학습으로 피드백 일반화 및 문턱값 자동 조정  
- **트레이너 모델링**: 인간이 실제로 인식하는 정책과 에이전트 정책 간 차이 보정  
- **다중 트레이너**: 편향·신뢰도 다른 피드백 통합 방법론 개발  
- **장기적 상호작용**: 피드백 지연·지속적 교육 시 안정성·수렴 이론 강화  

이러한 방향은 **인간과 AI 협업 학습**의 폭넓은 응용과 실용화를 앞당길 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/dad45d2b-703e-4ec7-86ab-fecac12819ae/1701.06049v2.pdf)
