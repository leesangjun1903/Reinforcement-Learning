# Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor

**핵심 주장 및 주요 기여**  
Soft Actor-Critic(SAC)은 **최대 엔트로피 강화학습(maximum entropy RL)** 프레임워크를 기반으로 한, **off-policy** actor-critic 알고리즘으로, 기대 보상뿐 아니라 정책의 엔트로피를 최대화함으로써  
- **샘플 효율성**을 획기적으로 개선하고  
- **수렴 안정성**을 크게 높이며  
- **하이퍼파라미터 민감도**를 완화  

시킵니다. 이로써 SAC는 DDPG, PPO 같은 기존 on-/off-policy 기법을 능가하는 성능과 안정성을 연속 제어 벤치마크에서 일관되게 달성합니다.[1]

***

## 1. 해결하고자 하는 문제  
기존의 모델-프리 딥 RL 알고리즘들은  
- **샘플 복잡도**가 매우 높아 수백만~수천만 스텝의 경험이 필요하고  
- **하이퍼파라미터**(학습률, 탐험 강도 등)에 극도로 민감하여 수작업 튜닝이 필수적  

입니다. 특히, on-policy 방식은 매 그라디언트 업데이트마다 새 경험을 요구해 비효율적이고, DDPG 같은 off-policy 기법은 **함수 근사 불안정성**으로 인해 수렴이 불안정합니다.[1]

***

## 2. 제안 방법

### 2.1 최대 엔트로피 강화학습 목표  
일반 RL 목표  

$$
J(\pi) = \mathbb{E}\bigl[\sum_t r(s_t,a_t)\bigr]
$$  

최대 엔트로피 RL 목표  

$$
J(\pi) = \mathbb{E}\Bigl[\sum_t \bigl(r(s_t,a_t) + \alpha\,\mathcal{H}(\pi(\cdot|s_t))\bigr)\Bigr]
$$  

온도 파라미터 $$\alpha$$는 보상과 엔트로피의 상대적 중요도를 조절합니다.[1]

### 2.2 소프트 정책 반복(Soft Policy Iteration)  
탭형 설정(tabular)에서  
1. **정책 평가:** 소프트 벨만 연산자 $$T^\pi$$를 통해 소프트 Q-함수 $$Q^\pi$$ 수렴 보장  
2. **정책 개선:** Kullback–Leibler 프로젝션으로 업데이트  

$$
\pi_{\text{new}} = \arg\min_{\pi'\in\Pi}D_{KL}\Bigl(\pi'(\cdot|s)\big\|\exp(Q^{\pi_{\text{old}}}(s,\cdot))/Z\Bigr)
$$  

이 과정을 반복하면 정책 클래스 $$\Pi$$ 내에서 최적 엔트로피 정책으로 수렴합니다.[1]

### 2.3 연속 도메인용 근사 SAC 알고리즘  
함수 근사를 위해  
- **Vψ**, **Qθ₁, Qθ₂**, **πφ** 네트워크 사용  
- **리플레이 버퍼**로 off-policy 업데이트  
- **타깃 네트워크**(지수 이동 평균)로 안정화  

각 네트워크 손실  
– 소프트 값 함수:  

$$
J_V = \mathbb{E}_{s\sim D}\Bigl[\tfrac12\bigl(V_\psi(s)-\mathbb{E}_{a\sim\pi_\phi}[Q_\theta(s,a)-\log\pi_\phi(a|s)]\bigr)^2\Bigr]
$$  

– 소프트 Q 함수:  

$$
J_Q = \mathbb{E}_{(s,a)\sim D}\Bigl[\tfrac12\bigl(Q_\theta(s,a)-[r+\gamma\,V_{\bar\psi}(s')]\bigr)^2\Bigr]
$$  

– 정책: 재파라미터화 기법으로  

$$
J_\pi = \mathbb{E}_{s\sim D,\epsilon}\bigl[\log\pi_\phi(f_\phi(\epsilon;s)|s)-Q_\theta(s,f_\phi(\epsilon;s))\bigr]
$$  

두 개의 Q-함수를 쓰는 **twin Q 아이디어**로 과대추정 편향을 완화합니다.[1]

***

## 3. 모델 구조  
정책 네트워크는 상태를 입력받아 평균·분산을 출력하는 가우시안 분포를 학습하며, 출력에 tanh를 적용해 행동 범위를 제한합니다.  
Critic은 두 개의 Q-네트워크와 하나의 V-네트워크로 구성되며, 목표 네트워크로 지수 이동 평균 방식을 사용합니다.[1]

***

## 4. 성능 향상 및 한계

### 4.1 성능 향상  
OpenAI Gym의 Hopper, Walker2d, HalfCheetah, Ant, Humanoid 벤치마크에서 SAC는  
- 더 빠른 학습 속도와 높은 최종 성능  
- 다양한 랜덤 시드 간 **일관된** 안정성  

을 보여주며, DDPG, PPO, SQL, TD3를 모두 능가합니다.[1]

### 4.2 한계  
- **보상 스케일**에 민감: 엔트로피와 보상 균형을 맞추기 위해 환경별 보상 스케일 튜닝 필요.[1]
- **하드웨어 요구**: 복잡한 신경망과 대규모 리플레이 버퍼로 연산 비용이 높음.  
- **이론적 확장**: 이차 정보(trust region 등)나 더욱 표현력 높은 정책 클래스 연구가 필요.[1]

***

## 5. 일반화 성능 향상 관점  
최대 엔트로피 목표는  
- 다양한 행동 모드를 학습 가능  
- 탐험(exploration) 강화로 지역 최적해 회피  
- **추정 오류에 대해 로버스트(robust)**  

특히, 예측 오차나 환경 변화에 강한 정책을 학습하여 실제 로봇 제어 등에서 **일반화 성능**을 높일 수 있습니다.[1]

***

## 6. 향후 영향 및 고려 사항  
- **연구 영향**: SAC는 안정적 off-policy 최대 엔트로피 RL 알고리즘으로 자리매김하며, 이후 TD3, RWR, AWAC 같은 변형 연구에 영감을 제공했습니다.  
- **고려 사항**:  
  - 보상 스케일 해석 및 자동 조정 기법  
  - Trust-region, 자연정책경사 등 **이차 최적화** 접목  
  - **분산 학습** 및 효율적 하드웨어 활용  
  - **정책 표현력** 강화(Flow, Normalizing Flows)  

이를 통해 실제 로봇 제어, 시뮬레이션 간 격차(sim-to-real) 극복, 멀티태스크 일반화 등으로 확장할 수 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ad65275b-f429-4c1f-97ed-ebe0aff18d66/1801.01290v2.pdf)
