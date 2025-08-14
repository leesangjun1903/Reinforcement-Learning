# Stop Regressing: Training Value Functions via Classification for Scalable Deep RL

## 1. 핵심 주장 및 주요 기여  
본 논문은 **기존의 MSE 회귀 대신 분류(classification) 방식으로 가치 함수를 학습**함으로써, 심층 강화학습(value-based RL)의 **성능·안정성·확장성**을 대폭 향상시킬 수 있음을 실험적으로 입증했다.  
- MSE 회귀가 가진 **노이즈 민감성**, **비정상성(non-stationarity)**, **표현력 한계** 문제를 분류 손실(categorical cross-entropy)로 대체함으로써 완화  
- 다양한 도메인(아타리, 멀티 태스크·멀티 게임, 로봇 제어, 체스 무탐색, Wordle 언어 에이전트) 및 아키텍처(SoftMoE, ResNet, Transformer)에서 **30–115%** 이상의 성능 개선 및 안정적 스케일업 달성  
- 특히 Imani & White(2018)의 **HL-Gauss** 히스토그램 레이블링이 Two-Hot, C51 등 기존 방식 대비 **일관되게 최상 성능**  

## 2. 문제 설정, 제안 방법, 모델 구조, 성능 향상 및 한계  

### 2.1 해결하고자 하는 문제  
- 가치 기반 RL에서 Q-값 회귀(regression) 손실(MSE)은  
  1) **노이즈가 섞인 TD 목표치**에 민감  
  2) **비정상적(target non-stationarity)**인 업데이트로 학습 불안정  
  3) **대형 네트워크**(ResNet, Transformer, MoE) 학습 시 성능 저하  
- 반면 대규모 분류(classification)는 cross-entropy 손실로 **안정적 확장**이 잘 알려져 있음  

### 2.2 제안 방법  
1) **Categorical Representation**  
   - Q-값을 직접 예측하는 대신, 지지점(𝑧₁,…,𝑧ₘ) 간격으로 나눈 범위 [𝑣ₘᵢₙ,𝑣ₘₐₓ] 위의 **categorical distribution** ˆpᵢ(s,a;θ)를 softmax 출력으로 모델링  
   - 예측 Q(s,a;θ)=∑ᵢˆpᵢ·zᵢ  

2) **Cross-Entropy TD 손실**  
   - TD 타깃 $(𝒯Q)(s,a)=r+γmaxₐ′Q(s′,a′;θ⁻)$ 를 categorical target 분포 $pᵢ(s,a;θ⁻)$ 로 변환  
   - 손실:  

$$ L(θ)=𝔼_D\bigl[-\sum_{i=1}^m p_i(s_t,a_t;θ^-) \log \hat p_i(s_t,a_t;θ)\bigr] $$

3) **Target Projection 기법**  
   - Two-Hot: 연속 타깃을 둘로 둘러싼 두 지지점에 선형 보간  
   - C51 (CDRL): 분포적 Bellman 투영  
   - **HL-Gauss**: 타깃을 μ=(𝒯Q)로 갖는 정규분포 N(μ,σ²)를 bin 경계로 적분하여 **label smoothing** 효과 확장  
     – σ/δ≈0.75 권장 (δ=bin width)  

### 2.3 모델 구조  
- **Single-task Atari**: DQN + Adam, 네트워크는 Nature CNN  
- **Mixture-of-Experts**: Impala penultimate 레이어를 SoftMoE(1,2,4,8 experts)로 교체  
- **Multi-task / Multi-game**: Impala-CNN → ResNet-{18,34,50,101}  
- **Wordle**: GPT-스타일 Transformer (125M) with CQL regularizer  
- **Chess**: Q-함수 distillation용 Transformer (9M,137M,270M)  
- **Robotics**: Vision-based Q-Transformer (60M)  

### 2.4 성능 향상  
- **Single-task Atari**: HL-Gauss +30% IQM↑, C51+10%, Two-Hot underperforms MSE  
- **Offline Atari**: MSE 학습 붕괴 안정화, HL-Gauss 최고 유지  
- **MoE**: 파라미터 수 독립적 +30% 상대 성능 향상  
- **ResNet Scale-Up**: MSE는 ResNet-34 이후 성능 저하, HL-Gauss는 지속 개선  
- **Wordle**: 성공률 +40%  
- **Chess Distill**: 270M 모델이 AlphaZero(400 MCTS)의 수준 근접  
- **Robotics**: 학습 효율·정상율 +67%  

### 2.5 한계  
- **이론적 근거 미흡**: 왜 cross-entropy가 최적화·표현력 향상에 유리한지 상세 분석 필요  
- **Additional Domains**: Continual RL, 대규모 사전학습→미세조정 등 미검증  
- **하이퍼파라미터**: σ/δ, bin 개수 민감도 및 자동 조정 기법 부재  

## 3. 모델의 일반화 성능 향상 가능성  
- **표현 학습**: Linear probing 실험에서 HL-Gauss 표현이 MSE 대비 더 나은 Q-값 재학습 지원  
- **과적합 방지**: HL-Gauss의 레이블 스무딩 효과로 noisy target에 덜 민감  
- **비정상성 대응**: SARSA(정적 정책)에서는 MSE 대비 이점 약화, Q-러닝(non-stationary)에서는 뚜렷  
- **스케일업**: 대형 네트워크에서도 안정적·지속적 성능 개선  

## 4. 향후 연구 영향 및 고려 사항  
- **알고리즘 설계**: Value-based RL의 표준 손실로 cross-entropy 채택 권장  
- **이론 연구**: Cross-entropy 최적화 동역학·representational bias 분석 필요  
- **적응형 레이블링**: σ/δ 자동 조절, bin range 동적 변경 연구  
- **도메인 확장**: Continual, 메타-RL, 멀티모달 에이전트 등에 분류 손실 효과 검증  
- **하드웨어 가속**: MSE 대신 CCE로 캐시·연산 패턴 달라질 수 있어 최적화 고려  

---  
**주요 시사점**: 단순히 회귀를 분류로 전환하는 것만으로도 Deep RL의 확장성과 일반화 성능을 획기적으로 개선할 수 있으며, 향후 RL 알고리즘 개발의 새로운 표준이 될 잠재력을 지닌다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/4ed20aa6-623f-49dc-98cf-7fabe83b31af/2403.03950v1.pdf
