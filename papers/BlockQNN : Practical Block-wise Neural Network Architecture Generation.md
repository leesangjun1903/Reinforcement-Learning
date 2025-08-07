# BlockQNN : Practical Block-wise Neural Network Architecture Generation | Image classification, Reinforcement Learning

## 1. 핵심 주장과 주요 기여  
이 논문은 **BlockQNN**이라 불리는 블록 단위 자동 신경망 생성 기법을 제안한다. 주요 기여는 다음과 같다.  
- Reinforcement Learning(Q-learning)과 ε-greedy 탐색을 활용하여 네트워크의 기본 블록 구조만 탐색함으로써 전체 탐색 공간을 획기적으로 축소.  
- 분산 비동기 학습 프레임워크 및 **early stop** 보상 정의(식 (6), (7))를 도입하여 32GPU, 3일 만에 CIFAR-10 상의 3.54% Top-1 오류율 달성.  
- CIFAR에서 학습한 블록 구조를 ImageNet으로 옮겨도 competitive한 성능(22.6% Top-1 오류율) 발휘, **강력한 일반화** 가능성 검증.

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
- 현대 CNN 아키텍처는 레이어 수·하이퍼파라미터 조합이 방대하여 **수작업 설계 불가능**  
- 기존 NAS/MetaQNN 방식은 전체 네트워크 탐색으로 **연산 비용(800GPU×28일) 과도**  
- CIFAR → ImageNet 등 **다양한 데이터셋 전이성 부족**  

### 2.2 제안 방법  
1) **블록 단위 탐색**:  
   - 전체 네트워크가 아닌, “Network Structure Code(NSC)”로 정의된 블록만 RL 탐색  
   - NSC = (layer index, operation type, kernel size, pred1, pred2)  
2) **Q-Learning**:  
   - 상태 $$s_t$$: 현재 NSC  
   - 행동 $$a_t$$: 다음 NSC 선택  
   - 보상:  블록 스택 후 학습된 네트워크의 검증 정확도 $$\displaystyle r_T$$, 중간보상 $$r_t = r_T / T$$ (식 (6))  
   - Q-값 업데이트:  

$$
       Q(s_t,a_t) \leftarrow (1-\alpha)Q(s_t,a_t) + \alpha\Bigl[r_t + \gamma\max_{a'}Q(s_{t+1},a')\Bigr]
     $$  

3) **Early Stop 보상 재정의** (식 (7)):

$$
     \text{reward} = \text{ACCEarlyStop} - \mu\log(\text{FLOPs}) - \rho\log(\text{Density}),
   $$  

계산복잡도(FLOPs)와 그래프 밀도(Density)를 페널티로 활용해 조기 중단 학습과 최종 성능의 상관성을 높임.  
4) **분산 비동기 프레임워크**:  
   - Master agent → Controller → 다수 Compute node로 병렬 학습 및 보상 수집  

### 2.3 모델 구조  
- “블록”을 여러 번 반복(Stack)하여 전체 네트워크 구성  
- CIFAR: 3×3 Pre-activation Convolutional Cell(PCC), 풀링층 삽입, N=4  
- ImageNet: CIFAR 블록 반복(N=3) + 추가 다운샘플링, 필터 [64,128,256,능 향상  
- CIFAR-10: 오류율 3.54% 달성(Top-1)  
- CIFAR-100: 오류율 18.06% 달성(Top-1)  
- ImageNet-1K: 오류율 22.6% 달성(Top-1)  
- NAS 대비 자원: 32GPU×3일 vs. 800GPU×28일  

### 2.5 한계  
- PCC 외 다양한 레이어 유형 탐색 미흡(Depthwise conv, dilated conv 등 미적용)  
- FLOPs·밀도 외 추가 자원 제약(메모리, 지연시간) 미반영  
- 객체탐지·세그멘테이션 등 다른 비전 과제에 대한 검증 부족  

## 3. 일반화 성능 향상 관련 고찰  
- 블록 설계⇒다양한 입력 크기/CIFAR→ImageNet 전이 가능  
- **블록 반복 구조** 덕에 입력 해상도·채널 수 조정만으로도 재학습 없이 재사용 가능  
- Early stop 보상에 복잡도 페널티 포함 ⇒ **과적합 억제** 및 가벼운 모델 구조 선호 유도  

## 4. 향후 연구에 미치는 영향 및 고려 사항  
- **모듈화된 블록 탐색** 패러다임은 AutoML 연구의 효율성을 획기적으로 개선  
- 다양한 연산자(Depthwise conv, self-attention)·하이브리드 블록 탐색으로 확장 가능  
- 자원 제약(메모리, 지연시간)·실시간 응용을 고려한 다중 보상 설계 필요  
- 탐색된 블록의 **이론적 분석**을 통한 구조적 특성 이해 및 해석 가능성 제고  

실제 AutoML 및 네트워크 경량화·전이 학습 연구에 본 논문의 블록 기반 Q-러닝 프레임워크를 적용·확장하면, 제한된 자원으로도 고성능 신경망을 효율적으로 설계할 수 있는 길이 열릴 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/660c4885-b6cd-4d5c-a214-50b9ce836287/1708.05552v3.pdf
