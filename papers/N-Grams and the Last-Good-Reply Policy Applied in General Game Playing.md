# N-Grams and the Last-Good-Reply Policy Applied in General Game Playing

**핵심 주장**  
이 논문은 범용 게임 플레이(General Game Playing, GGP)에서 몬테카를로 트리 탐색(MCTS)의 시뮬레이션 단계 성능을 크게 향상시킬 수 있는 두 가지 핵심 기법을 제안한다.  
1. **ϵ-탐욕(ϵ-Greedy) 전략**이 기존의 소프트맥스(Gibbs) 기댓치 기반 선택보다 시뮬레이션 효율을 높인다.  
2. **N-Gram Selection Technique (NST)**와 **Last-Good-Reply Policy (LGRP)**를 도입해, 단일 수(單一 수) 평가를 넘어 “유망한 수열(move sequences)”과 “직전 성공 응수(reply)”를 학습함으로써 시뮬레이션 지침(simulation guidance)을 개선한다.  

이들 기법을 CADIAPLAYER 에이전트에 통합해, 다양한 두·다인용(turn-based)·동시행동(simultaneous-move)·다인용 게임에서 평균 승률을 약 70%로 끌어올렸다.

***

# 1. 해결하고자 하는 문제

1. MCTS의 **플레이아웃(play-out)** 단계에서 무작위 시뮬레이션은 강력한 휴리스틱 기반 프로그램에 비해 성능이 저조하다.  
2. 기존 GGP 에이전트(CADIAPLAYER)는 MAST(Move-Average Sampling Technique) + 소프트맥스(Gibbs) 방식에 의존해,  
   -  “좋은 수”를 전역 Qh(a)로만 평가 → 맥락(context)을 반영하지 못함  
   -  Gibbs 선택 확률이 다른 수의 가치에 따라 변동 → 최적수 보장이 약함  

→ **맥락 정보 활용 & 안정적 선택**을 위한 새로운 시뮬레이션 지침 필요

***

# 2. 제안 방법

## 2.1 ϵ-Greedy vs. Gibbs Measure  
- **Gibbs**:  
  $$P(a) = \frac{\exp(Q_h(a)/\tau)}{\sum_b \exp(Q_h(b)/\tau)}$$  

  -  τ=온도 매개변수  
  -  단점: 높은 Qh 이동의 선택 확률이 타격수 Qh에 의존, 미보장  
- **ϵ-Greedy**:  
  -  확률 $$1-\epsilon$$: 최고 Qh(a) 수 선택  
  -  확률 $$\epsilon$$: 균등 무작위 선택  
→ **고정된 탐험·이용(explore-exploit) 균형** 보장

## 2.2 N-Gram Selection Technique (NST)  
- 플레이아웃마다 각 플레이어가 둔 **연속 수열(sequence) S** 길이 1,2,3을 기록  
- 각 S에 대해 시뮬레이션 보상 R(S) 의 평균을 전역 테이블에 누적  
- 시뮬레이션 중 & 트리의 신규 수 선택 시:  
  1. 후보 수 a에 대하여 생성 가능한 S₁(S 길이1), S₂(길이2), S₃(길이3) 조회  
  2. 횟수 ≥ k(=7)인 S₂,S₃만 포함하여 $$ \text{score}(a) = \tfrac{1}{m}\sum R(S_i)$$ 계산  
  3. 미등장 수는 점수 100 부여 → 탐험 유도  
- 선택: Gibbs 또는 ϵ-Greedy 적용

## 2.3 Last-Good-Reply Policy (LGRP, LGRF-2)  
- **단일 수의 성공 응수(best reply)**를 직전 1수·2수 시퀀스별로 저장  
- 시뮬레이션 종료 후 플레이어 보상이 현 최고 보상 이상이면 응수 저장, 그렇지 않으면 삭제(forgetting)  
- 시뮬레이션 중 & 신규 수 선택 시:  
  1. 직전 두 수에 대한 저장 응수 시도 → 합법이면 선택  
  2. 실패 시 직전 한 수 응수 시도  
  3. 모두 실패 시 **폴백(fallback)**: MAST-ϵGreedy 또는 NST-ϵGreedy

***

# 3. 모델 구조 및 수식

- **UCT 선택**:  

```math
    a^* = \arg\max_{a\in A(s)} \Bigl\{Q(s,a) + C\sqrt{\frac{\ln N(s)}{N(s,a)}}\Bigr\}
```

- **MAST Qh 업데이트**:  
  $$\displaystyle Q_h(a) \leftarrow \frac{\sum_{\text{playouts containing }a} R}{\text{count}(a)}$$  
- **NST R(S) 업데이트**:  
  $$\displaystyle R(S) \leftarrow \frac{\sum_{\text{playouts containing }S} R}{\text{count}(S)}$$  
- **LGRP 저장·삭제**  
  - 저장: $$R_{\text{this}} \ge \max_{\text{others}}R$$  
  - 삭제: $$R_{\text{this}} < \max_{\text{others}}R$$

***

# 4. 성능 향상 결과

- **ϵ-Greedy(ϵ=0.4)**는 5개 두인용 게임 평균에서 Gibbs 대비 4/5 게임에서 통계적 유의미 승률 개선  
- **NST**: 원본 MAST 대비 두인용 5게임 평균 승률 약 70% 달성  
- **LGRP+NST**: NST 단독과 유사한 성능, 일부 게임(Connect5, Breakthrough)서 우위  
- **시간 민감도**:  
  - 장시간(60+30s) vs 단시간(10+10s) 모두 NST/LGRP 큰 폭 성능 향상  
  - CPU 성능 향상 시 더 큰 상대적 이득 전망

***

# 5. 한계 및 향후 과제

1. **게임 특이성**: 전략별 최적 성능이 게임별로 다름 → 적합 전략 자동 선택 필요  
2. **온-라인 파라미터 튜닝**: ϵ, k 등 매개변수의 고정치 한계 → 실시간 조정 메커니즘 개발  
3. **비표준 턴 메커니즘**: noop 식별·취급 자동화 → Pentago 등 확장 지원  

***

**요약**: 본 논문은 GGP 에이전트의 시뮬레이션 성능을 ϵ-Greedy, N-Gram 기반 시뮬레이션, Last-Good-Reply 정책을 통해 크게 향상시켰으며, 다양한 게임에서 평균 70% 승률을 달성했다. 온-라인 전략 선택·파라미터 튜닝 연구가 후속 과제로 제시된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/072c8c62-7744-4e53-a96f-2147c1ec1eb6/NST-and-MAST.pdf
