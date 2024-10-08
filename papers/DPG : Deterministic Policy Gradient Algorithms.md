# Deterministic Policy Gradient Algorithms

# Abstract
Stochastic Policy $\pi(a \lvert s)$ 가 아닌 Deterministic Policy $\mu(s)$ 에서도 Policy Gradient Theorem이 성립함을 보이고 있다.  
Deterministic Policy란 결국 Stochastic Policy에서 Variance가 0인 특수한 경우라는 것을 증명하고, 이를 통해 Stochastic Policy Gradient가 적용되는 기존 방법론(Actor-Critic)에서도 Deterministic Policy를 사용할 수 있다는 것을 보이고 있다.  
Actor-Critic에서 Critic을 Function Approximator $Q^w(s,a)$ 로 하여 Performance Gradient를 구할 때, 실제 Q-Function이 아니어서 발생하는 Bias를 제거하는 조건을 제시하고 있다.  

Deterministic Policy Gradient를 제안한다.  
Policy Gradient는 stochastic policy에 대해서만 존재한다고 믿고 있었는데 Silver는 이 논문을 통해서 deterministic policy에 대해서도 정의될 수 있음을 보였다.

# Introduction
Policy Gradient는 action space가 연속일 때 주로 사용되는데, 이전까지는 stochastic policy에서만 정의되던 Policy Gradient를 deterministic policy에서도 정의했다.  
그 뿐 아니라 deterministic policy gradient가 action-value의 그래디언트를 따르는 model-free 형식으로 나타낼 수 있음을 보였다.  
Deterministic policy는 state가 주어졌을 때 어떠한 action을 취해야 하는지가 정해져있으므로 어떠한 정책의 state value를 추정하고싶으면 state space에서 고른 탐색을 하는 것으로 충분하다.  
Stochastic policy의 경우에는 state뿐 아니라 action space에서도 고르게 탐색을 해야 하는 것을 생각해보면 전자가 더 효율적이라는 것을 알 수 있다.

하지만 deterministic policy는 충분한 탐색이 힘들다는 단점이 있다.  
State value를 추정하기 위해서는 여러 state에서의 경험이 필요한데 action이 정해져있으니 다양한 정보가 쌓이지 않는 것이다.  
따라서 이 논문에서는 off-policy 방식을 사용하여 탐색을 한다.  
이와 관련된 exploration-exploitation에 대해서 본 논문은 아래와 같이 정리하고 있다.

- 충분한 탐색을 위해 stochastic behaviour policy를 사용하여 action을 정하고, DPG의 효율성을 위해 deterministic한 target policy를 학습한다.  

Action-value 함수 추정을 위해 DPG를 이용한 off-policy actor-critic 알고리즘을 유도하고, 추정한 action-value함수의 그래디언트 방향으로 policy parameter를 업데이트하는 방식이다.

# Background
## Preliminaries

여타 표기는 기존 강화학습의 표기와 동일하지만, 몇가지 다르게 쓰이는 점이 있어 소개한다. 먼저 $t$ 스텝부터 시작해서 받은 discounted 보상의 합은 
$r^\gamma_t=\sum^ \infty_{k=t} \gamma^{k-t}r(s_k,a_k)$ 으로 표기한다.  
또한,  정책 $\pi$를 따랐을 때 상태 $\(s\)$ 에서 $\(t\)$ 스텝을 거쳐 $\(s'\)$ 에 도달할 확률을 $\(p(s\rightarrow,t,\pi)\)$ 로 표현한다.  
자연스럽게 $\(s'\)$ 에 있을 확률을 discount한 것을 $\rho^\pi(s')\coloneqq \int_\mathcal{S}\sum^\infty_{t=1}\gamma^{t-1}p_1(s)p(s\rightarrow s',t,\pi)\,ds$ 라고 표현한다.  
$\(s\)$ 가 시작 상태일 확률 $(\(p_1(s)\))$ 에 대해서 $\(s'\)$ 에 도달할 수 있는 확률을 평균낸 것이다.  
그렇다면 강화학습 문제의 목표는 아래와 같이 정의가 된다.  
$\(\mathcal{A}=\mathbb{R}^m\)$이고 $\(\mathcal{S}\)$ 는 $\(\mathbb{R}^d\)$ 의 컴팩트한 부분집합이라고 가정하자.

```math
J(\pi_\theta)=\int_\mathcal{S}\rho^{\pi}(s)\int_\mathcal{A}\pi_\theta(s,a)r(s,a)\,da\,ds=\mathbb{E}_{s\sim\rho^\pi,a\sim\pi_\theta}[r(s,a)]
```

정책 $π$ 를 따랐을 때 모든 상태에서 얻을 수 있는 평균 보상이다.

## Stochastic Policy Gradient Theorem
Stochastic policy gradient method는 목적함수의 그래디언트 $\(\nabla_\theta J(\pi_\theta)\)$ 의 방향으로 정책의 파라미터를 업데이트 하는 방법이다. 여기에서 중요하게 쓰이는 정리는 stochastic policy gradient theorem인데, 아래와 같다.

```math
\nabla_\theta J(\pi_\theta)=\int_\mathcal{S}\rho^\pi(s)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\mid s)Q^\pi(s,a)\,da\,ds\\=\mathbb{E}_{s\sim \rho^\pi,a\sim\pi_\theta}[\nabla_\theta\log\pi_\theta(a\mid s)Q^\pi(s,a)]
```

목적함수의 그래디언트를 구하는 데 시작 상태의 분포가 들어가지 않는 점이 인상적인 부분이다.  
이 정리에서 나오는 $\(Q^\pi(s,a)\)$ 를 쉽게 알수가 없어서 여러 추정 방식을 사용한다.  
예를 들어, REINFORCE에서는 샘플에서 얻을 수 있는 $\(r^\gamma_t\)$ 로 $\(Q^\pi(s_t,a_t)\)$ 를 추정하여 $\(Q^\pi\)$ 를 Monte-Carlo방식으로 추정한다.

## Stochastic Actor-Critic Algorithm

Policy Gradient Theorem은 actor-critic algorithm에서 주로 사용된다.  
Actor는 위에서 정의한 $\(\nabla_\theta J(\pi_\theta)\)$ 의 방향으로 $\(\pi_\theta(s)\)$ 의 파라미터 $\(\theta\)$ 를 업데이트하여 적당한 policy를 찾아간다.  
$\(Q^\pi\)$ 의 추정을 위해서 $\(Q^w(s,a)\)$ 라는 모델을 따로 만든다.  
Critic은 TD-learning같은 policy evaluation을 사용하여 $\(Q^w(s,a)\approx Q^\pi(s,a)\)$을 달성하려고 한다.

$\(\nabla_\theta J(\pi_\theta)\)$ 에 $\(Q^\pi\)$ 대신 $\(Q^w\)$ 를 사용하면 bootstrapping의 효과가 있기 때문에 bias가 생길 수 있다.  
하지만 이 function approximator가 compatible하다면 bias가 없다는 것이 Sutton에 의해서 증명되었다.  
Compatible하다는 것은 아래와 같은 두 가지 조건을 만족하는 approximator를 뜻한다.

$(Q^w(s,a)=\nabla_\theta\log\pi_\theta(a\mid s)^\top w$ , 즉 stochastic policy $\nabla_\theta\log\pi_\theta(a\mid s)$ 의 feature에 대해서 선형적인 approximator여야 한다.  
파라미터 $\(w\)$ 를 찾는 방식이 
```math
\epsilon^2(w)=\mathbb{E}_{s\sim\rho^\pi,a\sim\pi_\theta}[(Q^w(s,a)-Q^\pi(s,a))^2]
```
를 최소화하는 방향이어야 한다.  

하지만 Comparing Policy-Gradient Algorithms : Sutton 2000 에 따르면 위의 두 가정을 모두 만족시키는 approximator는 REINFORCE알고리즘과 동일한 결과를 가지게 된다.  
$\(r^\gamma_t\)$ 를 사용하는 것과 똑같은 것이다.  
따라서, 2번 조건은 TD-learning 등 policy evaluation을 사용할 때 주로 완화된다.

## Off-Policy Actor-Critic
Off-policy 기법에서는 behaviour policy와 target policy가 따로 존재한다.  
On-policy방법으로 진행하면 state과 action에서 고른 정보를 얻기 힘들기 때문에 다양한 시도를 하는 behaviour policy를 따로 두는 것이다.  
이제 강화학습 문제의 목표는 조금 바뀌게 된다.  
우리가 구하고자 하는 target policy가 optimal policy임을 생각하자.  
Target policy의 value function의 평균을 구하고자하는 것은 이전과 같지만, 탐색을 하는 정책이 달라졌기 때문에 state distribution을 behaviour policy에 따라야 하는 것이다.

```math
J_\beta(\pi_\theta)=\int_\mathcal{S}\rho^\beta(s)V^\pi(s)\,ds\\
=\int_\mathcal{S}\int_\mathcal{A}\rho^\beta(s)\pi_\theta(a\mid s)Q^\pi(s,a)\,da\,ds
```

Off-Policy Actor-Critic 에서는 위 목표함수의 그래디언트를 아래와 같이 근사하였다.

```math
\nabla_\theta J_\beta(\pi_\theta)\approx\int_\mathcal{S}\int_\mathcal{A}\rho^\beta(s)\nabla_\theta\pi_\theta(a\mid s)Q^\pi(s,a)\,da\,ds = \\ \mathbb{E}_{s\sim\rho^\beta,a\sim\beta}[\frac{\pi_\theta(a\mid s)}{\beta_\theta(a\mid s)}\nabla_\theta\log\pi_\theta(a\mid s)Q^\pi(s,a)]
```

$\(\nabla_\theta Q^\pi(s,a)\)$에 해당하는 항을 스킵하면 위와 같은 식을 얻을 수 있다.  
Degris가 제안한 off-policy actor-critic에서 critic은 off-policy하게 gradient temporal-difference learning을 활용하여 state-value function $\(V^v(s)\approx V^\pi(s)\)$ 를 근사한다.  
Actor 또한 off-policy로 $\(\nabla_\theta J_\beta(\pi_\theta)\)$ 을 근사한 위의 식을 활용하여 policy parameter $\(\theta\)$ 를 업데이트한다.  
문제가 되는 $\(Q^\pi(s,a)\)$ 는 TD error $\(\delta_t=r_{t+1}+\gamma V^v(s_{t+1})-V^v(s_t)\)$ 로 근사한다.  
Actor와 critic 둘 다에서 importance sampling ratio $\(\frac{\pi_\theta(a\mid s)}{\beta_\theta(a\mid s)}\)$ 가 사용되었는데, action이 target policy $\(\pi\)$ 가 아닌 $\(\beta\)$ 에서 샘플링되었기 때문에 그 비율을 맞춰주는 역할을 한다.

# Gradients of Deterministic Policies
## Action-Value Gradients
Generalized Policy Iteration framework는 Policy Evaluation과 Policy Improvement 단계로 이루어져 있다.  
Policy Evaluation에서는 주어진 정책 하에서 각 상태-행동 value function을 추정하는 단계이며, Monte-Carlo방식이나 TD방식 등이 사용된다.  
Policy Improvement에서는 추정된 state-action value function을 기반으로 최적의 정책을 찾는다.  
행동이 finite한 경우에는 행동을 하나씩 보면서 기대 보상이 제일 큰 행동을 취하는 방식으로 policy를 만들면 되겠지만, (greedy maximisation) action space가 연속인 경우에도 이런 방식을 쓰기 위해서는 매 Policy Improvement step마다 (non-) convex optimization 문제를 풀어야 하기 때문에 상당히 비효율적이다.

따라서 $\(Q\)$ 를 최대화하는 것보다 $\(Q\)$ 의 그래디언트의 방향으로 정책을 업데이트 하는 방법을 사용한다.  
정확하게는 정책 파라미터 $\(\theta^{k+1}\)$ 를 $\(\nabla_\theta Q^{\mu^k}(s,\mu_\theta(s))\)$ 방향으로 업데이트 하는 것이다.  
각 state마다 다른 그래디언트를 반환할 것이기 때문에 state distribution에 대해서 평균을 취한다.

```math
\theta^{k+1}=\theta^k+\alpha\mathbb{E}_{s\sim\rho^{\mu^k}}[\nabla_\theta Q^{\mu^k}(s,\mu_\theta(s))]
```

Chain rule을 취하면 action에 대한 action-value의 그래디언트와 policy parameter에 대한 policy의 그래디언트로 나누어지는 것을 볼 수 있다.
```math
\theta^{k+1}=\theta^k+\alpha\mathbb{E}_{s\sim\rho^{\mu^k}}\Big[\nabla_\theta\mu_\theta(s)\nabla_a Q^{\mu^k}(s,a)\Big|_{a=\mu_\theta(s)} \Big]
```
이렇게 정책을 바꾸면, 그 정책에 따라서 state distribution $\(\rho^\mu\)$ 또한 변하게 되어서 위의 업데이트 방식이 올바르지 못하다고 생각할 수 있지만, stochastic policy gradient과 마찬가지로 state distribution에 의존하지 않는 그래디언트 표현식을 도출할 수 있다.

## Deterministic Policy Gradient Theorem
이제 제대로 DPG를 정의한다. 아래와 같이 정의된 강화학습 문제의 목적함수가 있다고 하자.

```math
J(\mu_\theta)=\mathbb{E}[r^\gamma_1\mid\mu]\\
=\int_\mathcal{S}\rho^\mu(s)r(s,\mu_\theta(s))\,ds=\mathbb{E}_{s\sim\rho^\mu}[r(s,\mu_\theta(s))]
```

- Theorem 1. MDP가 regularity condition을 만족한다면 deterministic policy gradient가 존재하고, 아래와 같이 나타낼 수 있다.
```math
\nabla_\theta J(\mu_\theta)=\int_\mathcal{S}\rho^\mu(s)\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)\mid_{a=\mu_\theta(s)}ds \\ =\mathbb{E}_{s\sim\rho^\mu}[\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)\mid_{a=\mu_\theta(s)}]
```
여기서 제안하는 regularity condition은 state distribution, policy, reward과 그 도함수가 모든 파라미터에 대해서 연속임을 가정하는 것이다.

## Limit of the Stochastic Policy Gradient
Stochastic policy에 결정적인 부분과 확률적인 부분이 있다고 한다면, 확률적인 부분을 줄이면 줄일수록 deterministic policy에 가까워진다는 생각을 할 수 있다.  
이 논문에서는 stochastic policy $\(\pi_{\mu_\theta,\sigma}\)$ 를 deterministic policy $\(\mu_\theta\)$ 와 분산 $\(\sigma\)$ 로 나누어서 생각한다.  
이 방식으로 $\(\sigma\rightarrow0\)$ 일 때 stochastic policy gradient가 deterministic policy gradient에 수렴한다는 것을 증명하였다.

- Theorem 2. MDP가 regularity condition을 만족하고 $\(\nu_\sigma\)$ 가 regular data approximation, $\(\sigma\)$ 가 분산을 조정하는 파라미터일 때 $\(\pi_{\mu_\theta,\sigma}(a\mid s)=\nu_\sigma(\mu_\theta(s),a)\)$ 인 stochastic policy $\(\pi_{\mu_\theta,\sigma}\)$ 가 있다고 하면 아래와 같은 등식이 성립한다.

```math
\underset{\sigma\downarrow0}{\lim}\nabla_\theta J(\pi_{\mu_\theta,\sigma})=\nabla_\theta J(\mu_\theta)
```
좌변은 stochastic policy gradient, 우변은 deterministic policy gradient를 나타낸다.

stochastic policy의 분산이 작아져서 0이 되면 그 그래디언트가 deterministic policy gradient에 수렴하는 것을 보인 것이다.

# Deterministic Actor-Critic Algorithm
Deterministic Policy Gradient를 정의했으니, actor-critic 알고리즘에 DPG를 적용한다.

## On-Policy Deterministic Actor-Critic
Deterministic policy에 on-policy방식으로 actor-critic을 진행하면 탐색이 제한되기 때문에 제대로 value function approximation이 힘들어질 수 있으나, 일단 적용 가능성을 위해 on-policy deterministic Actor-Critic 알고리즘을 만든다.

Stochastic Actor-Critic 과 마찬가지로, actor는 위에서 정의한 deterministic policy gradient에 따라서 policy parameter $\(\theta\)$ 를 업데이트 하고, critic은 on-policy기법인 SARSA를 사용하여 action-value function을 근사한다.  
알고리즘은 아래와 같다.

```math
\begin{aligned}
\delta_t=r_t+\gamma Q^w(s_{t+1},a_{t+1})-Q^w(s_t,a_t) \newline
w_{t+1}=w_t+\alpha_w\delta_t\nabla_w Q^w(s_t,a_t) \newline
\theta_{t+1}=\theta_t+\alpha_\theta\nabla_\theta\mu_\theta(s_t)\,\nabla_aQ^w(s_t,a_t)\mid_{a=\mu_\theta(s)}
\end{aligned}
```

## Off-Policy Deterministic Actor-Critic
탐색을 높이기 위해서 Off-policy Actor-Critic 을 제안한다.  
Deterministic target policy $\(\mu_\theta(s)\)$ 와 stochastic behaviour policy $\(\beta(a\mid s)\)$ 를 나누어서 생각한다.  
Stochastic off-policy Actor-Critic 때와 마찬가지로 강화학습 문제의 목표함수를 다음과 같이 정의한다.

```math
J_\beta(\mu_\theta)=\int_\mathcal{S}\rho^\beta(s)V^\mu(s)\,ds=\int_\mathcal{S}\rho^\beta(s)Q^\mu(s,\mu_\theta(s))\,ds
```

Stochastic Policy Gradient때와 마찬가지로 $\(\nabla_\theta Q^{\mu_\theta}(s,a)\)$ 항을 스킵하면 아래와 같이 목표함수의 그래디언트를 근사할 수 있다.
```math
\nabla_\theta J_\beta(\mu_\theta)\approx\int_\mathcal{S}\rho^\beta(s)\nabla_\theta\mu_\theta(a\mid s)Q^\mu(s,a)\,ds\\
=\mathbb{E}_{s\sim \rho^\beta}[\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)\mid_{a=\mu_\theta(s)}]
```
Actor-Critic알고리즘에서 actor는 위의 그래디언트 방향으로 policy parameter를 업데이트하고, critic은 off-policy기법인 Q-Learning을 통해서 action-value function을 근사한다.  
전체적인 알고리즘은 아래와 같다.

```math
\begin{aligned}
\delta_t=r_t+\gamma Q^w(s_{t+1},\mu_\theta(s_{t+1}))-Q^w(s_t,a_t)\\
w_{t+1}=w_t+\alpha_w\delta_t\nabla_w Q^w(s_t,a_t)\\
\theta_{t+1}=\theta_t+\alpha_\theta\nabla_\theta\mu_\theta(s_t)\,\nabla_aQ^w(s_t,a_t)\mid_{a=\mu_\theta(s)}
\end{aligned}
```
또한, $\(a_t\)$는 $\(\beta\)$에서 샘플링되지만 action-value function은 이 policy에서 off-policy하게 업데이트 된다는 것을 알 수 있다.  
Q-Learning target $\(\delta\)$ 가 $\(\mu\)$ 에서 나온 action을 기반으로 만들어지기 때문이다.  
$\(\mu\)$ 는 target policy이자 optimal policy를 학습하는 것이므로 첫줄의 $(Q^w(s_{t+1},\mu_\theta(s_{t+1}))\)$ 는 $\(\underset{a}{\text{max}}Q\)$ 를 근사하는 것이라고 생각할 수 있다.

정책이 deterministic하기 때문에 action distribution에서 취해지는 적분이 필요가 없어지게 되어 actor에서 importance sampling이 필요가 없다.  
비슷하게 Q-Learning을 사용하기 때문에 critic에서도 importance sampling이 필요가 없는 것을 알 수 있다.

## Compatible Function Approximation
앞서 stochastic policy gradient 에서 compatible function에 대해서 살펴보았다.  
이 부분에서는 deterministic policy gradient에서 $\(Q^w\)$ 가 compatible하기 위해 필요한 조건을 알아보고 compatible한 함수로 DPG A-C알고리즘을 만든다.  
그래디언트에 영향을 주지 않고 $\(\nabla_aQ^\mu\)$ 대신 사용할 수 있는 $\(\nabla_aQ^w\)$ 로 DPG Actor-Critic 알고리즘을 만드는 것이다.

- Theorem 3. $\(\nabla_\theta J_\beta(\theta)=\mathbb{E}[\nabla_\theta\mu_\theta(s)\,\nabla_aQ^w(s,a)\mid_{a=\mu_\theta(s)}]\)$ 인 $\(\mu_\theta(s)\)$ 에 $\(Q^w(s,a)\)$ 가 compatible하기 위해서는 아래와 같은 조건을 만족해야 한다.
```math
\begin{aligned}
1. \nabla_aQ^w(s,a)\mid_{a=\mu_\theta(s)}=\nabla_\theta\mu_\theta(s)^\top w  \\
2. \epsilon(s;\theta,w)=\nabla_aQ^w(s,a)\mid_{a=\mu_\theta(s)}-\nabla_aQ^\mu(s,a)\mid_{a=\mu_\theta(s)}일 때 파라미터 w는 \(MSE(\theta,w)=\mathbb{E}[\epsilon(s;\theta,w)^\top\epsilon(s;\theta,w)]\)를 최소화한다.
\end{aligned}
```


# Reference
- https://sylim2357.github.io/paper%20review/dpg/
- https://talkingaboutme.tistory.com/entry/RL-Review-Deterministic-Policy-Gradient-Algorithm
- https://enfow.github.io/paper-review/reinforcement-learning/model-free-rl/2020/12/25/DPG-deterministic_policy_gradient_algorithms/
