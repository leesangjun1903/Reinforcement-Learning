# What is Reinforcement Learning?

Consider you are teaching the dog to catch a ball, but you cannot teach the dog explicitly to
catch a ball, instead, you will just throw a ball, every time the dog catches a ball, you will
give a cookie.  
If it fails to catch a dog, you will not give a cookie. So the dog will figure out what actions it does that made it receive a cookie and repeat that action.

Similarly in an RL environment, you will not teach the agent what to do or how to do,
instead, you will give feedback to the agent for each action it does. The feedback may be
positive (reward) or negative (punishment). The learning system which receives the
punishment will improve itself. Thus it is a trial and error process. The reinforcement
learning algorithm retains outputs that maximize the received reward over time. In the
above analogy, the dog represents the agent, giving a cookie to the dog on catching a ball is
a reward and not giving a cookie is punishment.

There might be delayed rewards. You may not get a reward at each step. A reward may be
given only after the completion of the whole task. In some cases, you get a reward at each
step to find out that whether you are making any mistake.

An RL agent can explore for different actions which might give a good reward or it can
(exploit) use the previous action which resulted in a good reward. If the RL agent explores
different actions, there is a great possibility to get a poor reward. If the RL agent exploits
past action, there is also a great possibility of missing out the best action which might give a
good reward. There is always a trade-off between exploration and exploitation. We cannot
perform both exploration and exploitation at the same time. We will discuss exploration exploitation
dilemma detail in the upcoming chapters.

Say, If you want to teach a robot to walk, without getting stuck by hitting at the mountain,
you will not explicitly teach the robot not to go in the direction of mountain,

![](https://github.com/leesangjun1903/Reinforcement-Learning/blob/main/B09792_01_01.png)

Instead, if the robot hits and get stuck on the mountain you will reduce 10 points so that
robot will understand that hitting mountain will give it a negative reward so it will not go
in that direction again.

![](https://github.com/leesangjun1903/Reinforcement-Learning/blob/main/B09792_01_02.png)

And you will give 20 points to the robot when it walks in the right direction without getting
stuck. So robot will understand which is the right path to rewards and try to maximize the
rewards by going in a right direction.

![](https://github.com/leesangjun1903/Reinforcement-Learning/blob/main/B09792_01_03.png)
