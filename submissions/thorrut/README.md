# Centralized PPO Implementation

## Train

```sh
python train.py --help
```

## View train results

```sh
tensorboard --logdir runs
```

## Model explanation

During a step, one agent is coordinating the whole team of predators. The
framework as is allows that (a StudentAgent use the same global TeamAgent and
the latter guess that a step for the team is over when the same StudenAgent
call it again).

The TeamAgent works like this :

1. The first predator calling it makes it take a decision in function of its
   observations only:
   $$\Pi (a_{1} | s_{1})$$
2. The second predator receives an action in function of its observations **and
   also the chosen action for the previous predator**:
   $$\Pi (a_{2} | s_{2}, a_1)$$
3. The third takes also in account the decisions of all the predators before it
   for this step.
   $$\Pi (a_{3} | s_{3}, a_1, a_2)$$
4. The process is generalized if the number of predators (let's define it $L$)
   is higher.

We can then infer the probability of a *team action* $\bigcap_{i=1}^{L}
a_i$ in function of a global state $s$ of the SimpleTag environment for the
team at a given step, if we assume in our model that $s$ is a *team
observation* that can be put in touch to any the individual observations of the
predators :
$$
s \approx s_1 \approx s_2 \approx ... \approx s_L \Rightarrow
\Pi(\bigcap_{i=1}^{L} a_i | s) \approx \prod_{i=1}^{L}\Pi(a_i | s_i, a_1,\dots, a_{i-1})
$$

The $\Pi (a_i | s_i , a_1, \dots, a_{i-1} )$ distributions are learnt by the
policy network which internally uses a Recurrent Neural Network to compute
contextual features from the action of the already-processed predators. Those contextual features enrich hidden features got from the individual
observations within Multi-Perceptron Layers, thanks to a sum operation of the
hidden states.

With such an architecture, we sequentially generates the actions (as in a
language model) for each predator and, during training, we can compose the
final log probability of the team action and the entropy of the distribution of
team actions returned by a theoretical *team actor*. With assuming all the
predator observations are equivalent, we also use one observation to compute
the critic value. As the policy network compute hidden features from the
individual observations, those features from the observations of the first
predator are passed through the critic layer head.

In this setting, we have at each step one classical PPO agent ready to learn.

## Experimental notes

A model trained on quite enough episodes (200) with enough steps (cycles) per
epochs is three times better than a model trained on 800 episodes and 25 steps
per episode.

32.400 as average evaluation score (so the prey is touched once in the average
cases).
