# Adaptive cacheing
Bayesian adaptive cacheing for expensive models

# Notes
(Summarized from a note by @ngoodman 10/23/15)

### Set up

Say we have a (deterministic?) function F:X->Y that is expensive to compute and we expect to be calling it a lot on "similar" X values inside some other computation. (eg. RSA with continuous state space, so X is R^n while Y is a distribution over utterances or states.) 

Say we can write a prior over (potentially stochastic) functions X->Y as a probabilistic program P with latent parameters \theta.

We then make a cached version of F by:

```
cache(F,P) is the function(X):
  estimate mean and variance (or HDI?) of F'(X), with F'~P(theta).
  if variance is above a threshold:
    evaluate F(X)
    update theta by doing doubly-stochastic gradient of ELBO
    return F(X)
  else return mean
```

Note that we are doing variational inference on P. This is so that we have a posterior approximation that is online and with a fast update. Time is constant per evaluation of underlying function, but depends on number of q samples used to approximate gradient. (Note that we may need to do gradient steps on previous evaluation data as we go along, to keep from washing out what we've learned.)

### Getting started

This is related to the previous GP-cache idea, but with more general priors on the function. It would be good if the version where the prior is a GP is expressible... while there are variational techniques for inference in GPs, we probably also want to consider exact posteriors here, since they exist. 

Can we get the [neural adaptive basis function](http://arxiv.org/abs/1502.05700) regression thing (a la [Ryan Adams' group](http://hips.seas.harvard.edu)) by 

1. Take P to be a distribution on multilayer perceptrons.
2. Do mean field on the weights on all but the last layer (to get effective MAP optimization).
3. Maintain uncertainty on last layer. 

For the last part, maybe have the functions be uncertain and P contain weights and variances?

In terms of implementation, I'm imagining using webppl (or vipp) directly to handle inference on the approximating program. This should mean we can get a quite flexible version of this working very quickly. It may be that we'll have to be careful about overhead issues to get it working well (e.g. if P doesn't have structure change, can we compute the gradient tapes just once?)

### Extensions

There should be a straightforward extension to the case where our evaluation results in a noisy approximation to F (e.g. from running MCMC). Should think through.

For structured cases, we can encode knowledge in prior P: e.g. "this region will have the same value, whatever it is". or, "the function decomposes into factors".

This extends to functions where input and output spaces (X and Y) are not continuous (or, mixtures of discrete and continuous) as long as a plausible but flexible prior over functions can be given. e.g. are there cases where it helps for ordinal domains (tendency to assume that neighboring integers are similar)?

Above we considered fixed confidence bounds to decide whether or not to run F. This bound could adapt. For example, it could drop over time to result in asymptotic correctness. It could also be set to achieve a target computation rate: Aim for X evals of F per minute or something.

It might be possible to track the time taken by each evaluation of F, infer a surrogate to this time (or other resource) function, and use it as part of the decision of whether to run F(X) for a new X. I.e., one could do a resource-rational (bayesian decision theory) computation using the surrogate resource function.


### Applications / related areas

+ RL: value-function approximation is basically this, right?
+ POMDPs: treating belief distributions this way seems very promising. has anyone done it? also extends to multi-agent case.
parsing: apply to parsing with continuous PCFGs? eg what if we wanted to do a generative version of recursive NN stuff, where we don't know the parses ahead of time?
doubly intractable models.
+ ABC.
+ more general caching: minimizing network access in browsers?
+ really anywhere in CS where caching is used for expensive functions could benefit!
