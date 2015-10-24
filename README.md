# Adaptive cacheing
Bayesian adaptive cacheing for expensive models

# Notes

```mathjax
F:X->Y
```

Say we have a (deterministic?) function $F:X->Y$ that is expensive to compute and we expect to be calling it a lot on "similar" X values inside some other computation. (eg. RSA with continuous state space, so $$$X$$$ is $$$R^n$$$ while $$$Y$$$ is a distribution over utterances or states.) 

say we can write a prior over (potentially stochastic) functions X->Y as a probabilistic program P with latent parameters \theta.

we make a cached version of F by:

cache(F,P) is the function(X):
  estimate mean and  variance (or HDI?) of F'(X), with F'~P(\theta).
  if variance is above a threshold:
    evaluate F(X)
    update \theta by doing doubly- stochastic gradient of ELBO
    return F(X)
  else return mean

note that we are doing variational inference on P. this is so that we have a posterior approx that is online and with a fast update. time is constant per evaluation of underlying function, but depends on number of q samples used to approximate gradient. (note that we may need to do gradient steps on previous evaluation data as we go along, to keep from washing out what we've learned.)

this is related to the previous GP-cache idea, but with more general priors on the function. it would be good if the version where the prior is a GP is expressible... while there are variational techniques for inference in GPs, we probably also want to consider exact posteriors here, since they exist. 

can we get the neural adaptive basis function regression thing (ala ryan adam's group) by taking P to be a distribution on multilayer perceptrons, doing mean field on the weights on all but the last layer (to get effective MAP optimization) and then maintain uncertainty on last layer? for the last part, maybe have the functions be uncertain and P contains weights and variances?

there should be a straightforward extension to the case where our evaluation results in a noisy approximation to F (eg from running MCMC). should think through.

for structured cases we can encode knowledge in prior P: eg "this region will have the same value, whatever it is". or, "the function decomposes into factors".

this extends to functions where input and output spaces (X and Y) are not continuous (or mix discrete and continuous) as long as a plausible but flexible prior over functions can be given. eg. are there cases where it helps for ordinal domains (tendency to assume that neighboring integers are similar)?

above i wrote it with a fixed confidence bound to decide whether to run F. this bound could adapt, eg it could drop over time to result in asymptotic correctness. it could also be set to achieve a target computation rate: aim for X evals of F per minute or something.

it might be possible to track the time taken by each evaluation of F, infer a surrogate to this time (or other resource) function, and use it as part of the decision of whether to run F(X) for a new X. ie one can do a resource-rational (bayesian decision theory) computation using the surrogate resource function.

in terms of implementation, i'm imagining using webppl (or vipp) directly to handle inference on the approximating program. this should mean we can get a quite flexible version of this working very quickly. it may be that we'll have to be careful about overhead issues to get it working well (eg if P doesn't have structure change, can we compute the gradient tapes just once?).



applications / related areas:
RL: value-function approximation is basically this, right?
POMDPs: treating belief distributions this way seems very promising. has anyone done it? also extends to multi-agent case.
parsing: apply to parsing with continuous PCFGs? eg what if we wanted to do a generative version of recursive NN stuff, where we don't know the parses ahead of time?
doubly intractable models.
ABC.
more general caching: minimizing network access in browsers?
really anywhere in CS where caching is used for expensive functions could benefit!
