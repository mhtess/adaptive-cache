nestedbayes
===========

Thinking through "nested-query" WebPPL models -- those where conditioning is nested inside conditioning. These are called doubly intractable in stats.

Examples include Bayesian social cognition models (including pragmatics models), Bayesian data analysis of Bayesian cognitive models, and posterior inference of model parameters for undirected models.

## Surrogate (caching) method

The approach currently explored is to create a gaussian process (GP) surrogate function for the probability of a return value from the sub-query with given params (or for its normalizing constant, which is usually the hard part). We should test this approach more thoroughly, and compare to other techniques.

The surrogate is implemented by providing a generic GP-cache function that is applied to the `erp.score` of the marginal returned from the sub-query. When the cached function is called with given args, it: evaluates the GP, determines if the uncertainty (estimate variance) is above a threshold, if so it evaluates the underlying function and updates the GP, it then returns the GP estimate (mean) of the function value. 

Currently the uncertainty threshold is a fixed parameter. It could potentially be adapted based on desired computation rate of the cached function, or other objectives. If the uncertainty threshold falls (quickly enough) an MCMC algorithms using these estimates will remain assymptotically unbiased (under some assumptions, such as boundedness of the cahced function domain).

One relatively easy extension would be to provide the gradient of the GP, in order to make this surrogate method compatible with HMC (which requires gradients). This seems like a good idea for cases like Bayesian data analysis of a cognitive model, where there are often a lot of continuous parameters (e.g. subject-wise and item-wise random effects).

## Other methods

There are a number of other approaches for doubly-intractible MCMC, and it would be worth thinking more generally. For example, in [MCMC for doubly-intractable distributions](http://arxiv.org/pdf/1206.6848.pdf) they point out that an unbiased estimator for the intractable likelihood is enough to get an unbiased MH for the whole model (via pseudo-marginal MH). [This one](http://xxx.tau.ac.il/pdf/1306.4032.pdf) extends these ideas, as well as reviewing current approaches.

## Dependencies
Requires sylvester math library (used by gaussian process code). Do `npm install sylvester`.

## Acknowledgements 
Gaussian process code was adapted from  [gaussianprocess.js](https://github.com/scotthellman/gaussianprocess_js).
