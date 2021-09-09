
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
# Notes
_To view the following equations on GitHub, you will need the [MathaJax for GitHub plugin](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima?hl=en)_.
## Vectorization

```py
import numpy as np

a = np.random.rand(1000000)
b = np.random.rand(1000000)
v = np.random.rand(1000000)

# Instead of for loops that calculate c (c[i] = a[i]*b[i]), use:
c = np.dot(a, b)

# Instead of for loops that calculate c (u[i] = math.exp(v[i])), use:
u = np.exp(v)

# simirarly we have
np.abs(a)
np.maxiimum(a)

# so, all z are...
z = np.dot(w.T, X)

# For gradients, we can keep in mind...
A = np.sigma(z)
# and for gradients
dz = A - Y
db = (1/m)*np.sum(dw)
dw = (1/m)*X*dz.T

# numpy tips
# don't use (5, ) (rank 1 array) data shape - use (5, 1) with reshape for np.dot etc
# the behavios of vectors wil be easier to understand
asserta.shape(a) == (5, 1)
```

_See solutions of week 2 here: https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Week%202%20Quiz%20-%20Neural%20Network%20Basics.md_

## Neural Networks

#### Notation
* () for training examples
* [] for layer number
* {} for mini-batches od data
* 1 "epoch" is a single pass through the training set (iteration)
* := values gets updated
### NNs
NNs are like taking the liner regression and repeating it twice

We can initialize all weights to zero in logistic regression but not in NNs. All hidden units will be symmetrical, so they'll calculate the same function, and we want them to calculate different functions. Ideally, we want small initializations, so we can do:

```py
np.random.randn((2, 2)) * 0.01
```

## Deep Neural Networks
A 1-hidden, 1-node is more of "shallow" NN (like logistic regression). A "Deep" NN would be a 5-hidden layer NN. We denote the number of Layers with **L**, and **n** the units in a layer. We have activations for each one of the layers, as well as b values. So Instead of X, we have A[0] (activation of 0th layer).



### Dimensions
* $W^{[l]}: (n^{[l]}, n^{[l-1]})$
* $b^{[l]}: (n^{[l]}, 1)$
* $Z^{[l]}, A^{[l]}: (n^{[l]}, m)$

Same for derivative's dimensions, for back propagation.

In a way, the first layers of a NN calculate simpler functions (high level like hard lines in an image) and then they are composed together and they calculate more complex things (like details in an image like brows).


## High Bias - High Variance
### Identify them

Errors examples:

    Train: 1%  & Dev: 11% -> High Variance
    Train: 15% & Dev: 16% -> High Bias (bad train data performance)
    Train: 15% & Dev 30% -> Both High Variance and High Bias
    Train: 0.5% & Dev: 1% -> Low Bias & Low Variance s


### Basic recipie for improving ML results
* High Bias (training set performance bad) -> Bigger Network or train longer
* High Variance (validation set performance bad) -> More data or Regularization or dropout reguliration

### Regularization
Do it when we have high variance: Add a parameter at the end (penalize) of the Cost Function J(w[i], b[i]) like L2 (most popular). Can also add L1.

After L2 (or Frobenius Norm or weight decay) regularization w will be sparse, so it will have a lot of zeros, so lew memory to store them. λ is the regularization parameter. We usually set this when using cross validation, or defiining hyperparameters.

It's also called weight decay, as at the end the w is multiplying with something slightly less than 1.

Regularization reduces overfitting as it does not let the model fit to the data that much by adding something at the end. Reduced w means reduced z, so it tends to be closer to a linear function, which cannot fit very easily.

### Dropout regulirization method
Go through all layers of NN and set a probability of removing one or more node, so we have a smaller network.

One weight cannot rely on one feature, cause that can go away anytime, so the NN has to spread out weights and shrink them.

Implement Inverted Dropout:
```py
l=3
keep_probability = 0.8
d3 = np.random.randn(a3.shape[0], a3.shape[1]) < keep_probability
a3 = np.multiply(a3, d3)
a3 /= keep_probability
```
Notes
* Maybe we don't need to do a dropout on test time - we don't want our output to be random.
* We can also have different keep_probability for each layer.
* Downside: not well defined J, so  the J to # iterations plot won't work.

### More regulirization methods

#### Data Augmentation

**Example**: take a pic from yout examples, flip it, rotate it, spread it etc, and add all thos new images to the dataset.

#### Early stoppin

Plot # iteration to error, dev error and training error or J. When those two start seperate, that's the # of iterentions to stop at. That way we have a mid-size w, so again smaller w, so the NN does not iterate too many times to fit the data completely.


## Optimizations

### Exploding/vanishing gradients
Deep networks can have exponential small or big gradients (is a functions of L), so it will never finds 0 and training is very hard. The solution is to randomly initialize the weights.

The more weights, the smaller we want them to be so z is mid-size. We can do:

```py
w[i] = np.random.randn(shape_of_matrix)*np.sqrt(1/n[l-1]) #(or 2/n... for ReLU)
```
### Check your derivaties computation in back probagation
Use this ONLY yo debug.
Numerical checking of gradients:
* Take $W^{[i]}, b^{[i]}, ... W^{[L]}, b^{[L]}$ and reshape them into a bog vector $θ$.
* Take $dW^{[i]}, db^{[i]}, ... dW^{[L]}, db^{[L]}$ and reshape them into a bog vector $dθ$.
So now we have $J(θ)$.
* Do a for each:
    * find $dθ$ and check:
    * Calculate the Euclidian distance between $dθ$ and $dθ_{approx}$. if this is something like $10^{-7}$ should be fine

Remember to do grad check with regulirizations, and that this doesn't work with dropout - turn in on after debugging. Also, we can run grad check with random initilization and let it rin for a while.

## Optimization Algorithms - Make your algorithms run faster
### Mini-batch gradient descent
* Break the training set in mono-batches: $X^{\{1\}}, X^{\{2\}}, ..., X^{\{m\}}$, so we have $Y^{\{1\}}, Y^{\{2\}}, ..., Y^{\{m\}}$ as our predictions. We can run them all in the same time.

Implementing this, we would have a for loop for all the batches, and the equaztions would be calculated for each mini batch (Zs, bs, Js and Ys). In mini-batch approach, instead of having 1 gradient descenr for 1 epoch, we have 5000 (for batches).

    If mini-batch size = m: Batch Grdient Descent (use it for small sets, <= 2000)
    If mini-batch size = 1: Stohastic Gradient Descent
    Ideal mono-batch size: somewhere in the middle + vectorization -> much faster

Typical mini-batches: $2^6, 2^7, 2^8$.


### More optimization algs
#### Moving/Exponential weighted average
$V_t = β*V_{t-1} + (1 - β)* θ_t$

$β = weight$

$V = average$

Adapts more slower to data changes, but is used to have less data in memory and have faster training.

To avoid the delay, we do a **Bias correction**:

$V_t = {1/(1 - β_t)}$

No need to have Nias correction here.


#### Momentum
We want slower learninf rate vertically, and faster horizontally:

$V_t = β*V_{dw} + (1 - β)* db$

$w := w - aV_dw$

$b := b - aV_db$




## Tips
* Start with a small NN or even logistic regression.
* Debug Gradient descent by plotting J to # of iterations to see if J reduces monotonically.
* Normalize inputs: bring everything around zero by removing the average, and then normilize the variance with `x /= σ^2`.
    