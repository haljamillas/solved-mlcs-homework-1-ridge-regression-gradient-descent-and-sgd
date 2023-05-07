Download Link: https://assignmentchef.com/product/solved-mlcs-homework-1-ridge-regression-gradient-descent-and-sgd
<br>






<h1>1           Introduction</h1>

In this homework you will implement ridge regression using gradient descent and stochastic gradient descent. We’ve provided a lot of support Python code to get you started on the right track. References below to particular functions that you should modify are referring to the support code, which you can download from the website. If you have time after completing the assignment, you might pursue some of the following:

<ul>

 <li>Study up on numpy’s <a href="https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html">broadcasting</a> to see if you can simplify and/or speed up your code.</li>

 <li>Think about how you could make the code more modular so that you could easily try different loss functions and step size methods.</li>

 <li>Experiment with more sophisticated approaches to setting the step sizes for SGD (e.g. try out the recommendations in “Bottou’s SGD Tricks” on the website)</li>

 <li>Instead of taking 1 data point at a time, as in SGD, try minibatch gradient descent, where you use multiple points at a time to get your step direction. How does this effect convergence speed? Are you getting computational speedup as well by using vectorized code?</li>

 <li>Advanced: What kind of loss function will give us “quantile regression”?</li>

</ul>

<h1>2           Linear Regression</h1>

<h2>2.1           Feature Normalization</h2>

When feature values differ greatly, we can get much slower rates of convergence of gradient-based algorithms. Furthermore, when we start using regularization (introduced in a later problem), features with larger values are treated as “more important”, which is not usually what you want. One common approach to feature normalization is perform an affine transformation (i.e. shift and rescale) on each feature so that all feature values in the training set are in [0<em>,</em>1]. Each feature gets its own transformation. We then apply the same transformations to each feature on the test<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> set. It’s important that the transformation is “learned” on the training set, and then applied to the test set. It is possible that some transformed test set values will lie outside the [0<em>,</em>1] interval.

Modify function feature_normalization to normalize all the features to [0<em>,</em>1]. (Can you use numpy’s “broadcasting” here?) Note that a feature with constant value cannot be normalized in this way. Your function should discard features that are constant in the training set.

<h2>2.2           Gradient Descent Setup</h2>

In linear regression, we consider the hypothesis space of linear functions <em>h<sub>θ </sub></em>: <strong>R</strong><em><sup>d </sup></em>→ <strong>R</strong>, where

<em>h<sub>θ</sub></em>(<em>x</em>) = <em>θ<sup>T</sup>x,</em>

for <em>θ,x </em>∈ <strong>R</strong><em><sup>d</sup></em>, and we choose <em>θ </em>that minimizes the following “square loss” objective function:

<em> ,</em>

where (<em>x</em><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<em>x<sub>m</sub>,y<sub>m</sub></em>) ∈ <strong>R</strong><em><sup>d </sup></em>× <strong>R </strong>is our training data.

While this formulation of linear regression is very convenient, it’s more standard to use a hypothesis space of “affine” functions: <em>h<sub>θ,b</sub></em>(<em>x</em>) = <em>θ<sup>T</sup>x </em>+ <em>b,</em>

which allows a “bias” or nonzero intercept term. The standard way to achieve this, while still maintaining the convenience of the first representation, is to add an extra dimension to <em>x </em>that is always a fixed value, such as 1. You should convince yourself that this is equivalent. We’ll assume this representation, and thus we’ll actually take <em>θ,x </em>∈ <strong>R</strong><em><sup>d</sup></em><sup>+1</sup>.

<ol>

 <li>Let <em>X </em>∈ <strong>R</strong><em><sup>m</sup></em><sup>×(<em>d</em>+1) </sup>be the design matrix, where the <em>i</em>’th row of <em>X </em>is <em>x<sub>i</sub></em>. Let <em>y </em>=</li>

</ol>

(<em>y</em><sub>1</sub><em>,…,y<sub>m</sub></em>)<em><sup>T </sup></em>∈ <strong>R</strong><em><sup>m</sup></em><sup>×1 </sup>be the “response”. Write the objective function <em>J</em>(<em>θ</em>) as a matrix/vector expression, without using an explicit summation sign. [Being able to write expressions as matrix/vector expressions without summations is crucial to making implementations that are useful in practice, since you can use numpy (or more generally, an efficient numerical linear algebra library) to implement these matrix/vector operations orders of magnitude faster than naively implementing with loops in Python.]

<ol start="2">

 <li>Write down an expression for the gradient of <em>J </em>(again, as a matrix/vector expression, without using an explicit summation sign).</li>

 <li>In our search for a <em>θ </em>that minimizes <em>J</em>, suppose we take a step from <em>θ </em>to <em>θ </em>+ <em>ηh</em>, where <em>h </em>∈ <strong>R</strong><em><sup>d</sup></em><sup>+1 </sup>is the “step direction” (recall, this is not necessarily a unit vector) and <em>η </em>∈ (0<em>,</em>∞) is the “step size” (note that this is not the actual length of the step, which is <em>η</em>k<em>h</em>k). Use the gradient to write down an approximate expression for the change in objective function value <em>J</em>(<em>θ </em>+ <em>ηh</em>) − <em>J</em>(<em>θ</em>). [This approximation is called a “linear” or “first-order” approximation.]</li>

 <li>Write down the expression for updating <em>θ </em>in the gradient descent algorithm. Let <em>η </em>be the step size.</li>

 <li>Modify the function compute_square_loss, to compute <em>J</em>(<em>θ</em>) for a given <em>θ</em>. You might want to create a small dataset for which you can compute <em>J</em>(<em>θ</em>) by hand, and verify that your compute_square_loss function returns the correct value.</li>

 <li>Modify the function compute_square_loss_gradient, to compute ∇<em><sub>θ</sub>J</em>(<em>θ</em>). You may again want to use a small dataset to verify that your compute_square_loss_gradient function returns the correct value.</li>

</ol>

<h2>2.3           (OPTIONAL) Gradient Checker</h2>

For many optimization problems, coding up the gradient correctly can be tricky. Luckily, there is a nice way to numerically check the gradient calculation. If <em>J </em>: <strong>R</strong><em><sup>d </sup></em>→ <strong>R </strong>is differentiable, then for any vector <em>h </em>∈ <strong>R</strong><em><sup>d</sup></em>, the directional derivative of <em>J </em>at <em>θ </em>in the direction <em>h </em>is given by<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>

<em>.</em>

We can approximate this directional derivative by choosing a small value of <em>ε &gt; </em>0 and evaluating the quotient above. We can get an approximation to the gradient by approximating the directional derivatives in each coordinate direction and putting them together into a vector. In other words, take <em>h </em>= (1<em>,</em>0<em>,</em>0<em>,…,</em>0) to get the first component of the gradient. Then take <em>h </em>= (0<em>,</em>1<em>,</em>0<em>,…,</em>0) to get the second component. And so on. See <a href="http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization">http://ufldl.stanford.edu/wiki/index. </a><a href="http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization">php/Gradient_checking_and_advanced_optimization</a> for details.

<ol>

 <li>Complete the function grad_checker according to the documentation given. Alternatively, you may complete the function generic_grad_checker so that it works for any objective function. It should take as parameters a function that computes the objective function and a function that computes the gradient of the objective function. Note: Running the gradient checker takes extra time. In practice, once you’re convinced your gradient calculator is correct, you should stop calling the checker so things run faster.</li>

</ol>

<h2>2.4           Batch Gradient Descent<sup>3</sup></h2>

At the end of the skeleton code, the data is loaded, split into a training and test set, and normalized. We’ll now finish the job of running regression on the training set. Later on we’ll plot the results together with SGD results.

<ol>

 <li>Complete batch_gradient_descent.</li>

 <li>Now let’s experiment with the step size. Note that if the step size is too large, gradient descent may not converge<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a>. Starting with a step-size of 0<em>.</em>1, try various different fixed step sizes to see which converges most quickly and/or which diverge. As a minimum, try step sizes 0.5, 0.1, .05, and .01. Plot the value of the objective function as a function of the number of steps for each step size. Briefly summarize your findings.</li>

 <li>(Optional) Implement backtracking line search (google it). How does it compare to the best fixed step-size you found in terms of number of steps? In terms of time? How does the extra time to run backtracking line search at each step compare to the time it takes to compute the gradient? (You can also compare the operation counts.)</li>

</ol>

<h2>2.5           Ridge Regression (i.e. Linear Regression with <em>`</em><sub>2 </sub>regularization)</h2>

When we have a large number of features compared to instances, regularization can help control overfitting. Ridge regression is linear regression with <em>`</em><sub>2 </sub>regularization. The regularization term is sometimes called a penalty term. The objective function for ridge regression is

where <em>λ </em>is the regularization parameter, which controls the degree of regularization. Note that the bias parameter is being regularized as well. We will address that below.

<ol>

 <li>Compute the gradient of <em>J</em>(<em>θ</em>) and write down the expression for updating <em>θ </em>in the gradient descent algorithm. (Matrix/vector expression – no summations please.)</li>

 <li>Implement compute_regularized_square_loss_gradient.</li>

 <li>Implement regularized_grad_descent.</li>

 <li>For regression problems, we may prefer to leave the bias term unregularized. One approach is to change <em>J</em>(<em>θ</em>) so that the bias is separated out from the other parameters and left unregularized. Another approach that can achieve approximately the same thing is to use a very large number <em>B</em>, rather than 1, for the extra bias dimension. Explain why making <em>B </em>large decreases the effective regularization on the bias term, and how we can make that regularization as weak as we like (though not zero).</li>

 <li>(Optional) Develop a formal statement of the claim in the previous problem, and prove the statement.</li>

 <li>(Optional) Try various values of <em>B </em>to see what performs best in test.</li>

 <li>Now fix <em>B </em>= 1. Choosing a reasonable step-size (or using backtracking line search), find the that minimizes <em>J</em>(<em>θ</em>) over a range of <em>λ</em>. You should plot the training loss and the test loss (just the square loss part, without the regularization, in each case) as a function of <em>λ</em>. Your goal is to find <em>λ </em>that gives the minimum test loss. It’s hard to predict what <em>λ </em>that will be, so you should start your search very broadly, looking over several orders of magnitude. For example,. Once you find a range that works better, keep zooming in. You may want to have log(<em>λ</em>) on the <em>x</em>-axis rather than <em>λ</em>. [If you like, you may use sklearn to help with the hyperparameter search.]</li>

 <li>What <em>θ </em>would you select for deployment and why?</li>

</ol>

<h2>2.6           Stochastic Gradient Descent</h2>

When the training data set is very large, evaluating the gradient of the objective function can take a long time, since it requires looking at each training example to take a single gradient step. When the objective function takes the form of an average of many values, such as

(as it does in the empirical risk), stochastic gradient descent (SGD) can be very effective. In SGD, rather than taking −∇<em>J</em>(<em>θ</em>) as our step direction, we take −∇<em>f<sub>i</sub></em>(<em>θ</em>) for some <em>i </em>chosen uniformly at random from {1<em>,…,m</em>}. The approximation is poor, but we will show it is unbiased.

In machine learning applications, each <em>f<sub>i</sub></em>(<em>θ</em>) would be the loss on the <em>i</em>th example (and of course we’d typically write <em>n </em>instead of <em>m</em>, for the number of training points). In practical implementations for ML, the data points are randomly shuffled, and then we sweep through the whole training set one by one, and perform an update for each training example individually. One pass through the data is called an epoch. Note that each epoch of SGD touches as much data as a single step of batch gradient descent. You can use the same ordering for each epoch, though optionally you could investigate whether reshuffling after each epoch affects the convergence speed.

<ol>

 <li>Show that the objective function</li>

</ol>

can be written in the form by giving an expression for <em>f<sub>i</sub></em>(<em>θ</em>) that makes the two expressions equivalent.

<ol start="2">

 <li>Show that the stochastic gradient ∇<em>f<sub>i</sub></em>(<em>θ</em>), for <em>i </em>chosen uniformly at random from {1<em>,…,m</em>}, is an unbiased estimator of ∇<em>J</em>(<em>θ</em>). In other words, show that E[∇<em>f<sub>i</sub></em>(<em>θ</em>)] = ∇<em>J</em>(<em>θ</em>) for any <em>θ</em>. (Hint: It will be easier, notationally, to prove this for a general , rather than the specific case of ridge regression. You can start by writing down an expression for E[∇<em>f<sub>i</sub></em>(<em>θ</em>)]…)</li>

 <li><em>W </em>rite down the update rule for <em>θ </em>in SGD for the ridge regression objective function.</li>

 <li>Implement stochastic_grad_descent. (Note: You could potentially generalize the code you wrote for batch gradient to handle minibatches of any size, including 1, but this is not necessary.)</li>

 <li>Use SGD to find <em>θ<sub>λ</sub></em><sup>∗ </sup>that minimizes the ridge regression objective for the <em>λ </em>and <em>B </em>that you selected in the previous problem. (If you could not solve the previous problem, choose <em>λ </em>= 10<sup>−2 </sup>and <em>B </em>= 1). Try a few fixed step sizes (at least try <em>η<sub>t </sub></em>∈ {0<em>.</em>05<em>,.</em>005}. Note that SGD may not converge with fixed step size. Simply note your results. Next try step sizes that decrease with the step number according to the following schedules: and</li>

</ol>

. For each step size rule, plot the value of the objective function (or the log of the objective function if that is more clear) as a function of epoch (or step number, if you prefer) for each of the approaches to step size. How do the results compare? Some things to note: 1) In this case we are investigating the convergence rate of the optimization algorithm with different step size schedules, thus we’re interested in the value of the objective function, which√

includes the regularization term. 2) Sometimes the initial step size (1 for 1<em>/t </em>and 1<em>/ t</em>) is too aggressive and will get you into a part of parameter space from which you can’t recover. Although in this problem, this probably means you implemented something poorly, in general you can try starting at <em>t </em>= 10, for example, to skip the large initial steps. 3) As we’ll learn in an upcoming lecture, SGD convergence is much slower than GD once we get close to the minimizer. (Remember, the SGD step directions are very noisy versions of the GD step direction). If you look at the objective function values on a logarithmic scale, it may look like SGD will never find objective values that are as low as GD gets. In terminology we’ll learn in Lecture 2, GD has much smaller “optimization error” than SGD. However, this difference in optimization error is usually dominated by other sources of error (estimation error and approximation error). Moreover, for very large datasets, SGD (or minibatch GD) is much faster (by wall-clock time) than GD to reach a point that’s close [enough] to the minimizer.

<ol start="6">

 <li>(Optional) Try a stepsize rule of the form, where <em>λ </em>is your regularization constant, and <em>η</em><sub>0 </sub>a constant you can choose. How do the results compare?</li>

</ol>

<h1>3           Risk Minimization</h1>

<h2>3.1           Square Loss</h2>

<ol>

 <li>Let <em>y </em>be a random variable with a known distribution, and consider the square loss function <em>`</em>(<em>a,y</em>) = (<em>a </em>− <em>y</em>)<sup>2</sup>. We want to find the action <em>a</em><sup>∗ </sup>that has minimal risk. That is, we want to find <em>a</em><sup>∗ </sup>= argmin<em><sub>a </sub></em>E(<em>a </em>− <em>y</em>)<sup>2</sup>, where the expectation is with respect to <em>y</em>. Show that <em>a</em><sup>∗ </sup>= E<em>y</em>, and the Bayes risk (i.e. the risk of <em>a</em><sup>∗</sup>) is Var(<em>y</em>). In other words, if you want to try to predict the value of a random variable, the best you can do (for minimizing expected square loss) is to predict the mean of the distribution. Your expected loss for predicting the mean will be the variance of the distribution. [Hint: Recall that Var(<em>y</em>) = E<em>y</em><sup>2 </sup>− (E<em>y</em>)<sup>2</sup>.]</li>

 <li>Now let’s introduce an input. Recall that the expected loss or “risk” of a decision function <em>f </em>: X → A is</li>

</ol>

<em>R</em>(<em>f</em>) = E<em>`</em>(<em>f</em>(<em>x</em>)<em>,y</em>)<em>,</em>

where (<em>x,y</em>) ∼ <em>P</em><sub>X×Y</sub>, and the Bayes decision function <em>f</em><sup>∗ </sup>: X → A is a function that achieves the <em>minimal risk </em>among all possible functions:

<em>R</em>(<em>f</em><sup>∗</sup>) = inf <em>R</em>(<em>f</em>)<em>.</em>

<em>f</em>

Here we consider the regression setting, in which A = Y = <strong>R</strong>. We will show for the square loss <em>`</em>(<em>a,y</em>) = (<em>a </em>− <em>y</em>)<sup>2</sup>, the Bayes decision function is <em>f</em><sup>∗</sup>(<em>x</em>) = E[<em>y </em>| <em>x</em>], where the expectation is over <em>y</em>. As before, we assume we know the data-generating distribution <em>P</em><sub>X×Y</sub>.

<ul>

 <li>We’ll approach this problem by finding the optimal action for any given <em>x</em>. If somebody tells us <em>x</em>, we know that the corresponding <em>y </em>is coming from the conditional distribution <em>y </em>| <em>x</em>. For a particular <em>x</em>, what value should we predict (i.e. what action <em>a </em>should we produce) that has minimal expected loss? Express your answer as a decision function <em>f</em>(<em>x</em>), which gives the best action for any given <em>x</em>. In mathematical notation, we’re looking for <em>f</em><sup>∗</sup>(<em>x</em>) = argmin<em><sub>a </sub></em>E<sup>h</sup>(<em>a </em>− <em>y</em>)<sup>2 </sup>| <em>x</em><sup>i</sup>, where the expectation is with respect to <em>y</em>. (Hint: There is really nothing to do here except write down the answer, based on the previous question. But make sure you understand what’s happening…)</li>

 <li>In the previous problem we produced a decision function <em>f</em><sup>∗</sup>(<em>x</em>) that minimized the risk for each <em>x</em>. In other words, for any other decision function <em>f</em>(<em>x</em>), <em>f</em><sup>∗</sup>(<em>x</em>) is going to be at least as good as <em>f</em>(<em>x</em>), for every single <em>x</em>. In math, we mean</li>

</ul>

E<sup>h</sup>(<em>f</em><sup>∗</sup>(<em>x</em>) − <em>y</em>)<sup>2 </sup>| <em>x</em><sup>i </sup>≤ E<sup>h</sup>(<em>f</em>(<em>x</em>) − <em>y</em>)<sup>2 </sup>| <em>x</em><sup>i</sup><em>,</em>

for all <em>x</em>. To show that <em>f</em><sup>∗</sup>(<em>x</em>) is the Bayes decision function, we need to show that

E<sup>h</sup>(<em>f</em><sup>∗</sup>(<em>x</em>) − <em>y</em>)<sup>2</sup><sup>i </sup>≤ E<sup>h</sup>(<em>f</em>(<em>x</em>) − <em>y</em>)<sup>2</sup><sup>i</sup>

for any <em>f</em>. Explain why this is true. (Hint: Law of iterated expectations.)

<h2>3.2           [Optional] Median Loss</h2>

<ol>

 <li>(Optional) Show that for the absolute loss <em>`</em>(<em>y,y</em>ˆ ) = |<em>y </em>− <em>y</em>ˆ|, <em>f</em><sup>∗</sup>(<em>x</em>) is a Bayes decision function if <em>f</em><sup>∗</sup>(<em>x</em>) is the median of the conditional distribution of <em>y </em>given <em>x</em>. [Hint: As in the previous section, consider one <em>x </em>at time. It may help to use the following characterization of a median: <em>m </em>is a median of the distribution for random variable and</li>

</ol>

Note: This loss function leads to “median regression”. There are other loss functions that lead to “quantile regression” for any chosen quantile. (For partial credit, you may assume that the distribution of <em>y </em>| <em>x </em>is discrete or continuous. For full credit, no assumptions about the distribution.)

<a href="#_ftnref1" name="_ftn1">[1]</a> Throughout this assignment we refer to the “test” set. It may be more appropriate to call this set the “validation” set, as it will be a set of data on which we compare the performance of multiple models. Typically a test set is only used once, to assess the performance of the model that performed best on the validation set.

<a href="#_ftnref2" name="_ftn2">[2]</a> Of course, it is also given by the more standard definition of directional derivative, .

The form given gives a better approximation to the derivative when we are using small (but not infinitesimally small)

Sometimes people say “batch gradient descent” or “full batch gradient descent” to mean gradient descent, defined as we discussed in class. They do this to distinguish it from stochastic gradient descent and minibatch gradient descent, which they probably use as their default.

<a href="#_ftnref3" name="_ftn3">[3]</a> For the mathematically inclined, there is a theorem that if the objective function is convex and differentiable, and the gradient of the objective is Lipschitz continuous with constant <em>L &gt; </em>0, then gradient descent converges for fixed steps of size 1<em>/L </em>or smaller. See <a href="https://www.cs.cmu.edu/~ggordon/10725-F12/scribes/10725_Lecture5.pdf">https://www.cs.cmu.edu/~ggordon/10725-F12/scribes/10725_ </a><a href="https://www.cs.cmu.edu/~ggordon/10725-F12/scribes/10725_Lecture5.pdf">Lecture5.pdf</a><a href="https://www.cs.cmu.edu/~ggordon/10725-F12/scribes/10725_Lecture5.pdf">,</a> Theorem 5.1.