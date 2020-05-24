

## The pseudoinverse rule for constructing RNNs with specific fixed points 

In general, $x^\mu$ is a fixed point under input $u^\mu$ if:
$$
0 = \frac{dx}{dt} (x^\mu)= -x^\mu +J \phi(x^\mu) + u^\mu,
$$
i.e.,
$$
J \phi(x^\mu) = x^\mu - u^\mu, \\
J a^\mu = b^\mu.
$$

In the last line we made the assignments $a^\mu = \phi(x^\mu)$ and $b^\mu = x^\mu - u^\mu$. By the so-called pseudoinverse learning rule [1], we may make $a^\mu$s and $b^\mu$s into the columns of two matrices $A$ and $B$, and solve the least squares problem
$$
(*) \quad J A = B
$$
to obtain a connectivity matrix $\hat J$ which enforces these fixed points.

## Using the pseudoinverse learning rule to construct RNNs that compute Boolean functions

Suppose a network with $N$ recurrent neurons is to compute the boolean function $y = F(z)$, where $y$ and $z$ are Boolean-valued vectors. Under input
$$
u^\mu = W_{in}^{(1)} z^\mu \in R^N,
$$
we want to construct a fixed point $x^\mu$ such that $y^\mu = W^{out} x^{\mu}$. Assume there is negligible overlap between all of the input and output weights (i.e. they are mutually orthogonal).

How to choose the $x^\mu$s? As stated initially, we want to enforce
$$
(**)\quad  W_{out} B = Y,
$$
where the columns of $Y$ are the $y^\mu$s. So, $B$ itself is the solution to a squares problem, of which one solution is

$$
x^\mu = W_{out}^+ y^\mu + u^\mu.
$$

Then the optimal value $\hat J$ is the least-norm solution to the equation $(**)$, which can be given closed form via the pseudoinverse. Note that the rank of $\hat J$ is therefore upper bounded by both the number of readout vectors and the number of fixed points, as well as the rank of the matrix of output values $Y$.

Also note that any $J$ constructed to satisfy $(*)$ and $(**)$ will work. More succinctly, $J$ must satisfy
$$
W_{out} J A = Y 
$$

To recap, the recipe we use here is:

1) Given $W_{out}$, $W_{in}$ and the input-output pairs $(z^\mu ,y^\mu )$, let $u^\mu = W_{in} z^\mu$ and solve $(**)$ for $b^\mu$ and let $x^\mu = b^\mu + u^\mu$. 

2) Let $a^\mu = \phi(x^\mu)$ and solve $(*)$ for $J$ via the pseudoinverse rule.



### References

[1] Personnaz, L., Guyon, I., & Dreyfus, G. (1986). Collective computational properties of neural networks: New learning mechanisms. _Physical Review A_, 34(5), 4217.