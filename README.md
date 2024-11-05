(DISCLAIMER: THIS PACKAGE IS UNDERCONSTRUCTION. Hopefully one day I will find the time to package it properly and add a full documentation)

# split_quaternion
Split Quaternion package, originally based upon the Quaternionic library.

## Quick initiation to split quaternions
Split quaternions are a 4-dimensional associative and non-commutative algebra, where each element $q\in\mathbb{S}$ is written in the form
 > $$\text{q} = q_0 + \underbrace{q_1 i + q_2 j + q_3 k}_{v_q} = s_q + v_q$$.

The rule of the algebra can be resumed by the formula
> $$-i^2 = j^2 = k^2 = (ij)k = 1 $$.

Similarly to complex numbers, the conjugate and modulas of a split quaternion is defined as 
> $$\text{q}^\ast = q_0 - q_1 i - q_2 j - q_3 k = s_q - v_q$$
> $$|\text{q}|^2 = \text{q}\text{q}^\ast = q_0^2 + q_1^2 - q_2^2 - q_3^2 $$

We define the Lorentz pseud-scalar product as 
> $ (x\circ y) = x_1y_1 - x_2y_2-x_3y_3

and the Lorentz cross-product
> $$[x\otimes y] = \left(\begin{array}{c} x_2y_3 - x_3y_2 \\
> -x_3y_1 + x_1y_3 \\
> -x_1 y_2 + x_2y_1
\end{array}\right).$$

The addition and multiplication reads
> $$\text{p}+\text{q} = (s_p + s_q) + (v_p + v_q)$$
> $$\text{pq} = (s_p + v_p)(s_q + v_q) = s_p s_q + s_p v_q + s_q v_p - [v_p\otimes v_q] -  (v_p\circ v_q).$$

## Using the package
This package extends most numpy functionnalities with matrices of split quaternions.

```
import split_quaternions as sq
p1 = sq.ones((10, 10))
p2 = sq.zeros((10, 10))
q = sq.array([[1,2,3,4], [-1, -2, -3, -4]])
q.conj  # equivalent to q.conjugate
q.modulus  # equivalent to q.norm
```
