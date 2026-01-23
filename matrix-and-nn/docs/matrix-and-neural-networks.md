# Matrix and Neural Networks

---

## Part 1: Matrix Fundamentals

### a) How do you represent an $a_{ij}$ matrix?

An $a_{ij}$ matrix is represented as a matrix with $i$ rows and $j$ columns. The indices denote the position of each element:

$$
\begin{bmatrix}
a_{1,1} & a_{1,2} \\
a_{2,1} & a_{2,2} \\
a_{3,1} & a_{3,2} 
\end{bmatrix}
$$

---

### b) Represent a row vector and a column vector

**Row vector** — elements arranged horizontally:

$$\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$$

**Column vector** — elements arranged vertically:

$$\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$

---

### c) Matrix Operations

Given:
$$A = \begin{pmatrix} 1 & -2 & 3 \\ 4 & 5 & -6 \end{pmatrix} \quad \text{and} \quad B = \begin{pmatrix} 1 & -2 & 3 \\ 4 & 5 & -6 \end{pmatrix}$$

#### $A + B$
$$\begin{pmatrix} 2 & -4 & 6 \\ 8 & 10 & -12 \end{pmatrix}$$

#### $3A$
$$\begin{pmatrix} 3 & -6 & 9 \\ 12 & 15 & -18 \end{pmatrix}$$

#### $2A - 3B$
$$\begin{pmatrix} -1 & 2 & -3 \\ -4 & -5 & 6 \end{pmatrix}$$

#### $AB$
> [!warning] Undefined Operation
> Matrix multiplication cannot be performed — the dimensions do not match. For $AB$ to be valid, the number of columns in $A$ must equal the number of rows in $B$.

#### $A^T$ (Transpose)
$$\begin{pmatrix} 1 & 4 \\ -2 & 5 \\ 3 & -6 \end{pmatrix}$$

#### $AI$ (Identity)
> [!note] Identity Property
> Any matrix multiplied by the identity matrix equals itself.

$$AI = \begin{pmatrix} 1 & -2 & 3 \\ 4 & 5 & -6 \end{pmatrix}$$

---

## Part 2: Artificial Neural Networks Presentation

> [!info] Presentation Link
> [View the Presentation on Google Slides](https://docs.google.com/presentation/d/13tsobpSMuF19WWWzfXjexNoUIdMoUP3uakOSJY4WdJw/edit?slide=id.p#slide=id.p)

