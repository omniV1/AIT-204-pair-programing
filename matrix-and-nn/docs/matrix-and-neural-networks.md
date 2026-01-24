# Matrix and neural networks

## Part 1
a. How do you represent an a<sub>i</sub><sub>j</sub> matrix?<br>
An a<sub>i</sub><sub>j</sub> matrix is represented as a matrix with i rows and j columns. You can see it indexed as such:
$$
\begin{bmatrix}
a_{11} & a_{12} \\\\
a_{21} & a_{22} \\\\
a_{31} & a_{32} 
\end{bmatrix}
$$
<br>
b. Represent a row vector and a column vector<br>
A row vector goes horizontally as such:

$[1 \quad 2 \quad 3]$

A column vector goes vertically as such:

$\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$

<br>

c. Let A = $\begin{pmatrix} 1 & -2 & 3 \\ 4 & 5 & -6 \end{pmatrix}$ and B = $\begin{pmatrix} 1 & -2 & 3 \\ 4 & 5 & -6 \end{pmatrix}$ find:<br>
- A + B<br>
$\begin{pmatrix} 2 & -4 & 6 \\ 8 & 10 & -12 \end{pmatrix}$
- 3A<br>
$\begin{pmatrix} 3 & -6 & 9 \\ 12 & 15 & -18 \end{pmatrix}$
- 2A - 3B<br>
$\begin{pmatrix} -1 & 2 & -3 \\ -4 & -5 & 6 \end{pmatrix}$
- AB<br>
You cannot perform matrix multiplication on these as the dimensions do not match properly
- A<sup>T</sup><br>
$\begin{pmatrix} 1 & 4 \\ -2 & 5 \\ 3 & -6 \end{pmatrix}$
- AI<br>
Any matrix multiplied by the identity matrix is itself.<br>
$\begin{pmatrix} 1 & -2 & 3 \\ 4 & 5 & -6 \end{pmatrix}$
