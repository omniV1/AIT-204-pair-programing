<div style="border: 2px solid #333; padding: 20px 30px; margin-bottom: 30px; font-family: 'Georgia', serif;">
  <div style="text-align: center; margin-bottom: 15px;">
    <h2 style="margin: 0; font-size: 1.6em; letter-spacing: 1px;">Grand Canyon University</h2>
  <hr style="border: none; border-top: 1px solid #999; margin: 12px 0;">
  <table style="width: 100%; border-collapse: collapse; font-size: 0.95em;">
    <tr>
      <td style="padding: 4px 0;"><strong>Course:</strong> AIT-204</td>
      <td style="padding: 4px 0; text-align: right;"><strong>Instructor:</strong> Professor Artzi</td>
    </tr>
    <tr>
      <td style="padding: 4px 0;"><strong>Authors:</strong> Owen Lindsey &amp; Tyler Friesen</td>
      <td style="padding: 4px 0; text-align: right;"><strong>Date:</strong> enterdatehere</td>
    </tr>
  </table>
  <hr style="border: none; border-top: 1px solid #999; margin: 12px 0;">
  <h1 style="text-align: center; margin: 10px 0 0 0; font-size: 1.4em;">CNN Convolution &amp; Max Pooling Worksheet</h1>
</div>

---

## Problem A: Constructing the Convolution Output Matrix M

**Given:**

$$
A = \begin{bmatrix} a_{1,1} & a_{1,2} & a_{1,3} \\ a_{2,1} & a_{2,2} & a_{2,3} \\ a_{3,1} & a_{3,2} & a_{3,3} \end{bmatrix} \quad K = \begin{bmatrix} k_{1,1} & k_{1,2} \\ k_{2,1} & k_{2,2} \end{bmatrix}
$$

### Step 1: Determine the output dimensions

- Input matrix $A$ is $3 \times 3$
- Kernel $K$ is $2 \times 2$
- Output dimension formula (no padding, stride = 1): $(n - f + 1) \times (n - f + 1)$
- Output $M$ is $2 \times 2$

### Step 2: Identify the receptive fields

For each position in the output, identify which elements of $A$ the kernel overlaps:

| Output Position | Receptive Field (submatrix of A) |
|---|---|
| $M_{1,1}$ | $\begin{bmatrix} a_{1,1} & a_{1,2} \\ a_{2,1} & a_{2,2} \end{bmatrix}$ |
| $M_{1,2}$ | $\begin{bmatrix} a_{1,2} & a_{1,3} \\ a_{2,2} & a_{2,3} \end{bmatrix}$ |
| $M_{2,1}$ | $\begin{bmatrix} a_{2,1} & a_{2,2} \\ a_{3,1} & a_{3,2} \end{bmatrix}$ |
| $M_{2,2}$ | $\begin{bmatrix} a_{2,2} & a_{2,3} \\ a_{3,2} & a_{3,3} \end{bmatrix}$ |

### Step 3: Compute each output element (element-wise multiply then sum)

The convolution operation at each position: $M_{i,j} = \sum (\text{receptive field} \odot K)$

$$M_{1,1} = (a_{1,1} \cdot k_{1,1}) + (a_{1,2} \cdot k_{1,2}) + (a_{2,1} \cdot k_{2,1}) + (a_{2,2} \cdot k_{2,2})$$

$$M_{1,2} = (a_{1,2} \cdot k_{1,1}) + (a_{1,3} \cdot k_{1,2}) + (a_{2,2} \cdot k_{2,1}) + (a_{2,3} \cdot k_{2,2})$$

$$M_{2,1} = (a_{2,1} \cdot k_{1,1}) + (a_{2,2} \cdot k_{1,2}) + (a_{3,1} \cdot k_{2,1}) + (a_{3,2} \cdot k_{2,2})$$

$$M_{2,2} = (a_{2,2} \cdot k_{1,1}) + (a_{2,3} \cdot k_{1,2}) + (a_{3,2} \cdot k_{2,1}) + (a_{3,3} \cdot k_{2,2})$$

### Step 4: Write the final output matrix

$$M = \begin{bmatrix} M_{1,1} & M_{1,2} \\ M_{2,1} & M_{2,2} \end{bmatrix}$$

---

## Problem B: Convolution with Specific Values + Max Pooling

**Given:**

$$
A = \begin{bmatrix} 14 & 15 & 16 \\ 17 & 18 & 19 \\ 20 & 21 & 22 \end{bmatrix} \quad K = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

### Step 1: Determine the output dimensions (same logic as Problem A)

Output $M$ is $2 \times 2$

### Comparison: Input A vs Kernel K

$$
A = \begin{bmatrix} 14 & 15 & 16 \\ 17 & 18 & 19 \\ 20 & 21 & 22 \end{bmatrix} \qquad K = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

The kernel $K$ is smaller than $A$. It slides across $A$ in a $2 \times 2$ window, stopping at four positions. At each position, the overlapping elements are multiplied element-wise with $K$ and summed into a single output value.

### Step 2: Compute each element of the convolution output

**$M_{1,1}$:** Kernel overlaps top-left $2 \times 2$ region of $A$

$$M_{1,1} = (14 \cdot 1) + (15 \cdot 2) + (17 \cdot 3) + (18 \cdot 4) = 14 + 30 + 51 + 72 = 167$$

**$M_{1,2}$:** Kernel slides one step right

$$M_{1,2} = (15 \cdot 1) + (16 \cdot 2) + (18 \cdot 3) + (19 \cdot 4) = 15 + 32 + 54 + 76 = 177$$

**$M_{2,1}$:** Kernel slides one step down from top-left

$$M_{2,1} = (17 \cdot 1) + (18 \cdot 2) + (20 \cdot 3) + (21 \cdot 4) = 17 + 36 + 60 + 84 = 197$$

**$M_{2,2}$:** Kernel at bottom-right

$$M_{2,2} = (18 \cdot 1) + (19 \cdot 2) + (21 \cdot 3) + (22 \cdot 4) = 18 + 38 + 63 + 88 = 207$$

### Step 3: Write the convolution result

$$M = \begin{bmatrix} 167 & 177 \\ 197 & 207 \end{bmatrix}$$

### Step 4: Apply Max Pooling to find $M_p^T$

> **Recall:** Max pooling selects the **maximum value** from a defined pooling window.

Pooling window covers the entire $2 \times 2$ matrix $M$:

$$M_p = \max(167, 177, 197, 207) = 207$$

### Step 5: Transpose

$$M_p^T = 207$$

> **Note:** For a scalar (single value), the transpose is just the value itself.

---
