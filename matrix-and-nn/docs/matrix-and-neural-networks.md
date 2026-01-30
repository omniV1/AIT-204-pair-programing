# Matrix and Neural Networks

### Owen Lindsey & Tyler Friesen
### Professor Artzi
### Jan 21 2026
### AIT-204 
### Streamlit application link: https://mlp-owen.streamlit.app/

<div style="page-break-after: always;"></div>

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

<div style="page-break-after: always;"></div>

## Part 2: Artificial Neural Networks Presentation

> [!info] Presentation Link
> [View the Application in Streamlit](https://mlp-owen.streamlit.app/)

<div style="page-break-after: always;"></div>

## Part 3: Ethical Considerations

**Potential biases in training data impacting fairness and accuracy**

Medical science has historically underrepresented women's health compared to men's. This is a well documented disparity, diseases often present differently in females than in males, yet diagnostic criteria have largely been developed from male-centric studies. Furthermore, as society evolves and humans augment their physiology through technologies like Neuralink, entirely new categories of bias may emerge.

The data we rely on is already skewed. When we analyze patterns, we often miss or omit variables that fall outside the norm. These outliers become invisible, and the systems we build fail to accommodate them. 

---

**Safeguarding the privacy and confidentiality of data used in training neural networks.**

Privacy has long been considered an important part of American society, yet this foundation has steadily eroded. Today, mass data collection by both government agencies and private corporations is an open secret. Surveillance infrastructure like Flock Safety cameras, deployed in cities such as Phoenix and backed by defense contractors like Palantir, monitors citizens with little public oversight. Consent, in any meaningful sense, has become an afterthought.

Given this precedent, there is little reason to trust that AI companies will handle personal data with greater care. The training of neural networks requires vast quantities of data often scraped, purchased, or inferred without explicit user consent. If existing institutions have already normalized surveillance at scale, the burden falls on AI developers to prove they can do better. Until robust, enforceable privacy protections exist, skepticism is warranted.

---

**The need for transparency and interpretability in neural network decisions.**

Even if training data and model architectures were fully disclosed, the technical complexity involved would limit meaningful public engagement. Neural networks operate through millions of weighted connections and nonlinear transformations—concepts that require significant technical literacy to interpret.

This suggests that interpretability efforts may be most valuable not as tools for mass understanding, but as mechanisms for accountability—enabling regulators, auditors, and domain experts to evaluate AI systems on the public's behalf. Transparency, in this framing, serves less as a democratic ideal and more as an institutional safeguard.


---

**The broader societal implications and responsibilities when deploying neural networks.**

Ethical AI development ultimately depends on individual accountability. Developers who commit to using only ethically sourced datasets—avoiding pirated databases or data obtained without consent can control their own practices. Consideration of environmental costs, whether through carbon offsets or more efficient training methods, represents another dimension of responsible development.

However, individual ethics cannot govern an industry driven by growth, shareholder value, and competitive pressure. Large corporations operate under different incentives, and voluntary adoption of ethical constraints remains unlikely without external enforcement.

The societal responsibility, then, is twofold: individuals must hold themselves accountable, and collectively, the public must demand regulation that holds corporations to the same standard. The question is not whether ethical AI development is possible—it is. The question is whether it will become the norm or remain the exception.

---

**Addressing the potential consequences of error and inaccuracies in neural network predictions**

Return to the medical framework. Imagine a neural network delivers a terminal diagnosis, and it is wrong.

Public trust in medical institutions is already strained. Into this environment, we introduce AI systems making life or death predictions systems whose errors carry irreversible consequences. If a patient is told they have six months and dies in three, families will demand accountability. If told they have a week and they survive, they may have already quit their job, drained savings, or made decisions that cannot be undone. The direction of the error does not eliminate its harm.

<div style="page-break-after: always;"></div>

## Part 4: Understanding our ANN output

### Interpretation of MLP Output for Optimal Basketball Team Selection

The multilayer perceptron (MLP) trained on the NBA player dataset has successfully identified the top 5 players most suitable for an optimal team based on their physical characteristics and draft information. The predicted optimal team includes:

- Eddie Jones (LAL), 25 years old, 198.12 cm, 86.18 kg  
- Bryon Russell (UTA), 29 years old, 200.66 cm, 102.06 kg  
- David Wingate (SEA), 34 years old, 195.58 cm, 83.91 kg  
- Tim Thomas (PHI), 21 years old, 208.28 cm, 104.33 kg  
- Michael Redd (MIL), 21 years old, 198.12 cm, 97.07 kg  

These players were ranked by the MLP according to the probability of being labeled as "optimal," which reflects a combination of normalized height, weight, age relative to a prime performance age of 27, and draft round.

The MLP effectively learned patterns from the weakly supervised labels derived from the suitability scoring formula:

$$
\text{suitability\_score} = 0.35 \times \text{height\_norm} + 0.25 \times \text{weight\_norm} + 0.25 \times \text{age\_prime\_score} + 0.15 \times \text{draft\_score}
$$

This formula evaluates each player's overall suitability by weighting their physical attributes and draft information. The MLP learned these patterns without explicitly recalculating the formula, allowing for a nonlinear and flexible assessment of player suitability. 

The selected team demonstrates a balance of physical attributes and experience: younger players such as Tim Thomas and Michael Redd provide height and athleticism, while older players like David Wingate contribute experience. 

Overall, the output illustrates the ability of the MLP to prioritize players who meet multi-dimensional criteria and to generate a well-rounded team from a pool of 100 candidates. This demonstrates the practical application of artificial neural networks in decision-making tasks such as team selection, with potential for further enhancement by including additional performance statistics or multiple hidden layers for more complex feature interactions.

<div style="page-break-after: always;"></div>

## References

1. Criado Perez, C. (2019). *Invisible Women: Data Bias in a World Designed for Men*. Abrams Press.

2. Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447–453. https://doi.org/10.1126/science.aax2342

3. Electronic Frontier Foundation. (2023). *The Surveillance State: How the NSA and FBI Spy on Americans*. EFF. https://www.eff.org/nsa-spying

4. Zuboff, S. (2019). *The Age of Surveillance Capitalism: The Fight for a Human Future at the New Frontier of Power*. PublicAffairs.

5. Pew Research Center. (2023). *Public Trust in Government: 1958-2023*. https://www.pewresearch.org/politics/public-trust-in-government-1958-2023/

6. Strubell, E., Ganesh, A., & McCallum, A. (2019). Energy and Policy Considerations for Deep Learning in NLP. *Proceedings of the 57th Annual Meeting of the ACL*, 3645–3650. https://doi.org/10.18653/v1/P19-1355

7. Pew Research Center. (2022). *Americans' Trust in Scientists, Other Groups Declines*. https://www.pewresearch.org/science/2022/02/15/americans-trust-in-scientists-other-groups-declines/
