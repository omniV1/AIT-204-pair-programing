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

## Part 3: Ethical Considerations

**Potential biases in training data impacting fairness and accuracy**

Medical science has historically underrepresented women's health compared to men's. This is a well documented disparity, diseases often present differently in females than in males, yet diagnostic criteria have largely been developed from male-centric studies. Furthermore, as society evolves and humans augment their physiology through technologies like Neuralink, entirely new categories of bias may emerge.

The data we rely on is already skewed. When we analyze patterns, we often miss or omit variables that fall outside the norm. These outliers become invisible, and the systems we build fail to accommodate them. 
This is the critical distinction: when we train AI on human data, we are training on datasets that already carry the weight of historical bias.

---

**Safeguarding the privacy and confidentiality of data used in training neural networks.**

Privacy has long been considered a cornerstone of American society—yet this foundation has steadily eroded since the passage of the Patriot Act in 2001. Today, mass data collection by both government agencies and private corporations is an open secret. Surveillance infrastructure like Flock Safety cameras, deployed in cities such as Phoenix and backed by defense contractors like Palantir, monitors citizens with little public oversight. Consent, in any meaningful sense, has become an afterthought.

Given this precedent, there is little reason to trust that AI companies will handle personal data with greater care. The training of neural networks requires vast quantities of data often scraped, purchased, or inferred without explicit user consent. If existing institutions have already normalized surveillance at scale, the burden falls on AI developers to prove they can do better. Until robust, enforceable privacy protections exist.

---

**The need for transparency and interpretability in neural network decisions.**

Consider the self-checkout machine a tiny computer found in nearly every store on the planet. It has large buttons, clear instructions, and a straightforward purpose. And yet, humans operate it with moderate success at best. The older generation, in particular, despises these devices. They "take jobs from real workers." They create confusion. And why should customers pay the same prices while doing the labor themselves?

Frankly, I agree with that assessment.

Now imagine trying to explain neural networks to the average consumer: how they work, what data they ingest, how decisions are made. Would they understand? Would they care? The average American does not even participate in the political processes that directly shape their daily lives. If companies released their training data tomorrow and researchers distilled it into a simple newspaper column, how many people would actually read it?

This raises an uncomfortable question: if transparency requires an engaged and informed public to be meaningful, and that public largely does not exist, then what is the real value of interpretability? The answer may be that transparency is less about public understanding and more about accountability—creating a paper trail for regulators, researchers, and advocates who *will* do the reading.

---

**The broader societal implications and responsibilities when deploying neural networks.**

At the end of the day, I can only control what I do.

As a developer working with neural networks, I believe honesty matters. I will not scrape pirated databases. I will not use data that was obtained through deception, exploitation, or without consent. These are my lines, and I intend to hold them. Where possible, I will seek out ethically compiled datasets and consider the environmental cost as well, whether that means carbon offsets or choosing more efficient training methods.

But I am not naive. I can govern my own practices; I cannot govern an industry. Large corporations operate under different incentives: growth, shareholder value, competitive advantage. Expecting them to voluntarily adopt ethical constraints is optimistic at best. The societal responsibility, then, is twofold: individuals must hold themselves accountable, and collectively, we must demand regulation that holds corporations to the same standard.

The question is not whether ethical AI development is possible, it is. The question is whether it will be the norm or the exception.

---

**Addressing the potential consequences of error and inaccuracies in neural network predictions**

Return to the medical framework. Imagine a neural network delivers a terminal diagnosis, and it is wrong.

Public trust in medicine is already fragile. The COVID-19 pandemic exposed deep fractures: misinformation spread faster than the virus, institutions contradicted themselves, and millions of people walked away more skeptical of medical authority than ever before. Into this landscape, we introduce AI systems that make life or death predictions. The margin for error is thin.

People will not tolerate AI misdiagnosis, regardless of outcome. If a patient is told they have six months to live and dies in three, the family will demand answers. If they are told they have a week and survive, they may have quit their job, drained their savings, or said goodbyes they cannot take back. A "positive" error is still an error with consequences.

Personally, I know how I would react. If an algorithm told me I had a week to live, I would live that week to its fullest, and when I woke up on day eight, a lawsuit would be the least of someone's problems. The question is not just technical accuracy; it is about the irreversible decisions people make based on what they are told. 


## References

1. Criado Perez, C. (2019). *Invisible Women: Data Bias in a World Designed for Men*. Abrams Press.

2. Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447–453. https://doi.org/10.1126/science.aax2342

3. Electronic Frontier Foundation. (2023). *The Surveillance State: How the NSA and FBI Spy on Americans*. EFF. https://www.eff.org/nsa-spying

4. Zuboff, S. (2019). *The Age of Surveillance Capitalism: The Fight for a Human Future at the New Frontier of Power*. PublicAffairs.

5. Pew Research Center. (2023). *Public Trust in Government: 1958-2023*. https://www.pewresearch.org/politics/public-trust-in-government-1958-2023/

6. Strubell, E., Ganesh, A., & McCallum, A. (2019). Energy and Policy Considerations for Deep Learning in NLP. *Proceedings of the 57th Annual Meeting of the ACL*, 3645–3650. https://doi.org/10.18653/v1/P19-1355

7. Pew Research Center. (2022). *Americans' Trust in Scientists, Other Groups Declines*. https://www.pewresearch.org/science/2022/02/15/americans-trust-in-scientists-other-groups-declines/


