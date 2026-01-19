### Owen Lindsey & Tyler Friesen
### Professor Artzi
### Jan 21 2027
### AIT-204 
### Linear Regression

---

###  Part 1 Formula Application for Differentiation

**a. Interpret dy/dx geometrically:** 


dy/dx represents the slope of a tangent line to a curve at a specific point. 

In Linear Regression dy/dx shows how steep the loss function is at a particular point. when dy/dx is positive, the loss is trending upward or uphill. When negative the loss is decreasing or going downhill. If dy/dx is zero, we are at a flat point, potentially the minimum. 

The sign tells me which direction to move the weights, either left or right on the curve. The magnitude tells me how steep the slope is, larger values result in steeper slopes. 


![[dx_dy.png]]
source(https://xaktly.com/TheDerivative.html) Figure 9

---

**b. How many differentiation formulas do we have and what are they?** 

1. **Power Rule:**

	d/dx(x^n) = nx^(n-1)
	
		*(Bring down the exponent, reduce power by 1)*

2. **Constant:**

	 d/dx[c] = 0
	
		 *(Derivative of any constant is zero)*

3. **Constant Multiple:**

	 d/dx[cf(x)] = (c)df/dx
	
		 *(Pull the constant out front)*

4. **Sum Rule:**

	 d/dx[f + g] = f'(x) + g'(x)
	
		 *(Differentiate each term separately)*

5. **Product Rule:**

	d/dx[fg] = (f)dg/dx + (g)df/dx
	
		*(First times derivative of second, plus second times derivative of first)*

6. **Quotient Rule:**

	d/dx[f/g] = (g·df/dx - f·dg/dx) / g²
	
		*(Bottom times derivative of top, minus top times derivative of bottom, all over bottom squared)*

7. **Chain Rule:**

	 d/dx[f(g(x))] = f'(g(x)) * g'(x)
	
		 *(Derivative of outer function times derivative of inner function)*

8. **Exponential:**

	 d/dx[e^x] = e^x
	
		 *(e^x is its own derivative)*  
	

---
**c. Differentiate:** 

**Problem 1:**

**y = 4 + 2x - 3x^2 - 5x^3 - 8x^4 + 9x^5**

1. Apply the sum rule and differentiate each term separately

2. Constant term 

		d/dx[4] = 0

3. Linear term 

		d/dx[2x] = 2
	
4. Apply power rule to remaining terms
	
		d/dx[-3x^2] = -6x
		d/dx[-5x^3] = -15x^2
		d/dx[-8x^4] = -32x^3
		d/dx[9x^5] = 45x^4
	
5. Combine all terms

**Answer:**dy/dx = 2 - 6x -15x^2 - 32x3 + 45x^4

---
**Problem 2:**

**y = 1/x+3/x^2 + 2x^3 = x^(-1) + 3x^(-2) + 2x^(-3)**

1. Rewrite using negative exponents
	
		 This allows us to use the power rule 
	
2. Apply the power rule 
	
		 d/dx[x^n] = nx^n-1 
	
	 d/dx[x^-1] = -1 * x^-2 = -x^-2
	 
	 d/dx[3x^-2] = 3(-2)x^-3 = -6x^-3 
	 
	 d/dx[2x^-3] = 2(-3)x^-4 = -6x^-4
	  
3. Combine terms
	
	dy/dx = -x^-2 - 6x^-3 - 6x^-4 
	
4. Convert back to fractions (Optional but it looks nice)

**Answer:**dy/dx = -1/x^2 - 6/x^3 - 6/x^4 
 

---
**Problem 3:**

**y = ∛(3x^2)-1/√5x**

1. Rewrite with exponents:
	
     
	y = (3x²)^(1/3) - 1/(5x)^(1/2)
	
	y = 3^(1/3) · x^(2/3) - 1/(5^(1/2) · x^(1/2))
	
	y = 3^(1/3) · x^(2/3) - 5^(-1/2) · x^(-1/2)

3. Simplify:
   
	 y = 3^(1/3) · x^(2/3) - (1/√5) · x^(-1/2)

4. Apply power rule:
   
	dy/dx = 3^(1/3) · (2/3)x^(-1/3) - (1/√5) · (-1/2)x^(-3/2)
	
	dy/dx = (2/3) · 3^(1/3) · x^(-1/3) + (1/2√5) · x^(-3/2)

6. Combine and simplify:

**Answer:** dy/dx = (2 · ∛3)/(3x^(1/3)) + 1/(2√5 · x^(3/2))



---

**d. Define partial derivative**

A partial derivative measures the rate of change of a multivariable function 
with respect to one variable while holding all other variables constant. 

In linear regression, partial derivatives allow us to find how the loss 
function changes with respect to each individual parameter (like slope or 
intercept) independently, which tells us how to adjust each parameter to 
minimize the loss.

---

**e. Given the following functions find ∂z/∂x and ∂z/∂y**

**Problem 1:**

z = 2x² - 3xy + 4y²

1. Find ∂z/∂x: Treat y as a constant

	∂z/∂x = ∂/∂x[2x² - 3xy + 4y²]

	∂z/∂x = 4x - 3y + 0

	∂z/∂x = 4x - 3y

2. Find ∂z/∂y: Treat x as a constant

	∂z/∂y = ∂/∂y[2x² - 3xy + 4y²]

	∂z/∂y = 0 - 3x + 8y

	∂z/∂y = -3x + 8y

**Answer:** ∂z/∂x = 4x - 3y  and  ∂z/∂y = -3x + 8y

---

**Problem 2:**

z = x²/y + y²/x

1. Rewrite using negative exponents

	z = x²y⁻¹ + y²x⁻¹

	This allows us to use the power rule

2. Find ∂z/∂x: Treat y as a constant

	∂z/∂x = 2xy⁻¹ + y²(-1)x⁻²

	∂z/∂x = 2x/y - y²/x²

3. Find ∂z/∂y: Treat x as a constant

	∂z/∂y = x²(-1)y⁻² + 2yx⁻¹

	∂z/∂y = -x²/y² + 2y/x

**Answer:** ∂z/∂x = 2x/y - y²/x²  and  ∂z/∂y = -x²/y² + 2y/x

---

**Problem 3:**

z = e^(x² + xy)

1. Use chain rule for ∂z/∂x

	Let u = x² + xy, then z = eᵘ

	∂z/∂x = eᵘ · ∂u/∂x

	∂u/∂x = 2x + y

	∂z/∂x = e^(x²+xy) · (2x + y)

2. Use chain rule for ∂z/∂y

	Let u = x² + xy, then z = eᵘ

	∂z/∂y = eᵘ · ∂u/∂y

	∂u/∂y = x

	∂z/∂y = e^(x²+xy) · x

**Answer:** ∂z/∂x = (2x + y)e^(x²+xy)  and  ∂z/∂y = xe^(x²+xy)
