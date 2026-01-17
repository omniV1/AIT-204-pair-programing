### Owen Lindsey & Tyler Friesen
### Professor Artzi
### Jan 21 2027
### AIT-204 
### Linear Regression

---

###  Part 1 Formula Application for Differentiation

**a. Interpret dy/dx geometrically:** 

![[Pasted image 20260117141313.png]]

    dy/dx represents the slope of a tangent line to a curve at a specific point. 

    in Linear Regression dy/dx shows how steep the loss function is at a particular point. when dy/dx is positive, the loss is trending upward or uphill. When negative the loss is decreasing or going downhill. If dy/dx is zero, we are at a flat point, potentially the minimum. 

    The sign tells me which direction to move the weights, either left or right on the curve. The magnitude tells me how steep the slope is, larger values result in steeper slopes. 


![[dx_dy.png]]
source(https://xaktly.com/TheDerivative.html) Figure 9

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
	


**c. Differentiate:** 

y = 4 + 2x - 3x^2 - 5x^3 - 8x^4 + 9x^5

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
	
	dy/dx = 2 - 6x -15x^2 - 32x3 + 45x^4


y = 1/x+3/x^2 + 2x^3 = x^(-1) + 3x^(-2) + 2x^(-3)

1. Rewrite using negative exponents
	
	 This allows us to use the power rule 
	
1. Apply the power rule 
	
		 d/dx[x^n] = nx^n-1 
	 



y = ∛(3x^2)-1/√5x


**d. Define partial derivative**



**e. Given the following functions find ∂z/∂x and ∂z/∂y**


