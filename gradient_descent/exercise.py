import numpy as np
import math

def gradient_descent(math,cs):
    m_curr = b_curr = 0
    iterations = 1000000
    learning_rate = 0.0002
    n = len(math)
    cost_previous = 0

    for i in range(iterations):
        cs_predicted = m_curr * math + b_curr
        cost = (1 / n) * sum([val **2 for val in(cs - cs_predicted)**2])
        md = -(2 / n) * sum(math * (cs - cs_predicted))  
        bd = -(2 / n) * sum(cs - cs_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {},iterations {},cost {}".format(m_curr,b_curr,i,cost))

    cs_new = m_curr * 92 + b_curr
    print(cs_new)
 
math = np.array([92,56,88,70,80,49,65,35,66,67])
cs = np.array([98,68,81,80,83,52,66,30,68,73])

gradient_descent(math,cs)