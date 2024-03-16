import numpy as np

'''
min z = 12 - 3x1 -2x2 <=> max (-z) = - 12 + 3x1 + 2x2
3x1+x2 >= 5 <=> 3x1 + x2 - x3 = 5 where x3 is a surplus variable >= 0
(<= creates slack variables and >= creates surplus variables)
Phase 1 : 3x1 +x2 -x3 + w1 = 5 where w1 is an artificial variable

'''


class TwoPhaseSimplex:
    def __init__(self,z,equations) -> None:
        #number of equations will determine number of surplus variables
        number_eq = equations.shape[0]
        
        number_vars = z.shape[0]
        
        eq_lhs = equations[:,:-1]
        surplus_vars = np.identity(number_eq) * -1
        eq_rhs = np.reshape(equations[:,-1], (number_eq, 1))
        
        tableau_upper = np.hstack([eq_lhs, surplus_vars,eq_rhs])
        
        z_part = np.hstack([z,np.zeros(number_eq + 1)])
        
        #x1,x2 - number eq, x3,x4 - vars, 1 for rhs
        w = np.zeros(number_eq + number_vars + 1)
        for row in tableau_upper:
            w+=row
        w = w * -1
        
        self.tableau = np.vstack([tableau_upper, z_part, w])
        #where the z row is going to be in the tableau  
        self.z_index = number_eq
        
        self.xs = [f'x{i+1}' for i in range(number_eq + number_vars)]
        self.column_x = self.xs.copy()
        #artificial variables 
        self.row_vals = [f'w{i+1}' for i in range(number_eq)]
       
         
         
       
    def phase1(self):
        while (self.tableau[-1,:-1] < 0).any():
            pivot_col = np.argmin(self.tableau[-1,:-1])
            lowest_ration = np.inf
            pivot_row = 0
            for row_index, value in enumerate(self.tableau[:,pivot_col]):
                if value <= 0 or row_index == self.z_index:
                    continue
                else:
                    ratio = self.tableau[row_index, -1] / value
                    if ratio < lowest_ration:
                        lowest_ration = ratio
                        pivot_row = row_index
                        
            pivot_element = self.tableau[pivot_row, pivot_col]
            #print(f'pivot element: {pivot_element} and pivot row: {pivot_row} and pivot col: {pivot_col}')
         

            self.column_x[pivot_col], self.row_vals[pivot_row] = (
                self.row_vals[pivot_row],
                self.column_x[pivot_col],
            )
            
            
            updated_tableau = self.tableau.copy()
        
            for i, row in enumerate(self.tableau):
                    for j, value in enumerate(row):
                        if i == pivot_row and j == pivot_col:
                            updated_tableau[i, j] = 1 / value
                        elif j == pivot_col:
                            updated_tableau[i, j] = -value / pivot_element
                        elif i == pivot_row:
                            updated_tableau[i, j] = value / pivot_element
                        else:
                            updated_tableau[i, j] = (
                                value
                                - self.tableau[i, pivot_col]
                                * self.tableau[pivot_row, j]
                                / pivot_element
                            )

            self.tableau = updated_tableau
            print(self.tableau)
        
    def phase2(self):
        #remove the w columns 
        w_index = np.where(self.tableau[-1] == 1)[0]
        phase2_tableau = np.delete(self.tableau, w_index, 1)
        phase2_tableau = phase2_tableau[:-1]
        #Do the simplex method on this tableau
        
        #remove ws
        self.column_x = [x for x in self.column_x if not x.startswith('w')]
        
        while (phase2_tableau[-1,:-1] < 0).any():
            pivot_col = np.argmin(phase2_tableau[-1,:-1])
            lowest_ration = np.inf
            pivot_row = 0
            for row_index, value in enumerate(phase2_tableau[:,pivot_col]):
                if value <= 0:
                    continue
                else:
                    ratio = phase2_tableau[row_index, -1] / value
                    if ratio < lowest_ration:
                        lowest_ration = ratio
                        pivot_row = row_index
                        
            pivot_element = phase2_tableau[pivot_row, pivot_col]
            #print(f'pivot element: {pivot_element} and pivot row: {pivot_row} and pivot col: {pivot_col}')
         
         
            self.column_x[pivot_col], self.row_vals[pivot_row] = (
                self.row_vals[pivot_row],
                self.column_x[pivot_col],
            )
            updated_tableau = phase2_tableau.copy()
        
            for i, row in enumerate(phase2_tableau):
                    for j, value in enumerate(row):
                        if i == pivot_row and j == pivot_col:
                            updated_tableau[i, j] = 1 / value
                        elif j == pivot_col:
                            updated_tableau[i, j] = -value / pivot_element
                        elif i == pivot_row:
                            updated_tableau[i, j] = value / pivot_element
                        else:
                            updated_tableau[i, j] = (
                                value
                                - phase2_tableau[i, pivot_col]
                                * phase2_tableau[pivot_row, j]
                                / pivot_element
                            )
            phase2_tableau = updated_tableau
          
        z = phase2_tableau[-1, -1]
        x_val = []
        for x in self.xs:
            if x in self.column_x:
                x_val.append(0)
            else:
                index = self.row_vals.index(x)
                x_val.append(phase2_tableau[index, -1])
         
        sol_output = {x: round(x_value, 9) for x, x_value in zip(self.xs, x_val)}
        return f"Max z = {round(z*-1,3)} and {sol_output} "
                            
    def solve(self):
        self.phase1()
        return self.phase2()

#Input 
'''
min z = 2x1 + x2
subject to
3x1 + x2 >= 5
x1 + 2x2 >= 3
x1,x2 >= 0
'''

z = np.array([2, 1])
equations = np.array([[3, 1,5], [1, 2,3]])
    
s1 = TwoPhaseSimplex(z,equations)
print(s1.solve())
    
    
    
    