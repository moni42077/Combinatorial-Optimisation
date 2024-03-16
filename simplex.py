import numpy as np

'''
##Simplex problem e.g.
max z = 10x1 + 12x2
2x1 + 3x2 <= 1500
3x1 + 2x2 <= 1500
x1 + x2 <= 550
x1,x2 >= 0

where x1 + x2 <= 550 <=> x1 + x2 + x3 = 550 where x3 is a slack variable >= 0
'''


class Simplex:
    def __init__(self, equations, max_z) -> None:
        """
        A function to initialize the simplex tableau with given equations and maximum z value.
        Parameters:
            equations: numpy array representing the equations
            max_z: integer, maximum z value
        Return type: None
        """
        self.tableau = np.vstack([equations, np.append(max_z * -1, 0)])
        # the xs which we will be interested in e.g. x1, x2
        self.xs = [f"x{i+1}" for i in range(len(max_z))]
        # at the start x1 x2 are in the column 1 and 2 and we will keep track at where they are in the table
        self.column_x = self.xs.copy()

        # Instead of having x3: 0, x4: 1, x5: 2 we can have [x3, x4, x5]
        # tracks slack variables as x3,x4 and x5
        self.row_x = [
            f"x{i+1}" for i in range(len(max_z), len(max_z) + equations.shape[0])
        ]

    def __repr__(self) -> str:
        """
        A method to generate a representation of the linear programming model in a human-readable form.
        """
        output = "Maximise z =\t"
        max_z = self.tableau[-1][:-1]
        for i, x in enumerate(self.xs):
            if i == len(self.xs) - 1:
                output += f"{int(max_z[i])}{x} \n"
            else:
                output += f"{int(max_z[i])}{x} +\t"
        output += "subjected to \t"
        # display equations
        for eq in self.tableau[:-1]:
            output += "\n"
            for i, x in enumerate(eq):
                if i < len(eq) - 2:
                    output += f"{int(x)}{self.xs[i]} + \t"
                elif i == len(eq) - 2:
                    output += f"{int(x)}{self.xs[i]} \t"
                else:
                    output += f" <= {int(x)} \n"
        output += "\n"
        for i, x in enumerate(self.xs):
            if i == len(self.xs) - 1:
                output += f"{x}"
            else:
                output += f"{x},"
        output += "\t >= 0 "
        return output

    def solve(self, numerical=False):
        """
        A function that solves a linear programming problem using the simplex method.

        Parameters:
            numerical (bool): A flag indicating whether to return numerical values or a formatted string.

        Returns:
            if numerical == True:
                tuple: A tuple containing the maximum value of z and the values for x.
            else:
                str: A formatted string indicating the maximum value of z and the values of x.
        """
        while (self.tableau[-1][:-1] < 0).any():
            # pick most negative coefficient for z
            pivot_col = np.argmin(self.tableau[-1][:-1])
            # select row with smallest positive value for the pivot column
            lowest_ration = np.inf
            pivot_row = 0
            for row_index, value in enumerate(self.tableau[:, pivot_col]):
                if value <= 0:
                    continue
                else:
                    ratio = self.tableau[row_index, -1] / value
                    if ratio < lowest_ration:
                        lowest_ration = ratio
                        pivot_row = row_index

            pivot_element = self.tableau[pivot_row, pivot_col]

            # swap x in row and column

            self.column_x[pivot_col], self.row_x[pivot_row] = (
                self.row_x[pivot_row],
                self.column_x[pivot_col],
            )

            # Update the whole tableau with i representing rows and j - columns
            # Note without copy you will modify the original tableau and you wont be able to access the old values
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

        # Get values for x
        z = self.tableau[-1, -1]
        x_val = []
        for x in self.xs:
            if x in self.column_x:
                x_val.append(0)
            else:
                index = self.row_x.index(x)
                x_val.append(self.tableau[index, -1])

        if numerical == True:
            return z, x_val
        else:
            sol_output = {x: round(x_value, 9) for x, x_value in zip(self.xs, x_val)}
            return f"Max z = {z} and {sol_output} "


"""
#Some tests and examples

equations = np.array([[2.0, 3.0, 1500.0], [3.0, 2.0, 1500.0], [1.0, 1.0, 550.0]])
max_z = np.array([10.0, 12.0])

eq_2 = np.array([[2.0, 3.0, 1.0, 5.0], [4.0, 1.0, 2.0, 11], [3.0, 4.0, 2.0, 8.0]])
max_z2 = np.array([5.0, 4.0, 3.0])

s1 = Simplex(equations, max_z)
print(s1)
print(s1.solve())

s2 = Simplex(eq_2, max_z2)
print(s2)
print(s2.solve())

"""
