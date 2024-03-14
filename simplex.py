import numpy as np

##Simplex problem e.g.
# max z = 10x1 + 12x2
# 2x1 + 3x2 <= 1500
# 3x1 + 2x2 <= 1500
# x1 + x2 <= 550
# x1,x2 >= 0


class Simplex:
    def __init__(self, equations, max_z) -> None:
        self.tableau = np.vstack([equations, np.append(max_z * -1, 0)])
        # the xs which we will be interested in e.g. x1, x2
        self.xs = [f"x{i+1}" for i in range(len(max_z))]
        # at the start x1 x2 are in the column 1 and 2 and we will keep track at where they are in the table
        # self.column_x = {x:i for i,x in enumerate(self.xs)} Note can you a list instead similar to dic with key 0, 1, 2 etc
        self.column_x = self.xs.copy()
        # self.row_x = {f'x{j+1}' : i for i,j in enumerate(range(len(max_z),len(max_z) + equations.shape[0]))} Same as above:
        # Instead of having x3: 0, x4: 1, x5: 2 we can have [x3, x4, x5]
        # tracks artificial variables as x3,x4 and x5
        self.row_x = [
            f"x{i+1}" for i in range(len(max_z), len(max_z) + equations.shape[0])
        ]

    def __str__(self) -> str:
        # z = SUM max_z  * xs
        output = "Maximise z =\t"
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

            ##print(f'pivot_col={pivot_col}, pivot_row={pivot_row}, pivot_element={pivot_element}')
            # swap x in row and column

            self.column_x[pivot_col], self.row_x[pivot_row] = (
                self.row_x[pivot_row],
                self.column_x[pivot_col],
            )

            # Update the whole tableau with i representing rows and j - columns
            # Note without copy you will modify the original tableau and you wont be able to access the old values
            updated_tableau = self.tableau.copy()
            ##print(f'BEFORE FOR \n Tableau is \n {self.tableau} \n and updated tableau is \n {updated_tableau} \n ')
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
                        # print(f'For element {value}, id = ({i},{j})the new value is {self.tableau[i,j]} by doing {self.tableau[i,pivot_col]} * {self.tableau[pivot_row,j]} / {pivot_element} or ({i},{pivot_col}) * ({pivot_row},{j}) / {pivot_element}')

            ##print(f'AFTER FOR \n Tableau is \n {self.tableau} \n and updated tableau is \n {updated_tableau} \n ')
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

            # print(f'AFTER FOR \n Tableau is\n {self.tableau} \n and updated tableau is \n {updated_tableau}')




equations = np.array([[2.0, 3.0, 1500.0], [3.0, 2.0, 1500.0], [1.0, 1.0, 550.0]])
max_z = np.array([10.0, 12.0])

eq_2 = np.array([[2.0, 3.0, 1.0, 5.0], [4.0, 1.0, 2.0, 11], [3.0, 4.0, 2.0, 8.0]])
max_z2 = np.array([5.0, 4.0, 3.0])

s1 = Simplex(equations, max_z)
print(s1)
print(s1.solve())

s2 = Simplex(eq_2, max_z2)
print(s2.solve())