import random
import numpy as np
from sympy import *
import re
from copy import deepcopy
import matplotlib.pyplot as plt

def subindices(n):
    """
    Here we produce the subindices for the matrix elements.
    """
    return np.arange(1,n)+1

def symbols_and_variables_for_D(n):
    """
    Here we produce the list of symbols.
    """
    Subindx = subindices(n)
    #Creating empty sets we avoid duplicates
    save_variable = set()
    save_symbol = set()
    for i in np.arange(0,n-1):
        for j in np.arange(0,n-1):
            #Next we add the variables as strings in the sets.
            save_variable.add(f"N_p_{Subindx[i]}{Subindx[j]}")
            save_variable.add(f"N_p_{Subindx[j]}{Subindx[i]}")
            save_symbol.add(f"N_{Subindx[j]}{Subindx[i]}^+")
            save_symbol.add(f"N_{Subindx[i]}{Subindx[j]}^+")
    #Finally we sort the variables and the symbols:
    My_variable_list_sorted = sorted(save_variable)
    My_symbol_list_sorted = sorted(save_symbol)
    return My_variable_list_sorted,My_symbol_list_sorted

def symbolizing(variable_list,symbols_list):
    symbolic_mapping = {}
    for var_name, sym_name in zip(variable_list, symbols_list):
        symbolic_mapping[var_name] = symbols(sym_name)
    return symbolic_mapping

def substitution_test_1(n):
    Subindx = subindices(n)
    save_Ns=set()
    save_Ni1s=set()
    save_N1js=set()
    save_mjs=set()
    for i in np.arange(0,n-1):
        save_Ni1s.add(f"N_{Subindx[i]}1")
        save_N1js.add(f"N_1{Subindx[i]}")
        save_mjs.add(f"m{Subindx[i]}")
        for j in np.arange(0,n-1):
            save_Ns.add(f"N_{Subindx[i]}{Subindx[j]}")
            save_Ns.add(f"N_{Subindx[j]}{Subindx[i]}")
            Ns1="N_"+str(Subindx[i])+str(Subindx[j])
            Ns2="N_"+str(Subindx[j])+str(Subindx[i])
    Ns_list_sorted=sorted(save_Ns)
    Ni1s_list=sorted(save_Ni1s)
    N1js_list=sorted(save_N1js)
    mjs_list=sorted(save_mjs)
    return Ns_list_sorted,Ni1s_list,N1js_list,mjs_list

def kroneker_delta(index1,index2):
    """
    Defining the delta Kroneker 
    """
    if index1==index2:
        kroneker = 1
    else: 
        kroneker = 0
    return kroneker

def Add_the_kroneker_to_D(variable,symbol,n):  
    """
    This function adds the delta kroneker array to the array of symbols to create the correct list to do a matrix from
    """
    kroneker_save=[]
    for a in subindices(n):
        for b in subindices(n):
            kroneker_save.append(kroneker_delta(a,b)) #Here we have the array with delta kronekers
    Correct_symbolic_list_completed=[]
    maping=symbolizing(variable,symbol)
    for i in range(len(maping)):
        Correct_symbolic_list=maping[variable[i]]+kroneker_save[i] #here we add elementwise both arrays
        Correct_symbolic_list_completed.append(Correct_symbolic_list) #here is the complete array to "Matrizise"
    return Correct_symbolic_list_completed

def D_Matrix(n):
    """
    Here we use the list of symbols created with the function "symbols_and_variables_for_D" and create a (n-1)x(n-1) matrix
    """
    My_variable_list_sorted,My_symbol_list_sorted=symbols_and_variables_for_D(n)
    Correct_symbolic_list_completed=Add_the_kroneker_to_D(My_variable_list_sorted,My_symbol_list_sorted,n)
    Preliminar_Matrix = ([[]]*(n-1))
    k=0
    for ii in np.arange(1,n):
        Preliminar_Matrix[k] = Correct_symbolic_list_completed[k*(n-1):ii*(n-1)]
        k+=1
    return Matrix(Preliminar_Matrix)

def KBIs_Matrix(n,list_of_Gs):
    G_matrix = []
    for items in range(0,n):
        G_matrix.append(list_of_Gs[items*n:(items+1)*n])
    return G_matrix

def list_of_mij(n,list_of_densities):
    m_list=[]
    for m_index in range(1,n):
        m_list.append(list_of_densities[m_index]/list_of_densities[0])
    return m_list

def Nij_p_list(n,densities_list,G_matrix,m_list):
    N1x_list = [] #N12,N13,N14,...
    Nx1_list = [] #N21,N31,N41,...
    for N1x in range(1,n):
        N1x_list.append(densities_list[N1x]*G_matrix[0][N1x])
        Nx1_list.append(densities_list[0]*G_matrix[N1x][0])
    Nij_list=[]
    for i in range(1,n):
        Nij_list_row=[]
        for j in range(1,n):
            Nij_list_row.append(densities_list[j]*G_matrix[i][j])
        Nij_list.append(Nij_list_row)
    Nij_p=[]
    for a in range(0,n-1):
        Nij_p_row=[]
        for b in range(0,n-1):
            N_list_dummy = Nij_list[a][b]-N1x_list[b]+m_list[b]*(1+densities_list[0]*G_matrix[0][0]-Nx1_list[a])
            Nij_p_row.append(N_list_dummy)
        Nij_p.append(Nij_p_row)
    return Nij_p

def Nij_p_Matrix(Nij_p):
    Nij_p_Matrix_list = sum(Nij_p,[])
    Nij_p_Matrix_list_new=[]
    for ele in range(0,len(Nij_p_Matrix_list)):
        simplified_element=simplify(Nij_p_Matrix_list[ele])
        Nij_p_Matrix_list_new.append(simplified_element)
    return Nij_p_Matrix_list_new

def list_of_symbols_from_matrix(total_matrix):
    symbols_old_matrix = total_matrix.free_symbols
    list_of_symbols = []
    for row in total_matrix.tolist():
        for element in row:
            list_of_symbols.extend(element.free_symbols)
    ordered_symbols = list(dict.fromkeys(list_of_symbols))
    return ordered_symbols

def KBI_matrix_components_as_a_function_of_symbolic_densities(n,G_list):
    total_matrix = D_Matrix(n)
    ordered_symbols=list_of_symbols_from_matrix(total_matrix) #we need these ordered symbols to substitute them by their corresponding values.
    
    total_density_list = symbols(f"rho_1:{n+1}")

    G_matrix=KBIs_Matrix(n,G_list)
    m_list=list_of_mij(n,total_density_list)
    Nij_p=Nij_p_list(n,total_density_list,G_matrix,m_list)
    Nij_p_Matrix_list_new=Nij_p_Matrix(Nij_p)
    substitution_test=dict(zip(ordered_symbols,Nij_p_Matrix_list_new))
    new_matrix = total_matrix.subs(substitution_test)
    return new_matrix,m_list,total_density_list

def y_generator(n):
    Subindx = subindices(n)
    solution_test1=[]
    for i in np.arange(0,n-1):
        for j in np.arange(0,n-1):
            solution_test1.append(f"y_{Subindx[j]}{Subindx[i]}")
    return solution_test1

def b_generator(n):
    # Create an identity matrix of size n x n
    identity_matrix = eye(n-1)
    # Convert it into a list of column matrices
    column_matrices = [identity_matrix.col(i) for i in range(n-1)]
    return column_matrices

def solution_dictionaries(n):
    solution_test2=[]
    ys_from_1_to_n=[]
    solution_test=y_generator(n)
    Subindx = subindices(n)
    for i in np.arange(0,n-1):
        ys_from_1_to_n.append(f"y_1{Subindx[i]}")
        solution_test2.append(solution_test[(n-1)*i:(n-1)*(i+1)])
    symbols_combined={}
    for element in np.arange(len(solution_test2)):
        symbols_dictionary=symbolizing(solution_test2[element],solution_test2[element])
        symbols_combined=symbols_combined|symbols_dictionary
    ys_from_1_to_n_symbols=symbolizing(ys_from_1_to_n,ys_from_1_to_n)
    return solution_test2,symbols_combined,ys_from_1_to_n_symbols,ys_from_1_to_n

def density_decreaser(Number_of_points,densities_list):
    if Number_of_points<2:
        raise ValueError("We need at least 2 points: the start and the end!")
    From_list_to_array_rho = np.asarray(densities_list, dtype=float)
    Scale_from_1_to_0 = np.linspace(1.0, 0.0001*densities_list[-1], Number_of_points)[:, None]   # column vector, the density of the solute will decrease down to a small number (but not zero to avoid singularity problems!!)
    return (From_list_to_array_rho * Scale_from_1_to_0)

def densities_total_derivatives(density_solvent,other_densities):
    drhoi_drho1=[]
    for i in range(len(other_densities[0])):
        drhoi_drho1.append(np.gradient(other_densities[:,i],density_solvent).tolist())
    return drhoi_drho1


def water_activity(n,densities_list,G_list,final_density,Number_of_points):

    rho_1=symbols("rho_1") #symbol for water density
    
    density_1=densities_list[0]
    Other_than_water_densities = densities_list[1:]
    
    b=b_generator(n) #Generates an identity matrix (n-1)x(n-1)
    b_array = np.array(b,dtype=float) #Transforms the identity matrix in a numpy array for efficiency purposes
    Ns_list_sorted,Ni1s_list,N1js_list,mjs_list=substitution_test_1(n) #From here we need the mj as list of strings in mj_list
    #Since mjs_list does not come in order for n<=10, the next line orders the strings in the list:
    mjs_list = sorted(mjs_list, key=lambda x: int(re.search(r'\d+', x).group()))
    
    Density_array = np.linspace(density_1,final_density,Number_of_points)
     
    Other_than_water_Density_array=density_decreaser(Number_of_points,Other_than_water_densities)

    
    ### Other_than_water_Density_array = np.linspace(np.sum(list_of_densities[1:]),0,Number_of_points)
    derivatives_of_densities = np.array(densities_total_derivatives(Density_array,Other_than_water_Density_array))
    derivatives_of_densities2=[]
    for der in range(len(derivatives_of_densities[0])):
         derivatives_of_densities2.append(derivatives_of_densities[:,der])
    
    KBI_activity_integrand_total=[]
    
    for value_solvent_density, value_other_density,derivatives_of_densities_test in zip(Density_array,Other_than_water_Density_array,derivatives_of_densities2):
       
        #ADDED 13.10.2025:
        New_density_array = [value_solvent_density,*value_other_density] 
        derivatives_of_densities_list=[*derivatives_of_densities_test]
        
        new_matrix,m_list,total_density_list=KBI_matrix_components_as_a_function_of_symbolic_densities(n,G_list)

        subs_dict = {total_density_list[i]: New_density_array[i] for i in range(len(total_density_list))}
        Numeral_matrix=new_matrix.subs(subs_dict) #Matrix with densities substitued
        
        Numeral_matrix_in_array=np.array(Numeral_matrix,dtype=float) #Same with a numpy array form
        solutions = np.linalg.solve(Numeral_matrix_in_array,b_array) #solutions from the equation Mx=b, so x=solutions
        #We need the symbols of our solutions ('y12','y13','y22',...) and these are found in solution_test2 and ys_from_1_to_n_symbols:
        solution_test2,symbols_combined,ys_from_1_to_n_symbols,ys_from_1_to_n=solution_dictionaries(n)
        
        sol_values={} #assigned the corresponding value to each solution
        for i in range(n-1):
            sol_dic=dict(zip(solution_test2[i],solutions[i]))
            sol_values=sol_values|sol_dic

        values_of_mj = [new_matrix.subs(dict(zip(total_density_list, New_density_array))) for new_matrix in m_list] #m2,m3,m4,...
        mjs_symbols=dict(zip(mjs_list,values_of_mj)) #Here assigning a value to each key (mj)
        
        dict_1_to_n={} #Here we have a dictionary with all the solutions of the type y12, y13, y14, y15, ...
        for jj in np.arange(0,n-1):
            gg=0
            for ii in np.arange(0,n-1):
                gg=gg-mjs_symbols[mjs_list[ii]]*sol_values[solution_test2[jj][ii]]
            dict_1_to_n[ys_from_1_to_n[jj]]=gg #y12,y13,y14,...
        
        sol_11=0 #Finally, here we build the y11 solution
        for elem in ys_from_1_to_n:
            sol_11=sol_11-dict_1_to_n[elem]
        
        sol_11_simplified_dict={'y_11': sol_11} #we make a dictionary for this y11 solution
        
        final_solutions = sol_11_simplified_dict|dict_1_to_n|sol_values #Here we have all the solutions together in a dictionary form
        
        KBI_activity_integrand = (final_solutions['y_11'] - 1)/value_solvent_density + 1/(value_solvent_density+np.sum(value_other_density))
        
        Integrand_solutes_total=[]
        for number_of_components in range(len(New_density_array[1:])):
            Integrand_solutes=((final_solutions['y_1'+str(number_of_components+2)]/New_density_array[1:][number_of_components])+1/np.sum(New_density_array))*derivatives_of_densities_list[number_of_components]
            Integrand_solutes_total.append(Integrand_solutes)
        
        KBI_activity_integrand_all_contributions = np.sum(Integrand_solutes_total)+KBI_activity_integrand
        KBI_activity_integrand_total.append(KBI_activity_integrand_all_contributions[0])

    KBI_activity_integrand_total=np.array(KBI_activity_integrand_total,dtype=float)

    #Integration by Trapezoidal:
    trapezoidal_integral = np.trapz(KBI_activity_integrand_total,Density_array)

    return np.exp(-trapezoidal_integral)


def Latex_Analytical_expressions_for_potential_derivatives(n):
    MM=D_Matrix(n)
    list_of_Gij_ncomp = [f"G_{i}{j}" for i in range(1, n+1) for j in range(1, n+1)]
    list_densities = symbols(f"rho_1:{n+1}")
    m_list = list_of_mij(n,list_densities)
    G = symbols(' '.join(list_of_Gij_ncomp))
    G_matrix=KBIs_Matrix(n,G)

    Nij_p=Nij_p_list(n,list_densities,G_matrix,m_list)

    b=b_generator(n)
    y=y_generator(n)
    
    new_matrix,m_list,total_density_list=KBI_matrix_components_as_a_function_of_symbolic_densities(n,G)
    
    list_symbols_from_matrix=list_of_symbols_from_matrix(MM)
    
    Nij_p_Matrix2=Nij_p_Matrix(Nij_p)

    solution_test2,symbols_combined,ys_from_1_to_n_symbols,ys_from_1_to_n=solution_dictionaries(n)
    
    print("Computing Determinant and Adjugate...")
    D = MM.det(method='berkowitz')
    Adj = MM.adjugate()
    B_matrix = Matrix.hstack(*b)
    Numerator_Matrix = Adj * B_matrix
    numerators = {}
    Dsym = Symbol(f"D_{n}")
    
    n_vars = MM.shape[0]
    
    # Iterate over b vectors (columns), then variables (rows)
    for j in range(len(b)):
        for i in range(n_vars):
            var = solution_test2[0][i]
            numerators[(j, var)] = Numerator_Matrix[i, j] #/ D
            
    No_columns_format = {}

    for (j, var_name) in numerators.keys():
        # Extract the row index from variable name '\\mu_{i2}'
        row_idx = var_name[2]
        
        # Map j to column index: j=0->col=2, j=1->col=3, j=2->col=4
        col_idx = j + 2
        
        # Create new variable name
        new_var_name = f'\\mu_{{{row_idx}{col_idx}}}'
        
        # Store with new key
        No_columns_format[new_var_name] = numerators[(j, var_name)]
    
    numerators=No_columns_format
    
    lines = []
    for key, num in numerators.items():
        # We print x = (1/D) * Num
        lines.append(f"{key} &= \\frac{{1}}{{{latex(Dsym)}}}\\left({latex(num)}\\right) \\\\")
    
    # 2. Print the Determinant definition
    # lines.append("") # Spacer
    lines.append(f"{latex(Dsym)} &= {latex(D)} \\\\")
    
    body = "\n".join(lines)
    print("\\begin{align*}\n" + body + "\n\\end{align*}")

    
    