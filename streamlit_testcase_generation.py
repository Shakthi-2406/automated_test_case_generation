import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import warnings
import math
import graphviz
import networkx as nx
import gravis as gv

confusables = [
    ('^', '**'),
    ('acos', 'cos')
]

# remove confusables
def pre_process_equations(equations):
    for equation in equations:
        for confusable in confusables:
            equation['equation'] = equation['equation'].replace(confusable[0], confusable[1])
            if equation['condition']:
                equation['condition'] = equation['condition'].replace(confusable[0], confusable[1])

# generate the list of values for each independant variables
def pre_process_ranges(ranges, edge_cases_only = True):
    processed_ranges = {}
    for var, range_value in ranges.items():
        if isinstance(range_value[0], int) and isinstance(range_value[1], int):
            edge_size = 1
            processed_ranges[var] = list(np.arange(range_value[0], range_value[1] + 1, range_value[2]))
            if edge_cases_only:
                processed_ranges[var] = processed_ranges[var][:edge_size] + processed_ranges[var][-edge_size:]
        else:
            processed_ranges[var] = range_value
    return processed_ranges

# adding pre requisites required for calculating each dependant variable
def add_pre_requisites(equations):
    for equation in equations:
        equation['calculation_pre_requisites'] = []
        equation['calculation_pre_requisites_dependant'] = []
        equation['condition_pre_requisites'] = []
        equation['condition_pre_requisites_dependant'] = []
        rhs = equation['equation'].split('=')[1]
        equation['lhs'] = equation['equation'].split('=')[0].strip()
        condition = equation['condition']

        equation["dependant"] = condition is not None
        
        for var in variables:
            if var in rhs:
                equation['calculation_pre_requisites'].append(var)
                if var in dependant_variables:
                    equation['calculation_pre_requisites_dependant'].append(var)
            if condition and var in condition:
                equation['condition_pre_requisites'].append(var)
                if var in dependant_variables:
                    equation['condition_pre_requisites_dependant'].append(var)

def add_equation_conditions(equations):
    equations_dict = {}
    for equation in equations:
        lhs = equation['equation'].split('=')[0].strip()
        if lhs in equations_dict:
            equations_dict[lhs].append(equation)
        else:
            equations_dict[lhs] = [equation]
    return equations_dict

def check_cyclic_dependancy(equations):
    for equation in equations:
        lhs = equation['equation'].split('=')[0].strip()
        pre_requisites = set(equation['calculation_pre_requisites'] + equation['condition_pre_requisites'])
        # yet to implement

def topological_sort(dependant_variables, equations_dict):
    sorted_variables = []
    visited = set()

    def dfs(variable):
        if variable in visited:
            return
        visited.add(variable)

        equations = equations_dict.get(variable, [])

        for equation_info in equations:
            calculation_pre_requisites_dependant = equation_info['calculation_pre_requisites_dependant']

            for pre_req_variable in calculation_pre_requisites_dependant:
                dfs(pre_req_variable)

        sorted_variables.append(variable)

    for variable in dependant_variables:
        dfs(variable)

    return sorted_variables

def add_ranges_for_miscellaneous():
    miscellaneous_variables = [var for var in variables if var not in dependant_variables and var not in independant_variables]
    if len(miscellaneous_variables) > 0:
        warning = f'Variable/s missing for implementation: {miscellaneous_variables}'
        warnings.warn(warning)

        for miscellaneous_variable in miscellaneous_variables:
            start = 10
            end = 20
            step = 4
            warning = f'Assuming values for {miscellaneous_variable} as {start} to {end} with step {step}'
            print(warning[:])
            ranges[miscellaneous_variable] = [start, end, step]
            warnings.warn(warning)
            independant_variables.append(miscellaneous_variable)

def process_dependant_variables():
    dependant_variables = []
    for equation in equations:
        variable = equation['equation'].split('=')[0].strip()
        dependant_variables.append(variable)
    dependant_variables = list(set(dependant_variables))
    return dependant_variables

# generating different combinations of independant variables
def initialize_data_frame():
    global test_cases_df
    for combination in independant_variable_combinations:
        new_row = dict(zip(ranges.keys(), combination))
        new_row_df = pd.DataFrame([new_row])
        test_cases_df = pd.concat([test_cases_df, new_row_df], ignore_index=True)

# printing statistics
def print_statistics():
    for dependant_variable in dependant_variables:
        try:
            print(f"Statistics for {dependant_variable}:")
            print(f"Minimum: {test_cases_df[dependant_variable].min()}")
            print(f"Maximum: {test_cases_df[dependant_variable].max()}")
            print(f"Median: {test_cases_df[dependant_variable].median()}")
            print(f"Unique values: {test_cases_df[dependant_variable].nunique()}")
            print()
        except:
            pass

# show variable trend
def show_variable_trend():
    for variable in dependant_variables:
        print(variable)
        plt.hist(test_cases_df[variable])
        plt.title(variable.capitalize())
        plt.xlabel(f'{variable} value')
        plt.ylabel('count')
        plt.show() 

# generate html
def generate_html():
    test_cases_html = test_cases_df.to_html()
    with open("test_cases.html", "w", encoding="utf-8") as text_file:
        text_file.write(test_cases_html)    

# identify dependancy trend
def identify_dependancy_trend():
    try:
        num_variables = len(independant_variables)
        num_dependant_variables = len(dependant_variables)

        fig, axs = plt.subplots(num_dependant_variables, num_variables, figsize=(5*num_variables, 6*num_dependant_variables))

        if num_dependant_variables == 1:
            axs = axs.reshape(1, num_variables)

        for j, dependant_variable in enumerate(dependant_variables):
            for i, independant_variable in enumerate(independant_variables):
                axs[j, i].scatter(test_cases_df[independant_variable], test_cases_df[dependant_variable])
                axs[j, i].set_title(f'{independant_variable.capitalize()} vs {dependant_variable.capitalize()}')
                axs[j, i].set_xlabel(f'{independant_variable.capitalize()}')
                axs[j, i].set_ylabel(f'{dependant_variable.capitalize()}')

        plt.tight_layout()
        plt.savefig('dependant_vs_independant.pdf', format='pdf', dpi=2000)
        plt.show() 
    except:
        pass   

# fill dependant variables
def fill_dependant_variables():
    for variable in dependant_variables:
        equations = equations_dict.get(variable, [])
        for index, row in test_cases_df.iterrows():
            for equation_info in equations:
                equation = equation_info['equation']
                equation_rhs = equation.split('=')[1]
                condition = equation_info['condition']
                calculation_pre_requisites = equation_info['calculation_pre_requisites']
                condition_pre_requisites = equation_info['condition_pre_requisites'] 

                # Check if condition matches
                if not condition or eval(condition, row[:].to_dict()):
                    # Calculate value using eval
                    try:
                        calculated_value = eval(equation_rhs, namespace, row[:].to_dict())
                        test_cases_df.at[index, variable] = calculated_value
                        break  
                    except:
                        print(f'Eval failed for {equation_rhs}')
                        print(equation_rhs, row[calculation_pre_requisites].to_dict())

# create flow graph
def create_flow_graph():
    global flowchart_data
    for var, equations in equations_dict.items():
        for equation in equations:
            flowchart_data += [(pre_req_var, equation['equation'], equation['condition']) for pre_req_var in equation['condition_pre_requisites']]
 
    graph = graphviz.Digraph('Flowchart', format='png')
    
    for node_from, node_to, condition in flowchart_data:
        graph.node(node_from)
        graph.node(node_to)
        graph.edge(node_from, node_to, label=condition)

    graph.render(filename=r'flowchart', format='png', cleanup=True)

# initialize edge and edge labels
def initialize_edge_and_labels():
    global edges, edge_labels 
    for var, equations in equations_dict.items():
        for equation in equations:
            edges += [(pre_req_var, var, equation['condition'], equation['equation']) for pre_req_var in equation['condition_pre_requisites']]
            new_edge_labels = {(pre_req_var, var) : equation['condition'] for pre_req_var in equation['condition_pre_requisites']}
            edge_labels.update(new_edge_labels)    

# save to csv
def save_to_csv():
    test_cases_df.columns = test_cases_df.columns.str.replace('test_case_notest_case_no','')
    test_cases_df.to_csv('test_cases_generated.csv')    

# create interactive graph
def create_interactive_graph():
    edge_label_index = 2

    G = nx.DiGraph(directed=True)

    for edge in edges:
        G.add_node(edge[0])
        G.add_node(edge[edge_label_index])

    for edge in edges:
        G.add_edge(edge[0], edge[edge_label_index], label=edge[2])
        G.edges[edge[0], edge[edge_label_index]].update({'label': edge[2]})

    gv.d3(G, show_edge_label=True, edge_label_data_source='label',
        layout_algorithm_active=True,
        use_collision_force=True,
        collision_force_radius=60,
        edge_label_rotation=0,
        edge_label_size_factor=1,
        edge_label_font='bold',
        zoom_factor=1.5,
        many_body_force_strength=10)

# create graph image
def create_graph_image():
    try:
        pos = nx.spring_layout(G)
        options = {
            'node_color': 'yellow',
            'node_size': 300,
            'width': 1,
            'arrowstyle': '-|>',
            'arrowsize': 12,
            'labels' : {node: node for node in G.nodes()}
        }
        nx.draw_networkx(G, pos, arrows=True, **options)
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_color='red'
        )
    except:
        pass

# generate namespace
def initialize_namespace():
    global namespace
    math_functions = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'copysign', 'cos', 'cosh', 'degrees', 'exp',
                      'fabs', 'floor', 'fmod', 'frexp', 'hypot', 'isfinite', 'isinf', 'isnan', 'ldexp', 'log',
                      'log10', 'modf', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']

    namespace = {func: getattr(math, func) for func in math_functions}



# input given 
FRD = '''
Auto_Drift_Estimate - [enabled, not_enabled]
a0 - [1, 2]
a1 - [2, 2]
a2 - [3, 5]
a3 - [0, 2]
a4 - [0, 1]
a5 - [1, 2]
a6 - [0, 2]
actual_frequency - [7000, 7010]
present_temperature - [-10, 50]


If Auto_Drift_Estimate Enabled
If -10 < present_temperature < 55 then
delfbyf0 = a0 + a1 * (present_temperature - temp0) + a2 * (present_temperature - temp0)^2 + a3 * (present_temperature - temp0)^3 + a4 * (present_temperature - temp0)^4 + a5 * (present_temperature - temp0)^5 + a6 * (present_temperature - temp0)^6
Where
f0 = (1/0.000125)
theoretical_frequency = f0
deltaf = actual_frequency - theoretical_frequency
temp0 = 25 deg
The above logic shall be computed every 7 sec as the temperature reading shall happen every 7 sec.
deltaf = f0 * delfbyf0
new_time_period = 1/(f0 + deltaf)
delta_time_period = new_time_period - t0
'''

def get_few_shot_prompting_response(FRD):
    # output from few shot prompting
    dict_from_few_shot_prompting ={
    "variables": ["A", "Axg", "Ayg", "Azg", "Roll", "Rollx", "Rolly", "Rollz", "dQ", "Theta", "TermAngCorThr", "ALSPhase", "DQ", "RefQ", "RefQi", "QRollBias", "RefW", "ζ", "Operation"],
    "equations": [
        {"equation": "Roll = (Axg* Ayg * Azg) / ((Axg + Ayg + Azg))", "condition": None},
        {"equation": "Theta = acos(Rolly)", "condition": None},
        {"equation": "Theta = TermAngCorThr * sin(Theta) * 3.14159 / 180", "condition": "ALSPhase == 'descent' and abs(Theta) > TermAngCorThr"},
        {"equation": "DQ[1] = Rollz * sin(Theta / 2) / sqrt(Rollx^2 + Rollz^2)", "condition": "(Rollx^2 + Rollz^2) > ζ"},
        {"equation": "DQ[2] = 0.0", "condition": "(Rollx^2 + Rollz^2) > ζ"},
        {"equation": "DQ[3] = -Rollx * sin(Theta / 2) / sqrt(Rollx^2 + Rollz^2)", "condition": "(Rollx^2 + Rollz^2) > ζ"},
        {"equation": "DQ[4] = cos(Theta / 2)", "condition": "(Rollx^2 + Rollz^2) > ζ"},
        {"equation": "Status = Quo", "condition": "not ((Rollx^2 + Rollz^2) > ζ)"},
        {"equation": "RefQ = RefQi + A + dQ", "condition": "Operation == 'Flight'"},
        {"equation": "RefQ = RefQi + A + dQ + A + QRollBias", "condition": "Operation == 'LTDT testing'"},
        {"equation": "RefW = [0, 0, 0]", "condition": None}
    ],
    "ranges": {
        "Operation": ['LTDT testing', 'Flight'],
        "ALSPhase": ['descent']
    }
    }

    return dict_from_few_shot_prompting


def init():
    global edge_cases_only, limit, edges, edge_labels, namespace, flowchart_data
    edge_cases_only = True
    limit = 20
    edges = []
    edge_labels = {}
    namespace = {}
    flowchart_data = []


def generate_test_cases(FRD):
    init()
    global variables, equations, ranges, equations_dict, independant_variables, dependant_variables, test_cases_df, independant_variable_combinations

    dict_from_few_shot_prompting = get_few_shot_prompting_response(FRD)
    # extracting the necessary details from few shot prompting api call 
    variables = dict_from_few_shot_prompting['variables']
    equations = dict_from_few_shot_prompting['equations']
    ranges = dict_from_few_shot_prompting['ranges']
    independant_variables = [var for var in ranges.keys()]
    dependant_variables = process_dependant_variables()
    add_ranges_for_miscellaneous()

    # pre processing
    pre_process_equations(equations)
    add_pre_requisites(equations)
    ranges = pre_process_ranges(ranges, edge_cases_only)
    equations_dict = add_equation_conditions(equations)
    print(dependant_variables)
    dependant_variables = topological_sort(dependant_variables, equations_dict)
    print(dependant_variables)
    variables = independant_variables+dependant_variables

    # creating data frame for the test cases
    test_cases_df = pd.DataFrame(columns=variables)
    independant_variable_combinations = list(itertools.islice(itertools.product(*ranges.values()), limit))
    initialize_data_frame()
    initialize_namespace()
    fill_dependant_variables()
    print(test_cases_df.describe(include = 'all').T)
    # show_variable_trend()
    print_statistics()
    save_to_csv()
    generate_html()
    identify_dependancy_trend()
    initialize_edge_and_labels()
    create_interactive_graph()
    create_graph_image()       
    create_flow_graph()


if __name__ == '__main__':
    generate_test_cases()
