import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import warnings
import math
import graphviz
import shutil
import os
import networkx as nx
import streamlit as st
import gravis as gv
import streamlit.components.v1 as components
from few_shot_prompting import get_few_shot_prompting_response as get_few_shot_response

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
def pre_process_ranges(ranges, edge_cases_only=True):
    processed_ranges = {}
    for var, range_value in ranges.items():
        if isinstance(range_value[0], int) and isinstance(range_value[1], int):
            edge_size = 1
            step = range_value[2] if range_value[2] != 0 else 1  # Check if step is zero
            processed_ranges[var] = list(np.arange(range_value[0], range_value[1] + 1, step))
            if edge_cases_only:
                processed_ranges[var] = processed_ranges[var][:edge_size] + processed_ranges[var][-edge_size:]
        else:
            processed_ranges[var] = range_value
    return processed_ranges

# adding pre requisites required for calculating each dependant variable
def add_pre_requisites(equations, dependant_variables, variables):
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

def add_ranges_for_miscellaneous(variables, dependant_variables, independant_variables, ranges, streamlit=False):
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
            if streamlit:
                st.warning(f' Warning: Assuming values for {miscellaneous_variable} as {start} to {end} with step {step}', icon='⚠️')

def process_dependant_variables(equations):
    dependant_variables = []
    for equation in equations:
        variable = equation['equation'].split('=')[0].strip()
        dependant_variables.append(variable)
    dependant_variables = list(set(dependant_variables))
    return dependant_variables

# generating different combinations of independant variables
def initialize_data_frame(test_cases_df, independant_variable_combinations, ranges):
    for combination in independant_variable_combinations:
        new_row = dict(zip(ranges.keys(), combination))
        new_row_df = pd.DataFrame([new_row])
        test_cases_df = pd.concat([test_cases_df, new_row_df], ignore_index=True)
    return test_cases_df

# get statistics
def get_statistics(dependant_variables, test_cases_df):
    statistics = ''
    for dependant_variable in dependant_variables:
        try:
            statistics += f"Statistics for {dependant_variable}:\n"
            statistics += f"Minimum: {test_cases_df[dependant_variable].min()}\n"
            statistics += f"Maximum: {test_cases_df[dependant_variable].max()}\n"
            statistics += f"Median: {test_cases_df[dependant_variable].median()}\n"
            statistics += f"Unique values: {test_cases_df[dependant_variable].nunique()}\n\n"
        except:
            pass
    return statistics


# get styled HTML statistics
def get_styled_html_statistics(dependant_variables, test_cases_df):
    styled_statistics = '<style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid #dddddd; text-align: left; padding: 8px;} th {background-color: #f2f2f2;}</style>'
    for dependant_variable in dependant_variables:
        try:
            styled_statistics += f"<h3>Statistics for {dependant_variable}</h3>"
            styled_statistics += "<table>"
            styled_statistics += f"<tr><td>Minimum</td><td>{test_cases_df[dependant_variable].min()}</td></tr>"
            styled_statistics += f"<tr><td>Maximum</td><td>{test_cases_df[dependant_variable].max()}</td></tr>"
            styled_statistics += f"<tr><td>Median</td><td>{test_cases_df[dependant_variable].median()}</td></tr>"
            styled_statistics += f"<tr><td>Unique values</td><td>{test_cases_df[dependant_variable].nunique()}</td></tr>"
            styled_statistics += "</table><br>"
        except:
            pass
    return styled_statistics


def show_variable_trend(dependant_variables, test_cases_df, streamlit=False):
    for variable in dependant_variables:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(test_cases_df[variable])
        ax.set_title(variable.capitalize())
        ax.set_xlabel(f'{variable} value')
        ax.set_ylabel('count')
        if streamlit:
            st.pyplot(fig)
        plt.close(fig)

# generate html
def generate_html(test_cases_df, path):
    test_cases_html = test_cases_df.to_html()
    with open(path, "w", encoding="utf-8") as text_file:
        text_file.write(test_cases_html)    

# identify dependancy trend
def identify_dependancy_trend(dependant_variables, equations_dict, test_cases_df, streamlit=False):
    try:
        num_dependant_variables = len(dependant_variables)
        calc_dependant_variables = {}
        num_variables = 0
        
        for var in dependant_variables:
            for equation in equations_dict[var]:
                if var in calc_dependant_variables:
                    calc_dependant_variables[var] += equation['calculation_pre_requisites']
                else: 
                    calc_dependant_variables[var] = equation['calculation_pre_requisites']
            calc_dependant_variables[var] = list(set(calc_dependant_variables[var]))
            num_variables = max(num_variables, len(calc_dependant_variables[var]))
                    
        fig, axs = plt.subplots(num_dependant_variables, num_variables, figsize=(5*num_variables, 6*num_dependant_variables))

        if num_dependant_variables == 1:
            axs = axs.reshape(1, num_variables)

        for j, dependant_variable in enumerate(dependant_variables):
            for i, independant_variable in enumerate(calc_dependant_variables[var]):
                axs[j, i].scatter(test_cases_df[independant_variable], test_cases_df[dependant_variable])
                axs[j, i].set_title(f'{independant_variable.capitalize()} vs {dependant_variable.capitalize()}')
                axs[j, i].set_xlabel(f'{independant_variable.capitalize()}')
                axs[j, i].set_ylabel(f'{dependant_variable.capitalize()}')

        plt.tight_layout()
        if streamlit:
            st.pyplot(fig)
        # plt.savefig('dependant_vs_independant.pdf', format='pdf', dpi=2000)
        plt.close()
        # plt.show() 
    except:
        pass

# fill dependant variables
def fill_dependant_variables(dependant_variables, equations_dict, test_cases_df, namespace):
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
    return test_cases_df

# create flow graph
def backup_create_flow_graph(flowchart_data, directory_path, equations_dict):
    for var, equations in equations_dict.items():
        for equation in equations:
            flowchart_data += [(pre_req_var, equation['equation'], equation['condition']) for pre_req_var in equation['condition_pre_requisites']]
 
    graph = graphviz.Digraph('Flowchart', format='png')
    
    for node_from, node_to, condition in flowchart_data:
        graph.node(node_from)
        graph.node(node_to)
        graph.edge(node_from, node_to, label=condition)

    graph.render(filename=r'flowchart', format='png', cleanup=True)

def delete_directory_contents(directory_path):
    try:
        shutil.rmtree(directory_path)
        os.makedirs(directory_path)  
        print(f"Contents of {directory_path} deleted successfully.")
    except Exception as e:
        print(f"Error: {e}")
        

def create_flow_graph(flowchart_data, directory_path, equations_dict):
    flowchart_groups = {}
    delete_directory_contents(directory_path)
    
    for var, equations in equations_dict.items():
        for equation in equations:
            pre_req_vars = equation['condition_pre_requisites']
            key = tuple(sorted(pre_req_vars + [var]))

            if key not in flowchart_groups:
                flowchart_groups[key] = []

            flowchart_groups[key].append((pre_req_vars, var, equation['equation'], equation['condition']))

    graph_index = 1
    for key, flowchart_data_group in flowchart_groups.items():
        flowchart_name = f'{directory_path}/{graph_index}'
        
        graph = graphviz.Digraph(flowchart_name, format='png') 

        for pre_req_vars, var, equation, condition in flowchart_data_group:
            graph.node(equation)
            for node_from in pre_req_vars:
                graph.node(node_from)
                graph.edge(node_from, equation, label=condition)

        graph.render(filename=flowchart_name, format='png', cleanup=True)
        graph_index += 1
        flowchart_path = flowchart_name + '.png'


# initialize edge and edge labels
def initialize_edge_and_labels(edges, edge_labels, equations_dict):
    for var, equations in equations_dict.items():
        for equation in equations:
            edges += [(pre_req_var, var, equation['condition'], equation['equation']) for pre_req_var in equation['condition_pre_requisites']]
            new_edge_labels = {(pre_req_var, var) : equation['condition'] for pre_req_var in equation['condition_pre_requisites']}
            edge_labels.update(new_edge_labels)    

# save to csv
def save_to_csv(test_cases_df, path):
    test_cases_df.to_csv(path)    

# create interactive graph
def create_interactive_graph(edges, streamlit=False):
    edge_label_index = 2

    G = nx.DiGraph(directed=True)

    for edge in edges:
        G.add_node(edge[0])
        G.add_node(edge[3])
        G.nodes[edge[3]]['color'] = 'red'

    for edge in edges:
        G.add_edge(edge[0], edge[3], label=edge[2])
        G.edges[edge[0], edge[3]].update({'label': edge[2]})

    fig = gv.d3(G, show_edge_label=True, edge_label_data_source='label',
        layout_algorithm_active=True,
        use_collision_force=True,
        collision_force_radius=70,
        edge_label_rotation=0,
        edge_label_size_factor=0.7,
        edge_label_font='monospace',
        zoom_factor=1.5,
        many_body_force_strength=10)
    
    if streamlit:
        components.html(fig.to_html(), height=700)
    
# create graph image
def create_graph_image(edges, edge_labels, streamlit=False):
    try:
        edge_label_index = 2
        G = nx.DiGraph(directed=True)

        for edge in edges:
            G.add_node(edge[0])
            G.add_node(edge[1])
        for edge in edges:
            G.add_edge(edge[0], edge[1], label=edge[2])
            G.edges[edge[0], edge[1]].update({'label': edge[2]})
        
        pos = nx.spring_layout(G)
        options = {
            'node_color': 'yellow',
            'node_size': 300,
            'width': 1,
            'arrowstyle': '-|>',
            'arrowsize': 12,
            'labels' : {node: node for node in G.nodes()}
        }
        fig, ax = plt.subplots()
        nx.draw_networkx(G, pos, arrows=True, **options)
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_color='red'
        )
        if streamlit:
            st.pyplot(fig)
        plt.close(fig)
    except:
        pass

# generate namespace
def initialize_namespace():
    math_functions = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'copysign', 'cos', 'cosh', 'degrees', 'exp',
                      'fabs', 'floor', 'fmod', 'frexp', 'hypot', 'isfinite', 'isinf', 'isnan', 'ldexp', 'log',
                      'log10', 'modf', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']

    namespace = {func: getattr(math, func) for func in math_functions}
    return namespace


def get_few_shot_prompting_response(FRD = ''):
    print('waiting for request response')
    # output from few shot prompting
    dict_from_few_shot_prompting = get_few_shot_response(FRD)
    print(dict_from_few_shot_prompting)
    return dict_from_few_shot_prompting


def generate_test_cases(FRD = ''):
    edge_cases_only = True
    limit = 20
    edges = []
    edge_labels = {}
    namespace = {}
    flowchart_data = []

    dict_from_few_shot_prompting = get_few_shot_response(FRD=FRD)
    
    # extracting the necessary details from few shot prompting api call 
    variables = dict_from_few_shot_prompting['variables']
    equations = dict_from_few_shot_prompting['equations']
    ranges = dict_from_few_shot_prompting['ranges']
    independant_variables = [var for var in ranges.keys()]
    
    dependant_variables = process_dependant_variables(equations)
    add_ranges_for_miscellaneous(variables, dependant_variables, independant_variables, ranges)

    # pre processing
    pre_process_equations(equations)
    add_pre_requisites(equations, dependant_variables, variables)
    
    ranges = pre_process_ranges(ranges, edge_cases_only)
    equations_dict = add_equation_conditions(equations)
    print(dependant_variables)
    
    dependant_variables = topological_sort(dependant_variables, equations_dict)
    print(dependant_variables)
    variables = independant_variables+dependant_variables

    # creating data frame for the test cases
    test_cases_df = pd.DataFrame(columns=variables)
    independant_variable_combinations = list(itertools.islice(itertools.product(*ranges.values()), limit))
    test_cases_df = initialize_data_frame(test_cases_df, independant_variable_combinations, ranges)
    
    namespace = initialize_namespace()
    test_cases_df = fill_dependant_variables(dependant_variables, equations_dict, test_cases_df, namespace)
    
    print(test_cases_df.describe(include = 'all').T)
    
    statistics = get_statistics(dependant_variables, test_cases_df)
    print(statistics)
    
    save_to_csv(test_cases_df, path='test_cases_generated.csv')
    generate_html(test_cases_df, path="test_cases.html")
    
    
    create_flow_graph(flowchart_data=flowchart_data,
                      directory_path='flowchart',
                      equations_dict=equations_dict)  
    
    # show_variable_trend(dependant_variables)
    # identify_dependancy_trend(dependant_variables, independant_variables, test_cases_df)
    # initialize_edge_and_labels(edges, edge_labels, equations_dict)
    # create_interactive_graph(edges)
    # create_graph_image(edges, edge_labels) 
    

if __name__ == '__main__':
    # input given 
    FRD = '''
    Auto_Drift_Estimate - [enabled, not_enabled]
    a0 - [1, 2, 1]
    a1 - [2, 2, 1]
    a2 - [3, 5, 1]
    a3 - [0, 2, 1]
    a4 - [0, 1, 1]
    a5 - [1, 2, 1]
    a6 - [0, 2, 1]
    actual_frequency - [7000, 7010, 3]
    present_temperature - [-10, 50, 8]
    _RANGE_END_TAG_

    If Auto_Drift_Estimate Enabled
    If -10 < present_temperature < 55 then
    delfbyf0 = a0 + a1 * (present_temperature - temp0) + a2 * (present_temperature - temp0)^2 + a3 * (present_temperature - temp0)^3 + a4 * (present_temperature - temp0)^4 + a5 * (present_temperature - temp0)^5 + a6 * (present_temperature - temp0)^6
    Where
    f0 = (1/0.000125)
    theoretical_frequency = f0
    deltaf = actual_frequency - theoretical_frequency
    temp0 = 25 
    The above logic shall be computed every 7 sec as the temperature reading shall happen every 7 sec.
    deltaf = f0 * delfbyf0
    new_time_period = 1/(f0 + deltaf)
    delta_time_period = new_time_period - t0
    '''
    generate_test_cases(FRD)
