import time
import pandas as pd
# from few_shot_prompting import get_few_shot_prompting_response as get_few_shot_response
from prompting import get_few_shot_prompting_response as get_few_shot_response
from testcase_generation import *

print("-----------TestCaseGenerator----------")

FRD_input = '''     
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
    delta_time_period = new_time_period - t0'''
    
auth_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjp7ImlkIjoiMTE0MTY3Mzc5MzMxNTY5NTEyNjM2In0sImlhdCI6MTcyNjY4ODc4MywiZXhwIjoxNzI3NzY4NzgzfQ.5Zs3Bqe4qmiYRtxEStQx_irYboksR9QNwundB77at0s"

edge_cases_only = True
limit = 500

edges = []
edge_labels = {}
namespace = {}
flowchart_data = []

retry_limit = 5
retry_count = 0

while retry_count < retry_limit:
    try:
        print("Generating Testcases... Please wait a minute...")
        dict_from_few_shot_prompting = get_few_shot_response(FRD=FRD_input, auth_token=auth_token)
        break
    except error as e:
        retry_count += 1
        print(f"An error occurred: {str(e)}")
        print(f"Retrying... Attempt {retry_count}/{retry_limit}")

if retry_count == retry_limit:
    print(f"Reached maximum retry limit ({retry_limit}). Please check for issues.")
else:
    print("Testcases generated successfully!")

# extracting the necessary details from few shot prompting api call 
variables = dict_from_few_shot_prompting['variables']
equations = dict_from_few_shot_prompting['equations']
ranges = dict_from_few_shot_prompting['ranges']
independant_variables = [var for var in ranges.keys()]

dependant_variables = process_dependant_variables(equations)
add_ranges_for_miscellaneous(variables, dependant_variables, independant_variables, ranges, streamlit=False)

# pre processing
pre_process_equations(equations)
add_pre_requisites(equations, dependant_variables, variables)
ranges = pre_process_ranges(ranges, edge_cases_only)
equations_dict = add_equation_conditions(equations)
print(dependant_variables)
dependant_variables = topological_sort(dependant_variables, equations_dict)
print(dependant_variables)

# Ḍisplaying dependant and independant variables
print(f"Independant variables: {', '.join(independant_variables)}")
print(f"Dependant variables: {', '.join(dependant_variables)}")

variables = independant_variables+dependant_variables
print(variables)

# Display Flowcharts
create_flow_graph(flowchart_data=flowchart_data,
                    directory_path='flowchart',
                    equations_dict=equations_dict) 
 
image_directory = "flowchart"
image_files = [f for f in os.listdir(image_directory) if f.endswith((".png", ".jpg", ".jpeg"))]
print("Flowcharts generated for better visualisation")
for no, image_file in enumerate(image_files):
    image_path = os.path.join(image_directory, image_file)

# creating data frame for the test cases
print(variables)
test_cases_df = pd.DataFrame(columns=variables)
independant_variable_combinations = list(itertools.islice(itertools.product(*ranges.values()), limit))
test_cases_df = initialize_data_frame(test_cases_df, independant_variable_combinations, ranges)
namespace = initialize_namespace()

# Filling the test cases dataframe and displaying it
test_cases_df = fill_dependant_variables(dependant_variables, equations_dict, test_cases_df, namespace)
print("Generated Test Cases:")
print(test_cases_df)

# Describe Data Frame
print(test_cases_df.describe(include = 'all').T)

# Display Statistics
statistics = get_statistics(dependant_variables, test_cases_df)
print("Statistics:")
print(statistics)

try:
    show_variable_trend(dependant_variables, test_cases_df, streamlit=False)
    identify_dependancy_trend(dependant_variables, equations_dict, test_cases_df, streamlit=False)
except:
    print('There was some error with rendering the variable trend')
        
initialize_edge_and_labels(edges, edge_labels, equations_dict)
create_graph_image(edges, edge_labels, streamlit=False) 

create_interactive_graph(edges, streamlit=False)
test_cases_df.to_csv('test_cases_generated.csv', index=False)
test_cases_df.to_html('test_cases_generated.html', index=False)

print("Test cases saved to test_cases_generated.csv")
print("Test cases saved to test_cases_generated.html")