import time
import streamlit as st
import pandas as pd
from few_shot_prompting import get_few_shot_prompting_response as get_few_shot_response
from testcase_generation import *
import streamlit_ext as ste

st.set_page_config(
    page_title="TestCaseGenerator",
    page_icon="🚀"
)

st.markdown("""
<style>
div[data-baseweb="input-container"] textarea {
    resize: vertical !important;
    overflow: hidden !important;
    height: 100px !important; /* Set an initial height */
    spellcheck: false !important;
}
</style>
""", unsafe_allow_html=True)

# Streamlit UI components
st.title("Test Case Generator")

# Create placeholders for user input
FRD_input = st.text_area('''Enter the FRD (Functional Requirement Document):''', '''     Auto_Drift_Estimate - [enabled, not_enabled]
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
    delta_time_period = new_time_period - t0''',placeholder="Copy and paste FRD here...", height=300)

edge_cases_only = ste.checkbox("Edge Cases Only", True)
limit = ste.slider("Test cases count", min_value=50, max_value=1000, value=50)

if st.button("Generate Test Cases"):
    edges = []
    edge_labels = {}
    namespace = {}
    flowchart_data = []
    
    retry_limit = 5
    retry_count = 0
    
    while retry_count < retry_limit:
        try:
            with st.spinner("Generating Testcases... Please wait a minute..."):
                dict_from_few_shot_prompting = get_few_shot_response(FRD=FRD_input)
            break
        except Exception as e:
            retry_count += 1
            st.write(f"An error occurred: {str(e)}")
            st.write(f"Retrying... Attempt {retry_count}/{retry_limit}")
    
    if retry_count == retry_limit:
        st.write(f"Reached maximum retry limit ({retry_limit}). Please check for issues.")
        break
    else:
        success = st.success("Testcases generated successfully!")
        time.sleep(2)
        success.empty()
    
    # extracting the necessary details from few shot prompting api call 
    variables = dict_from_few_shot_prompting['variables']
    equations = dict_from_few_shot_prompting['equations']
    ranges = dict_from_few_shot_prompting['ranges']
    independant_variables = [var for var in ranges.keys()]
    
    dependant_variables = process_dependant_variables(equations)
    add_ranges_for_miscellaneous(variables, dependant_variables, independant_variables, ranges, streamlit=True)

    # pre processing
    pre_process_equations(equations)
    add_pre_requisites(equations, dependant_variables, variables)
    ranges = pre_process_ranges(ranges, edge_cases_only)
    equations_dict = add_equation_conditions(equations)
    print(dependant_variables)
    dependant_variables = topological_sort(dependant_variables, equations_dict)
    print(dependant_variables)

    # Ḍisplaying dependant and independant variables
    st.write(f"Independant variables: {', '.join(independant_variables)}")
    st.write(f"Dependant variables: {', '.join(dependant_variables)}")

    variables = independant_variables+dependant_variables
    print(variables)
    
    # Display Flowcharts
    create_flow_graph(flowchart_data=flowchart_data,
                      directory_path='flowchart',
                      equations_dict=equations_dict)  
    image_directory = "flowchart"
    image_files = [f for f in os.listdir(image_directory) if f.endswith((".png", ".jpg", ".jpeg"))]
    st.title("Flowcharts for better visualisation")
    for no, image_file in enumerate(image_files):
        image_path = os.path.join(image_directory, image_file)
        st.image(image_path, caption=f'flowchart-{no}', use_column_width=True)


    # creating data frame for the test cases
    print(variables)
    test_cases_df = pd.DataFrame(columns=variables)
    independant_variable_combinations = list(itertools.islice(itertools.product(*ranges.values()), limit))
    test_cases_df = initialize_data_frame(test_cases_df, independant_variable_combinations, ranges)
    namespace = initialize_namespace()
    

    # Filling the test cases dataframe and displaying it
    test_cases_df = fill_dependant_variables(dependant_variables, equations_dict, test_cases_df, namespace)
    st.header("Generated Test Cases:")
    st.dataframe(test_cases_df, height=800)
    
    # Describe Data Frame
    st.dataframe(test_cases_df.describe(include = 'all').T, width=800)
    
    # Display Statistics
    statistics = get_styled_html_statistics(dependant_variables, test_cases_df)
    st.header("Statistics:")
    st.markdown(statistics, unsafe_allow_html=True)
    
    
    # # Download button for excel
    # st.download_button(
    #     label="Download Test Cases as Excel",
    #     data=test_cases_df.to_excel(index=False),
    #     file_name='test_cases_generated.xlsx',
    #     key="download_html"
    # )  
    
    try:
        show_variable_trend(dependant_variables, test_cases_df, streamlit=True)
        identify_dependancy_trend(dependant_variables, equations_dict, test_cases_df, streamlit=True)
    except:
        st.write('There was some error with rendering the variable trend')
            
    initialize_edge_and_labels(edges, edge_labels, equations_dict)
    create_graph_image(edges, edge_labels, streamlit=True) 
    
    create_interactive_graph(edges, streamlit=True)
    
    # Download button for CSV format
    ste.download_button(
        label="Download Test Cases as CSV",
        data=test_cases_df.to_csv(index=False).encode('utf-8'),
        file_name='test_cases_generated.csv',
    )
    
    # Download button for HTML format
    ste.download_button(
        label="Download Test Cases as HTML",
        data=test_cases_df.to_html(index=False),
        file_name='test_cases_generated.html',
    )

