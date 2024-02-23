from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from config import *

llm = CTransformers(
            model='llama-2-7b.ggmlv3.q8_0.bin',
            model_type='llama',
            config={
                "max_new_tokens":512,
                "temperature":0.3,
                "context_length":2048
            }
        )


def get_equation_dict(frd_document):
    prompt = PromptTemplate(
        input_variables=["input_frd","output_frd","document","example","output_condition"], template=EQUATION_PROMPT_TEMPLATE
    )
    formatted_prompt = prompt.format_prompt(
                     input_frd=equation_input_frd,
                     output_frd=equation_output_frd,
                     document=frd_document,
                     example=equation_example,
                     output_condition=output_condition)
    print(formatted_prompt)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    equation_dict_response = llm_chain.predict(
                     input_frd=equation_input_frd,
                     output_frd=equation_output_frd,
                     document=frd_document,
                     example=equation_example,
                     output_condition=output_condition)
    dict_string = equation_dict_response[equation_dict_response.find('{'):equation_dict_response.rfind('}')+1]
    return eval(dict_string)


def get_ranges_dict(frd_document):
    prompt = PromptTemplate(
        input_variables=["input_frd","output_frd","document","example","output_condition"], template=RANGE_PROMPT_TEMPLATE
    )
    formatted_prompt = prompt.format_prompt(
                     input_frd=range_input_frd,
                     output_frd=range_output_frd,
                     document=frd_document,
                     example=range_example,
                     output_condition=output_condition)
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    range_dict_response = llm_chain.predict(
                     input_frd=range_input_frd,
                     output_frd=range_output_frd,
                     document=frd_document,
                     example=range_example,
                     output_condition=output_condition)
    dict_string = range_dict_response[range_dict_response.find('{'):range_dict_response.rfind('}')+1]
    return eval(dict_string)


def get_few_shot_prompting_response(FRD):
    few_shot_response = get_equation_dict(FRD)
    part1 = FRD[:FRD.find('_RANGE_END_TAG_')]
    
    if FRD.find('_RANGE_END_TAG_') != -1:
      ranges_few_shot_response = get_ranges_dict(part1)
      few_shot_response.update(ranges_few_shot_response)
    else:
      few_shot_response['ranges'] = {}
      
    pprint.pprint(few_shot_response)
    return few_shot_response


if __name__ == "__main__":
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
    response = get_few_shot_prompting_response(FRD)
    print(response)