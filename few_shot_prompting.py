import requests
import json
import pprint
import copy
from config import *

model_name = "openai/gpt-4-1106-preview"
model_name = "anthropic/claude-instant-1"

url = 'https://api.getknit.ai/v1/router/run'
headers = {
    'x-auth-token': "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjp7ImlkIjoiMTE0MTY3Mzc5MzMxNTY5NTEyNjM2In0sImlhdCI6MTcyNjY4ODc4MywiZXhwIjoxNzI3NzY4NzgzfQ.5Zs3Bqe4qmiYRtxEStQx_irYboksR9QNwundB77at0s",
    'Content-Type': 'application/json'
}

ranges_system_prompt = '''
You are a python json range generator.
You will be given a part of functional requirement document of a satellite.
Each line of the document has the following structure - variable name - [start value, end value, step value (optional)]
Your task is to analyse the given document.
- Identify ALL the range values.
- ALL the range values should be ONLY BE CONSTANTS (RAW VALUES of string or int or float).
- Ignore the units like sec, m, deg if present.
- ALL values ranges should be assignable to a python variable without any error.
- You will be given example input document and example output JSON.
- The key-value pairs of the output json should be 'variable_name':tuple(start, end, step(optional))
You have to return the output in the following JSON format.
    Example 1:
    Input document: "
    Auto_drift - ["enable", "disable"]
    operation - ["fight", "no"]
    curvature - [100, 102, 1]
    "

    Output:
      ```json
      {
                'Auto_drift' : ("enable", "disable"), 
                'operation' : ("fight", "no"),
                'curvature' : (100, 102, 1)
      }```

    Example 2:
    Input document: "
    speed - [50, 100, 20]
    acceleration - [0, 5, 2] 
    "
    
    Output:
    ```json{
        "speed": [50, 100, 20],
        "acceleration": [0, 5, 2]
      }
    ```

    Example 3:
    Input Document:
    "
    temperature - [-10, 50, 10]
    humidity - [30, 70, 8]
    "
    Output:
    ```json{
        "temperature": [-10, 50, 10],
        "humidity": [30, 70, 8]
      }```
'''


range_data = {
    "messages": [
        {
            "role": "system",
            "content": ranges_system_prompt
        },
        {
            "role": "user",
            "content": '''Now generate output for the input document: {}'''
        }
    ],
    "model": {
	    "name": model_name
    },
    "variables": []
}

equations_system_prompt = '''
You are a python equation generator.
You will be given a functional requirement document of a satellite.
The document will contain conditions and equations of technicalities related to satellites in natural language.
Your task is to analyse the given document.
- Identify ALL the variables present and ALL the equations present in the document.
- For the equations, all the conditions and equation should be a python executable command or line which can be executed by eval.
- Give equations in the order of their executions. Use None instead of null. Identify ALL the variables present.
- You will be given example input document and example output JSON.
You have to return the output in the following JSON format.

Example 1:
                Input document: "
                Auto_drift - [0, 1, 0.5]
                operation - ["fight", "no"]
                curvature - [100, 102, 1]
                _RANGE_END_TAG_

                beta = 100
                temp0 = 40 deg
                If Auto_drift is 0,
                then beam = curvature + 5 + temp
                else,
                beam = curvature * 5
                "

                Output:
                {
                "variables" : ['beta', 'temp0', 'Auto_drift', 'curvature', 'beam'],
                "equations" : [
                           {'equation' : 'beta = 100', 'condition': None}, 
                           {'equation' : 'temp0 = 40', 'condition': None}, 
                           {'equation' : 'beam = curvature + 5 + temp', 'condition': 'Auto_drift == 0 and operation == "fight"'}, 
                           {'equation':  'beam = curvature * 5', 'condition': 'Auto_drift != 0'} 
                 ]
                 }

                Example 2:
                Input document: "
                speed - [50, 100, 20]
                acceleration - [0, 5, 2] 
                _RANGE_END_TAG_
                
                if speed > 75, 
                then distance = speed * time
                else, 
                distance = 0.5 * acceleration * time^2
                "
                Output:
                {
                  "variables": ["speed", "acceleration", "distance", "time"],
                  "equations": [
                    {"equation": "distance = speed * time", "condition": "speed > 75"},
                    {"equation": "distance = 0.5 * acceleration * time^2", "condition": "speed <= 75"}
                  ]
                }

                Example 3:
                Input Document:
                "
                temperature - [-10, 50, 10]
                humidity - [30, 70, 8]
                
                _RANGE_END_TAG_
                base_hod  = temperature + 3
                if temperature < 0 and humidity > 50,
                then weather = "Cold and Humid"
                else,
                weather = "Moderate"
                "
                Output:
                {
                  "variables": ["temperature", "humidity", "weather"],
                  "equations": [
                    {"equation": 'base_hod  = temperature + 3', "condition": None},
                    {"equation": 'weather = "Cold and Humid"', "condition": "temperature < 0 and humidity > 50"},
                    {"equation": 'weather = "Moderate"', "condition": "not (temperature < 0 and humidity > 50)"}
                  ]
                }
'''

data = {
    "messages": [
        {
            "role": "system",
            "content": equations_system_prompt
        },
        {
            "role": "user",
            "content": '''Please Generate output for the input document: {}'''
        }
    ],
    "model": {
	    "name": model_name
    },
    "variables": []
}

def pre_process(FRD):
    to_remove = ['\n','\\','"',"'",'{','}']
    for char in to_remove:
        FRD = FRD.replace(char, ' ')
    return FRD

def get_few_shot_prompting_response(
  FRD,
  auth_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjp7ImlkIjoiMTE0MTY3Mzc5MzMxNTY5NTEyNjM2In0sImlhdCI6MTcyNjY4ODc4MywiZXhwIjoxNzI3NzY4NzgzfQ.5Zs3Bqe4qmiYRtxEStQx_irYboksR9QNwundB77at0s"
  ):
    global url, headers, data, range_data
    
    part1 = FRD[:FRD.find('_RANGE_END_TAG_')]
    
    data_copy = copy.deepcopy(data)
    range_data_copy = copy.deepcopy(range_data)
        
    # FRD = pre_process(FRD)
    headers['x-auth-token'] = auth_token
    data_copy['messages'][1]['content'] = data_copy['messages'][1]['content'].format(FRD)
    
    response = requests.post(url, headers=headers, json=data_copy)
    response_dict = response.json()
    print(response_dict)
    dict_string = response_dict['responseText'][response_dict['responseText'].find('{'):response_dict['responseText'].rfind('}')+1]

    few_shot_response = eval(dict_string)
    
    if FRD.find('_RANGE_END_TAG_') != -1:
      range_data_copy['messages'][1]['content'] = range_data_copy['messages'][1]['content'].format(part1)
      response = requests.post(url, headers=headers, json=range_data_copy)
      response_dict = response.json()
      dict_string = response_dict['responseText'][response_dict['responseText'].find('{'):response_dict['responseText'].rfind('}')+1]

      ranges_few_shot_response = eval(dict_string)

      few_shot_response['ranges'] = ranges_few_shot_response
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
    