import requests
import json
import pprint
import copy


url = 'https://api.getknit.ai/v1/router/run'
headers = {
    'x-auth-token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjp7ImlkIjoiMTAwNzI1NTA4NDU2MTk5OTc4ODAzIn0sImlhdCI6MTcwNjQ3MTQzMCwiZXhwIjoxNzA3NTUxNDMwfQ.7e8C-gU0fsaOH6o5Pno5xQILfljYJM_Wd6bb-OhUCWE',
    'Content-Type': 'application/json'
}

range_data = {
    "messages": [
        {
            "role": "system",
            "content": '''You have to analyse the given Functional requirement document snippet and going to return me a json of this format 
{"ranges": {'d' : (2, 100, 3), 'mode' : ("on", "off"), 'temp' : (3, 7, 1), ... } }
You are going to analyse the document, identify ALL the range values should ONLY BE CONSTANTS (RAW VALUES of string or int or float).
Ignore the units like sec, m, deg if present. Give the output in the specified JSON format.'''
        },
        {
            "role": "assistant",
            "content": '''
                Example 1:
                Input document: "
                Auto_drift - [0, 1, 0.5]
                operation - ["fight", "no"]
                curvature - [100, 102, 1]
                "

                Output:
                {
                 "ranges": {
                           'Auto_drift' : (0, 1, 0.5), 
                           'operation' : ("fight", "no"),
                           'curvature' : (100, 102, 1)
                   }
                 }

                Example 2:
                Input document: "
                speed - [50, 100, 20]
                acceleration - [0, 5, 2] 
                "
                
                Output:
                {
                  "ranges": {
                    "speed": [50, 100, 20],
                    "acceleration": [0, 5, 2]
                  }
                }

                Example 3:
                Input Document:
                "
                temperature - [-10, 50, 10]
                humidity - [30, 70, 8]
                "
                Output:
                {
                  "ranges": {
                    "temperature": [-10, 50, 10],
                    "humidity": [30, 70, 8]
                  }
                }

            '''
        },
        {
            "role": "user",
            "content": '''Now generate output for the input document: {}'''
        }
    ],
    "model": {
	    "name": "openai/gpt-4-1106-preview"
    },
    "variables": []
}

data = {
    "messages": [
        {
            "role": "system",
            "content": '''You have to analyse the given Functional requirement document snippet and going to return me a json of this format 
{"variables" : ['a', 'b',....],
"equations" : [{'equation' : 'b = 2*d', 'condition': None}, {'equation':  'a = b+c', 'condition': 'b>0'} .... {}]}
You are going to analyse the document, identify ALL the variables present, ALL the equations present.
Give equations in the order of their executions. Use None instead of null.
Give the output in the specified JSON format.'''
        },
        {
            "role": "assistant",
            "content": '''
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
        },
        {
            "role": "user",
            "content": '''Now generate output for the input document: {}'''
        }
    ],
    "model": {
	    "name": "openai/gpt-4-1106-preview"
    },
    "variables": []
}

def pre_process(FRD):
    to_remove = ['\n','\\','"',"'",'{','}']
    for char in to_remove:
        FRD = FRD.replace(char, ' ')
    return FRD

def get_few_shot_prompting_response(FRD):
    global url, headers, data, range_data
    
    part1 = FRD[:FRD.find('_RANGE_END_TAG_')]
    
    data_copy = copy.deepcopy(data)
    range_data_copy = copy.deepcopy(range_data)
        
    # FRD = pre_process(FRD)
    data_copy['messages'][2]['content'] = data_copy['messages'][2]['content'].format(FRD)
    range_data_copy['messages'][2]['content'] = range_data_copy['messages'][2]['content'].format(part1)
    
    response = requests.post(url, headers=headers, json=data_copy)
    response_dict = response.json()
    dict_string = response_dict['responseText'][response_dict['responseText'].find('{'):response_dict['responseText'].rfind('}')+1]

    few_shot_response = eval(dict_string)
    
    response = requests.post(url, headers=headers, json=range_data_copy)
    response_dict = response.json()
    dict_string = response_dict['responseText'][response_dict['responseText'].find('{'):response_dict['responseText'].rfind('}')+1]
    
    ranges_few_shot_response = eval(dict_string)

    few_shot_response.update(ranges_few_shot_response)
    
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
    