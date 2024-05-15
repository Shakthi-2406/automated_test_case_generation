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
RANGE_END_TAG

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

EQUATION_FRD_OUTPUT = {
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
  ]}

output_condition = "Start your response with the json open brace '{' and end the response with '}'"

USER_PROMPT = """
Input Document: {document}.
"""


equation_example = """{"variables" : ['a', 'b',....],
"equations" : [{'equation' : 'b = 2*d', 'condition': None},
                {'equation':  'a = b+c', 'condition': 'b>0'},
                {'equation':  'a = b+c', 'condition': 'mode == "enabled" and speed > 120'}.... {}]
}"""

equation_input_frd = """
{
'Auto_drift' : (0, 1, 0.5), 
'operation' : ("fight", "no"),
'curvature' : (100, 102, 1)
}
_RANGE_END_TAG_

beta = 100
temp0 = 40 deg
If Auto_drift is 0 and operation is fight,
then beam = curvature + 5 + temp0
else,
beam = curvature * 5
"""

equation_output_frd = """
{
"variables": ['beta', 'temp0', 'Auto_drift', 'curvature', 'beam'],
"equations": [
{'equation': 'beta = 100', 'condition': None}, 
{'equation': 'temp0 = 40', 'condition': None}, 
{'equation': 'beam = curvature + 5 + temp', 'condition': 'Auto_drift == 0 and operation == "fight"'}, 
{'equation':  'beam = curvature * 5', 'condition': 'Auto_drift != 0 or operation != "fight"'} 
]
}
"""

EQUATION_SYS_PROMPT_TEMPLATE = """
You are a python equation generator.
You will be given a functional requirement document of a satellite.
The document will contain conditions and equations of technicalities related to satellites in natural language.
Your task is to analyse the given document.
- Identify ALL the variables present and ALL the equations present in the document.
- For the equations, all the conditions and equation should be a python executable command or line which can be executed by eval.
- Give equations in the order of their executions. Use None instead of null. Identify ALL the variables present.
- You will be given example input document and example output JSON.
You have to return the output in the following JSON format.
```json
{example}
```
Example input document: {input_frd} 
Example output json: {output_frd}
{output_condition}
"""

EQUATION_PROMPT_TEMPLATE = f"""
<s>[INST]\n<<SYS>>\n{EQUATION_SYS_PROMPT_TEMPLATE}\n<</SYS>>\n\n{USER_PROMPT}[/INST]\nOutput Json: 
"""




range_example = """{"ranges": {'d' : (2, 100, 3), 'mode' : ("on", "off"), 'temp' : (3, 7, 1), ... } }"""

range_input_frd = """
Auto_drift - [enable, disable]
operation - ["fight", "no"]
curvature - [100, 102, 1]
"""

range_output_frd = """
```json{
'Auto_drift' : ("enable", "disable"), 
'operation' : ("fight", "no"),
'curvature' : (100, 102, 1)
}```
"""

RANGE_SYS_PROMPT_TEMPLATE = """
You are a python json range generator.
You will be given a part of functional requirement document of a satellite.
The document will contain ranges or options of technicalities related to satellites in natural language.
Your task is to analyse the given document.
- Identify ALL the range values.
- ALL the range values should be ONLY BE CONSTANTS (RAW VALUES of string or int or float).
- Ignore the units like sec, m, deg if present.
- ALL values ranges should be assignable to a python variable without any error.
- You will be given example input document and example output JSON.
You have to return the output in the following JSON format.
```json
{example}
```
Example input document: {input_frd} 
Example output json: {output_frd}
{output_condition}
"""

RANGE_PROMPT_TEMPLATE = f"""
<s>[INST]\n<<SYS>>\n{RANGE_SYS_PROMPT_TEMPLATE}\n<</SYS>>\n\n{USER_PROMPT}[/INST]\nOutput Json: 
"""