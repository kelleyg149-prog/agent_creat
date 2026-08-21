import json
print("开始")

json_text = '''
{
    "membrane": "GO-01",
    "voltage": 2.0,
    "Cl_flux": 15.6
}
'''
print(type(json_text))
data=json.loads(json_text)
print(type(data))
print(f"{data['membrane']},{data['voltage']},{data['Cl_flux']}")
data["voltage"]=3.0
data_new=json.dumps(data,ensure_ascii=False)
print(type(data_new))