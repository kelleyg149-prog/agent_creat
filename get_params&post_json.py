import requests
import json

url="https://jsonplaceholder.typicode.com/posts"
header={"Content-Type": "application/json"}
data = {
    "membrane": "GO-01",
    "voltage": 3.0,
    "Cl_flux": 18.2
}
response=requests.post(url,json=data,headers=header)

print("状态",response.status_code)
print('url:',response.url)
result=response.json()
results=json.dumps(result,ensure_ascii=False,indent=4)
print(results)
print(result["membrane"])