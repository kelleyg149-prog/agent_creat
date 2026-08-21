import requests

url = "https://api.github.com"

response = requests.get(url)

data = response.json()

print(type(data))

print(data["current_user_url"])
print(data["repository_url"])