import requests

url = "https://data.cdc.gov/resource/hn4x-zwk7.json?$limit=5"
response = requests.get(url)

print("Status code:", response.status_code)
print("Data:")
print(response.json())
