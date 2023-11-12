import requests


s_date = '1402,8,1'
e_date = '1402,8,30'

data = {
    'start_time': s_date,
    'end_time': e_date
}

url = "http://127.0.0.1:8000/sale_predict/"

response = requests.get(url, json=data)

print(response.status_code)
print(response.json())
