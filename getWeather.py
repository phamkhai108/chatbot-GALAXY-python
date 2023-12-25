import requests
import spacy

DEFAULT_API_KEY = "a686c1bf4b5fcf9fbeb5c114dc9f382e"

def weather(text):

    nlp = spacy.load("en_core_web_sm")


    doc = nlp(text)


    cities = []
    for ent in doc.ents:
        if ent.label_ == "GPE":  
            cities.append(ent.text)

    if not cities:
        return weather_api('Hanoi')
    else:
        return weather_api(cities[0])

def weather_api(city, api_key=DEFAULT_API_KEY):
    base_url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {
        'q': f"{city}",
        'limit': 1,
        'appid': api_key
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()

        if response.status_code == 200 and data:
            latitude = data[0]['lat']
            longitude = data[0]['lon']
            
            weather_result = get_weather(city, latitude, longitude, api_key)
            return weather_result
        else:
            return "Không thể lấy dữ liệu thời tiết. Vui lòng thử lại sau."
    
    except requests.RequestException:
        return "Không thể lấy dữ liệu thời tiết. Vui lòng thử lại sau."

def get_weather(city, lat, lon, api_key=DEFAULT_API_KEY):
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()

        if response.status_code == 200:
            temperature = data['main']['temp']
            return f"Nhiệt độ hiện tại tại {city}: {temperature} °C"
        else:
            return "Không thể lấy dữ liệu thời tiết. Vui lòng thử lại sau."
    
    except requests.RequestException:
        return "Không thể lấy dữ liệu thời tiết. Vui lòng thử lại sau."


