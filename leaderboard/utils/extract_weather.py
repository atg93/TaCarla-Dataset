from bs4 import BeautifulSoup
import os
import sys
import carla
import pickle

file_path = '/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/modified_data/routes_training.xml'
sys.path.append(file_path)

file_path = '/cta/users/tgorgulu/git/leaderboard2.0/leaderboard/modified_data/routes_training.xml'
sys.path.append(file_path)


with open(file_path, 'r') as file:
    xml_data = file.read()

# Parsing the XML data
soup = BeautifulSoup(xml_data, 'xml')

# Extracting <weathers>...</weathers> section
weathers = soup.find_all('weather')

# Converting the extracted section back to a string for display or further processing
#weathers_str = str(weathers_section)
weather_list = []
key_list = ['cloudiness', 'precipitation', 'precipitation_deposits', 'wetness', 'wind_intensity', 'sun_azimuth_angle', 'sun_altitude_angle', 'fog_density']
for index, weather in enumerate(weathers):
    element_dict = {}
    for key in key_list:
        element_dict.update({key:weather[key]})
    weather_list.append(frozenset(element_dict.items()))

predefined_list = [carla.WeatherParameters.ClearNoon,carla.WeatherParameters.ClearSunset,carla.WeatherParameters.CloudyNoon,
            carla.WeatherParameters.CloudySunset,carla.WeatherParameters.WetNoon,carla.WeatherParameters.WetSunset,
            carla.WeatherParameters.MidRainyNoon,carla.WeatherParameters.MidRainSunset,
            carla.WeatherParameters.WetCloudyNoon,carla.WeatherParameters.WetCloudySunset,
            carla.WeatherParameters.HardRainNoon,carla.WeatherParameters.HardRainSunset,
            carla.WeatherParameters.SoftRainNoon,carla.WeatherParameters.SoftRainSunset,
            carla.WeatherParameters.ClearNight,carla.WeatherParameters.CloudyNight,carla.WeatherParameters.WetNight,
            carla.WeatherParameters.MidRainyNight,carla.WeatherParameters.HardRainNight,
            carla.WeatherParameters.SoftRainNight]

for index, pre_weather in enumerate(predefined_list):
    new_weather_dict = {}
    for key in key_list:
        new_weather_dict.update({key:getattr(pre_weather,key)})
    weather_list.append(frozenset(element_dict.items()))
    asd = 0
weather_set = set(weather_list)
print(weather_set)
print(len(weather_set))
for weather in weather_set:
    weather_dict = dict(weather)
    """new_weather_class = carla.WeatherParameters(cloudiness=float(weather_dict['cloudiness']),
                            precipitation=float(weather_dict['precipitation']),
                            precipitation_deposits=float(weather_dict['precipitation_deposits']),
                            wind_intensity=float(weather_dict['wind_intensity']),
                            sun_azimuth_angle=float(weather_dict['sun_azimuth_angle']),
                            sun_altitude_angle=float(weather_dict['sun_altitude_angle']),
                            fog_density=float(weather_dict['fog_density']),
                            wetness=float(weather_dict['wetness']))"""

    #with open('/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/weather_class.pickle', 'ab') as file:
    #    pickle.dump(weather_dict, file)

asd = 0