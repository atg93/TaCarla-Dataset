import pickle
import xml.etree.ElementTree as ET
import re


def prettify(element, indent='   '):
    """
    Generate a pretty-printed XML string for the Element.
    Elements will be indented with the given string.
    """
    original_string = ET.tostring(element, 'unicode')
    lines = original_string.splitlines()
    new_lines = []
    current_indent = ""
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("</"):
            current_indent = current_indent[:-len(indent)]
            new_lines.append(current_indent + stripped_line)
        elif stripped_line.startswith("<") and not stripped_line.startswith("<?xml"):
            new_lines.append(current_indent + stripped_line)
            if not (stripped_line.endswith("/>") or stripped_line.startswith("<?xml")):
                current_indent += indent
        else:
            new_lines.append(current_indent + stripped_line)

    # Adding empty line specifically after route closing tag as an example
    formatted_lines = []
    for line in new_lines:
        formatted_lines.append(line)
        if '</route>' in line:
            formatted_lines.append('')  # Add an empty line after each route closing tag

    return "\n".join(formatted_lines)



with open('/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/tl_locations.pickle', 'rb') as file:
    data = pickle.load(file)

location_list = []
for index in range(4):
    for sample in list(data.values()):
        try:
            location_list.append(sample[index])
        except:
            pass



# Create the root element
routes = ET.Element("routes")
adding_list = []
routes_data = []

for index in range(50):

    # Populate waypoints for this route
    waypoint_list = []
    for loc_number, loc in enumerate(location_list):
        x, y, z = loc[0], loc[1], loc[2]
        if [x, y, z] in adding_list:
            continue
        waypoint_list.append({"x": str(x), "y": str(y), "z": str(z)})
        adding_list.append([x, y, z])
        if len(waypoint_list) >= 20:
            break

    route = {
        "id": str(index),
        "town": "Town12",
        "waypoints": waypoint_list}

    routes_data.append(route)

weather_elements = [
    '<weather route_percentage="0" cloudiness="5.0" precipitation="0.0" precipitation_deposits="50.0" wetness="0.0" wind_intensity="10.0" sun_azimuth_angle="-1.0" sun_altitude_angle="15.0" fog_density="10.0"/>',
    '<weather route_percentage="100" cloudiness="5.0" precipitation="0.0" precipitation_deposits="0.0" wetness="0.0" wind_intensity="10.0" sun_azimuth_angle="-1.0" sun_altitude_angle="45.0" fog_density="2.0"/>'
]
# Start of the XML file as a string
xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n'

for route in routes_data:
    # Add the route element
    xml_content += f'   <route id="{route["id"]}" town="{route["town"]}">\n'
    xml_content += '      <!-- Dummy tl route -->\n'

    xml_content += '      <weathers>\n'
    for weather in weather_elements:
        # Add each weather element, indented appropriately
        xml_content += f'         {weather}\n'
    xml_content += '      </weathers>\n'

    xml_content += '      <waypoints>\n'
    for wp in route["waypoints"]:
        # Add each waypoint
        xml_content += f'         <position x="{wp["x"]}" y="{wp["y"]}" z="{wp["z"]}" />\n'
    xml_content += '      </waypoints>\n'
    xml_content += '      <scenarios>\n      </scenarios>\n'
    xml_content += '   </route>\n\n'  # Note the extra newline for spacing between routes

xml_content += '</routes>'

# Write the manually constructed XML to a file
with open("example_routes_with_spacing.xml", "w", encoding="utf-8") as file:
    file.write(xml_content)



asd = 0