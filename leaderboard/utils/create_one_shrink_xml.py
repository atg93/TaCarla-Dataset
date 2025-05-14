import xml.etree.ElementTree as ET
import os


def merge_selected_xml_files(directory, keyword, output_file):
    # Create a new root element
    combined_root = ET.Element("CombinedXML")

    # Get all XML files in the directory
    files = [file for file in os.listdir(directory) if file.endswith(".xml") and keyword in file.split('_')]

    for file in files:
        file_path = os.path.join(directory, file)
        # Parse the current XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Append the content of the current XML to the combined root
        combined_root.append(root)

    # Write the combined XML to the output file
    combined_tree = ET.ElementTree(combined_root)
    combined_tree.write(output_file, encoding="utf-8", xml_declaration=True)


# Directory containing XML files
data_type = "training"
if data_type == "validation":
    directory_path = "/workspace/tg22/remote-pycharm/leaderboard2.0/shrinked_dummy_val/"  # Update this path to your XML files directory
elif data_type == "training":
    directory_path = "/workspace/tg22/remote-pycharm/leaderboard2.0/shrinked_split_trigger_data/"  # Update this path to your XML files directory
assert os.path.exists(directory_path)

keyword_list = list(set([key.split('.')[0].split('_')[1] for key in os.listdir(directory_path)]))


os.makedirs("/workspace/tg22/remote-pycharm/leaderboard2.0/single_scenario_" + data_type + "_xml/", exist_ok=True)

files = os.listdir(directory_path)
for keyword in keyword_list:
    # Keyword to filter XML files
    #keyword = "Accident"
    #keyword = "ConstructionObstacle"

    # Output file
    output_xml = "/workspace/tg22/remote-pycharm/leaderboard2.0/single_scenario_" + data_type + "_xml/merged_" + keyword + ".xml"

    # Merge XML files
    merge_selected_xml_files(directory_path, keyword, output_xml)
    print(f"Merged XML saved to {output_xml}")
