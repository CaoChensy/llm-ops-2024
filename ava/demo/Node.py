import os
import glob
import json
import xmltodict

paths = glob.glob("demo/dataset/multi/*/nodetree.xml")
print(paths)
output_path = "demo/nodes/"
os.makedirs(output_path, exist_ok=True)

for path in paths:
    with open(path, "r", encoding='utf-8') as f:
        xml_content = f.read()
    data_dict = xmltodict.parse(xml_content)

    current_dir = path.split('\\')[-2]
    json.dump(data_dict, open(output_path + f"{current_dir}_node.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=4)
