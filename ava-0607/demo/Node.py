import glob
import json
import xmltodict

paths = glob.glob("demo/dataset/multi/*/nodetree.xml")

for path in paths:
    with open(path, "r", encoding='utf-8') as f:
        xml_content = f.read()
    data_dict = xmltodict.parse(xml_content)
    output_path = path.split('\\')[-2]
    json.dump(data_dict, open("demo/nodes/" + f"{output_path}_node.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=4)
