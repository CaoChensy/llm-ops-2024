import os
import re
import glob
import json
import copy
import numpy as np
from tqdm import tqdm


def reveal_node(nod, input_path):
    lst = []

    def reveal(nod, last_name, lst):
        if last_name:
            names = "-".join([last_name, nod["@name"]])
        else:
            names = nod["@name"]
        #         print(names)
        tags = copy.deepcopy(nod)
        tags.pop("node", None)
        try:
            c_path = re.sub("\\\\topics", "", nod["url"])
            c_path = re.sub(".html", ".txt", c_path)
            c_path = re.sub(".htm", ".txt", c_path)
            current_path = os.path.join(input_path, c_path)
            print(current_path)
            with open(current_path, "r", encoding="utf-8") as f:
                content = f.read()
            f.close()
        except Exception as e:
            print(f"error_msg:{e}")
            print(nod["@url"])

        lst.append([names, content, tags])

        if "node" in nod:
            if isinstance(nod["node"], list):
                for n in nod["node"]:
                    reveal(n, names, lst)
            else:
                reveal(nod["node"], names, lst)
        else:
            return None

    init_name = re.split("\\\\|/", input_path)[-2].split("_")[0]
    reveal(nod, init_name, lst)

    return lst


def chunk_text(all_lst, max_lens=2000):
    chunked_all_lst = []
    for lst in tqdm(all_lst, desc="文本切分"):
        text = lst[1]
        if len(text) > max_lens:
            txt_lst = [t for t in re.split("\n", text) if not bool(re.compile(r"^\s*$").match(t))]

            # 判断每列开头是中文和数字在所有列的占比（排除代码和类似的干扰）
            num_ch = [t for t in txt_lst if re.match("[\u4e00-\u9fff]|\d", t.strip())]
            num_ch_ratio = len(num_ch) / len(txt_lst)

            # 判断文本长度的差异性
            txt_lens = [len(t) for t in txt_lst]
            median_lens = np.median(txt_lens)
            differs = abs(np.mean([tl - median_lens for tl in txt_lens]))

            if differs < 10 and num_ch_ratio > 0.7:
                chunks = txt_lst[1:]
            else:
                block = []
                chunks = []
                for txt in txt_lst[1:]:
                    if len("\n\n".join(block+[txt])) > max_lens:
                        chunks.append("\n\n".join(block))
                        block = [txt]

                    elif len(txt) < median_lens and re.match("[\u4e00-\u9fff]|\d", txt.strip()) and len(block) > 2:
                        chunks.append("\n\n".join(block))
                        block = [txt]

                    elif re.match("\d", txt.strip()) or re.findall("【\d】", txt.strip()) or bool(re.compile(r"^\s*$").match(txt)):
                        chunks.append("\n\n".join(block))
                        block = [txt]

                    else:
                        block.append(txt)
                chunks.append("\n\n".join(block))
            for chunk in chunks:
                chunked_all_lst.append([lst[0], chunk, lst[2]])
        else:
            chunked_all_lst.append(lst)

    return chunked_all_lst


if __name__ == "__main__":
    input_paths = glob.glob("demo/dataset/output/multi/*/")
    print(input_paths)
    node_paths = glob.glob("demo/nodes/*")
    full_text_path = "demo/documents/"
    output_path = "demo/chunked/"
    max_lens = 8000

    for idx, node_path in enumerate(node_paths):

        # 加载节点
        nodes = json.load(open(node_path, encoding="utf-8"))
        save_path = re.split("\\\\|/", node_path)[-1].split("_node.")[-2]

        root_nodes = nodes['nodes']['node']
        all_lst = []
        # 展开节点
        for node in tqdm(root_nodes, desc=f"展开{save_path}节点"):
            all_lst += reveal_node(node, input_paths[idx])

        # os.makedirs(full_text_path, exist_ok=True)
        # output = os.path.join(full_text_path, save_path + ".json")
        # json.dump(all_lst, open(output, "w", encoding="utf-8"), indent=4, ensure_ascii=False)

        chunked_text = chunk_text(all_lst, max_lens)

        os.makedirs(output_path, exist_ok=True)
        output = os.path.join(output_path, save_path + ".json")
        json.dump(chunked_text, open(output, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
