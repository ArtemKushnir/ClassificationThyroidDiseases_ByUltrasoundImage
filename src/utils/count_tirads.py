import re
import xml.etree.ElementTree as ET

from src.utils.get_paths import get_join_paths, get_paths_with_extension

DEFAULT_PATH = "../../data/image_data/"


def count_tirads(folder_path: str = DEFAULT_PATH) -> dict[str, int]:
    tirads_dict = {}

    xml_paths = get_paths_with_extension(folder_path, "xml")

    full_paths = get_join_paths(folder_path, xml_paths)

    for xml_path in full_paths:
        root = ET.parse(xml_path)
        tirads = root.find("tirads").text
        tirads = tirads if tirads is not None else "1"
        tirads = re.findall(r"\d+", tirads)[0]
        tirads_dict[tirads] = tirads_dict.get(tirads, 0) + 1
    return tirads_dict
