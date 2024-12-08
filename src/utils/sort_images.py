import re
import shutil
import xml.etree.ElementTree as ET

from src.utils.get_paths import get_join_paths, get_paths_with_extension

DEFAULT_FOLDER_PATH = "../../data/image_data/"
DEFAULT_PATH = "../../data/clear_and_cropped_image_data/"


def count_file_tirads(folder_path: str = DEFAULT_FOLDER_PATH) -> dict[str, int]:
    file_tirads_dict = {}

    xml_paths = get_paths_with_extension(folder_path, "xml")

    full_paths = get_join_paths(folder_path, xml_paths)

    for xml_path in full_paths:
        number = xml_path[19:-4]
        root = ET.parse(xml_path)
        tirads = root.find("tirads").text
        tirads = tirads if tirads is not None else "1"
        tirads = int(re.findall(r"\d+", tirads)[0])
        file_tirads_dict[number] = tirads

    return file_tirads_dict


def move_image(folder_path: str = DEFAULT_PATH) -> None:
    image_paths = get_paths_with_extension(folder_path, "jpg")

    full_paths = get_join_paths(folder_path, image_paths)

    for i, image_path in enumerate(full_paths):
        image_number = image_paths[i][8:-6]
        file_tirads_dict = count_file_tirads()
        tirads = file_tirads_dict[image_number]
        if tirads <= 3:
            shutil.move(image_path, folder_path + f"normal/{image_paths[i]}")
        else:
            shutil.move(image_path, folder_path + f"malignant/{image_paths[i]}")
