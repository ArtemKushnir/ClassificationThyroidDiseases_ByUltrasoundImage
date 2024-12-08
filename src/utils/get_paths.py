import os


def get_paths(folder_path: str) -> list[str]:
    return os.listdir(folder_path)


def get_paths_with_extension(folder_path: str, extension: str) -> list[str]:
    return [file_name for file_name in os.listdir(folder_path) if file_name.endswith(extension)]


def get_join_paths(folder_path, file_paths):
    return [os.path.join(folder_path, file_path) for file_path in file_paths]
