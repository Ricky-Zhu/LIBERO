import libero.libero.utils.download_utils as download_utils
from libero.libero import benchmark, get_libero_path, set_libero_default_path

download_dir = get_libero_path("datasets")
datasets = "libero_spatial" # Can specify "all", "libero_goal", "libero_spatial", "libero_object", "libero_100"

libero_datasets_exist = download_utils.check_libero_dataset(download_dir=download_dir)

if not libero_datasets_exist:
    download_utils.libero_dataset_download(download_dir=download_dir, datasets=datasets)