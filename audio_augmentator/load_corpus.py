from urllib.parse import urlencode
from zipfile import BadZipFile, ZipFile
from io import BytesIO
import os
import requests
from tqdm import tqdm


def load_dataset_from_yandex_disk(load_url,
                                  dataset_path,
                                  base_url="https://cloud-api.yandex.net/v1/disk/public/resources/download?",
                                  dataset_name=""):
    if dataset_name == "":
        dataset_name = load_url
    final_url = base_url + urlencode(dict(public_key=load_url))
    response = requests.get(final_url)
    download_url = response.json()['href']

    print(f"Start downloading corpus {load_url}")
    response = requests.get(download_url, stream=True)
    total = int(response.headers.get('content-length', 0))

    if response.ok:
        try:
            bytes_input = BytesIO()

            with tqdm(desc=dataset_name, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
                for data in response.iter_content(chunk_size=1024 * 1024):
                    size = bytes_input.write(data)
                    bar.update(size)

            z = ZipFile(bytes_input)
            z.extractall(dataset_path)
            full_path = os.path.abspath(dataset_path)
            print(f"Dataset {dataset_name} successfully loaded and saved to '{full_path}. "
                  f"Use it as noises_dataset= argument'")
            return True
        except BadZipFile as ex:
            print('Dataset downloading error: {}'.format(ex))
            return False


datasets_info = {
    "noises_corpus": {
        "link": "https://disk.yandex.ru/d/z24vJQQiuSZO4w",
        "path": "./noises_dataset",
        "checkpoint_name": "dataset_dict.json"
    },
}

for corpus_name, dataset_params in datasets_info.items():
    if not os.path.isdir(os.path.join(dataset_params["path"], dataset_params["checkpoint_name"])):
        load_dataset_from_yandex_disk(load_url=dataset_params["link"],
                                      dataset_path=dataset_params["path"],
                                      dataset_name=corpus_name)
