#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 08/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import zipfile
import io
import requests
import os

from conf import urls, VERBOSE


def extract_data(response, names):
    """
    Extract data from the response and generate tsv files.
    :param response: the downloaded file
    :param names: the names of the files contained in the dataset
    :return:
    """
    for name in names:
        data = []
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            for line in zip_ref.open(f"ml-1m/{name}.dat"):
                data.append(str(line, "utf-8").replace("::", "\t"))
        os.makedirs("data", exist_ok=True)
        with open(f"data/{name}.tsv", "w") as f:
            f.writelines(data)


def perform_loading():
    """
    Perform data loading
    :return:
    """
    print("download data" if VERBOSE else "")
    response = requests.get(urls["movielens1m"])
    print("data downloaded..." if VERBOSE else "")
    extract_data(response, ["ratings"])


