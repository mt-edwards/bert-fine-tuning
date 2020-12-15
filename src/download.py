# import dependencies
import wget
import os
import zipfile

# data source
url = "https://nyu-mll.github.io/CoLA/cola_public_1.1.zip"

# download data
if not os.path.exists("data/cola_public"):
    wget.download(url, "data/cola_public.zip")
    with zipfile.ZipFile("data/cola_public.zip") as zip_ref:
        zip_ref.extractall("data")
    os.remove("data/cola_public.zip")
