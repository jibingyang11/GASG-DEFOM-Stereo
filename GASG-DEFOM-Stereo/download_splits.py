import urllib.request
import os

os.makedirs("splits", exist_ok=True)
url = "https://raw.githubusercontent.com/nianticlabs/monodepth2/master/splits/eigen_benchmark/test_files.txt"
urllib.request.urlretrieve(url, "splits/eigen_test_files.txt")