import urllib.request
import os

def main():
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    os.makedirs("./data/MNIST/raw", exist_ok=True)

    for f in files:
        path = f"./data/MNIST/raw/{f}"
        print(f"Downloading {f}...")
        urllib.request.urlretrieve(base_url + f, path)
        print(f"Downloaded {f}")

    print("Download complete. You can now start the training.")

if __name__ == "__main__":
    main()
