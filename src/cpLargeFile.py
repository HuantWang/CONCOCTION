import os
import tarfile
import shutil
import sys

def extract_tar_gz(tar_gz_file, destination_dir):
    with tarfile.open(tar_gz_file, 'r:gz') as tar:
        tar.extractall(destination_dir)

def copy_files(src_dir, dest_dir):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_path = os.path.join(root, file)
            relative_path = os.path.relpath(src_path, src_dir+"/CONCOCTION_largeFile")

            dest_path = os.path.join(dest_dir, relative_path)

            # os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            print(dest_path)
            print(src_path)
            shutil.copy2(src_path, dest_path)


# compressed_file = "/CONCOCTION_largeFile/CONCOCTION_largeFile.tar.gz"


# extracted_dir = "/CONCOCTION_largeFile/CONCOCTION_largeFile"

# destination_dir = "/CONCOCTION/src"

compressed_file=sys.argv[1]
extracted_dir=sys.argv[2]
destination_dir= os.path.dirname(os.path.realpath(__file__))


extract_tar_gz(compressed_file, extracted_dir)


copy_files(extracted_dir, destination_dir)
