# script.py
import shutil
from google.colab import drive
import os

def mount_and_process():
    if not os.path.ismount('/content/drive'):
      # Mount Google Drive
      print("Mounting Google Drive...")
      drive.mount('/content/drive')

      # Copy files from Google Drive to the Colab local workspace
      print("Copying files from Google Drive...")
      shutil.copy("/content/drive/MyDrive/ucsd.zip", "/content/ucsd.zip")
      shutil.copy("/content/drive/MyDrive/ori.zip", "/content/ori.zip")
      shutil.copy("/content/drive/MyDrive/DenseNet_224_up_uncrop.h5", "/content/DenseNet_224_up_uncrop.h5")

      # Extract the contents of the zip files
      print("Extracting zip files...")
      get_ipython().system('7z x /content/ucsd.zip -o/content/ucsd')
      get_ipython().system('7z x /content/ori.zip -o/content/ori')
      print("Files extracted successfully.")
    else:
      print("Google Drive is already mounted and files exacted.")
