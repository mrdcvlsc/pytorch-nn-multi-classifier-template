download_dataset:
	python3 dataloader.py

serve_tensorboard:
	tensorboard --logdir=runs

install_kaggle_windows:
	pip install kaggle

install_kaggle_linux_mac:
	pip install --user kaggle

download_cats_vs_dogs_dataset:
	kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset

# remove broken images:
# - Cat/666.jpg
# - Dog/11702.jpg