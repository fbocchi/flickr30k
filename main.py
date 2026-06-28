import csv
import random
import json
from collections import defaultdict

INPUT_FILE = "flickr30k-dataset/captions.txt"
OUTPUT_FILE = "captions_split.json"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

random.seed(42)

# 1. parsing CSV corretto
image_to_captions = defaultdict(list)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f)

    header = next(reader, None)  # salta "image,caption"

    for row in reader:
        if len(row) < 2:
            continue

        image = row[0].strip()
        caption = row[1].strip()

        image_to_captions[image].append(caption)

# 2. lista immagini
images = list(image_to_captions.keys())
random.shuffle(images)

n = len(images)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_images = images[:n_train]
val_images = images[n_train:n_train + n_val]
test_images = images[n_train + n_val:]

# 3. costruzione dataset
dataset = {"train": {}, "val": {}, "test": {}}

for img in train_images:
    dataset["train"][img] = image_to_captions[img]

for img in val_images:
    dataset["val"][img] = image_to_captions[img]

for img in test_images:
    dataset["test"][img] = image_to_captions[img]

# 4. salva JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("OK")
print(len(train_images), len(val_images), len(test_images))