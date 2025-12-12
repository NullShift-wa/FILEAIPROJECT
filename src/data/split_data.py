import os 
import shutil 
import random 
father_dir = r"D:\AI_PROJECT\Assets\raw"
train_dir  = r"D:\AI_PROJECT\Assets\processed\train"
val_dir    = r"D:\AI_PROJECT\Assets\processed\val"
split = 0.8
random_seed = 42
os.makedirs(train_dir,exist_ok = True )
os.makedirs(val_dir,exist_ok = True )
random.seed(random_seed)
for i in os.listdir(father_dir):
    j = os.path.join(father_dir,i)
    if not os.path.isdir(j): 
        continue
    images = [f for f in os.listdir(j) if f.lower().endswith((".jpg",".jpeg",".png"))] 
    if not images:
        continue
    random.shuffle(images)
    split_index = int(len(images)*split)
    train_images = images[:split_index]
    val_images = images[split_index:]
    os.makedirs(os.path.join(train_dir,i),exist_ok = True)
    os.makedirs(os.path.join(val_dir,i),exist_ok = True)
    for ig in train_images:
        shutil.copy(os.path.join(j,ig),os.path.join(train_dir,i,ig))
    for ig in val_images:
        shutil.copy(os.path.join(j,ig),os.path.join(val_dir,i,ig))

print("Done") 