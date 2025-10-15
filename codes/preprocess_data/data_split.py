import os
import json
# ========================== params ====================================
base_path = './AASCE_rawdata/boostnet_labeldata/' # your data path
target_path = './AASCE_processed/'
val_images_path = './AASCE_processed/val_images.json'
# ======================================================================

val_images = json.load(open(val_images_path, 'r'))
val = list(map(lambda x: x.split('/')[-1], val_images))
# print(f'Number of validation images: {len(val_images)}')
# print(val)

test_images = json.load(open(val_images_path.replace('val','test'), 'r'))
test = list(map(lambda x: x.split('/')[-1], test_images))



for dirpath, dirnames, filenames in os.walk(base_path + 'data/'):
# for dirpath, dirnames, filenames in os.walk(base_path + 'labels/'):
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # get image name
            if filename in test:
                full_image_path = os.path.join(dirpath, filename)
                # move from training to test
                # os.rename(full_image_path, os.path.join(target_path, 'test', filename))
                # print(f'Move {full_image_path} to {full_image_path.replace("training","test")}')
                test_path = full_image_path.replace('training','test')
                os.makedirs(os.path.dirname(test_path), exist_ok=True)
                os.rename(full_image_path, test_path)

                full_image_path = full_image_path.replace('/data/','/labels/')+'.mat'
                test_path = test_path.replace('/data/','/labels/')+'.mat'
                os.makedirs(os.path.dirname(test_path), exist_ok=True)
                os.rename(full_image_path, test_path)

            
