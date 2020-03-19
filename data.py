'''
data.py
'''
import tensorflow as tf
import json
import os

f = open('../config/config.json','r')
config = json.load(f)
f.close()
AUTOTUNE = tf.data.experimental.AUTOTUNE

def decode_img(img):
#     img = tf.io.read_file(file_path)
#     print(img)
    img = tf.image.decode_jpeg(img,channels = 3)
#     print(img)
    img = tf.image.convert_image_dtype(img,tf.float32)
#     print(img)
    if config['img_size']:
        img = tf.image.resize(img,[config['img_size'],config['img_size']])
    return img

def get_label(file_path):
    parts = tf.strings.split(file_path,os.path.sep)
    return 1 if str(parts[-2]).startswith('tumor_') else 0

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img,label

#we glob a dir and get all the image endswith tif
def GetDataset(dir_path,batch_size,shuffle_size = 10):
    f = tf.data.Dataset.list_files(os.path.join(dir_path,'*.%s'%config['fmt']))
    dataset = f.map(process_path,num_parallel_calls = AUTOTUNE)
    dataset.repeat()
    dataset = dataset.shuffle(shuffle_size)#
    dataset = dataset.batch(batch_size)
    #dataset = dataset.prefetch(buffer_size = AUTOTUNE)
    return dataset
def print_batch(dataset):
    data_batch,label_batch = next(iter(dataset))
    print(data_batch.numpy(),label_batch.numpy())

if __name__ == '__main__':
    dataset = GetDataset('../Dataset/',4)
    print_batch(dataset)
