import six
import os
import pickle
import numpy as np
import tensorflow as tf

DATA_DIR = './images_all/images/'
OUTPUT_DIRECTORY = './tf_records/'
IMAGES_PER_RECORD_SHARD = 256
NAME = 'val'

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if isinstance(value, six.string_types):
    value = six.binary_type(value,'utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image, label, filename):
  example = tf.train.Example(features=tf.train.Features(feature={
          'image/encoded': _bytes_feature(image),
          'image/label': _int64_feature(label),
          'image/filename': _bytes_feature(filename)}))
  return example


def main():


  with open('./validation_labels_new.pkl','rb') as f:
          labels = pickle.load(f)
  filenames = list(labels.keys())
  num_files = len(filenames)
  num_shards = num_files//IMAGES_PER_RECORD_SHARD
  current_file = 0
  #print(num_files)
  
  for shard_num in range(num_shards):
          output_file = '%s-%.5d-of-%.5d' % (NAME, shard_num, num_shards)
          output_file = os.path.join(OUTPUT_DIRECTORY, output_file)
          with tf.python_io.TFRecordWriter(output_file) as writer:
                  for record_num in range(IMAGES_PER_RECORD_SHARD):               
                          with tf.gfile.FastGFile(os.path.join(DATA_DIR,filenames[current_file]),'rb') as f:
                                  image_data=f.read()     
                          label = labels[filenames[current_file]]
                          example = _convert_to_example(image_data, label,filenames[current_file])
                          print("===========================================================")
                          print("Filename = %s"%(filenames[current_file]))
                          print("Label Written = %s"%(label))
                          print("Pickle label  = %s"%(labels[filenames[current_file]]))
                          print("===========================================================")
                          writer.write(example.SerializeToString())
                          current_file+=1
          print('Finished writing shard %d of %d' % (shard_num, num_shards))                   
  print("Used %s files of %s"%(current_file, num_files)) 



if __name__ == '__main__':
        main()
