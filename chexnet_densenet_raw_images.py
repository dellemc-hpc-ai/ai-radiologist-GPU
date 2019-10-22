""" 
This script runs cheXNet training, expects the data to be in the form of raw images. Please update paths to reflect images downloaded from the NIH dataset.  
"""

import tensorflow as tf
import os
import argparse
import pickle
import numpy as np
from time import time
from tensorflow.keras import backend as K
from tensorflow.keras.applications import DenseNet121, resnet50
from tensorflow.keras.utils import multi_gpu_model, get_file
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras import optimizers
from tensorflow.data import Dataset
from sklearn.metrics import roc_auc_score
import horovod.tensorflow.keras as hvd


# shuffle buffer for tf data
_SHUFFLE_BUFFER = 500

# actual number of training images for chexnet
_NUM_TRAINING_IMAGES = 77871

# actual number of validation images for chexnet
_NUM_VAL_IMAGES = 8653


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='./images_all/images/',
    help='The directory where the input data is stored.')

parser.add_argument(
    '--model_dir', type=str, default='./saved_weights/',
    help='The directory where the model will be stored.')

parser.add_argument(
    '--batch_size', type=int, default='16',
    help='Batch size for SGD')

parser.add_argument(
    '--image_size', type=int, default='256',
    help='Image size')

parser.add_argument(
    '--opt', type=str, default='adam',
    help='Optimizer to use (adam, sgd, rmsprop, adagrad, adadelta, adamax, nadam)')

parser.add_argument(
    '--momentum', type=float, default='0.0',
    help='Momentum rate for SGD optimizer')

parser.add_argument(
    '--nesterov', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
    help='Use Nesterov momentum for SGD optimizer')

parser.add_argument(
    '--lr', type=float, default='1e-3',
    help='Learning rate for optimizer')

parser.add_argument(
    '--epochs', type=int, default=15,
    help='Number of epochs to train')

# https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_preprocessing.py 
# use tensorflows bfloat16
parser.add_argument(
    '--use_float16', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
    help='Boolean to use bfloat16')

# A value between 0 and 1 that indicates what fraction of the
# available GPU memory to pre-allocate for each process.  1 means
# to pre-allocate all of the GPU memory, 0.5 means the process
# allocates ~50% of the available GPU memory. This is just a workaround for 
# CUDA_ERROR_OUT_OF_MEMORY: out of memory error.
parser.add_argument(
    '--gpu_memory_fraction', type=float, default=1.0,
    help='Describes how much of GPU capacity to be used')

parser.add_argument(
    '--parallel_calls', type=int, default=10,
    help='Number of workers to preprocess image'
)

parser.add_argument(
    '--datasets_private_threads', type=int, default=None,
    help='Number of private threads to tf data.'
)

parser.add_argument(
    '--tf_data_experimental_slack', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
    help='Turn on tf data experimental slack?.'
)

parser.add_argument(
    '--skip_eval', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True,
    help='Defines if evaluation must be skipped while training'
)

parser.add_argument(
    '--write_weights', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
    help='Defines whether to Write weights to disk'
)

parser.add_argument(
    '--enable_tensorboard', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
    help='Defines whether to enable logging to tensorboard.'
)


with open('./training_labels_new.pkl', 'rb') as f:
    # get all training elements as a dictionary - has filename:multi class label pairs.
    training_data = pickle.load(f)
# get all training files as a list.
training_files = list(training_data.keys())



with open('./validation_labels_new.pkl', 'rb') as f:
    # get all validation elements as a dictionary - has filename:multiclass label pairs.
    validation_data = pickle.load(f)
# get all validation files as a list.
validation_files = list(validation_data.keys())
    
    
# generate a dictionary that has all elements - training and validation files.
labels = dict(training_data.items())
labels.update(validation_data.items())



def get_filenames(is_training, data_dir):
   """Return filenames for dataset.""" 
   if is_training:    
       return [
             os.path.join(data_dir, filename) 
             for filename in training_files]
   else: 
       return [
             os.path.join(data_dir, filename) 
             for filename in validation_files]    


def get_labels(is_training):
   """Return labels for dataset.""" 
   if is_training:    
       return [
             labels[filename] 
             for filename in training_files]
   else: 
       return [
             labels[filename] 
             for filename in validation_files]    



def record_parser(filename, label, is_training, dtype):
    """ Parses and preprocesses an example proto containing a training example of an image."""


    image = tf.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([1024, 1024, 3])
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    image = tf.image.resize_images(image, [FLAGS.image_size, FLAGS.image_size])
    
    # randomly flip the image while training
    if is_training:
        image = tf.image.random_flip_left_right(image)

    image = image / 255.0
    image = tf.cast(image, dtype)
    print("Cast successful! Image dtype : %s " % (image.dtype))
    return image, label


def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=1,
                           dtype=tf.float16,
                           datasets_num_private_threads=None,
                           num_parallel_batches=1,
                           ):
    """Given a Dataset with raw records, return an iterator over the records.

        Args:
          dataset: A Dataset representing raw records
          is_training: A boolean denoting whether the input is for training.
          batch_size: The number of samples per batch.
          shuffle_buffer: The buffer size to use when shuffling records. A larger
            value results in better randomness, but smaller values reduce startup
            time and use less memory.
          parse_record_fn: A function that takes a raw record and returns the
            corresponding (image, label) pair.
          num_epochs: The number of epochs to repeat the dataset.
          dtype: Data type to use for images/features.
          datasets_num_private_threads: Number of threads for a private
            threadpool created for all datasets computation.
          num_parallel_batches: Number of parallel batches for tf.data.

        Returns:
          Dataset of (image, label) pairs ready for iteration.
    """
    # Prefetches a batch at a time to smooth out the time taken to load input
    # files for shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # Shuffles records before repeating to respect epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        steps_per_epoch =_NUM_TRAINING_IMAGES//batch_size

    # take care while lambda x: (str(x).lower() in ['true','1', 'yes'])building validation dataset
    else:
        steps_per_epoch =_NUM_VAL_IMAGES//batch_size	 

    # Repeats the dataset for the number of epochs to train. 
    # Multiplying by the factor steps_per_epoc // hvd.size() to 
    # prevent running out of data problem.
    dataset = dataset.repeat(num_epochs*steps_per_epoch//hvd.size())
    
    # Parses the raw records into images and labels.
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda filename, label: parse_record_fn(filename, label,  is_training, dtype),
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=True
            ))

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    # Defines a specific size thread pool for tf.data operations.
    if datasets_num_private_threads:
        tf.logging.info('datasets_num_private_threads: %s',
                        datasets_num_private_threads)
        dataset = threadpool.override_threadpool(
            dataset,
            threadpool.PrivateThreadPool(
                datasets_num_private_threads,
                display_name='input_pipeline_thread_pool'))

    return dataset


def input_fn(is_training,
             data_dir,
             batch_size,
             dtype,
             num_epochs=1,
             datasets_num_private_threads=None,
             num_parallel_batches=5,
             ):
    """Input function which provides batches for train or eval.
   
    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.
      dtype: Data type to use for images/features
      datasets_num_private_threads: Number of private threads for tf.data.
      num_parallel_batches: Number of parallel batches for tf.data.
      parse_record_fn: Function to use for parsing the records.
      
    Returns:
      A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    labels = get_labels(is_training)
    dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
    
    # shard the dataset if it makes sense
    if hvd.size()>1:
        print('Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
                    hvd.rank(), hvd.size())) 
   
        dataset = dataset.shard(hvd.size(), hvd.rank()) 
      
    if is_training:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

    # Convert to individual records.
    # cycle_length = 10 means 10 files will be read and deserialized in parallel.
    # This number is low enough to not cause too much contention on small systems
    # but high enough to provide the benefits of parallelization. You may want
    # to increase this number if you have a large number of CPU cores.
    #dataset = dataset.apply(tf.data.experimental.parallel_interleave(
    #   tf.data.TFRecordDataset, cycle_length=10))
    
 
    return process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=record_parser,
        num_epochs=num_epochs,
        dtype=dtype,
        datasets_num_private_threads=datasets_num_private_threads,
        num_parallel_batches=num_parallel_batches
    )

def chexnet_model(FLAGS):
    """ Builds the chexnet model using specifics from FLAGS. Returns a compiled model."""
    base_model = DenseNet121(include_top=False,
                            weights='imagenet',
                            input_shape=(FLAGS.image_size, FLAGS.image_size, 3))
     
   
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(14, activation='sigmoid', bias_initializer='ones')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    if FLAGS.opt == 'adam':
        opt = optimizers.Adam(lr=FLAGS.lr)
    elif FLAGS.opt == 'sgd':
        opt = optimizers.SGD(lr=FLAGS.lr, momentum=FLAGS.momentum, nesterov=FLAGS.nesterov)
    elif FLAGS.opt == 'rmsprop':
        opt = optimizers.RMSProp(lr=FLAGS.lr)
    elif FLAGS.opt == 'adagrad':
        opt = optimizers.Adagrad(lr=FLAGS.lr)
    elif FLAGS.opt == 'adadelta':
        opt = optimizers.Adadelta(lr=FLAGS.lr)
    elif FLAGS.opt == 'adamax':
        opt = optimizers.Adamax(lr=FLAGS.lr)
    elif FLAGS.opt == 'nadam':
        opt = optimizers.Nadam(lr=FLAGS.lr)
    else:
        print("No optimizer selected. Using Adam.")
        opt = optimizers.Adam(lr=FLAGS.lr)

    hvd_opt = hvd.DistributedOptimizer(opt)

    model.compile(loss='binary_crossentropy',
                  optimizer=hvd_opt,
                  metrics=['accuracy'])

    return model


def config_gpu():
    """"
    Setup the GPUs to support for running with more memory as opposed to pre allocated memory.
    Controls the percentage use of memory. Sets up process such that, typically one process acts on one gpu.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
    session = tf.Session(config=config)
    K.set_session(session)

    
def main():
    
    hvd.init()

    config_gpu()

    np.random.seed(hvd.rank())
   
    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0
    #verbose = 1
    # display specs of the system hardware
    print("Running with the following config:")
    for item in FLAGS.__dict__.items():
        print('%s = %s' % (item[0], str(item[1])))

    # get the compiled chexnet model
    model = chexnet_model(FLAGS)

    # Path to weights file
    weights_file = FLAGS.model_dir + '/lr_{:.3f}_bz_{:d}'.format(FLAGS.lr,
                                                                 FLAGS.batch_size) + '_loss_{val_loss:.3f}_epoch_{epoch:02d}.h5'

    # Setup callbacks
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=0.01,
                                   cooldown=0, patience=1, min_lr=1e-15, verbose=0)
    model_checkpoint = ModelCheckpoint(weights_file, monitor="val_loss", save_best_only=True,
                                       save_weights_only=True, verbose=1)

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),

    ]

    # Horovod: write weights on the first worker.
    if hvd.rank() == 0 and FLAGS.write_weights and not FLAGS.skip_eval:
        callbacks.append(model_checkpoint)

    # Reduce the learning rate if training plateaues, this works only when there is validation 
    # data available.
    if not FLAGS.skip_eval:
        callbacks.append(lr_reducer)   
    
    # enable logging with tensorboard if asked.
    if FLAGS.enable_tensorboard:
        callbacks.append(TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=64))
        
   
    # build the training dataset.
    train_input_dataset = input_fn(
        is_training=True,
        data_dir=FLAGS.data_dir,
        batch_size=FLAGS.batch_size,
        dtype=tf.float16 if FLAGS.use_float16 else tf.float32,
        num_epochs=FLAGS.epochs,
        datasets_num_private_threads=FLAGS.datasets_private_threads,
    )

    # build the validation dataset when skip validation is not True.
    eval_input_dataset = None
    val_steps = None
    if not FLAGS.skip_eval:
        eval_input_dataset = input_fn(
            is_training=False,
            data_dir=FLAGS.data_dir,
            batch_size=FLAGS.batch_size,
            dtype=tf.float16 if FLAGS.use_float16 else tf.float32,
            num_epochs=FLAGS.epochs,
            )

        # set the validation steps only if validation is not skipped. 
        val_steps = (_NUM_VAL_IMAGES//FLAGS.batch_size)// hvd.size()

    
  
    
    # time before training
    start = time()
    
    # Horovod: the training will randomly sample 1 / N batches of training data and
    # 3 / N batches of validation data on every worker, where N is the number of workers.
    # Over-sampling of validation data helps to increase probability that every validation
    # example will be evaluated.
    model.fit(
        train_input_dataset,
        steps_per_epoch= (_NUM_TRAINING_IMAGES//FLAGS.batch_size)// hvd.size(),
        epochs=FLAGS.epochs,
        validation_data=eval_input_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=verbose)

    # time after training
    end = time()


if __name__ == '__main__':
    FLAGS, _ = parser.parse_known_args()
    main()
