import os
import sys

import tensorflow as tf
import tensorpack as tp
import tensorpack.dataflow as df

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from transformer.networks.classifier import Classifier
from transformer.networks.image_transformer_cifar10 import ImageTransformerCifar10
from transformer.utils.devices import Devices

def get_input_fn(name, batch_size=32):
    image_size = 32

    is_training = name == 'train'
    ds = df.dataset.Cifar10(name, shuffle=is_training)
    augmentors = [
        tp.imgaug.CenterPaste((40, 40)),
        tp.imgaug.RandomCrop((32, 32)),
        tp.imgaug.Flip(horiz=True),
        #tp.imgaug.MapImage(lambda x: (x - pp_mean)/128.0),
    ]
    if is_training:
        ds = df.RepeatedData(ds, -1)
        ds = tp.AugmentImageComponent(ds, augmentors)
    ds = tp.AugmentImageComponent(ds, [tp.imgaug.Resize((image_size, image_size))])
    ds = df.MapData(ds, tuple)  # for tensorflow.data.dataset
    ds.reset_state()

    def input_fn():
        with tf.name_scope('dataset'):
            dataset = tf.data.Dataset.from_generator(
                ds.get_data,
                output_types=(tf.float32, tf.int64),
                output_shapes=(tf.TensorShape([image_size, image_size, 3]), tf.TensorShape([]))
            ).batch(batch_size)
        return dataset
    return input_fn

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=None)
    parser.add_argument('--gpus', type=int, nargs='*', default=[0])
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    device_info = Devices.get_devices(gpu_ids=args.gpus)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5' if args.debug else '3' 
    tf.logging.set_verbosity(tf.logging.DEBUG if args.debug else tf.logging.INFO)
    tf.logging.info('\nargs: %s\ndevice info: %s', args, device_info)

    input_functions = {
        'train': get_input_fn('train', 64),
        'eval': get_input_fn('test', 64)
    }

    model_fn = Classifier.get('transformer', ImageTransformerCifar10)
    session_config = tf.ConfigProto()
    # session_config.gpu_options.allocator_type="BFC"
    # session_config.log_device_placement=True
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        save_summary_steps=10,
        session_config=session_config
    )
    hparams = {}
    hparams['weight_decay'] = 0.0001
    hparams['optimizer'] = tf.train.MomentumOptimizer(
        learning_rate=0.0001,
        momentum=0.9
    )
    hparams = tf.contrib.training.HParams(**hparams)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=hparams
    )

    for epoch in range(50):
        estimator.train(input_fn=input_functions['train'], steps=(50000 // 32))
        estimator.evaluate(input_fn=input_functions['eval'], steps=100)
