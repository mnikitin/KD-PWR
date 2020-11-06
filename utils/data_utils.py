from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.mxnet import DALIClassificationIterator


def unpack_batch(batch):
    data = [b.data[0] for b in batch]
    label = [b.label[0] for b in batch]
    return data, label


class HybridRecPipe(Pipeline):
    def __init__(self, db_prefix, input_shape, batch_size, data_params, for_train,
                 num_threads, device_id, num_shards):
        super(HybridRecPipe, self).__init__(batch_size,
                                            num_threads,
                                            device_id,
                                            seed=12+device_id,
                                            prefetch_queue_depth=2)
        self.for_train = for_train
        self.input = ops.MXNetReader(path=[db_prefix + '.rec'], index_path=[db_prefix + '.idx'],
                                     random_shuffle=data_params['shuffle'] if for_train else False,
                                     shard_id=device_id,
                                     num_shards=num_shards)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(input_shape[1], input_shape[2]),
                                            mean=data_params['mean'] if isinstance(data_params['mean'], list) else
                                                 [data_params['mean'] for i in range(input_shape[0])],
                                            std=data_params['std'] if isinstance(data_params['std'], list) else
                                                [data_params['std'] for i in range(input_shape[0])])
        if self.for_train:
            self.rotate = ops.Rotate(device="gpu", interp_type=types.INTERP_LINEAR)
            self.color = ops.ColorTwist(device='gpu')
            self.rng_angle = ops.Uniform(range=(-float(data_params['max_rotate_angle']),
                                                +float(data_params['max_rotate_angle'])))
            self.rng_contrast = ops.Uniform(range=(1.0-data_params['contrast'], 1.0+data_params['contrast']))
            self.rng_brightness = ops.Uniform(range=(1.0-data_params['brightness'], 1.0+data_params['brightness']))
            self.rng_saturation = ops.Uniform(range=(1.0-data_params['saturation'], 1.0+data_params['saturation']))
            self.rng_hue = ops.Uniform(range=(1.0-data_params['hue'], 1.0+data_params['hue']))
            self.coin = ops.CoinFlip(probability=0.5) if data_params['rand_mirror'] else 0

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        if self.for_train:
            images = self.rotate(images, angle=self.rng_angle())
            images = self.color(images, brightness=self.rng_brightness(), contrast=self.rng_contrast(),
                                        saturation=self.rng_saturation(), hue=self.rng_hue())
            output = self.cmnp(images, mirror=self.coin())
        else:
            output = self.cmnp(images)
        return [output, labels.gpu()]


def get_rec_data_iterators(train_db_prefix, val_db_prefix, input_shape, batch_size, data_params, devices):
    num_threads = 2
    num_shards = len(devices)
    train_pipes = [HybridRecPipe(train_db_prefix, input_shape, batch_size, data_params, True,
                                 num_threads, device_id, num_shards) for device_id in range(num_shards)]
    # Build train pipeline to get the epoch size out of the reader
    train_pipes[0].build()
    print("Training pipeline epoch size: {}".format(train_pipes[0].epoch_size("Reader")))
    # Make train MXNet iterators out of rec pipelines
    dali_train_iter = DALIClassificationIterator(train_pipes, train_pipes[0].epoch_size("Reader"), auto_reset=True)
    if val_db_prefix:
        val_pipes = [HybridRecPipe(val_db_prefix, input_shape, batch_size, data_params, False,
                                num_threads, device_id, num_shards) for device_id in range(num_shards)]
        # Build val pipeline get the epoch size out of the reader
        val_pipes[0].build()
        print("Validation pipeline epoch size: {}".format(val_pipes[0].epoch_size("Reader")))
        # Make val MXNet iterators out of rec pipelines
        dali_val_iter = DALIClassificationIterator(val_pipes, val_pipes[0].epoch_size("Reader"), auto_reset=True)
    else:
        dali_val_iter = None
    return dali_train_iter, dali_val_iter
