from keras_applications import get_submodules_from_kwargs

from ._common_blocks import Conv2dBn
from ._utils import freeze_model
from ..backbones.backbones_factory import Backbones

backend = None
layers = None
models = None
keras_utils = None


# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

import keras.backend as K
from keras.models import Model
from keras.layers import Conv2DTranspose as Transpose
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Concatenate


def handle_block_names(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def ConvRelu(filters,
             kernel_size,
             use_batchnorm=False,
             conv_name='conv',
             bn_name='bn',
             relu_name='relu'):

    def layer(x):

        x = Conv2D(filters,
                   kernel_size,
                   padding="same",
                   name=conv_name,
                   use_bias=not(use_batchnorm))(x)

        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)

        x = Activation('relu', name=relu_name)(x)

        return x
    return layer


def Conv2DUpsample(filters,
                   upsample_rate,
                   kernel_size=(3,3),
                   up_name='up',
                   conv_name='conv',
                   **kwargs):

    def layer(input_tensor):
        x = UpSampling2D(upsample_rate, name=up_name)(input_tensor)
        x = Conv2D(filters,
                   kernel_size,
                   padding='same',
                   name=conv_name,
                   **kwargs)(x)
        return x
    return layer


def Conv2DTranspose(filters,
                    upsample_rate,
                    kernel_size=(4,4),
                    up_name='up',
                    **kwargs):

    if not tuple(upsample_rate) == (2,2):
        raise NotImplementedError(
            'Conv2DTranspose support only upsample_rate=(2, 2), got {}'.format(upsample_rate))

    def layer(input_tensor):
        x = Transpose(filters,
                      kernel_size=kernel_size,
                      strides=upsample_rate,
                      padding='same',
                      name=up_name)(input_tensor)
        return x
    return layer


def UpsampleBlock(filters,
                  upsample_rate,
                  kernel_size,
                  use_batchnorm=False,
                  upsample_layer='upsampling',
                  conv_name='conv',
                  bn_name='bn',
                  relu_name='relu',
                  up_name='up',
                  **kwargs):

    if upsample_layer == 'upsampling':
        UpBlock = Conv2DUpsample

    elif upsample_layer == 'transpose':
        UpBlock = Conv2DTranspose

    else:
        raise ValueError('Not supported up layer type {}'.format(upsample_layer))

    def layer(input_tensor):

        x = UpBlock(filters,
                    upsample_rate=upsample_rate,
                    kernel_size=kernel_size,
                    use_bias=not(use_batchnorm),
                    conv_name=conv_name,
                    up_name=up_name,
                    **kwargs)(input_tensor)

        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)

        x = Activation('relu', name=relu_name)(x)

        return x
    return layer


def DecoderBlock(stage,
                 filters=32,
                 use_batchnorm=True,
                 upsample_layer='transpose'):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)


        x = UpsampleBlock(filters=filters,
                          kernel_size=(4, 4),
                          upsample_layer=upsample_layer,
                          upsample_rate=(2, 2),
                          use_batchnorm=use_batchnorm,
                          conv_name=conv_name + '1',
                          bn_name=bn_name + '1',
                          up_name=up_name + '1',
                          relu_name=relu_name + '1')(input_tensor)

        x = ConvRelu(filters,
                     kernel_size=(3, 3),
                     use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2',
                     bn_name=bn_name + '2',
                     relu_name=relu_name + '2')(x)

        x = ConvRelu(filters,
                     kernel_size=(3, 3),
                     use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '3',
                     bn_name=bn_name + '3',
                     relu_name=relu_name + '3')(x)

        return x
    return layer

def DecoderBranch(branch,
                  num_blocks,
                  channel_scale=1,
                  use_batchnorm=True,
                  upsample_layer='transpose'):

    def layer(input_tensor, skip_tensor=None):

        if skip_tensor is not None:
            x = Concatenate(axis=-1)([input_tensor, skip_tensor])
        else:
            x = input_tensor

        num_filters = K.int_shape(x)[-1] // channel_scale
        for i in range(num_blocks):
            stage = branch+str(i+1)
            x = DecoderBlock(stage,
                             num_filters,
                             use_batchnorm,
                             upsample_layer)(x)
            if i==0:
                skip_branch = x
        if num_blocks == 0:
            skip_branch = None

        x = Conv2D(1, 3, padding="same", name=branch+"_logit")(x)
        x = Conv2DTranspose(1, (2, 2), up_name=branch+"_resize")(x)
        pred = Activation("sigmoid", name=branch+"_pred")(x)

        return x, pred, skip_branch
    return layer

def build_cascadenet(backbone,
                     classes,
                     skip_connection_layers,
                     channel_scale=[1, 1, 1, 1, 1],
                     use_batchnorm=True,
                     activation="sigmoid"):
    """
    """
    input_ = backbone.input
    x = backbone.output

    skips = [backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers]

    x_lst = [x] + skips


    B5, B5_pred, B5_skip = DecoderBranch("B5", 4, channel_scale[4], use_batchnorm, 'transpose')(x_lst[0], None)
    B4, B4_pred, B4_skip = DecoderBranch("B4", 3, channel_scale[3], use_batchnorm, 'transpose')(x_lst[1], B5_skip)
    B3, B3_pred, B3_skip = DecoderBranch("B3", 2, channel_scale[2], use_batchnorm, 'transpose')(x_lst[2], B4_skip)
    B2, B2_pred, B2_skip = DecoderBranch("B2", 1, channel_scale[1], use_batchnorm, 'transpose')(x_lst[3], B3_skip)
    B1, B1_pred, B1_skip = DecoderBranch("B1", 0, channel_scale[0], use_batchnorm, 'transpose')(x_lst[4], B2_skip)

    concat_pred = Concatenate(axis=-1)([B5, B4, B3, B2, B1])
    concat_pred = Conv2D(classes, 1, padding="same", name="fusion", activation=activation)(concat_pred)

    model = Model(input_, [concat_pred, B1_pred, B2_pred, B3_pred, B4_pred, B5_pred])

    return model


# ---------------------------------------------------------------------
#  LinkNet Decoder
# ---------------------------------------------------------------------

def build_linknet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform'
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model

def Cascadenet(backbone_name='resnet34',
            input_shape=(None, None, 3),
            classes=1,
            channel_scale=[1, 1, 1, 1, 1],
            activation='sigmoid',
            encoder_weights='imagenet',
            encoder_freeze=False,
            encoder_features='default',
            decoder_use_batchnorm=True,
            **kwargs):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_cascadenet(backbone,
                          classes,
                          encoder_features,
                          activation=activation,
                          channel_scale=channel_scale,
                          use_batchnorm=decoder_use_batchnorm)

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone)

    model.name = 'cascadenet-{}'.format(backbone_name)

    return model
# ---------------------------------------------------------------------
#  LinkNet Model
# ---------------------------------------------------------------------

def Linknet(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(None, None, None, None, 16),
        decoder_use_batchnorm=True,
        **kwargs
):


    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_linknet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
