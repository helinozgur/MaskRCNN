

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

from mrcnn import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Fayda fonksiyonları
############################################################

def log(text, array=None):
    """Bir metin mesajı yazdırır. Ve isteğe bağlı olarak, bir Numpy dizisi sağlanırsa,
     şeklini, minimum ve maksimum değerlerini yazdırır.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Keras BatchNormalization sınıfını, merkezi bir yerin gerekirse değişiklikler yapmasına izin verecek şekilde genişletir.

     Toplu normalleştirme, gruplar küçükse eğitim üzerinde olumsuz bir etkiye sahiptir, bu nedenle bu katman genellikle dondurulur
    (Yapılandırma sınıfında ayarlanarak) ve doğrusal katman olarak işlev görür.
    """
    def call(self, inputs, training=None):
        """
        training değerleri hakkında not:
             None: BN katmanlarını eğit. Bu normal mod
             False: BN katmanlarını dondur. Batch size küçük olduğunda iyidir
             True: (kullanma). Çıkarımlar yaparken bile eğitim modunda katmanı ayarlar
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """Çekirdek ağ(Backbone Network), çeşitli ağların parçalarını birbirine bağlayan ve
     farklı LAN'lar veya alt ağlar arasında bilgi alışverişi için bir yol sağlayan bir bilgisayar ağının parçasıdır.

    Çekirdek ağının her aşamasının genişliğini ve yüksekliğini hesaplar.

    Returns:
        [N, (height, width)]. N, aşamaların sayısıdır
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])


############################################################
#  Resnet Graph: Artık değerlerin (residual value) sonraki katmanlara besleyen blokların (residual block) modele eklenmesiyle oluşmaktadır.
############################################################



def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """
    Identity_block kısa yoldan konvolüsyonel katmanı olmayan bir bloktur.
    # Argümanlar
        input_tensor: input tensörü
        kernel_size: varsayılan 3, ana yoldaki orta konvolüsyon katmanının çekirdek boyutu.
        filters: tamsayıların listesi, ana yoldaki 3 konvolüsyon katmanının nb_filters'ı.
        stage: tam sayı, geçerli aşama etiketi, katman adları oluşturmak için kullanılır.
        block: 'a', 'b' ..., geçerli blok etiketi, katman adları oluşturmak için kullanılır.
        use_bias: Dönüşüm katmanlarında önyargı kullanmak veya kullanmamak.
        train_bn: Boolean. Toplu Norm katmanlarını eğitir veya dondurur.
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block, kısayolda bir dönüşüm katmanına sahip olan bloktur
    # Argümanlar
        input_tensor: input tensörü
        kernel_size: varsayılan 3, ana yoldaki orta konvolüsyon katmanının çekirdek boyutu.
        filters: tamsayıların listesi, ana yoldaki 3 konvolüsyon katmanının nb_filters'ı.
        stage: tam sayı, geçerli aşama etiketi, katman adları oluşturmak için kullanılır.
        block: 'a', 'b' ..., geçerli blok etiketi, katman adları oluşturmak için kullanılır.
        use_bias: Dönüşüm katmanlarında önyargı kullanmak veya kullanmamak.
        train_bn: Boolean. Toplu Norm katmanlarını eğitir veya dondurur.
     3. aşamadan itibaren, ana yoldaki ilk dönüşüm katmanının subsample= (2,2) ile olur ve
     kısayolun da subsample= (2,2) olması gerekir
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """ResNet graph yapısı:
        architecture: resnet50 veya resnet101 olabilir
        stage5: Boolean. False ise, ağın stage5'i oluşturulmaz
        train_bn: Boolean. Toplu Norm katmanlarını eğitir veya dondurur
    """
    assert architecture in ["resnet50", "resnet101"]
    # Aşama 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Aşama 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Aşama 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Aşama 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Aşama 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer(RPN):Nesnenin bulunduğu bölge için "önerileri" oluşturmak için, küçük bir ağ,
#  son evrişimli katmanın çıktısı olan bir evrişimli özellik haritası üzerinden kaydırılır.
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Verilen deltaları verilen kutulara uygular.
     boxes: [N, (y1, x1, y2, x2)] güncellenecek kutular
     deltas: [N, (dy, dx, log (dh), log (dw))] uygulanacak iyileştirmeler
    """
    # Y, x, h, w'ye dönüştürme
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Deltaların uygulanması
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Tekrar y1, x1, y2, x2 şekline dönüştürme
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):

    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KE.Layer):
    """Sabit puanları alır ve teklif olarak geçmek için bir alt küme seçer
     ikinci aşamaya. Filtreleme, çapa puanlarına göre yapılır ve
     örtüşmeleri kaldırmak için maksimum olmayan bastırma. Ayrıca sınırlama uygular
     çapalar için kutu inceltme deltaları.

     Inputlar:
         rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
         rpn_bbox: [batch, num_anchors, (dy, dx, log (dh), log (dw))]
         çapalar: [batch, num_anchors, (y1, x1, y2, x2)] normalleştirilmiş koordinatlarda çapalar

     Returns:
         Normalleştirilmiş koordinatlarda teklifler [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):

        scores = inputs[0][:, :, 1]

        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])

        anchors = inputs[2]

        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])


        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])


        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = utils.batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Katmanı
############################################################

def log2_graph(x):
    """Log2'nin Uygulanması. TF'nin yerel bir uygulaması yoktur."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    """IÖzellik piramidinin birden çok düzeyinde ROI Havuzlamasını uygular.

     Parametreler:
     - pool_shape: çıktı havuzuna alınmış bölgelerin [pool_height, pool_width]. Genellikle [7, 7]

     Inputlar:
     - normalleştirilmiş kutular: [batch, num_boxes, (y1, x1, y2, x2)]
              koordinatlar. Yeterli değilse, muhtemelen sıfırlarla doldurulmuş
              diziyi doldurmak için kutular.
     - image_meta: [toplu, (meta veri)] Resim ayrıntıları. Compose_image_meta () öğesine bakın
     - feature_maps: Piramidin farklı seviyelerinden özellik haritalarının listesi.
                     Her biri [toplu iş, yükseklik, genişlik, kanallar]

     Outputlar:
     Şekilde havuzlanmış bölgeler: [batch, num_boxes, pool_height, pool_width, kanallar].
     Genişlik ve yükseklik, katmandaki pool_shape'e özgü olanlardır.
     yapıcı.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):

        boxes = inputs[0]

        image_meta = inputs[1]

        feature_maps = inputs[2:]

        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]

        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)


        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)


            box_indices = tf.cast(ix[:, 0], tf.int32)

            box_to_level.append(ix)

            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        pooled = tf.concat(pooled, axis=0)

        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


############################################################
#  Hedef Tespit Katmanı
############################################################

def overlaps_graph(boxes1, boxes2):
    """İki grup kutu arasında IoU çakışmasını hesaplar..
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. boxes2 ve boxes1 i karolara ayırır. Bu döngü kullanmadar her boxes1 ile boxes2
    # değerlerini karşılaştırmamızı sağlar.

    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Kesişme noktalarını hesaplama
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Birlikleri hesaplama
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. IoU hesaplayın ve [kutular1, kutular2] olarak yeniden şekillendirme
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Tek bir görüntü için algılama hedefleri oluşturur. Her biri için hedef sınıf kimliklerini,
    sınırlayıcı kutu deltalarını ve maskeleri alt örneklerle sunar ve oluşturur.

     Inputlar:
         proposals: normalleştirilmiş koordinatlarda [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)].
         Yeterli teklif yoksa sıfır dolgulu olabilir.
         gt_class_ids: [MAX_GT_INSTANCES] int sınıf kimlikleri
         gt_boxes: normalleştirilmiş koordinatlarda [MAX_GT_INSTANCES, (y1, x1, y2, x2)].
         gt_masks: boole türünün [yükseklik, genişlik, MAX_GT_INSTANCES].

     Returns: Hedef ROI'ler ve ilgili sınıf kimlikleri, sınırlayıcı kutu kaymaları ve maskeler.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] normalleştirilmiş koordinatlarda
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Tamsayı sınıf kimlikleri. Sıfır yastıklı.
        deltalar: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log (dh), log (dw))]
        maskeler: [TRAIN_ROIS_PER_IMAGE, yükseklik, genişlik]. Maskeler kutu sınırlarına kırpıldı ve
         sinir ağı çıktı boyutuna yeniden boyutlandırıldı.

     Not: Yeterli hedef ROI yoksa, döndürülen diziler sıfır dolgulu olabilir.
    """

    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    overlaps = overlaps_graph(proposals, gt_boxes)

    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Pozitif ROI'ler, GT kutusu ile> = 0,5 IoU olanlardır
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negatif ROI'ler, her GT kutusunda <0.5 olanlardır.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. % 33 pozitif hedefleyin
    # Positif ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negatif ROIs. Pozitif/negatif oranı korumak için yeterince ekleyin.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Seçilen ROIs topla
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # pozitif ROIs GT boxes a ata.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Pozitif ROIs için bbox ayrıntılandırmasını hesaplayın
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # GT maskelerine pozitif ROIs atayın
    # Maskeleri değiştir [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Her ROI için doğru maskeyi seçin
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    #Maske hedeflerini hesapla
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # ROI koordinatlarını normalleştirilmiş görüntü alanından dönüştürün
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # Ekstra boyutu maskelerden çıkarın.
    masks = tf.squeeze(masks, axis=3)

    # GT maskelerinin ikili çapraz entropi kaybıyla kullanılması için
    # 0 veya 1 olması için 0,5'teki eşik maskesi pikselleri ayarlama.
    masks = tf.round(masks)

    # Negatif ROI'leri ve sıfırlarla negatif ROI'ler için kullanılmayan bbox deltalarını ve maskeleri ekleyin.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """Önerileri alt örnekler ve hedef kutu ayrıntılandırması, class_ids,
    ve her biri için maskeler.

    Inputlar:
    proposals: normalleştirilmiş koordinatlarda [toplu, N, (y1, x1, y2, x2)]. Belki
               Yeterli teklif yoksa sıfır doldurulmalıdır.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Tamsayı sınıf kimlikleri.
    gt_boxes: Normalleştirilmiş koordinatlarda [toplu, MAX_GT_INSTANCES, (y1, x1, y2, x2)].
    gt_masks: boole türünün [batch, height, width, MAX_GT_INSTANCES]

    Returns: Hedef ROI'ler ve ilgili sınıf kimlikleri, sınırlayıcı kutu kaymaları ve maskeler.
    rois: [toplu, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] normalleştirilmiş olarak
          koordinatlar
    target_class_ids: [toplu, TRAIN_ROIS_PER_IMAGE]. Tamsayı sınıf kimlikleri.
    target_deltas: [toplu, TRAIN_ROIS_PER_IMAGE, (dy, dx, günlük (dh), günlük (dw)]
    target_mask: [toplu, TRAIN_ROIS_PER_IMAGE, yükseklik, genişlik]
                 Maskeler bbox sınırlarına kırpıldı ve sinir ağı çıktı boyutuna yeniden boyutlandırıldı.

    Not: Yeterli hedef ROI yoksa, döndürülen diziler sıfır dolgulu olabilir.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


############################################################
#  Tespit Katmanı
############################################################

def refine_detections_graph(rois, probs, deltas, window, config):
    """Sınıflandırılmış teklifleri hassaslaştırın ve çakışmaları filtreleyin ve nihai olarak geri dönün
     tespitler.

     Girişler:
         rois: [N, (y1, x1, y2, x2)] normalleştirilmiş koordinatlarda
         problar: [N, sınıf_sayısı]. Sınıf olasılıkları.
         deltalar: [N, sınıf_sayısı, (dy, dx, günlük (dh), günlük (dw))]. Sınıfa özgü
                 sınırlayıcı kutu deltaları.
         pencere: (y1, x1, y2, x2) normalleştirilmiş koordinatlarda. Görüntünün dolgusu hariç görüntüyü içeren kısmı.

     Koordinatların normalleştirildiği [num_detections, (y1, x1, y2, x2, class_id, score)] şeklindeki algılamaları döndürür.
    """

    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)

    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)

    deltas_specific = tf.gather_nd(deltas, indices)

    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)

    refined_rois = clip_boxes_graph(refined_rois, window)

    keep = tf.where(class_ids > 0)[:, 0]

    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Verilen sınıfın ROI'lerine Maksimum Olmayan Bastırma uygulayın."""

        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]

        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)

        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))

        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)

        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)

    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])

    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]

    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)


    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)


    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    """Sınıflandırılmış teklif kutularını ve sınırlayıcı kutu deltalarını alır ve son algılama kutularını döndürür.

     Returns:
     Koordinatların normalleştirildiği [toplu iş, sayısal algılama, (y1, x1, y2, x2, sınıf_kimliği, sınıf_çekimi)].
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """ Region Proposal Network hesaplama grafiğini oluşturur.

     feature_map: omurga özellikleri [toplu iş, yükseklik, genişlik, derinlik]
     anchors_per_location: özellik haritasındaki piksel başına çapa sayısı
     anchor_stride: Çapa yoğunluğunu kontrol eder. Tipik olarak 1 (çapalar
                    özellik haritasındaki her piksel) veya 2 (diğer her piksel).

     Returns:
         rpn_class_logits: [batch, H * W * anchors_per_location, 2] Çapa sınıflandırıcı günlükleri (softmax'tan önce)
         rpn_probs: [batch, H * W * anchors_per_location, 2] Çapa sınıflandırıcı olasılıkları.
         rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log (dh), log (dw))] Deltalar olacak
    """

    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)


    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Region Proposal Network'ın Keras modelini oluşturur.
     RPN grafiğini, paylaşım ağırlıklarıyla birden çok kez kullanılabilmesi için sarar.

     anchors_per_location: özellik haritasındaki piksel başına çapa sayısı
     anchor_stride: Çapa yoğunluğunu kontrol eder. Tipik olarak 1 (özellik haritasındaki her piksel için bağlantı) veya 2 (diğer her piksel).
     derinlik: Omurga özellik haritasının derinliği.

     Bir Keras Model nesnesi döndürür. Çağrıldığında model çıktıları şunlardır:
     rpn_class_logits: [batch, H * W * anchors_per_location, 2] Çapa sınıflandırıcı günlükleri (softmax'tan önce)
     rpn_probs: [batch, H * W * anchors_per_location, 2] Çapa sınıflandırıcı olasılıkları.
     rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log (dh), log (dw))] Çapalara uygulanacak deltalar.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    """Özellik piramidi ağının ve regresör kafalarının hesaplama grafiğini oluşturur.

     rois: [toplu, num_rois, (y1, x1, y2, x2)] Normalleştirilmiş koordinatlarda teklif kutuları.
     feature_maps: Piramidin farklı katmanlarından [P2, P3, P4, P5] özellik haritalarının listesi. Her birinin farklı bir çözünürlüğü vardır.
     image_meta: [toplu, (meta veri)] Resim ayrıntıları. Compose_image_meta () öğesine bakın
     pool_size: ROI Pooling'den oluşturulan kare özellik haritasının genişliği.
     sınıf_sayısı: sonuçların derinliğini belirleyen sınıf sayısı
     train_bn: Boole. Toplu Norm katmanlarını eğitin veya dondurun
     fc_layers_size: 2 FC katmanının boyutu

     Returns:
         günlükler: [batch, num_rois, NUM_CLASSES] sınıflandırıcı günlükleri (softmax'tan önce)
         problar: [batch, num_rois, NUM_CLASSES] sınıflandırıcı olasılıkları
         bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log (dh), log (dw))] Teklif kutularına uygulanacak deltalar
    """

    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)

    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)

    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Feature Pyramid Network'ün maske başlığının hesaplama grafiğini oluşturur.

     rois: [toplu, num_rois, (y1, x1, y2, x2)] Normalleştirilmiş koordinatlarda teklif kutuları.
     feature_maps: Piramidin farklı katmanlarından özellik haritalarının listesi,
                   [P2, P3, P4, P5]. Her birinin farklı bir çözünürlüğü vardır.
     image_meta: [toplu, (meta veri)] Resim ayrıntıları. Compose_image_meta () öğesine bakın
     pool_size: ROI Pooling'den oluşturulan kare özellik haritasının genişliği.
     sınıf_sayısı: sonuçların derinliğini belirleyen sınıf sayısı
     train_bn: Boole. Toplu Norm katmanlarını eğitin veya dondurun

     Retun: Maskeler [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """

    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Smooth-L1 kaybını uygular.
     y_true ve y_pred tipik olarak: [N, 4], ancak herhangi bir şekilde olabilir.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN çapa sınıflandırıcı kaybı.

     rpn_match: [toplu iş, çapalar, 1]. Sabit eşleme türü. 1 = pozitif,
                -1 = negatif, 0 = nötr çapa.
     rpn_class_logits: [toplu iş, çapalar, 2]. BG / FG için RPN sınıflandırıcı günlükleri.
    """

    rpn_match = tf.squeeze(rpn_match, -1)

    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)

    indices = tf.where(K.not_equal(rpn_match, 0))

    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)

    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """RPN sınırlayıcı kutu kayıp grafiğini döndürün.

    config: model yapılandırma nesnesi.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Kullanılmayan bbox deltalarını doldurmak için 0 dolgusu kullanır.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Maske RCNN'nin sınıflandırıcı kafası için kayıp.

    target_class_ids: [batch, num_rois]. Tamsayı sınıf kimlikleri. Diziyi doldurmak için sıfır dolgu kullanır.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Görüntünün veri kümesindeki sınıflar için 1 ve veri kümesinde olmayan
     sınıflar için 0 değerine sahiptir.
    """

    target_class_ids = tf.cast(target_class_ids, 'int64')

    pred_class_ids = tf.argmax(pred_class_logits, axis=2)

    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    loss = loss * pred_active

    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Maske R-CNN sınırlayıcı kutu iyileştirmesi için kayıp hesaplama.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)


    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Maske kafası için ikili çapraz entropi kaybını maskeleyin.

    target_masks: [batch, num_rois, height, width].
        0 veya 1 değerlerinin float32 tensörü. Diziyi doldurmak için sıfır dolgu kullanır.
    target_class_ids: [batch, num_rois]. Tamsayı sınıf kimlikleri. Sıfır yastıklı.
    pred_masks: [batch, proposals, height, width, num_classes] 0'dan 1'e değerlere sahip float32 tensörü.
    """

    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))

    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Bir görüntü için kesin referans verilerini yükleyin ve geri döndürün (görüntü, maske, sınırlayıcı kutular).

    augment: (kullanımdan kaldırıldı. Bunun yerine büyütme kullanın). Doğruysa rastgele uygula
        görüntü büyütme. Şu anda, yalnızca yatay çevirme sunulmaktadır.
    augmentation: İsteğe bağlı. Bir imgaug (https://github.com/aleju/imgaug) büyütme.
        Örneğin, imgaug.augmenters.Fliplr (0.5) geçmek görüntüleri çevirir
        sağ / sol% 50 zaman.
    use_mini_mask: False ise aynı yüksekliğe sahip tam boyutlu maskeler döndürür
        ve orijinal görüntü olarak genişlik. Bunlar büyük olabilir, örneğin
        1024x1024x100 (100 örnek için). Mini maskeler daha küçüktür, genellikle
        224x224 boyutundadır ve sınırlayıcı kutusu çıkarılarak oluşturulur.
        nesnesi ve MINI_MASK_SHAPE olarak yeniden boyutlandırın.

    Retuns:
    image: [yükseklik, genişlik, 3]
    shape: görüntünün yeniden boyutlandırılmadan ve kırpılmadan önceki orijinal şekli.
    class_ids: [instance_count] Tamsayı sınıf kimlikleri
    bbox: [örnek_sayısı, (y1, x1, y2, x2)]
    mask: [yükseklik, genişlik, örnek_sayısı]. Use_mini_mask True olmadıkça, yükseklik ve genişlik görüntününkilerdir,
    bu durumda MINI_MASK_SHAPE içinde tanımlanırlar.
    """

    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)


    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)


    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Maskelere hangi artırıcıların uygulanacağını belirler."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS


        image_shape = image.shape
        mask_shape = mask.shape

        det = augmentation.to_deterministic()
        image = det.augment_image(image)

        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))

        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"

        mask = mask.astype(np.bool)

    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]

    bbox = utils.extract_bboxes(mask)

    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Aşama 2 sınıflandırıcı ve maske kafalarının eğitimi için hedefler oluşturun.
     Bu normal eğitimde kullanılmaz. Hata ayıklama veya Mask RCNN kafalarını RPN kafasını
     kullanmadan eğitmek için kullanışlıdır.

    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [örnek sayısı] Tamsayı sınıf kimlikleri
    gt_boxes: [örnek sayısı, (y1, x1, y2, x2)]
    gt_masks: [yükseklik, genişlik, örnek sayısı] Kesin referans maskeleri. Tam boy veya mini maskeler olabilir.

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Sınıfa özgü bbox iyileştirmeleri.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). bbox sınırlarına ve sinir ağı çıktı boyutuna
    yeniden boyutlandırıldı.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
        gt_masks.dtype)

    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
        (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
        (gt_boxes[:, 3] - gt_boxes[:, 1])

    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(
            gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(
        overlaps.shape[0]), rpn_roi_iou_argmax]

    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids

    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids

    keep = np.concatenate([keep_fg_ids, keep_bg_ids])

    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:

        if keep.shape[0] == 0:
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])

    bboxes /= config.BBOX_STD_DEV

    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1

            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)

            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Anchors ve GT kutuları göz önüne alındığında, örtüşmeleri hesaplayın ve karşılık gelen GT kutuları
    ile eşleşecek şekilde iyileştirmek için pozitif bağlantıları ve deltaları tanımlayın.

      anchors: [sayı_anchors, (y1, x1, y2, x2)]
     gt_class_ids: [num_gt_boxes] Tamsayı sınıf kimlikleri.
     gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

     Returns:
     rpn_match: [N] (int32), çapalar ve GT kutuları arasında eşleşir.
                1 = pozitif bağlantı, -1 = negatif bağlantı, 0 = nötr
     rpn_bbox: [N, (dy, dx, log (dh), log (dw))] Sabit bbox deltaları.
    """

    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)

    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1

    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
    rpn_match[gt_iou_argmax] = 1
    rpn_match[anchor_iou_max >= 0.7] = 1

    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    ids = np.where(rpn_match == 1)[0]
    ix = 0
    for i, a in zip(ids, anchors[ids]):
        gt = gt_boxes[anchor_iou_argmax[i]]

        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Bölge teklif ağına benzer yatırım getirisi teklifleri üretir
     üretecekti.

     image_shape: [Yükseklik, Genişlik, Derinlik]
     count: Oluşturulacak ROI sayısı
     gt_class_ids: [N] Tamsayı temel doğruluk sınıfı kimlikleri
     gt_boxes: [N, (y1, x1, y2, x2)] Piksel cinsinden kesin referans kutuları.

     Returns: [count, (y1, x1, y2, x2)] ROI kutuları piksel cinsinden.
    """

    rois = np.zeros((count, 4), dtype=np.int32)

    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
    """Görüntüleri ve ilgili hedef sınıf kimliklerini döndüren bir jeneratör,
    sınırlayıcı kutu deltaları ve maskeler.

    veri kümesi: Verilerin seçileceği Veri Kümesi nesnesi
    config: Model yapılandırma nesnesi
    karıştır: True ise, örnekleri her dönemden önce karıştırır
    augment: (kullanımdan kaldırıldı. Bunun yerine büyütme kullanın). Doğruysa, rastgele görüntü büyütme uygulayın.
    augmentation: İsteğe bağlı. Bir imgaug (https://github.com/aleju/imgaug) büyütme.
        Örneğin, imgaug.augmenters.Fliplr (0.5) geçmek görüntüleri çevirir
        sağ / sol% 50 zaman.
    random_rois: Eğer> 0 ise, bu durumda
                 ağ sınıflandırıcı ve maske kafaları. Eğitim ise kullanışlıdır
                 RPN'siz Maske RCNN kısmı.
    batch_size: Her çağrıda kaç resim döndürülecek
    Detection_targets: True ise, algılama hedefleri oluşturun (sınıf kimlikleri, bbox
        deltalar ve maskeler). Genellikle hata ayıklama veya görselleştirmeler için, çünkü
        eğitimde algılama hedefleri DetectionTargetLayer tarafından oluşturulur.
    no_augmentation_sources: İsteğe bağlı. Hariç tutulacak kaynakların listesi
        büyütme. Kaynak, bir veri kümesini tanımlayan dizedir ve
        Dataset sınıfında tanımlanmıştır.

    Bir Python oluşturucu döndürür. Sonraki () arandığında,
    jeneratör iki liste, girdi ve çıktı döndürür. İçerikler
    Listelerin% 'si alınan argümanlara göre farklılık gösterir:
    inputlistesi:
    - images: [toplu iş, H, W, C]
    - image_meta: [toplu, (meta veri)] Resim ayrıntıları. Compose_image_meta () öğesine bakın
    - rpn_match: [batch, N] Tam sayı (1 = pozitif bağlantı, -1 = negatif, 0 = nötr)
    - rpn_bbox: [toplu, N, (dy, dx, log (dh), log (dw))] Sabit bbox deltaları.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Tamsayı sınıf kimlikleri
    - gt_boxes: [toplu, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [toplu iş, yükseklik, genişlik, MAX_GT_INSTANCES]. Use_mini_mask True olmadıkça, yükseklik ve genişlik
    görüntününkilerdir, bu durumda MINI_MASK_SHAPE içinde tanımlanırlar.

    output listesi: Normal eğitimde genellikle boştur. Ancak, algılama_hedefleri Doğru ise, çıktılar listesi
    hedef sınıf_itleri, bbox deltaları ve maskeleri içerir.
    """
    b = 0
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    while True:
        try:
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            image_id = image_ids[image_index]

            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              augmentation=None,
                              use_mini_mask=config.USE_MINI_MASK)
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt(dataset, config, image_id, augment=augment,
                                augmentation=augmentation,
                                use_mini_mask=config.USE_MINI_MASK)

            if not np.any(gt_class_ids > 0):
                continue

            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)

            if random_rois:
                rpn_rois = generate_random_rois(
                    image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                        build_detection_targets(
                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                if random_rois:
                    batch_rpn_rois = np.zeros(
                        (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros(
                            (batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        batch_mrcnn_class_ids = np.expand_dims(
                            batch_mrcnn_class_ids, -1)
                        outputs.extend(
                            [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    """Maske RCNN modeli işlevselliğini kapsüller.

     Gerçek Keras modeli, keras_model özelliğindedir.
    """

    def __init__(self, mode, config, model_dir):
        """mod: "eğitim" veya "çıkarım"
         config: Config sınıfının bir Alt sınıfı
         model_dir: Eğitim günlüklerini ve eğitimli ağırlıkları kaydetme rehberi
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Mask R-CNN mimarisi oluşturun.
             input_shape: Giriş görüntüsünün şekli.
             mod: "eğitim" veya "çıkarım". Modelin girdi ve çıktıları buna göre farklılık gösteriyor.
        """
        assert mode in ['training', 'inference']

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        if mode == "training":

            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)

            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)

            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)

            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":

            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                             stage5=True, train_bn=config.TRAIN_BN)


        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])

        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)

        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors
#RPN Modeli Oluşturma
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":

            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)

            if not config.USE_RPN_ROIS:
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)

                target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            rois, target_class_ids, target_bbox, target_mask =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Son eğitilmiş modelin son kontrol noktası dosyasını bulur.
         model dizini.
         Returns:
             Son kontrol noktası dosyasının yolu
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        dir_name = os.path.join(self.model_dir, dir_names[-1])

        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """İlgili Keras işlevinin değiştirilmiş sürümü ile
         çoklu GPU desteğinin eklenmesi ve dışlama yeteneği
         bazı katmanlar yüklemeden.
         exclude: dışlanacak katman adlarının listesi
        """
        import h5py
        try:
            from keras.engine import saving
        except ImportError:
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """ImageNet eğitimli ağırlıkları Keras'tan indirir.
         Ağırlıklar dosyasının yolunu döndürür.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):

        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):

        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Model günlük dizinini ve dönem sayacını ayarlar.

         model_path: Yok ise veya bu kodun kullandığından farklı bir biçimse, yeni bir günlük dizini ayarlayın ve
          0'dan epochs'u başlatın. Aksi takdirde, dosya adından günlük dizinini ve epoch sayacını çıkarın.
        """
        self.epoch = 0
        now = datetime.datetime.now()

        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Model eğitilir.
         train_dataset, val_dataset: Eğitim ve doğrulama Veri kümesi nesneleri.
         learning_rate: Eğitim yapılacak öğrenme oranı
         epochs: Eğitim dönemi sayısı. Önceki eğitim dönemlerinin önceden yapılmış olarak kabul edildiğini unutmayın, bu nedenle bu,
                 bu partikülerden ziyade toplamda eğitilecek çağlar
                 aramak.
         layers: Eğitilecek katmanların seçilmesini sağlar. Olabilir:
             - Eğitilecek katman adlarıyla eşleşecek normal bir ifade
             - Bu önceden tanımlanmış değerlerden biri:
               heads: Ağın RPN, sınıflandırıcı ve maske kafaları
               tümü: Tüm katmanlar
               3+: Resnet'i 3. aşama ve üstü eğitin
               4+: Resnet'i 4. aşama ve üstü eğitin
               5+: Resnet'i eğitin 5. aşama ve üstü

        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        if custom_callbacks:
            callbacks += custom_callbacks

        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1
        ww = wx2 - wx1
        scale = np.array([wh, ww, wh, ww])

        boxes = np.divide(boxes - shift, scale)

        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        full_masks = []
        for i in range(N):
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0):
        """Algılama ardışık düzenini çalıştırır.

         images: Muhtemelen farklı boyutlardaki görsellerin listesi.

         Resim başına bir dikt olacak şekilde bir diktat listesi verir. Dikte şunları içerir:
         rois: [N, (y1, x1, y2, x2)] algılama sınırlayıcı kutuları
         class_ids: [N] int sınıf kimlikleri
         puanlar: [N] sınıf kimlikleri için kayan olasılık puanları
         masks: [H, W, N] örnek ikili maskeler
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        molded_images, image_metas, windows = self.mold_inputs(images)

        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Algılama ardışık düzenini çalıştırır, ancak
         zaten kalıplanmış. Çoğunlukla modeli hata ayıklamak ve incelemek için kullanılır.

         molded_images: load_image_gt () kullanılarak yüklenen görüntülerin listesi
         image_metas: resim meta verileri, ayrıca load_image_gt () tarafından döndürülür

         Resim başına bir dikt olacak şekilde bir diktat listesi verir. Dikte şunları içerir:
         rois: [N, (y1, x1, y2, x2)] algılama sınırlayıcı kutuları
         class_ids: [N] int sınıf kimlikleri
         puanlar: [N] sınıf kimlikleri için kayan olasılık puanları
         masks: [H, W, N] örnek ikili maskeler
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE,\
            "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def get_anchors(self, image_shape):
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            self.anchors = a
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """Hesaplama grafiğinde bir TF tensörünün atasını bulur.
         tensor: TensorFlow sembolik tensör.
         isim: Bulunacak üst tensörün adı
         kontrol edildi: Dahili kullanım için. Grafikte gezinirken döngüleri önlemek için zaten
         aranmış olan tensörlerin bir listesi.
        """
        checked = checked if checked is not None else []
        if len(checked) > 500:
            return None
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        layers = []
        for l in self.keras_model.layers:
            l = self.find_trainable_layer(l)
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Verileni hesaplayan hesaplama grafiğinin bir alt kümesini çalıştırır
         çıktılar.

         image_metas: Sağlanırsa, görüntülerin zaten olduğu varsayılır
             kalıplanmış (yani yeniden boyutlandırılmış, yastıklı ve normalleştirilmiş)

         outputs: Hesaplanacak tuplelar listesi (ad, tensör). Tensörler sembolik TensorFlow tensörleridir ve
          isimler kolay takip içindir.

         Sıralı bir sonuç emri verir. Anahtarlar, girişte alınan isimlerdir ve değerler Numpy dizileridir.
        """
        model = self.keras_model

        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Veri Formatlama
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Bir görüntünün niteliklerini alır ve onları tek bir 1B dizisine yerleştirir.

     image_id: Resmin int kimliği. Hata ayıklama için kullanışlıdır.
     original_image_shape: [H, W, C] yeniden boyutlandırmadan veya doldurmadan önce.
     image_shape: [H, W, C] yeniden boyutlandırma ve doldurmadan sonra
     window: (y1, x1, y2, x2) piksel cinsinden. Görüntünün gerçek görüntünün olduğu alan (dolgu hariç)
     scale: Orijinal görüntüye uygulanan ölçekleme faktörü (float32)
     active_class_ids: Görüntünün geldiği veri kümesinde bulunan class_ids listesi. Tüm sınıfların tüm veri kümelerinde
     bulunmadığı birden çok veri kümesinden görüntüler üzerinde eğitim alıyorsanız kullanışlıdır.
    """
    meta = np.array(
        [image_id] +
        list(original_image_shape) +
        list(image_shape) +
        list(window) +
        [scale] +
        list(active_class_ids)
    )
    return meta


def parse_image_meta(meta):
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Bileşenlerine görüntü özniteliklerini içeren bir tensörü çözümler.
     Daha fazla ayrıntı için compose_image_meta () konusuna bakın.

     meta: [toplu, meta uzunluk] burada meta uzunluğunun NUM_CLASSES değerine bağlı olduğu

     çıktı olarak ayrıştırılmış tensörlerin bir diktesini verir.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)



def trim_zeros_graph(boxes, name='trim_zeros'):
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Kutuları piksel koordinatlarından normalleştirilmiş koordinatlara dönüştürür.
     boxes: piksel koordinatlarında [..., (y1, x1, y2, x2)]
     shapes: [..., (yükseklik, genişlik)] piksel cinsinden

     Not: Piksel koordinatlarında (y2, x2) kutunun dışındadır. Ancak normalleştirilmiş koordinatlarda kutunun içindedir.

     Returns:
         [..., (y1, x1, y2, x2)] normalleştirilmiş koordinatlarda
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Kutuları normalleştirilmiş koordinatlardan piksel koordinatlarına dönüştürür.
     Boxes: [..., (y1, x1, y2, x2)] normalleştirilmiş koordinatlarda
     shapes: [..., (yükseklik, genişlik)] piksel cinsinden

     Not: Piksel koordinatlarında (y2, x2) kutunun dışındadır. Ancak normalleştirilmiş koordinatlarda kutunun içindedir.

     Returns:
         piksel koordinatlarında [..., (y1, x1, y2, x2)]
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
