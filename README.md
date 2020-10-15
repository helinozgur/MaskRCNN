# MaskRCNN

Mask R-CNN, makine öğreniminde veya bilgisayarla görmede örnek bölümleme problemini çözmeyi amaçlayan derin bir sinir ağıdır.
Başka bir deyişle, bir görüntü veya videodaki farklı nesneleri ayırabilir.

Faster R-CNN
Daha hızlı R-CNN, görüntü özelliklerini çıkarmak için bir CNN özellik çıkarıcı kullanır. 
Ardından, ilgi alanları (ROI'ler) oluşturmak için bir CNN bölgesi teklif ağı kullanır. Bunları sabit boyuta dönüştürmek için RoI havuzlaması uyguluyoruz. 
Daha sonra, sınıflandırma ve sınır kutusu tahmini yapmak için tamamen bağlantılı katmanlara beslenir.

KAYNAKÇA:

https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
https://github.com/avolkov1/keras_experiments/blob/master/keras_exp/multigpu/
https://github.com/fchollet/keras/blob/master/keras/utils/training_utils.py
https://medium.com/@tibastar/mask-r-cnn-d69aa596761f
https://github.com/matterport/Mask_RCNN
https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272
