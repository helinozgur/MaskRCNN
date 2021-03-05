# MaskRCNN ile Nesne Tanıma

Mask R-CNN:

Mask R-CNN, bilgisayarlı görüde nesneleri bölütlemek için kullanılan bir derin sinir
ağıdır. İki aşaması vardır:
1. Girdi görüntüsüne göre bir nesnenin olabileceği bölgeler hakkında tahminler
üretir.
2. Nesnenin sınıfını öngörür, sınırlayıcı kutuyu arıtır ve ilk aşama tahminine
dayanarak nesnenin piksel düzeyinde bir maske oluşturur.
Mask R-CNN temel olarak Faster R-CNN’in bir uzantısıdır. Faster R-CNN, nesne
tanımada yaygın olarak kullanılır.
Faster R-CNN, görüntülerden özellik haritaları çıkarmak için önce bir ConvNet
kullanır. Bu özellik haritaları, daha sonra aday çerçeveleri döndüren bir Region
Proposal Network(RPN)’den geçirilir. Daha sonra, tüm adayları aynı boyuta getirmek
için bu aday sınırlayıcı çerçevelerine bir ROI pooling katmanı uygulanır. Son olarak
öneriler, nesnelerin sınırlayıcı çerçevelerini sınıflandırmak ve çıktılamak için fully
connected katmandan geçirilir.
Aşağıdaki görseldeki gibi nesnenin orada olabileceği düşünülen bir sürü çerçeve
oluşur.


![1](https://user-images.githubusercontent.com/52162324/110106443-be001e80-7dba-11eb-84e9-5526a3fce1ec.PNG)


Daha sonra, aynı sınıf için üst üste gelen kutuları gruplandırır ve yalnızca en yüksek
tahmini seçer. Böylece, aynı nesne için kopyaları önler. Nihai olarak aşağıdaki gibi
nesnelerimiz çerçeve içerisine alınır.


![2](https://user-images.githubusercontent.com/52162324/110106509-d2441b80-7dba-11eb-992d-95e6596a08a0.PNG)


Ve Mask R-CNN ile son halini almış olur

![3](https://user-images.githubusercontent.com/52162324/110106572-e425be80-7dba-11eb-895c-f36508965c84.PNG)



#### PROJE DEMO ÇIKTISI:

![4](https://user-images.githubusercontent.com/52162324/110106822-336bef00-7dbb-11eb-83b1-8b80d2370491.PNG)


#### KAYNAKÇA:

https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012

https://github.com/avolkov1/keras_experiments/blob/master/keras_exp/multigpu/

https://github.com/fchollet/keras/blob/master/keras/utils/training_utils.py

https://medium.com/@tibastar/mask-r-cnn-d69aa596761f

https://github.com/matterport/Mask_RCNN

https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272
