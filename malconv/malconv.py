#!/usr/bin/python
'''defines the MalConv architecture.
Adapted from https://arxiv.org/pdf/1710.09435.pdf
Things different about our implementation and that of the original paper:
 * The paper uses batch_size = 256 and SGD(lr=0.01, momentum=0.9, decay=UNDISCLOSED, nesterov=True )
 * The paper didn't have a special EOF symbol
 * The paper allowed for up to 2MB malware sizes, we use 1.0MB because of memory on a Titan X
 '''

def main(): 
    from keras.layers import Dense, Conv1D, Activation, GlobalMaxPooling1D, Input, Embedding, Multiply
    from keras.models import Model
    from keras import backend as K
    from keras import metrics
    import multi_gpu
    import os
    import math
    import random
    import argparse
    import os
    import numpy as np
    import requests

    batch_size = 100
    input_dim = 257 # every byte plus a special padding symbol
    padding_char = 256

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', help='number of GPUs', default=1)

    args = parser.parse_args()
    ngpus = int(args.gpus)

    if os.path.exists('malconv.h5'):
        print("restoring malconv.h5 from disk for continuation training...")
        from keras.models import load_model
        basemodel = load_model('malconv.h5')
        _, maxlen, embedding_size = basemodel.layers[1].output_shape
        input_dim
    else:
        maxlen = 2**20 # 1MB
        embedding_size = 8 

        # define model structure
        inp = Input( shape=(maxlen,))
        emb = Embedding( input_dim, embedding_size )( inp )
        filt = Conv1D( filters=128, kernel_size=500, strides=500, use_bias=True, activation='relu', padding='valid' )(emb)
        attn = Conv1D( filters=128, kernel_size=500, strides=500, use_bias=True, activation='sigmoid', padding='valid')(emb)
        gated = Multiply()([filt,attn])
        feat = GlobalMaxPooling1D()( gated )
        dense = Dense(128, activation='relu')(feat)
        outp = Dense(1, activation='sigmoid')(dense)

        basemodel = Model( inp, outp )

    basemodel.summary() 

    print("Using %i GPUs" %ngpus)

    if ngpus > 1:
        model = multi_gpu.make_parallel(basemodel,ngpus)
    else:
        model = basemodel

    from keras.optimizers import SGD
    model.compile( loss='binary_crossentropy', optimizer=SGD(lr=0.01,momentum=0.9,nesterov=True,decay=1e-3), metrics=[metrics.binary_accuracy] )

    def bytez_to_numpy(bytez,maxlen):
        b = np.ones( (maxlen,), dtype=np.uint16 )*padding_char
        bytez = np.frombuffer( bytez[:maxlen], dtype=np.uint8 )
        b[:len(bytez)] = bytez
        return b

    def getfile_service(sha256,url=None,maxlen=maxlen):
        if url is None:
            raise NotImplementedError("You must provide your own url for getting file bytez by sha256")
        r = requests.get( url, params={'sha256':sha256} )
        if not r.ok:
            return None
        return bytez_to_numpy( r.content, maxlen )        

    def generator( hashes, labels, batch_size, shuffle=True ):
        X = []
        y = []
        zipped = list(zip(hashes, labels))
        while True:
            if shuffle:
                random.shuffle( zipped )
            for sha256,l in zipped:
                x = getfile_service(sha256)
                if x is None:
                    continue
                X.append( x )
                y.append( l )
                if len(X) == batch_size:
                    yield np.asarray(X,dtype=np.uint16), np.asarray(y)
                    X = []
                    y = []

    import pandas as pd
    train_labels = pd.read_csv('ember_training.csv.gz')
    train_labels = train_labels[ train_labels['y'] != -1 ] # get only labeled samples
    labels = train_labels['y'].tolist()
    hashes = train_labels['sha256'].tolist()

    from sklearn.model_selection import train_test_split
    hashes_train, hashes_val, labels_train, labels_val = train_test_split( hashes, labels, test_size=200 )

    train_gen = generator( hashes_train, labels_train, batch_size )
    val_gen = generator( hashes_val, labels_val, batch_size )

    from keras.callbacks import LearningRateScheduler

    base = K.get_value( model.optimizer.lr )
    def schedule(epoch):
        return base / 10.0**(epoch//2)

    model.fit_generator(
        train_gen,
        steps_per_epoch=len(hashes_train)//batch_size,
        epochs=10,
        validation_data=val_gen,
        callbacks=[ LearningRateScheduler( schedule ) ],
        validation_steps=int(math.ceil(len(hashes_val)/batch_size)),
    )

    basemodel.save('malconv.h5')

    test_labels = pd.read_csv('ember_test.csv.gz')
    labels_test = test_labels['y'].tolist()
    hashes_test = test_labels['sha256'].tolist()

    test_generator = generator(hashes_test,labels_test,batch_size=1,shuffle=False)
    test_p = basemodel.predict_generator( test_generator, steps=len(test_labels), verbose=1 )


if __name__ == '__main__':
    print('*'*80)
    print('''
This is nonfunctional demonstration code that is provided for convenience. It shows
- The MalConv structure used in our paper
- Training procedure used in the paper
- How to load the weights for the MalConv model that we used.

It may be made functional by modifying the code to retrieve file contents by sha256
from a user-defined URL.

You may use the provided weights under the Ember AGPL-3.0 license included in the parent directory.
We also ask that you cite the original MalConv paper and refer to the Ember paper as the implementation.

(1) E. Raff, J. Barker, J. Sylvester, R. Brandon, B. Catanzaro, C. Nicholas, "Malware Detection by Eating a Whole EXE", in ArXiv e-prints. Oct. 2017.

@ARTICLE{raff2017malware,
  title={Malware detection by eating a whole exe},
  author={Raff, Edward and Barker, Jon and Sylvester, Jared and Brandon, Robert and Catanzaro, Bryan and Nicholas, Charles},
  journal={arXiv preprint arXiv:1710.09435},
  year={2017}
}

(2) H. Anderson and P. Roth, "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models”, in ArXiv e-prints. Apr. 2018.

@ARTICLE{2018arXiv180404637A,
  author = {{Anderson}, H.~S. and {Roth}, P.},
  title = "{EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models}",
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1804.04637},
  primaryClass = "cs.CR",
  keywords = {Computer Science - Cryptography and Security},
  year = 2018,
  month = apr,
  adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180404637A},
}
''')
    print('*'*80)

    #main() # uncomment this line after fixing the URL NotImplementedError above