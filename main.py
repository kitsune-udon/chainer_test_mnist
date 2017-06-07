import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

class MyModel(chainer.Chain):
    def __init__(self):
        super(MyModel, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(1, 9, 5, pad=2, stride=1)
            self.c2 = L.Convolution2D(9, 9, 5, pad=2, stride=1)
            self.bn1 = L.BatchNormalization(9)
            self.bn2 = L.BatchNormalization(9)
            self.l1 = L.Linear(None, 50)
            self.l2 = L.Linear(None, 10)
    def __call__(self, x):
        batchsize = x.shape[0]
        h0 = self.bn1(self.c1(x))
        h1 = F.max_pooling_2d(F.relu(h0), 3, stride=2)
        h2 = self.bn2(self.c2(h1))
        h3 = F.max_pooling_2d(F.relu(h2), 3, stride=2)
        h4 = self.l1(F.reshape(h3, (batchsize, -1)))
        h5 = self.l2(h4)
        return F.tanh(h5)

def main():
    batchsize = 100
    max_epoch = 10
    n_outputs = 10
    train, test = chainer.datasets.get_mnist(withlabel=True, ndim=3)

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

    model = L.Classifier(MyModel())

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer)

    trainer = training.Trainer(updater, (max_epoch, 'epoch'))
    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.run()

main()
