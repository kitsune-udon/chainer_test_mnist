import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import argparse

class MyMLP(chainer.Chain):
    def __init__(self, n_channels_of_first_cnn, n_channels_of_second_cnn,
            kernelsize_of_first_cnn, kernelsize_of_second_cnn,
            n_units_of_hidden_layer, kernelsize_of_pooling):
        n_ch1, n_ch2 = n_channels_of_first_cnn, n_channels_of_second_cnn
        k1, k2 = kernelsize_of_first_cnn, kernelsize_of_second_cnn
        self.k_pool = kernelsize_of_pooling
        super(MyMLP, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(1, n_ch1, k1, pad=k1 // 2, stride=1)
            self.c2 = L.Convolution2D(n_ch1, n_ch2, k2, pad=k2 // 2, stride=1)
            self.bn1 = L.BatchNormalization(n_ch1)
            self.bn2 = L.BatchNormalization(n_ch2)
            self.bn3 = L.BatchNormalization((n_units_of_hidden_layer,))
            self.l1 = L.Linear(None, n_units_of_hidden_layer)
            self.l2 = L.Linear(None, 10)
    def __call__(self, x):
        batchsize = x.shape[0]
        h0 = self.bn1(self.c1(x))
        h1 = F.max_pooling_2d(F.relu(h0), self.k_pool, stride=2)
        h2 = self.bn2(self.c2(h1))
        h3 = F.max_pooling_2d(F.relu(h2), self.k_pool, stride=2)
        h4 = self.l1(F.reshape(h3, (batchsize, -1)))
        h5 = self.l2(self.bn3(h4))
        return F.tanh(h5)

class MyMLP2(chainer.Chain): # Using average pooling. This is fast.
    def __init__(self, n_channels_of_first_cnn, n_channels_of_second_cnn,
            kernelsize_of_first_cnn, kernelsize_of_second_cnn,
            n_units_of_hidden_layer, kernelsize_of_pooling):
        n_ch1, n_ch2 = n_channels_of_first_cnn, n_channels_of_second_cnn
        k1, k2 = kernelsize_of_first_cnn, kernelsize_of_second_cnn
        self.k_pool = kernelsize_of_pooling
        super(MyMLP2, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(1, n_ch1, k1, pad=k1 // 2, stride=2)
            self.c2 = L.Convolution2D(n_ch1, n_ch2, k2, pad=k2 // 2, stride=2)
            self.bn1 = L.BatchNormalization(n_ch1)
            self.bn2 = L.BatchNormalization(n_ch2)
            self.bn3 = L.BatchNormalization((n_units_of_hidden_layer,))
            self.l1 = L.Linear(None, n_units_of_hidden_layer)
            self.l2 = L.Linear(None, 10)
    def __call__(self, x):
        batchsize = x.shape[0]
        h0 = self.bn1(self.c1(x))
        h1 = F.relu(h0)
        h2 = self.bn2(self.c2(h1))
        h3 = F.relu(h2)
        h4 = F.average_pooling_2d(h3, self.k_pool, stride=1, pad=self.k_pool // 2)
        h5 = self.l1(F.reshape(h4, (batchsize, -1)))
        h6 = self.l2(self.bn3(h5))
        return F.tanh(h6)

def main():
    parser = argparse.ArgumentParser(description='CNN test for MNIST')
    parser.add_argument('--batchsize', type=int, default=100,
            help='Number of samples in a mini-batch')
    parser.add_argument('--epoch', type=int, default=10,
            help='Number of iterations of taking the whole training set')
    parser.add_argument('--gpu', type=int, default=-1,
            help='GPU ID')
    parser.add_argument('--n_units_of_hidden_layer', type=int, default=50,
            help='Number of units of the hidden layer')
    parser.add_argument('--n_channels_of_first_cnn', type=int, default=9,
            help='Number of channels of the first CNN layer\'s output')
    parser.add_argument('--n_channels_of_second_cnn', type=int, default=9,
            help='Number of channels of the second CNN layer\'s output')
    parser.add_argument('--kernelsize_of_first_cnn', type=int, default=5,
            help='Kernel size of the first CNN layer')
    parser.add_argument('--kernelsize_of_second_cnn', type=int, default=5,
            help='Kernel size of the second CNN layer')
    parser.add_argument('--kernelsize_of_pooling', type=int, default=3,
            help='Kernel size of the pooling function')
    parser.add_argument('--mlp_type', type=int, default=2,
            help='MLP type')
    args = parser.parse_args()
    print("args:{}".format(vars(args)))

    train, test = chainer.datasets.get_mnist(withlabel=True, ndim=3)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    if args.mlp_type == 1:
        mlp = MyMLP
    elif args.mlp_type == 2:
        mlp = MyMLP2

    model = L.Classifier(mlp(args.n_channels_of_first_cnn, args.n_channels_of_second_cnn,
        args.kernelsize_of_first_cnn, args.kernelsize_of_second_cnn,
        args.n_units_of_hidden_layer, args.kernelsize_of_pooling))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.run()

main()
