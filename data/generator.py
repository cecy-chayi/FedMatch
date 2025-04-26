import os
import cv2
import time
import random
import numpy as np
import tensorflow as tf

from misc.utils import *
from torchvision import datasets,transforms

class DataGenerator:

    def __init__(self, args):
        """ Data Generator

        Generates batch-iid and batch-non-iid under both
        labels-at-client and labels-at-server scenarios.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """

        self.args = args
        self.base_dir = os.path.join(self.args.dataset_path, self.args.task) 
        self.shape = (32,32,3)

    def generate_data(self):
        print('generating {} ...'.format(self.args.task))
        start_time = time.time()
        self.task_cnt = -1
        self.is_labels_at_server = True if 'server' in self.args.scenario else False
        self.is_imbalanced = True if 'imb' in self.args.task else False
        x, y = self.load_dataset(self.args.dataset_id)
        self.generate_task(x, y)
        print(f'{self.args.task} done ({time.time()-start_time}s)')

    def load_dataset(self, dataset_id):
        temp = {}
        if self.args.dataset_id_to_name[dataset_id] == 'cifar_10':
            temp['train'] = datasets.CIFAR10(self.args.dataset_path, train=True, download=True) 
            temp['test'] = datasets.CIFAR10(self.args.dataset_path, train=False, download=True) 
            x, y = [], []
            for dtype in ['train', 'test']:
                for image, target in temp[dtype]:
                    x.append(np.array(image))
                    y.append(target)
        elif self.args.dataset_id_to_name[dataset_id] == 'fmnist':
            temp['train'] = datasets.FashionMNIST(self.args.dataset_path, train=True, download=True)
            temp['test'] = datasets.FashionMNIST(self.args.dataset_path, train=False, download=True)
            x, y = [], []
            for dtype in ['train', 'test']:
                for image, target in temp[dtype]:
                    image_expanded = np.expand_dims(np.array(image), axis=-1)
                    x.append(image_expanded)
                    y.append(target)
        x, y = self.shuffle(x, y)
        print(f'{self.args.dataset_id_to_name[self.args.dataset_id]} ({np.shape(x)}) loaded.')
        return x, y

    def generate_task(self, x, y):
        x_train, y_train = self.split_train_test_valid(x, y)
        self.split_s_and_u(x_train, y_train)

    def split_train_test_valid(self, x, y):
        self.num_examples = len(x)
        self.num_train = self.num_examples - (self.args.num_test+self.args.num_valid) 
        self.num_test = self.args.num_test
        self.labels = np.unique(y)
        # train set
        x_train = x[:self.num_train]
        y_train = y[:self.num_train]
        # test set
        x_test = x[self.num_train:self.num_train+self.num_test]
        y_test = y[self.num_train:self.num_train+self.num_test]
        y_test = tf.keras.utils.to_categorical(y_test, len(self.labels))
        l_test = np.unique(y_test)
        self.save_task({
            'x': x_test,
            'y': y_test,
            'labels': l_test,
            'name': f'test_{self.args.dataset_id_to_name[self.args.dataset_id]}'
        })
        # valid set
        x_valid = x[self.num_train+self.num_test:]
        y_valid = y[self.num_train+self.num_test:]
        y_valid = tf.keras.utils.to_categorical(y_valid, len(self.labels))
        l_valid = np.unique(y_valid)
        self.save_task({
            'x': x_valid,
            'y': y_valid,
            'labels': l_valid,
            'name': f'valid_{self.args.dataset_id_to_name[self.args.dataset_id]}'
        })
        return x_train, y_train

    def split_s_and_u(self, x, y):
        x, y = self.shuffle(x, y)
        num_classes = self.args.num_classes
        num_clients = self.args.num_clients
        alpha = 0.8
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idxs = np.where(y == c)[0]
            np.random.shuffle(idxs)
            proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
            proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            split_indices = np.split(idxs, proportions)
            for i, client_idx in enumerate(split_indices):
                client_indices[i].extend(client_idx)

        for cid in range(num_clients):
            client_idx = np.array(client_indices[cid])
            client_targets = y[client_idx]

            train_labeled_idxs, train_unlabeled_idxs = self.dirichlet_x_u_split(client_targets)

            global_labeled_idx = client_idx[train_labeled_idxs]
            global_unlabeled_idx = client_idx[train_unlabeled_idxs]

            xl = x[global_labeled_idx]
            yl = y[global_labeled_idx]
            self.save_task({
                'x': xl,
                'y': tf.keras.utils.to_categorical(yl, len(self.labels)),
                'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                'labels': np.unique(yl)
            })
            xu = x[global_unlabeled_idx]
            yu = y[global_unlabeled_idx]
            self.save_task({
                'x': xu,
                'y': tf.keras.utils.to_categorical(yu, len(self.labels)),
                'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                'labels': np.unique(yu)
            })

    def dirichlet_x_u_split(self, labels):
        labels = np.array(labels)
        n_total = len(labels)
        n_labeled = int(n_total * self.args.labeled_ratio)

        all_indices = np.arange(n_total)
        np.random.shuffle(all_indices)

        labeled_idx = all_indices[:n_labeled]
        unlabeled_idx = all_indices[n_labeled:]

        return labeled_idx, unlabeled_idx

    def save_task(self, data):
        np_save(base_dir=self.base_dir, filename=f"{data['name']}.npy", data=data)
        print(f"filename:{data['name']}, labels:[{','.join(map(str, data['labels']))}], num_examples:{len(data['x'])}")
    
    def shuffle(self, x, y):
        idx = np.arange(len(x))
        random.seed(self.args.seed)
        random.shuffle(idx)
        return np.array(x)[idx], np.array(y)[idx]











        
