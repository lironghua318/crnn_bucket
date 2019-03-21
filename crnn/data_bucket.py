import sys
#sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx
import time
import cv2
import random
import os.path
import bisect
from tqdm import tqdm

# class TextIter(mx.io.DataIter):

class TextIter(mx.io.DataIter):
    def __init__(self, dataset_path, image_path, image_set,batch_size,
                 init_states,num_label=12,data_shape=[32,240],buckets=[5,10,20,30]):
        super(TextIter, self).__init__()
        self.batch_size = batch_size
        self.data_name=('data')
        self.label_name=('label')
        self.dataset_path=dataset_path
        self.image_path=image_path
        self.image_set=image_set
        # self.content = self.read_content(path)
        # print (path+'records number : ',len(self.content))
        self.num_label=num_label
        self.buckets = buckets
        self.bucket_items, self.data_plan,self.samples=self.get_bucket()
        # self.imagelabels=get_label_from_content(self.content,num_label)
        self.default_bucket_key = max(buckets)
        self.factor=4
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, 3,data_shape[0],data_shape[1]))] + init_states
        self.provide_label = [('label', (batch_size,self.num_label ))]
        self.current=0
        self.batch_size=batch_size
        self.size=len(self.data_plan)
        random.shuffle(self.data_plan)
    def get_bucket(self):
        path = os.path.join(self.dataset_path, '%s.txt' % self.image_set)
        print(path)
        bucket_items = {}
        with open(path) as ins:
            content = ins.read()
            content = content.split('\n')
            content.pop()
        buckets_len = len(self.buckets)
        # buckets_len = 9
        for i in range(buckets_len):
            bucket_items[i] = []
        for i,line in enumerate(content):
            line = line.strip()
            ll = line.split(' ')
            path = ll[0]
            label = ll[1:]
            label = [int(l)+1 for l in label]
            buck = bisect.bisect_left(self.buckets, len(label)*2)
            bucket_items[buck].append([path,label])
        data_plan = []
        for bucket_idx in bucket_items:
            length_bucket = len(bucket_items[bucket_idx])
            print("bucket " + " length :", length_bucket)
            for idx in range(length_bucket // self.batch_size):
                data_plan.append([bucket_idx, idx])
        return bucket_items,data_plan,len(content)

    def preprocess(self, img_path, index):
        # index for self.bucket
        # print(img_path)
        img = cv2.imread(os.path.join(self.dataset_path,self.image_path,img_path))
        height,width = img.shape[0],img.shape[1]
        ratio = height/32
        width = int(width/ratio)
        img_ori = cv2.resize(img, (self.buckets[index]*8, 32))
        # img_pad = np.ones((32, self.buckets[index]*8, 3)) * 255
        # img_pad[:, :width, :] = img_ori
        img = img_ori.astype('float32')
        img = img - 127.5
        img *= 0.0078125
        img = np.transpose(img, axes=(2, 0, 1))
        return img

    def iter_next(self):
        return self.current < self.size

    def next(self):
        if self.iter_next():
            # start=time.time()
            i=self.current
            init_state_names = [x[0] for x in self.init_states]
            current_batch=self.data_plan[i]
            buck_idx=current_batch[0]
            img_idx=current_batch[1]
            batch_datas = self.bucket_items[buck_idx][img_idx*self.batch_size:img_idx*self.batch_size+self.batch_size]
            data = []
            label = []
            for batch_data in batch_datas:
                image_path = batch_data[0]
                image_label = batch_data[1]
                img = self.preprocess(image_path,buck_idx)
                # label_tmp = np.ones(self.buckets[buck_idx])
                # label_tmp[:len(image_label)] = image_label
                # label_tmp = list(label_tmp)
                data.append(img)
                label.append(image_label)

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            # data_names = ['data']  + init_state_names
            # label_names = ['label']
            # data_batch = SimpleBatch(data_names, data_all, label_names, label_all,self.buckets[buck_idx])
            self.current+=1
            # end=time.time()

            return mx.io.DataBatch(data=data_all, label=label_all,
                                   pad=0, bucket_key=self.buckets[buck_idx],
                                   provide_data=[mx.io.DataDesc(name=self.data_name,
                                                                shape=data_all[0].shape)] + self.init_states,
                                   provide_label=[mx.io.DataDesc(name=self.label_name, shape=label_all[0].shape)])

            # return data_batch
        else:
            raise StopIteration

    def num_samples(self):
      return self.samples

    def reset(self):
        self.current = 0
        random.shuffle(self.data_plan)



