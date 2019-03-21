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

# mx.rnn.BucketSentenceIter
# def default_read_content(path):
#     with open(path) as ins:
#         content = ins.read()
#         content = content.split('\n')
#         content.pop()
#         return content




# def get_label_from_content(content):
#     labels=[]
#     for line in content:
#         label=line.split(' ')[1:-1]
#         tmp = []
#         for i in label :
#             if i == 0:
#                 pass
#                 tmp.append(-1)
#             else:
#                 tmp.append(i)
#         labels.append(tmp)
#     return labels

# def get_image_batch(paths,data_root):
#     data=[]
#     base_hight=32
#     max_ratio=25
#
#     for path in paths:
#         img=cv2.imread(data_root+'/'+path)
#         shape=img.shape
#         hight=shape[0]
#         width=shape[1]
#         ratio=(1.0*width/hight)
#         if ratio>max_ratio:
#             ratio=max_ratio
#         if ratio<1:
#             ratio=1
#         img=cv2.resize(img,(int(32*ratio),32))
#         hight=32
#         width=int(32*ratio)
#         assert hight==base_hight
#         img=np.transpose(img,(2,0,1))
#         if width % hight !=0:
#             padding_ratio=(min(int(ratio+1),max_ratio))
#             new_img=np.zeros((3,base_hight,base_hight*padding_ratio))
#             for i in range(3):
#                 padding_value = int(np.mean(img[i][:][-1]))
#                 z=np.ones((base_hight,base_hight*padding_ratio-width))*padding_value
#                 new_img[i]=np.hstack((img[i],z))
#             img=new_img
#         else:
#             img=img
#         data.append(img)
#     return np.array(data)
# class SimpleBatch(object):
#     def __init__(self, data_names, data, label_names, label, bucket_key):
#         self.data = data
#         self.label = label
#         self.data_names = data_names
#         self.label_names = label_names
#         self.bucket_key = bucket_key
#         self.pad = 0
#         self.index = None
#     @property
#     def provide_data(self):
#         return [(n, x.shape) for n, x in zip(self.data_names, self.data)]
#
#     @property
#     def provide_label(self):
#         return [(n, x.shape) for n, x in zip(self.label_names, self.label)]



# class TextIter(mx.io.DataIter):
class TextIter(object):
    def __init__(self, dataset_path, image_path, image_set,batch_size,
                 init_states,num_label=100,data_shape=[32,800],buckets=[5,10,20,30,40,50,60,70,100]):
        self.batch_size = batch_size
        self.data_name=('data')
        self.label_name=('label')
        self.dataset_path=dataset_path
        self.image_path=image_path
        self.image_set=image_set
        # self.content = self.read_content(path)
        # print (path+'records number : ',len(self.content))
        self.num_label=num_label
        self.bucket_items, self.data_plan=self.get_bucket()
        # self.imagelabels=get_label_from_content(self.content,num_label)
        self.default_bucket_key = max(buckets)
        self.factor=4
        self.buckets=[5,10,20,30,40,50,60,70,100]
        self.imagepaths=np.array(self.imagepaths)
        self.imagelabels=np.array(self.imagelabels)
        self.bucket_images,self.bucket_labels,self.data_plan=self.make_buckets()
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, 3,data_shape[0],data_shape[1]))] + init_states
        self.provide_label = [('label', (batch_size,self.num_label ))]
        self.all_idx=range(len(self.imagepaths))
        self.current=0
        self.batch_size=batch_size
        self.size=len(self.data_plan)
        print('buckets')
        random.shuffle(self.data_plan)
    def get_bucket(self):
        path = os.path.join(self.dataset_path, '%s.txt' % self.image_set)
        bucket_items = {}
        with open(path) as ins:
            content = ins.read()
            content = content.split('\n')
            content.pop()
        print('size',self.size)
        buckets_len = len(self.buckets)
        # buckets_len = 9
        for i in range(buckets_len):
            bucket_items[i] = []
        for i,line in enumerate(content):
            line = line.strip()
            ll = line.split(' ')
            path = ll[0]
            label = ll[1:]
            buck = bisect.bisect_left(self.buckets, len(label)*3.5)
            bucket_items[buck].append([path,label])
        data_plan = []
        for bucket_idx in bucket_items:
            length_bucket = len(bucket_items[bucket_idx])
            print("bucket " + " length :", length_bucket)
            for idx in range(length_bucket // self.batch_size):
                data_plan.append([bucket_idx, idx])
        return bucket_items,data_plan



    def preprocess(self, img_path, index):
        # index for self.bucket
        img = cv2.imread(img_path)
        height,width = img.shape[0],img.shape[1]
        ratio = height/32
        width = width/ratio
        img_ori = cv2.resize(img, (width, 32))
        img_pad = np.ones((32, self.buckets[index]*8, 3)) * 255
        img_pad[:, :width, :] = img_ori
        img = img_pad.astype('float32')
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
      return self.imagepaths.shape[0]

    def reset(self):
        self.current = 0
        random.shuffle(self.data_plan)



    # def make_buckets(self):
    #     buckets_len=len(self.buckets)
    #     bucket_image_paths=[]
    #     for i in range(buckets_len):
    #         bucket_image_paths.append([])
    #
    #     buck = bisect.bisect_left(self.buckets, width // 8)

    # def make_buckets(self):
    #     print ("making buckets")
    #     buckets_len=len(self.buckets)
    #     bucket_images=[]
    #     bucket_labels=[]
    #     for i in range(buckets_len):
    #         bucket_images.append([])
    #         bucket_labels.append([])
    #     data_plan=[]
    #     max_ratio=25
    #     # data = []
    #     # print(self.imagepaths)
    #     for label_idx,img in tqdm(enumerate(self.imagepaths)):
    #         if not os.path.exists(os.path.join(self.dataset_path,self.image_path,img)):
    #             continue
    #         try:
    #             image=cv2.imread(os.path.join(self.dataset_path,self.image_path,img))
    #         except:
    #             continue
    #         shape=image.shape
    #         hight=shape[0]
    #         width=shape[1]
    #         ratio=(1.0*width/hight)
    #         if ratio>max_ratio:
    #             ratio=max_ratio
    #         if ratio<1:
    #             ratio=1
    #         base_hight = 32
    #         image = cv2.resize(image, (int(32 * ratio), 32))
    #         hight = 32
    #         width = int(32 * ratio)
    #         assert hight == base_hight
    #         image = np.transpose(image, (2, 0, 1))
    #         buck = bisect.bisect_left(self.buckets, width // 8)
    #         new_width = self.buckets[buck]*8
    #
    #         if width < new_width:
    #             new_img = np.zeros((3, base_hight, new_width))
    #             for i in range(3):
    #                 padding_value = int(np.mean(image[i][:][-1]))
    #                 z = np.ones((base_hight, new_width - width)) * padding_value
    #                 new_img[i] = np.hstack((image[i], z))
    #             image = new_img
    #         else:
    #             image = image
    #
    #         bucket_images[buck].append(image)
    #
    #         label_ = [l for l in self.imagelabels[label_idx]]
    #         label_tmp = [0 for _ in range(0, self.buckets[buck]-len(label_))]
    #         label_+=label_tmp
    #         bucket_labels[buck].append(label_)
    #     for bucket_idx,i in enumerate(bucket_images):
    #         length_bucket=len(i)
    #         print ("bucket "+" length :",length_bucket)
    #         for idx in range(length_bucket//self.batch_size):
    #             data_plan.append([bucket_idx,idx])
    #
    #     return bucket_images,bucket_labels,data_plan


