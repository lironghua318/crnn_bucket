# decoding: utf-8
import cv2
import mxnet as mx
import numpy as np
from PIL import Image
import bisect
from crnn.symbols.crnn import crnn_lstm

class TextIter(mx.io.DataIter):
    def __init__(self, image_path,batch_size,
                 init_states,num_label=15,data_shape=[32,240],buckets=[5,10,20,30]):
        super(TextIter, self).__init__()
        self.data_name=('data')
        self.label_name=('label')
        self.image_path=image_path
        self.batch_size = batch_size
        self.num_label=num_label
        self.buckets = buckets
        self.bucket_items, self.data_plan=self.get_bucket()
        # self.imagelabels=get_label_from_content(self.content,num_label)
        self.default_bucket_key = max(buckets)
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, 3,data_shape[0],data_shape[1]))] + init_states
        self.provide_label = [('label', (batch_size,self.num_label ))]
        self.current=0
        self.size=len(self.data_plan)
        # random.shuffle(self.data_plan)
    def get_bucket(self):
        bucket_items = {}
        for path in self.image_path:
            image = cv2.imread(path)
            h = image.shape[0]
            w = image.shape[1]
            ratio = h/32
            w_new = int(w/ratio)
            comp = np.array([abs(w_new-bucket*8) for bucket in self.buckets])
            print(comp)
            buck = np.where(comp[:] == np.min(comp))[0][0]
            print(buck)
            # if  w_new//8 >= self.buckets[-1]:
            #     buck = len(self.buckets)-1
            # else:
            #     buck = bisect.bisect_left(self.buckets, w_new//8)-1
            # print(buck)
            bucket_items[buck]=[path]
        data_plan = []
        for bucket_idx in bucket_items:
            length_bucket = len(bucket_items[bucket_idx])
            print("bucket " + " length :", length_bucket)
            for idx in range(length_bucket):
                data_plan.append([bucket_idx, idx])
        return bucket_items,data_plan

    def preprocess(self, img_path, index):
        # index for self.bucket
        # print(img_path)
        img = cv2.imread(img_path)
        img_ori = cv2.resize(img, (self.buckets[index]*8, 32))
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
            # init_state_names = [x[0] for x in self.init_states]
            current_batch=self.data_plan[i]
            buck_idx=current_batch[0]
            img_idx=current_batch[1]
            batch_datas = self.bucket_items[buck_idx][img_idx*self.batch_size:img_idx*self.batch_size+self.batch_size]
            data = []
            label = []
            for batch_data in batch_datas:
                image_path = batch_data
                # image_label = batch_data[1]
                img = self.preprocess(image_path,buck_idx)
                data.append(img)
                label.append(np.zeros(self.buckets[buck_idx]//2, int))

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            self.current+=1
            return mx.io.DataBatch(data=data_all, label=label_all,
                                   pad=0, bucket_key=self.buckets[buck_idx],
                                   provide_data=[mx.io.DataDesc(name=self.data_name,
                                                                shape=data_all[0].shape)] + self.init_states,
                                   provide_label=[mx.io.DataDesc(name=self.label_name, shape=label_all[0].shape)])


def sym_gen(seq_len):
    return crnn_lstm(seq_len)

class predict():
    def __init__(self, images, data_shape, model_name, from_epoch,num_hidden,sym_gen,num_label):
        ctx = [mx.cpu(0)]
        self.char_lists=['blank','0','1','2','3','4','5','6','7','8','9']
        self.img = images
        self.BATCH_SIZE = 1
        self.num_hidden = num_hidden
        # self.seq_len = seq_len
        num_lstm_layer = 2
        self.data_shape = data_shape
        init_c = [('l%d_init_c' % l, (self.BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer * 2)]
        init_h = [('l%d_init_h' % l, (self.BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer * 2)]
        init_states = init_c + init_h
        # self.to_predict = OCRIter(self.BATCH_SIZE, 5991, data_shape, num_label, init_states, images)
        self.to_predict = TextIter(images,self.BATCH_SIZE,init_states)
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, from_epoch)
        self.module = mx.mod.BucketingModule(sym_gen=sym_gen, default_bucket_key=30, context=ctx)

        # provide_data = [('data', (self.BATCH_SIZE, 3, 32, 800))] + init_states
        # provide_label = [('label', (self.BATCH_SIZE, num_label))]
        #self.to_predict = OCRIter(self.BATCH_SIZE, len(charset) + 1, data_shape, num_label, init_states, images)
        self.module.bind(self.to_predict.provide_data, self.to_predict.provide_label, for_training=False)
        # model.bind(data_shapes=provide_data, label_shapes=provide_label, for_training=False)
        self.module.init_params(arg_params=arg_params, aux_params=aux_params)

    def __get_string(self, label_list):
        # Do CTC label rule
        # CTC cannot emit a repeated symbol on consecutive timesteps
        ret = []
        label_list2 = [0] + list(label_list)
        for i in range(len(label_list)):
            c1 = label_list2[i]
            c2 = label_list2[i + 1]
            if c2 == 0 or c2 == c1:
                continue
            ret.append(c2)
        s = ''
        for l in ret:
            c = self.char_lists[l]
            s += c
        return s

    def run(self):
        for input in self.to_predict:
            if input is not None:
                self.module.forward(input, is_train=False)
                preds = self.module.get_outputs()
                pred_label = preds[0].asnumpy().argmax(axis=1)
                # print(pred_label)
                strs=''
                for i in range(len(pred_label)):
                    # print(prob[i])
                    max_index = pred_label[i]
                    # print(max_index)
                    if i < len(pred_label) - 1 and pred_label[i] == pred_label[i + 1]:
                        continue
                    if max_index != 0:
                        strs += self.char_lists[max_index]
                print(strs)
            else:
                break


if __name__ == '__main__':
    files = [
        # 'han.jpg'
        # '0000.jpg'
        '9529.jpg'
        # '28216766.jpg'
        # '996527452444.jpg'
    ]
    # images = [cv2.imread(x,0) for x in files]
    # images, data_shape, model_name, from_epoch, num_hidden, sym_gen):
    my_predictor = predict(files,(240,32),'model/digit',40,100,sym_gen,15)#23-5990
    my_predictor.run()

