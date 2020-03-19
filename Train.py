#train.py
#you can use python train.py --config ***/default = ../config/config.json
#some imitate from https://github.com/baidu-research/NCRF/blob/master/wsi/bin/train.py
import tensorflow as tf
import dcrf
import data
import json
import logging
import argparse
import time
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--config',default = '../config/config.json')
args = parser.parse_args()
config = args.config
f = open(config,'r')
config = json.load(f)
f.close()


train_dataset = data.GetDataset(config['train'],config['batch_size'])       #define dataset loader
valid_dataset = data.GetDataset(config['valid'],config['batch_size'])
# for data,label in train_dataset:
#     print(data.numpy(),label.numpy())
# print(train_dataset)
model = dcrf.DCRF()                                                             #define our inference model
optimizer = tf.keras.optimizers.SGD(learning_rate = config['lr'],
                                    momentum = config['momentum'])         #define the optimizer

# @tf.function()    
def train(summary,summary_writer):
#     train_dataset = data.GetDataset('../Dataset/',4)  
    time_now = time.time()
#     global train_dataset
    for i,(datas,labels) in enumerate(train_dataset):
#         print(datas)
#         sess = tf.compat.v1.Session()
#         sess.run(datas)
#         datas,labels = next(iter(train_dataset))
#         print(datas.numpy())
        with tf.GradientTape() as Tape:
            inference = model(datas)
            print(inference)
            probs = tf.sigmoid(inference)
            #here we get (batch_size,patch_size,1) represents the probility
            loss = -tf.reduce_sum(tf.math.log(probs))
            predict = tf.cast((probs>=0.5),tf.int32)
        grad = Tape.gradient(loss,model.trainable_variables)
        #print(model.trainable_variables[0])
        print(grad[0])
        #grad[0] = tf.cast(grad[0],tf.int32)
        optimizer.apply_gradients(zip(grad,model.trainable_variables))
        labels = tf.tile(tf.reshape(labels,(-1,1)),(1,inference.shape[1]))
        acc = tf.reduce_sum(tf.cast(tf.equal(labels,predict),tf.float32)) / (labels.shape[0]*labels.shape[1])
#         auc = tf.metrics.AUC()
#         auc.update_state(labels,predict)
        f = open(config['inference'] + 'epoch%d_batch_%d.npy'%(summary['epoch']+1,i+1),'wb')
        pickle.dump(probs.numpy(),f)
        f.close()
        time_spent = time.time() - time_now
        time_now = time.time()
        logging.info(
            '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
            'Training Acc : {:.3f}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                summary['step'] + 1, loss.numpy(), acc.numpy(), time_spent))

        
        summary['step'] += 1
        if summary['step'] % config['log_every'] == 0:
            with summary_writer.as_default():
                tf.summary.scalar('train/loss', loss.numpy(), summary['step'])
                tf.summary.scalar('train/acc', acc.numpy(), summary['step'])
#                 tf.summary.scalar('

    summary['epoch'] += 1
    return summary

# @tf.function()
def valid(summary):
    loss_sum = 0
    acc_sum = 0
    for i,(datas,labels) in enumerate(valid_dataset):
        #when do valid,we should return the loss and acc until one epoch ends
        inference = model(datas)
        #print(inference.shape)
        probs = tf.sigmoid(inference)
        predict = tf.cast((probs>=0.5),tf.int32)
        print(predict.shape)
        loss = -tf.reduce_sum(tf.math.log(probs))
        #labels = tf.tile(tf.reshape(labels,(-1,1)),(1,inference.shape[0]))
        acc = tf.reduce_sum(tf.cast(tf.equal(labels,predict),tf.float32)) / (labels.shape[0]*labels.shape[1])
        loss_sum += loss.numpy()
        acc_sum += acc.numpy()
    
    summary['loss'] = loss_sum / (i+1)
    summary['acc'] = acc_sum / (i+1)
    return summary

def main():
    #we should calculate the acc and auc and froc of the model,each epoch we do valid on validation set
    logging.basicConfig(level=logging.INFO)
    try:
        train_log_dir = config['train_log']
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#         test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        summary_train = {'epoch': 0, 'step': 0}
        summary_valid = {'loss': float('inf'), 'acc': 0}
        
        for eps in range(config['epochs']):
            #f = open(config['inference'] + 'epoch%d.npy'%eps,'wb')
            summary_train = train(summary_train,train_summary_writer)
            #f.close()
            model.save_weights(config['weight_path'])
            time_now = time.time()
            summary_valid = valid(summary_valid)
            time_spent = time.time() - time_now

            logging.info(
                '{}, Epoch : {}, Step : {}, Validation Loss : {:.5f}, '
                'Validation Acc : {:.3f}, Run Time : {:.2f}'
                .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                    summary_train['step'], summary_valid['loss'],
                    summary_valid['acc'], time_spent))
            with train_summary_writer.as_default():
                tf.summary.scalar('valid/loss', summary_valid['loss'], step=eps)
                tf.summary.scalar('valid/accuracy', summary_valid['acc'], step=eps)
    except KeyboardInterrupt:
        model.save_weights(config['weight_path'])  #if ctrl + c ,save model_weights

if __name__ == '__main__':
    main()
