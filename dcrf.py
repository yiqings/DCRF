#model.py
#implement DCRF inference 
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
#import tensorflow.keras.applications.resnet as resnet
import tensorflow.keras.layers as layers
import cv2
import matplotlib.pyplot as plt
import json
import io
import time
import unittest
import argparse

tf.keras.backend.set_floatx('float32')

def bilinear(img,p):
    '''
    implement bilinear process of roi deform
    Args:
        img:an image with size (h,w,channel) array_like
        p:the location of the pixel to commute (maybe fractional) array_like with shape(bins,2)
        
    Returns:
        value of the pixel in the img ,with shape(3,)
    '''
    h,w,c = img.shape
    bins,_ = p.shape
    values = np.zeros(shape = (bins,3),dtype = np.float32)
#     assert h == w
    img = tf.transpose(img,(2,0,1))
    for i in range(bins):
        time_now = time.time()
        x = np.zeros((h*w,2),np.float32)
        x[:,0] = np.repeat(np.arange(h),w)
        x[:,1] = np.tile(np.arange(w),h)
        #print(p.shape)
        g = G(x,p[i].reshape(1,2)).reshape(h,w)                               #shape (h*w,1)   
        #print(g.shape)
        #print(img)
        out = np.split(img,c,axis = 0)
        values[i] = np.array([np.sum(out[i]*g)for i in range(c)])
        #print(time.time()-time_now)
    return values            #return sum of each channel


def G(p,q):
    '''
    Args:
        p:a point,array_like with shape(m,2) 
        q:a point,array_like with shape(1,2)
        
    Returns:
        value of the kernel with shape (m,1)
    '''
    f = lambda x,y:np.maximum(0,1-np.abs(x-y))
    return f(p[:,0],q[:,0])*f(p[:,1],q[:,1])



class RoiDeform(keras.layers.Layer):
 
  def __init__(self, k,crop_size = 224,**kwargs):
    super(RoiDeform, self).__init__(**kwargs)
    self.k = k
    self.crop_size = crop_size
 
  def build(self, input_shape):
    shape = tf.TensorShape((2, self.k,self.k,2))
    # Create a trainable weight variable for this layer.
    self.delta_p =  self.add_weight(
                                  name='delta_p',
                                  dtype = tf.float32,
                                  shape = shape,
                                  initializer = tf.keras.initializers.TruncatedNormal(stddev = 20),
                                  trainable=True)
    print(self.delta_p.shape)
    # Be sure to call this at the end
#     super(Roi_Deform, self).build(input_shape)
  def call(self, inputs):
    '''
    Args:
        inputs: a tensor with shape(batch_size,img_height,img_width,img_channel)
    
    Returns:
        a tensor with shape(batch_size,num_patches,crop_height,crop_width,img_channel)
    '''
    batch_size,img_height,img_width,img_channel = inputs.shape
#     index = 0
    y = np.zeros((batch_size,self.k*self.k,self.crop_size,self.crop_size,img_channel),dtype = np.float32)
    for t in range(batch_size):
        print(t)
        index = 0
        for i in range(self.k):
            for j in range(self.k):
                # we use (img_height-crop_size*k)//2,(img_width-crop_size*k)//2表示起点
                pl = (img_height - self.crop_size*self.k)//2
                pr = (img_width - self.crop_size*self.k)//2
                #now we define the span patch
                span_1 = np.arange(pl + i*self.crop_size ,pl + (i + 1)*self.crop_size ) + self.delta_p[t,i,j,0]
                span_2 = np.arange(pr + j*self.crop_size ,pr + (j + 1)*self.crop_size ) + self.delta_p[t,i,j,1]
                p = np.zeros(shape = (self.crop_size*self.crop_size,2),dtype = np.float32)
                p[:,0] = np.tile(span_1,(1,self.crop_size))
                p[:,1] = np.tile(span_2,(1,self.crop_size))
                y[t,index] = bilinear(inputs[t],p).reshape((self.crop_size,self.crop_size,3))
                plt.imshow(y[t,index])
                plt.savefig('./test_%i.png')
                print('save successfully')
                index += 1
    return y


class CRF(keras.layers.Layer):
 
    def __init__(self, num_patches,iteration = 10, **kwargs):
        self.num_patches = num_patches
        self.iteration = iteration
        super(CRF, self).__init__(**kwargs)
 
    def build(self, input_shape):
        shape = tf.TensorShape((1, self.num_patches,self.num_patches))
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='W',
                                      shape=shape,
                                      initializer=tf.keras.initializers.zeros(),
                                      trainable=True)
        # Be sure to call this at the end
#         super(Roi_Deform, self).build(input_shape)
 
    def call(self, delta_p,logits):
        '''
        Args:
            delta_p:a tensor with shape(batch_size,num_grid,num_grid,2)
            logits:a tensor with shape(batch_size,num_patches,1)
            note that num_grid*num_grid = num_patches
        Returns:
            a tensor with shape(batch_size,num_patches,1)
        '''
        batch_size,num_grid,_,_ = delta_p.shape
        feats = tf.reshape(delta_p,(batch_size,num_grid*num_grid,2))
        feats_norm = tf.norm(feats,axis = 2,keepdims = True)#shape(batch_size,num_patches,1)
        pairwise_norm = tf.matmul(feats_norm,
                                  tf.transpose(feats_norm, (0,2,1)))#shape(batch_size,num_patches,num_patches)
        pairwise_dot = tf.matmul(feats, tf.transpose(feats, (0,2,1)))
        # cosine similarity between feats
        pairwise_sim = pairwise_dot / pairwise_norm
        # symmetric constraint for CRF weights
        feats_pow = tf.tile(tf.pow(feats_norm,2),(1,1,num_grid*num_grid))
        guassian = tf.exp(-(feats_pow+tf.transpose(feats_pow,(0,2,1)))/2)
        W_sym = (self.W + tf.transpose(self.W, (0,2,1))) / 2
        pairwise_potential = guassian * (1-pairwise_sim) * W_sym#shape(batch_size,num_patches,num_patches)
        unary_potential = logits

        for i in range(self.iteration):
            # current Q after normalizing the logits
            probs = tf.transpose(tf.sigmoid(logits), (0,2,1))
            # taking expectation of pairwise_potential using current Q
            pairwise_potential_E = tf.reduce_sum(
                probs * pairwise_potential - (1 - probs) * pairwise_potential,
                2, keepdims=True)
            logits = unary_potential + pairwise_potential_E

        return logits
    
class BasicBlock(keras.layers.Layer):
    def __init__(self,filter_num,stride=1):
        super(BasicBlock, self).__init__()
        self.conv1=layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
        self.bn1=layers.BatchNormalization()
        self.relu=layers.Activation('relu')

        self.conv2=layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride!=1:
            self.downsample=keras.models.Sequential()
            self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride))
        else:
            self.downsample=lambda x:x
    def call(self,input,training=None):
        out=self.conv1(input)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        identity=self.downsample(input)
        output=layers.add([out,identity])
        output=tf.nn.relu(output)
        return output

class ResNet(keras.Model):
    def __init__(self,layer_dims,num_classes=1,num_patches = 9,use_crf = True):
        super(ResNet, self).__init__()
        # 预处理层
        self.stem=keras.models.Sequential([
            layers.Conv2D(64,(3,3),strides=(1,1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')
        ])
        # resblock
        self.layer1=self.build_resblock(64,layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1],stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # there are [b,512,h,w]
        # 自适应
        self.avgpool=layers.GlobalAveragePooling2D()
        self.fc=layers.Dense(num_classes)
        self.crf = CRF(num_patches) if use_crf else None


    def call(self,input,delta_p = None,training=None):
        batch_size,patch_size,crop_size,crop_size,channel = input.shape
        x = tf.reshape(input,(-1,crop_size,crop_size,channel))
        x=self.stem(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        # [b,c]
        x=self.avgpool(x)
        feats = tf.reshape(x,(x.shape[0],-1))
        logits=self.fc(feats)
        feats = tf.reshape(feats,(batch_size,patch_size,-1))
        logits = tf.reshape(logits,(batch_size,patch_size,-1))
        if self.crf:
            logits = self.crf(delta_p,logits)
        logits = tf.squeeze(logits)
        return logits

    def build_resblock(self,filter_num,blocks,stride=1):
        res_blocks= keras.models.Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num,stride))
        # just down sample one time
        for pre in range(1,blocks):
            res_blocks.add(BasicBlock(filter_num,stride=1))
        return res_blocks

def resnet18():
    return  ResNet([2,2,2,2])


                                  
class DCRF(keras.Model):
    '''
    implement DERF model ,it will receive images and return possibility of images being tumor
    '''
    def __init__(self,k = 3):
        super(DCRF,self).__init__()
        self.roi_deform = RoiDeform(k)
        self.resnet = resnet18()

    def call(self,inputs):
        '''
        Args:
            inputs: a tensor with shape(batch_size,img_size,img_size,3)
        
        Returns:
            logits with shape(batch_size,grid_size)
        '''
        infer = self.roi_deform(inputs)#(batch_size,patc h_size,224,224,3).reshape(batch_size*patch_size,224,224,3)->
#         print(infer.dtype)
        #print(infer)
        delta_p = self.roi_deform.get_weights()[0]
        out = self.resnet(infer,delta_p)#(k,embedding_size)->reshape(batch_size,patch_size,embedding_size)
        return out


if __name__ == '__main__':
    x = tf.zeros(shape = (1,768,768,3),dtype = tf.float32)
    model = DCRF()
    model(x)
    train = model.trainable_variables
    #print(len(train))
    variable_names = [v.name for v in train]
    print(variable_names)
    model.summary() 
