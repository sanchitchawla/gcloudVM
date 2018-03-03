
import tensorflow as tf
import numpy as np
import pandas as pd
from multiprocessing import Process,Manager
import time
graph=tf.Graph()

start_time = time.time()



# Define the CNN model. Give the train data it will train the model and after training it will use test data to predict the test label
def CNN_Part(number,training_data,training_labels,validation_data,return_dict):
#Paramters


        print ("start")
        num_lables=10
        image_size=28
        
        learning_rate=1e-2
        batch_size=50
        
        #Read Data
        #data=pd.read_csv('train.csv',header=0)
         

        
        #helper methods
        def weights(shape):
            return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
        
        def biases(shape):
            return tf.Variable(tf.constant(0.1,shape=shape))
        
        def conv2d(data,wts):
            return tf.nn.conv2d(data,wts,strides=[1,1,1,1],padding='SAME',use_cudnn_on_gpu=False)
        
        def max_pooling(value):
            return tf.nn.max_pool(value,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
        
        
        
        #Define the CNN model
        global graph
        graph=tf.Graph()
        with graph.as_default():
        
        
            tf_train_data=tf.placeholder(tf.float32,shape=[batch_size,image_size*image_size])
            tf_train_labels=tf.placeholder(tf.float32,shape=[batch_size,num_lables])
        
            tf_valid_data=tf.constant(validation_data)
            tf_test_data=tf.placeholder(tf.float32,shape=[batch_size,image_size*image_size])
            
            
            layer1_w=weights([5,5,1,32])
            layer1_b=biases([32])
        
            #Layer2
            layer2_w=weights([5,5,32,64])
            layer2_b=biases([64])
        
            #fully connected
            fully_w=weights([7*7*64,1024])
            fully_b=biases([1024])
            final_w=weights([1024,num_lables])
            final_b=biases([num_lables])
        
        
            def model(data):
        
     
                image=tf.reshape(data,[-1,image_size,image_size,1])
        
                hidden1_conv=tf.nn.relu(conv2d(image,layer1_w)+layer1_b)
                hidden1_pool=max_pooling(hidden1_conv)
        
                hidden2_conv=tf.nn.relu(conv2d(hidden1_pool,layer2_w)+layer2_b)
                hidden2_pool=max_pooling(hidden2_conv)
        
                hidden2_pool_flat=tf.reshape(hidden2_pool,[-1,7*7*64])
                hidden_fully=tf.nn.relu(tf.matmul(hidden2_pool_flat,fully_w)+fully_b)
                
                #dropout
                hidden_final=tf.nn.dropout(hidden_fully,keep_prob=1)
                
                
                return  tf.matmul(hidden_final,final_w)+final_b
        
            print ("start2")
            logits=model(tf_train_data)
            
            loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = tf_train_labels))
            
            optimization=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        
            train_predections=tf.nn.softmax(logits)            
            valid_parameter=model(tf_valid_data)
            
            
           
        # Give the parameter for training steps
        num_steps=200
        
        
        
        
        with tf.Session(graph=graph) as session:
            
            #Initialize the model
            tf.initialize_all_variables().run()
            print('initialized')
        
            counter=0
            for step in range(num_steps):
                if counter>len(training_data):
                    prem=np.arange(len(training_data))
                    np.random.shuffle(prem)
                    training_data=training_data[prem]
                    training_labels=training_labels[prem]
                    counter=0
                offset = (step * batch_size) % (training_labels.shape[0] - batch_size)
                batch_data = training_data[offset:(offset + batch_size), :]
                batch_labels = training_labels[offset:(offset + batch_size), :]
        
                feed_dict={tf_train_data:batch_data,tf_train_labels:batch_labels}
                
                print ("Step",counter)

                # Train the data with one batch of the data and updata the weights in every step
                # Then when training process has been finished, use the trained model to predict the test label and return the weighs from last layer
                _,parameter= session.run([optimization,valid_parameter], feed_dict=feed_dict)
                

          
                counter+=1

        
        # return the weights 
        return_dict[number]=parameter
        
        print (parameter)
        return parameter



def splitdata(data):


        #Split data into Training data1, Training data2, and Test Data
        data1=data[1:2000]
        data2=data[10000:20000]
        data3=data[20000:30000]
        
        images1=np.array(data1.drop('label',axis=1)).astype(np.float32)
        images1=np.multiply(images1,1.0/255.0)
        
        labels1=np.array(data1.label).astype(np.float32)        
        labels1=(np.arange(10)==labels1[:,None]).astype(np.float32)
        
        
        
        images2=np.array(data2.drop('label',axis=1)).astype(np.float32)
        images2=np.multiply(images2,1.0/255.0)
        
        labels2=np.array(data2.label).astype(np.float32)        
        labels2=(np.arange(10)==labels2[:,None]).astype(np.float32)
        
        
        
        images3=np.array(data3.drop('label',axis=1)).astype(np.float32)
        images3=np.multiply(images3,1.0/255.0)
        labels3=np.array(data3.label).astype(np.float32)        
        labels3=(np.arange(10)==labels3[:,None]).astype(np.float32)

        

        training_data1=images2
        training_labels1=labels2
        
        training_data2=images3
        training_labels2=labels3
        
        
        
        validation_data=images1
        validation_labels=labels1
        
        
        return (training_data1,training_labels1,training_data2,training_labels2,validation_data,validation_labels)
    



# Give the CNN weights, this function will build the CNN model and return the training accuracy 
def Caculate_P(someweight,validation_data,validation_labels):
#Paramters
        num_lables=10
        image_size=28
        
        learning_rate=1e-2
        batch_size=50
        
        #helper methods
        def weights(shape):
            return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
        
        def biases(shape):
            return tf.Variable(tf.constant(0.1,shape=shape))
        
        def conv2d(data,wts):
            return tf.nn.conv2d(data,wts,strides=[1,1,1,1],padding='SAME',use_cudnn_on_gpu=False)
        
        def max_pooling(value):
            return tf.nn.max_pool(value,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
        
        def accuracy(predictions, labels):
          return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                  / predictions.shape[0])



        # Define the CNN model and parameter
        with graph.as_default():
        
        
            tf_train_data=tf.placeholder(tf.float32,shape=[batch_size,image_size*image_size])
            tf_train_labels=tf.placeholder(tf.float32,shape=[batch_size,num_lables])
        
            tf_valid_data=tf.constant(validation_data)
            tf_test_data=tf.placeholder(tf.float32,shape=[batch_size,image_size*image_size])
            # tf_valid_labels=tf.placeholder(tf.float32,shape=[baCNN_Parttch_size,num_lables])
        
            layer1_w=weights([5,5,1,32])
            layer1_b=biases([32])
        
            #Layer2
            layer2_w=weights([5,5,32,64])
            layer2_b=biases([64])
        
            #fully connected
            fully_w=weights([7*7*64,1024])
            fully_b=biases([1024])
            final_w=weights([1024,num_lables])
            final_b=biases([num_lables])
        

            def model(data):
        
                image=tf.reshape(data,[-1,image_size,image_size,1])
        
                hidden1_conv=tf.nn.relu(conv2d(image,layer1_w)+layer1_b)
                hidden1_pool=max_pooling(hidden1_conv)
        
                hidden2_conv=tf.nn.relu(conv2d(hidden1_pool,layer2_w)+layer2_b)
                hidden2_pool=max_pooling(hidden2_conv)
        
                hidden2_pool_flat=tf.reshape(hidden2_pool,[-1,7*7*64])
                hidden_fully=tf.nn.relu(tf.matmul(hidden2_pool_flat,fully_w)+fully_b)
                
                #dropout
                hidden_final=tf.nn.dropout(hidden_fully,keep_prob=1)
                
                
                return  tf.matmul(hidden_final,final_w)+final_b
        

           
         
        with tf.Session(graph=graph) as session:
            

            position=tf.nn.softmax(someweight)
            
            position=position.eval()
            


            #Position is the encrypted predicted label        
            print (position)      
                            
            print('Validation accuracy: %.1f%%' % accuracy(position, validation_labels))  
            
            
            return position




      



if __name__ == '__main__': 
     time0=time.clock()
     print (time0)
     manager = Manager()
     return_dict = manager.dict()
     jobs = []
    
    #read the data 
     data=pd.read_csv('train_test.csv',header=0)
     
     #split data into training data1, training data2, test data
     dataset=splitdata(data)
     

     #parallel computing 

     #Use first two CPU cores to caculate the first part of training data
     batch1=Process(target=CNN_Part,args=(1,dataset[0],dataset[1],dataset[4],return_dict))
     jobs.append(batch1)
     batch1.start()
    
     #Use another two CPU cores to caculate another part of training data
     batch2=Process(target=CNN_Part,args=(2,dataset[2],dataset[3],dataset[4],return_dict)) 
     jobs.append(batch2)
     batch2.start()
 
     for proc in jobs:
        proc.join()
      

     # Add weight1 from data1 to weight2 from data2.
     # Then get the final weight

     finalpare=return_dict.get(1)+return_dict.get(2)
     # Put the final weight into the function to build the CNN model and return the training accuracy

     Caculate_P(finalpare,dataset[4],dataset[5])
     
     save=time.time() - start_time
     

     # Print the running time
     print ("Total Time:", save)
 
