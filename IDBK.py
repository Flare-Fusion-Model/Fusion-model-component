def identity_block(X,f,filters,stage,block):
    
    conv_name_base = "res"+str(stage)+block+"_branch"
    bn_name_base = "bn"+str(stage)+block+"_branch"
 
    F1,F2,F3 = filters
 
    X_shortcut = X
 
    X = Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding="valid",
               name=conv_name_base+"2a",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)   
    X = Activation("relu")(X)
 
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding="same",
               name=conv_name_base+"2b",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)
 
    X = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding="valid",
               name=conv_name_base+"2c",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)
 
    X = layers.add([X,X_shortcut])
    X = Activation("relu")(X)
 
    return X
 
tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    A_prev = tf.placeholder("float",shape=[3,4,4,6])
    X = np.random.randn(3,4,4,6)
    A = identity_block(A_prev,f=2,filters=[2,4,6],stage=1,block="a")
    sess.run(tf.global_variables_initializer())
    out = sess.run([A],feed_dict={A_prev:X,K.learning_phase():0})
    print("out = "+str(out[0][1][1][0]))
