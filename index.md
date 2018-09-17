It is super convenient to train a model and get some analytical results in Python. And you can serve the model in Java applications use 

### PMML
A PMML file seems a bit old fashioned. However it's friendly to more libraries not equipping Java API.

[@jpmml](https://github.com/jpmml) has been indeed helpful in PMML conversions:

- scikit-learn to PMML: https://github.com/jpmml/sklearn2pmml
- R to PMML: https://github.com/jpmml/r2pmml
- PySpark to PMML: https://github.com/jpmml/pyspark2pmml

And if you wonder how PMML file looks like:

```
<PMML version="4.2.1" xmlns="http://www.dmg.org/PMML-4_2"><Header><Timestamp>2018-03-05 11:00:50.239006</Timestamp></Header><DataDictionary><DataField dataType="string" name="class" optype="categorical"><Value value="y0"/></DataField>...</DataDictionary>
<NeuralNetwork activationFunction="logistic" functionName="classification"><MiningSchema><MiningField name="class" usageType="target"/>...</NeuralNetwork>
</PMML>
```

### Tensorflow API
Tensorflow has been widely used in model training these days. And its generated `.pb` file could be called via its Java API.
Note that you should define placeholders so that you could use them in Java program later. Here is an example implemented in hotel recommendation system:
```
# Define placeholder
X = tf.placeholder(tf.float32, shape = [None, 118], name="features")
y = tf.placeholder(tf.int32, shape = [None, 2], name = "click")

# Define variables
w1 = weight_variable([118, hidden_units_1])
b1 = bias_variable([hidden_units_1])
w2 = weight_variable([hidden_units_1, hidden_units_2])
b2 = bias_variable([hidden_units_2])
w3 = weight_variable([hidden_units_2, hidden_units_3])
b3 = bias_variable([hidden_units_3])
w4 = weight_variable([hidden_units_3, hidden_units_4])
b4 = bias_variable([hidden_units_4])
w5 = weight_variable([hidden_units_4, 2])
b5 = bias_variable([2])

# Define network
# Hidden layer
z1 = tf.add(tf.matmul(X, w1), b1)
a1 = tf.nn.relu(z1)
z2 = tf.add(tf.matmul(a1, w2), b2)
a2 = tf.nn.relu(z2)
z3 = tf.add(tf.matmul(a2, w3), b3)
a3 = tf.nn.relu(z3)
z4 = tf.add(tf.matmul(a3, w4), b4)
a4 = tf.nn.softmax(z4)
z5 = tf.add(tf.matmul(a4, w5), b5)
y_pred = tf.nn.softmax(z5, name = "prediction")
```
Also we need define suitable loss function and optimizer. Then we can train and save model file in Python:

```
builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
with tf.Session() as sess:
    if os.path.exists(model_dir + "checkpoint"):
        saver.restore(sess,model_dir + model_name)
        print("model load successfully")
    else:
        sess.run(init)
        writer = tf.summary.FileWriter("./data/graph", sess.graph)
    #Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(x_train)/batch_size)

        for i in range(total_batch):
            batch_x = x_train[i*batch_size: (i+1)*batch_size]
            batch_y = y_train[i*batch_size: (i+1)*batch_size]
            _, c = sess.run([optimizer, loss], feed_dict = {X: batch_x, y: batch_y})
            avg_cost += c / total_batch
        epoch_train_accuracy = accuracy.eval(feed_dict={X: x_train, y: y_train})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost), "Train_accuracy={:.4f}".format(epoch_train_accuracy))
    print("Optimization Finished!")
    builder.save()```
    
    
    Then we will get folder:
    
    ```admins-MacBook-Pro:tf_model chloe$ tree .
.
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index
1 directory, 3 filesUse generated files
```
    
   Let us use it in Java:
```
    trainingMatrix = PrepareCandidate.trainingMatrix(hotelId,
                candidateID,
                partialFeatures,
                site,
                source,
                Double.valueOf(partnerPrice),
                hotelPrice,
                checkinDate,
                checkoutDate); # prepare feature matrix
       
        final String basePath = getClass().getResource("/")
                .getPath() + "tf_model";

        Session session = SavedModelBundle.load(basePath, "htlrec").session();

        Tensor x = Tensor.create(trainingMatrix);
        System.out.println(Arrays.toString(trainingMatrix[0]));
        float[][] results = new float[redisCandidates.length][1];
        Tensor pred = Tensor.create(results);

        float[][] outputs = session.runner()
                .feed("features", x)
                .feed("click", pred)
                .fetch("prediction")
                .run()
                .get(0)
                .copyTo(new float[redisCandidates.length][2]);
                
```        
Sorting the scores we will get the ranking results. The code is much neater than PMML ones, also it's batch training and the whole process is pretty quick (30ms for a 50*118 matrix in our case). The drawback is it only supports `.pb` file.
There are other methods that we could use python model in Java program, such as [DL4J](https://deeplearning4j.org/docs/latest/keras-import-overview) supports Keras. Also you can choose Scala other than Python+Java.
