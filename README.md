### GAN - Generative Adversarial Network

##### 구분자 Discriminator
```python
    def discriminator(inputs):
        hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    
        return output
```

##### 생성자 Generator
```python
    def generator(noise_z):
        hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
        output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
    
        return output
```

##### 이미지와 노이즈-이미지가 있다. 생성자는 노이지-이미지를 받아 실제 이미지와 같은 크기(Size)의 결과값을 출력하도록 한다. 구분자는 진짜와 얼마나 가까운가를 판단하는 값으로 0~1 사이의 값을 출력한다. GAN의 핵심은 생성자로 만든 노이지-이미지를 구분자가 가짜(0)이라고 판단하게끔, 그리고 진짜(1)라고 판단하게끔 하여 균형을 맞추도록 하는 것이다.

```python
    G = generator(Z)
    D_gene = discriminator(G)
    D_real = discriminator(X)
    
    loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1-D_gene))
    loss_G = tf.reduce_mean(tf.log(D_gene))

    train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
    train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)
```