from chainer.datasets import mnist
# CPUで画像を保存したいときは、AGGで宣言させて、画像の保存コマンドを使う(予想)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# データセットがダウンロード済みでなければ、ダウンロードも行う
train_val, test = mnist.get_mnist(withlabel=True, ndim=1)

# データの例示
x, t = train_val[0]  # 0番目の (data, label) を取り出す
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()
print('label:', t)
