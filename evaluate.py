import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,roc_auc_score
import scikitplot as skplt
import matplotlib.pyplot as plt


data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

# run test data through model
preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(
            CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
plt.show()


predic=model.predict(test_points)
predic=tf.argmax(predic,-1)
s=confusion_matrix(test_labels,predic)
disp=ConfusionMatrixDisplay(s)
disp.plot()
#Paste the path where you want to save the confusion matrix
plt.savefig("conf_matrix.png") 


skplt.metrics.plot_roc_curve(test_labels,tf.keras.utils.to_categorical(predic,num_classes=NUM_CLASSES), figsize=(14,10), 
                             title='ROC Curve')
#Paste the path you want to save the roc-curve to.
plt.savefig("roc_bri_tri_lung_4096.png")
plt.show()

roc_auc_score(tf.keras.utils.to_categorical(test_labels,num_classes=3),tf.keras.utils.to_categorical(predic,num_classes=3))