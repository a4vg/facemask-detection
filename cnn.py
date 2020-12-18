from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import utils

class CNN():
  def __init__(self, w=None, h=None, train_dirs=None, test_dirs=None, modelfile=None, preprocessfile=None):
    if modelfile is None:
      self.w = w
      self.h = h
      self.preprocess(train_dirs, test_dirs, preprocessfile)
      self.build()
    else:
      self.model = load_model( modelfile )

  def preprocess(self, train_dirs, test_dirs, preprocessfile):
    print("[IN PROGRESS] Preprocessing data")

    if preprocessfile is not None:
      with open(preprocessfile, "rb") as f:
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.classes = pickle.load(f)
      return

    print("[IN PROGRESS] Preprocessing train data")
    X_train, labels = utils.load_dataset(train_dirs,
                    [utils.resize(50,50), img_to_array],
                    verbose=1000
                  )
    self.X_train = X_train.astype( "float" )/255.0
    self.classes = list(set(labels))
    self.Y_train = utils.binarize_labels(labels, self.classes)
    print("[DONE] Preprocess train data")

    print("[IN PROGRESS] Preprocessing test data")
    X_test, labels = utils.load_dataset(train_dirs,
                    [utils.resize(50,50), img_to_array],
                    verbose=1000
                  )
    self.X_test = X_test.astype( "float" )/255.0
    self.classes = list(set(labels))
    self.Y_test = utils.binarize_labels(labels, self.classes)
    print("[DONE] Preprocess test data")

    with open("preprocess.pickle", "wb") as f:
      pickle.dump([self.X_train, self.X_test, self.Y_train, self.Y_test, self.classes], f)

    print("[DONE] Preprocess data")

  def build(self):
    print("[IN PROGRESS] Building model")

    self.model = Sequential([
      Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=self.X_train[0].shape),
      Conv2D(64, kernel_size=(3,3), activation='relu'),
      MaxPooling2D( pool_size=(2,2) ),

      Flatten(),
      Dense(64, activation='relu'),
      Dense(20, activation='relu'),
      Dense( len(self.classes), activation='softmax' )
    ])

    self.model.compile( loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics = ['accuracy'] )

    print("[DONE] Build model")

  def fit(self, epochs, model_exportfile, metric_exportimg, verbose=1):
    print("[IN PROGRESS] Fitting model")

    print("[IN PROGRESS] Actually fitting model")
    results = self.model.fit( self.X_train, self.Y_train, validation_data=(self.X_test, self.Y_test), epochs=epochs, verbose=verbose )
    print("[DONE] Actually fit model")

    self.model.save( model_exportfile )
    self.plot_metrics(results, epochs, metric_exportimg)
    print(f"[DONE] Fit model.\nModel saved to {model_exportfile}. Metrics plot in {metric_exportimg}")
    print("[DETAILS] Classes: ", self.classes)

  def plot_metrics(self, results, epochs, metric_exportimg):
    plt.style.use('ggplot')
    plt.figure( figsize=(24,12) )
    plt.plot( np.arange(0,epochs), results.history['loss'], label='entrenamiento:loss' )
    plt.plot( np.arange(0,epochs), results.history['val_loss'], label='validación:loss' )
    plt.plot( np.arange(0,epochs), results.history['accuracy'], label='entrenaiento:accuracy' )
    plt.plot( np.arange(0,epochs), results.history['val_accuracy'], label='validación:accuracy' )
    plt.title('Modelo CNN')
    plt.xlabel('# épocas')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.savefig(metric_exportimg)


  def validate(self, validation_dirs):
    print("[IN PROGRESS] Validating model")
    print("[IN PROGRESS] Loading dataset ")
    X_valid, labels = utils.load_dataset(validation_dirs,
                    [utils.resize(50,50), img_to_array],
                    verbose=1000
                  )
    X_valid = X_valid.astype( "float" )/255.0
    Y_valid = utils.binarize_labels(labels, self.classes)
    print("[DONE] Load dataset")

    print("[IN PROGRESS] Predicting ")
    predicciones = self.model.predict( X_valid )
    print("[DONE] Predict")

    print("[DONE] Validating model\n")

    print( classification_report(np.array(Y_valid).argmax(axis=1),
            predicciones.argmax(axis=1), target_names=self.classes)
    )

  def predict_single(self, img_path):
    X = utils.preprocess_single(cv2.imread( img_path ), [utils.resize(50,50), img_to_array])
    X = X.astype( "float" )/255.0
    prediction = self.model.predict( X )
    print("Prediction:", prediction.argmax(axis=1))



'''
# Train
cnn = CNN(50, 50, train_dirs=["dataset/Train/WithMask", "dataset/Train/WithoutMask"], test_dirs=["dataset/Test/WithMask", "dataset/Test/WithoutMask"])
cnn.fit(10, "shallow.hdf5", "results.png", verbose=1)
'''

'''
# Validate trained
cnn = CNN(modelfile="out/cnn.hdf5")
cnn.classes = ['WithoutMask', 'WithMask']
cnn.validate(["dataset/Validation/WithMask", "dataset/Validation/WithoutMask"])
'''



