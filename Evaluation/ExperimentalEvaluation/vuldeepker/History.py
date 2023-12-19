from keras.callbacks import Callback
import time
class LossHistory(Callback):
	# def on_train_begin(self, logs={}):
	# 	self.losses = []
	# def on_batch_end(self, batch, logs={}):
	# 	self.losses.append(logs.get('loss'))
	def on_train_begin(self, logs={}):
		self.t0 = time.time()
		self.time = []
		self.acc = []

	# def on_batch_end(self, batch, logs={}):
	# 	self.time.append(time.time()-self.t0)
	# 	self.acc.append(logs.get('accuracy'))

	def on_epoch_end(self, epoch, logs={}):
		self.time.append(time.time()-self.t0)
		self.acc.append(logs.get('val_accuracy'))