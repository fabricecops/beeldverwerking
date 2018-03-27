
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,Callback,CSVLogger
from keras.callbacks import TensorBoard, History
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from src.models.tests.model.FS_manager import FS_manager
import time
import warnings
import numpy as np


class model(FS_manager):

    def __init__(self,dict_p=None,path=None):

        if(dict_p != None):
            FS_manager.__init__(self, dict_p['path_save'])
            self.pickle_save(dict_p,'dict')
            self.dict_p  = dict_p

        elif(path != None):
            FS_manager.__init__(self,path)
            self.dict_p = self.pickle_load(self.return_path_dict())
        else:
            print('WRONG CONF, GIVE PATH OR DICT')
        print(self.return_TB_command())
    #### return functions ####
    def save_model(self):

        path = self.return_path_model()
        self.model.save(path)

    def return_model(self):

        return self.model

    def load_model(self):

        return load_model(self.return_path_model())

    def return_TB_command(self):
        path    = '~/Dropbox/Code/Beeldverwerking/git'+self.return_path_TB()[1:]
        command = 'tensorboard --logdir '+path

        return command

    #### private functions #####
    def conf_optimizer(self,optimizer):
        if optimizer == 'adam':
            return Adam(lr=self.dict_p['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    def create_callbacks(self):
        callbacks = []

        if (self.dict_p['MT'] == True):
            time_CB = TimeHistory()
            callbacks.append(time_CB)

        if(self.dict_p['hist'] == True):
            hist   = History()
            callbacks.append(hist)

        if(self.dict_p['ES'] == True):
            ES         = EarlyStopping(patience=self.dict_p['ES_patience'],
                                    verbose=0)
            callbacks.append(ES)

        if(self.dict_p['TB'] == True):
            test_datagen = ImageDataGenerator(rescale=self.dict_p['rescale'])

            validation_generator = test_datagen.flow_from_directory(
                self.dict_p['validation_data_dir'],
                target_size=(self.dict_p['img_width'], self.dict_p['img_height']),
                batch_size=self.dict_p['batch_size'],
                color_mode=self.dict_p['colormode'],
                shuffle=self.dict_p['shuffle_val'])


            tensorboard = TensorBoardWrapper(validation_generator,20,
                                      log_dir=self.return_path_TB(),
                                      histogram_freq=1,
                                      write_graph=True,
                                      write_images=True,
                                      embeddings_freq=0,
                                      write_grads=True,
                                      embeddings_layer_names=True,
                                      embeddings_metadata=True)

            # tensorboard = TensorBoard(
            #                           log_dir=self.return_path_TB(),
            #                           histogram_freq=0,
            #                           write_graph=True,
            #                           write_images=True,
            #                           embeddings_freq=0,
            #                           write_grads=False,
            #                           embeddings_layer_names=True,
            #                           embeddings_metadata=True)
            callbacks.append(tensorboard)

        if(self.dict_p['MC']== True):
            Model_c     = ModelCheckpoint(self.return_path_model(), mode = 'min',verbose=1,save_best_only=True)
            callbacks.append(Model_c)

        if(self.dict_p['LR_P'] == True):
            R_LR_plat   = ReduceLROnPlateau(monitor      = 'val_loss',
                                      factor       = self.dict_p['LR_factor'],
                                      patience     = self.dict_p['LR_patience'],
                                      verbose      = 0,
                                      mode         = 'auto',
                                      epsilon      = 0.0001,
                                      cooldown     = 0,
                                      min_lr       = 0)
            callbacks.append(R_LR_plat)


        if(self.dict_p['TH_stopper']==True):
            TH_stopper = Stop_reach_trehshold(value=self.dict_p['TH_value'])
            callbacks.append(TH_stopper)

        if(self.dict_p['ESR'] == True):
            Early_ratio = EarlyStoppingratio(value= self.dict_p['early_ratio_val'], verbose = 1)
            callbacks.append(Early_ratio)

        if(self.dict_p['CSV'] == True):
            csv         = CSVLogger(filename=self.return_path_CSV(),
                                    append  =self.dict_p['CSV_append'])
            callbacks.append(csv)

        return callbacks

#### callbacks ##############
class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps     # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class EarlyStoppingratio(Callback):
    def __init__(self, monitor=['val_loss','loss'], value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):

        current_val   = logs.get(self.monitor[0])
        current_train = logs.get(self.monitor[1])

        ratio         = current_val/current_train

        if ratio is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if ratio > self.value:
            if self.verbose > 0:
                print()
                print("Epoch %05d: early stopping RATIO STOPPER" % epoch)
            self.model.stop_training = True

class Stop_reach_trehshold(Callback):
    def __init__(self, monitor=['val_acc'], value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):

        val_acc = logs.get(self.monitor[0])

        if val_acc is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if val_acc > self.value:
            if self.verbose > 0:
                print()
                print("Epoch %05d: early stopping THRESHOLD STOPPER" % epoch)
            self.model.stop_training = True

class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.model.save(filepath, overwrite=True)

