from in_out import DataGenerator, load_x_y, sample_x
from model import Point2Pose, Latent2Pose, Center, Normal
from schedules import onetenth_50_75, onetenth_75_100, onetenth_150_175
import plot

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mean_absolute_error

import os
os.environ["TF_KERAS"] = "1"
from keras_radam import RAdam
from keras_lookahead import Lookahead

import os
import numpy as np

'''
def eval():
    pred = model.predict(x)

    print("GT: {} {}".format(y['center'][0], y['norm'][0]))
    print("PR: {} {}".format(pred[0][0], pred[1][0]))

    points2obj(x[0],"output/val_pc.obj")
    anno2obj(y['center'][0], y['norm'][0],"output/val_gt.obj")
    anno2obj(pred[0][0], pred[1][0],"output/val_pred.obj")
'''

def evaluate(x,y,model_path="results"):
    y = {'center': y[:,:3], 'norm': y[:,3:]}
    #y = y[:,:3]
    #y = y[:,3:]
    model = models.load_model(os.path.join(model_path,"end2end.h5"), compile=True)
    score = model.evaluate(x, y, verbose=1)
    return model.metrics_names, score

def predict(x,model_path="results"):
    model = models.load_model(os.path.join(model_path,"end2end.h5"), compile=False)
    print("[INFO] predicting...")
    pred = model.predict(x)
    return pred

def encode(x,model_path="results"):
    model = models.load_model(os.path.join(model_path,"end2end_encoder.h5"), compile=False)
    print("[INFO] encoding...")
    encs = model.predict(x)
    return encs

def fit_fine_gen(train_gen,val_gen,n_train,n_val,n_points=1024,n_latent=16,epochs=200,batch_size=64,lr=0.01,model_path=None,output_path="results"):
    # init model
    model = models.load_model(model_path, compile=False)

    for layer in model.layers:
	       layer.trainable = True

    model.summary()

    opt = Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss={'center': 'mean_squared_error', 'norm': 'mean_squared_error'},
                  loss_weights = {"center": 1.0, "norm": 1.0},
                  metrics=['mae'])

    #checkpoint = ModelCheckpoint('./models/pointnet.h5', monitor='val_loss',
    #                             save_weights_only=True, save_best_only=True, verbose=1)

    history = model.fit_generator(train_gen.generateXY(),
                                  steps_per_epoch=n_train // batch_size,
                                  epochs=epochs,
                                  validation_data=val_gen.generateXY(),
                                  validation_steps=n_val // batch_size,
                                  callbacks=[onetenth_150_175(lr)],
                                  verbose=1)

    plot.plot_history([('', history)], output_path, 'end2end')
    model.save(os.path.join(output_path,"end2end.h5"))

def fit_head_gen(train_gen,val_gen,n_train,n_val,n_points=1024,n_latent=16,epochs=200,batch_size=64,lr=0.01,encoder_path=None,freeze=False,output_path="results"):
    # init model
    encoder = models.load_model(encoder_path, compile=False)
    center = Center(encoder.output)
    norm = Normal(encoder.output)
    model = Model(inputs=encoder.input, outputs=[center, norm])

    if freeze:
        for layer in encoder.layers:
    	       layer.trainable = False

    model.summary()
    encoder.summary()

    opt = Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss={'center': 'mean_squared_error', 'norm': 'mean_squared_error'},
                  loss_weights = {"center": 1.0, "norm": 1.0},
                  metrics=['mae'])

    if epochs < 101:
        lr_callback = onetenth_50_75(lr)
    else:
        lr_callback = onetenth_150_175(lr)

    history = model.fit_generator(train_gen.generateXY(),
                                  steps_per_epoch=n_train // batch_size,
                                  epochs=epochs,
                                  validation_data=val_gen.generateXY(),
                                  validation_steps=n_val // batch_size,
                                  callbacks=[lr_callback],
                                  verbose=1)

    plot.plot_history([('', history)], output_path, 'end2end_head')
    model.save(os.path.join(output_path,"end2end.h5"))
    encoder.save(os.path.join(output_path,"end2end_encoder.h5"))


def fit_gen(train_gen,val_gen,n_train,n_val,model,n_points=1024,n_latent=16,epochs=100,batch_size=64,lr=0.01,output_path="results", split="complete_split"):
    # init model
    #if model_path == None:
    #    model, encoder = Point2Pose(cloud_shape = (n_points, 3), n_latent = n_latent)
    #else:
    #    model = models.load_model(os.path.join(model_path,"end2end.h5"), compile=False)
    #    encoder = models.load_model(os.path.join(model_path,"end2end_encoder.h5"), compile=False)

    #model = Point2Pose2(cloud_shape = (n_points, 3))
    #model.summary()
    #encoder.summary()

    opt = Lookahead(RAdam(learning_rate=lr),sync_period=5,slow_step=0.5)
    #opt = Adam(lr=lr)
    if split == "complete_split":
        model.compile(optimizer=opt,
                      loss='mean_squared_error',
                      metrics=['mae'])
    else:
        model.compile(optimizer=opt,
                      loss={'center': 'mean_squared_error', 'norm': 'mean_squared_error'},
                      loss_weights = {"center": 1.0, "norm": 1.0},
                      metrics=['mae'])

    #checkpoint = ModelCheckpoint('./models/pointnet.h5', monitor='val_loss',
    #                             save_weights_only=True, save_best_only=True, verbose=1)

    history = model.fit_generator(train_gen.generateXY(),
                                  steps_per_epoch=n_train // batch_size,
                                  epochs=epochs,
                                  validation_data=val_gen.generateXY(),
                                  validation_steps=n_val // batch_size,
                                  #callbacks=[onetenth_150_175(lr)],
                                  verbose=1)

    plot.plot_history([('', history)], output_path, 'end2end')
    model.save(os.path.join(output_path,"end2end.h5"))
    #encoder.save(os.path.join(output_path,"end2end_encoder.h5"))

if __name__ == '__main__':
    TRAIN = 0
    EVAL = 1

    batch_size = 64

    train_file = 'data/train.txt'
    val_file = 'data/val.txt'
    test_file = 'data/test.txt'

    n_points = 1024


    # load ply from list(crops wiht more than n_points)
    X_train, y_train = load_x_y(train_file)
    x_train = sample_x(X_train, n_points)
    np.save('output/x_train.npy',x_train)

    X_val, y_val = load_x_y(val_file)
    x_val = sample_x(X_val, n_points)
    np.save('output/x_val.npy',x_val)

    X_test, y_test = load_x_y(test_file)
    x_test = sample_x(X_test, n_points)
    np.save('output/x_test.npy',x_test)


    model, encoder = Point2Pose(cloud_shape = (n_points, 3), n_latent = 16)

    print("Number of samples: {} train, {} val".format(X_train.shape[0],X_val.shape[0]))

    if TRAIN:
        # generators that samples n_points from clouds in list and make batches
        train_gen = DataGenerator(X_train, y_train, batch_size=batch_size, augment=False)
        val_gen = DataGenerator(X_val, y_val, batch_size=batch_size, augment=False)
        fit_gen(train_gen, val_gen, X_train.shape[0], X_val.shape[0], model)

    if EVAL:
        #z_train = encode(x_train)
        #np.save("output/train_z.npy",z_train)
        #z_val = encode(x_val)
        #np.save("output/val_z.npy",z_val)

        # placeholdes to avoid https://github.com/keras-team/keras/issues/12916
        y_pred = K.placeholder([None, 3])
        y_true = K.placeholder([None, 3])
        loss_fn = mean_absolute_error(y_true, y_pred)

        print("computing per sample loss on training set")
        pred_train = predict(x_train)
        train_p0_losses = []
        train_norm_losses = []
        for y, y_p0, y_n in zip(y_train,pred_train[0],pred_train[1]):
            #pred_p0_loss = K.eval(mean_absolute_error(K.variable(np.expand_dims(y[:3], axis=0)),K.variable(np.expand_dims(y_p0, axis=0))))
            pred_p0_loss = K.get_session().run(loss_fn, feed_dict={y_true: np.expand_dims(y[:3], axis=0), y_pred: np.expand_dims(y_p0, axis=0)})
            train_p0_losses.append(pred_p0_loss)
            pred_norm_loss = K.get_session().run(loss_fn, feed_dict={y_true: np.expand_dims(y[3:], axis=0), y_pred: np.expand_dims(y_n, axis=0)})
            train_norm_losses.append(pred_norm_loss)
        np.save("output/train_p0_loss.npy",train_p0_losses)
        np.save("output/train_norm_loss.npy",train_norm_losses)

        print("computing per sample loss on validation set")
        pred_val = predict(x_val)
        val_p0_losses = []
        val_norm_losses = []
        for y, y_p0, y_n in zip(y_val,pred_val[0],pred_val[1]):
            pred_p0_loss = K.get_session().run(loss_fn, feed_dict={y_true: np.expand_dims(y[:3], axis=0), y_pred: np.expand_dims(y_p0, axis=0)})
            val_p0_losses.append(pred_p0_loss)
            pred_norm_loss = K.get_session().run(loss_fn, feed_dict={y_true: np.expand_dims(y[3:], axis=0), y_pred: np.expand_dims(y_n, axis=0)})
            val_norm_losses.append(pred_norm_loss)
        np.save("output/val_p0_loss.npy",val_p0_losses)
        np.save("output/val_norm_loss.npy",val_norm_losses)

        print("computing per sample loss on test set")
        pred_test = predict(x_test)
        test_p0_losses = []
        test_norm_losses = []
        for y, y_p0, y_n in zip(y_test,pred_test[0],pred_test[1]):
            pred_p0_loss = K.get_session().run(loss_fn, feed_dict={y_true: np.expand_dims(y[:3], axis=0), y_pred: np.expand_dims(y_p0, axis=0)})
            test_p0_losses.append(pred_p0_loss)
            pred_norm_loss = K.get_session().run(loss_fn, feed_dict={y_true: np.expand_dims(y[3:], axis=0), y_pred: np.expand_dims(y_n, axis=0)})
            test_norm_losses.append(pred_norm_loss)
        np.save("output/test_p0_loss.npy",test_p0_losses)
        np.save("output/test_norm_loss.npy",test_norm_losses)

        np.save('output/pred_train.npy',np.concatenate((pred_train[0], pred_train[1]), axis=1))
        np.save('output/y_train.npy',y_train)
        np.save('output/pred_val.npy',np.concatenate((pred_val[0], pred_val[1]), axis=1))
        np.save('output/y_val.npy',y_val)
        np.save('output/pred_test.npy',np.concatenate((pred_test[0], pred_test[1]), axis=1))
        np.save('output/y_test.npy',y_test)
