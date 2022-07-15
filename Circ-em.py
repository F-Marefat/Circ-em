from getData import *
import os
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# for not using gpu, cuda-visible = -1
# to use gpu, cuda-visible = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Embedding import *
import matplotlib as mpl
mpl.use('Agg')
from sklearn.model_selection import KFold
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

def CircEM(parser):
    protein = parser.protein
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    embed_model_path = parser.embed_model_path
    embed_vector_dim = parser.embed_vector_dim
    input_dim1 = parser.input_dim1
    model_type = parser.model_type
    degree = parser.degree
    model_path = "models/"

    tf.keras.backend.clear_session()
    tf.random.set_seed(5005)
    p_len = [46, 44, 42, 40]
    embedding_weights = embedding_matrix(model_type, embed_model_path, embed_vector_dim)
    input_layer = tf.keras.layers.Input(shape=(input_dim1,))
    embed_layer = tf.keras.layers.Embedding(input_dim=embedding_weights.shape[0],
                                            output_dim=embedding_weights.shape[1],
                                            weights=[embedding_weights], trainable=False)(input_layer)

    conv3 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, padding='valid', activation="relu",use_bias=True,name="conv3")(embed_layer)
    max_pool3 = tf.keras.layers.MaxPooling1D(pool_size=p_len[0], name="pool3")(conv3)
    conv5 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='valid', activation="relu",use_bias=True,name="conv5")(embed_layer)
    max_pool5 = tf.keras.layers.MaxPooling1D(pool_size=p_len[1], name="pool5")(conv5)
    conv7 = tf.keras.layers.Conv1D(filters=128, kernel_size=7, padding='valid', activation="relu",use_bias=True,name="conv7")(embed_layer)
    max_pool7 = tf.keras.layers.MaxPooling1D(pool_size=p_len[2], name="pool7")(conv7)
    merge1 = tf.keras.layers.concatenate([max_pool3,max_pool5,max_pool7], name="merge1")
    merged_flat = tf.keras.layers.Flatten()(merge1)
    drop1 = tf.keras.layers.Dropout(0.25, name="drop1")(merged_flat)
    hidden1 = tf.keras.layers.Dense(512,activation='relu', name="hidden1")(drop1)  # 4096?
    drop_hidden1 = tf.keras.layers.Dropout(0.5, name="drop_hidden1")(hidden1)
    dense_drop_hidden1 = tf.keras.layers.Dense(256, activation='relu', name="dense_drop_hidden1")(drop_hidden1)
    merge2 = tf.keras.layers.concatenate([dense_drop_hidden1, drop1, hidden1], name="merge2")
    output = tf.keras.layers.Dense(2,activation="sigmoid",name="output")(merge2)

    model_func = tf.keras.Model(inputs=input_layer, outputs=output)
    model_func.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),metrics=['accuracy']) # adam
    print(model_func.summary())
    print('loading data')
    trainXeval, x_test, trainYeval, y_test = dealwithembeddata(model_type, protein, embed_model_path, embed_vector_dim,
                                                               degree, input_dim1)
    y_test = y_test[:, 1]
    kf = KFold(n_splits=2)
    aucs = []
    for train_index, eval_index in kf.split(trainXeval):
        train_X = trainXeval[train_index]
        train_y = trainYeval[train_index]
        eval_X = trainXeval[eval_index]
        eval_y = trainYeval[eval_index]
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=model_path + protein + "_" , verbose=1, save_best_only=True)
        model_func.fit(train_X, train_y, batch_size=batch_size, epochs=n_epochs,verbose=1, validation_data=(eval_X,eval_y),callbacks=[checkpointer])
        print('predicting the model')
        y_pred = model_func.predict(x_test)
        ytrue = y_test
        y_pred = np.argmax(y_pred, axis=-1)
        auc = roc_auc_score(ytrue, y_pred)
        aucs.append(auc)

    return save_results_to_txtfile(ytrue, y_pred, aucs, protein,model_path,model_func.count_params())

def save_results_to_txtfile(ytrue, y_pred, aucs, protein,model_path,parameters):
    address = model_path + str(protein)+"_"
    with open(address,"w") as writer:
        writer.write(str(parameters))
    acc = accuracy_score(ytrue, y_pred)
    precision = precision_score(ytrue, y_pred)
    recall = recall_score(ytrue, y_pred)
    fscore = f1_score(ytrue, y_pred)
    MCC = matthews_corrcoef(ytrue, y_pred)
    return np.mean(aucs), acc, precision, recall, fscore, MCC

def parse_arguments(parser):
    parser.add_argument('--protein', type=str, default="",help='the protein for training model')
    parser.add_argument('--batch_size', type=int, default=50, help='The size of a single mini-batch (default value: 50)')
    parser.add_argument('--n_epochs', type=int, default=30, help='The number of training epochs (default value: 30)')
    parser.add_argument('--embed_vector_dim', type=int, default=30, help='')
    parser.add_argument('--embed_model_path', type=str, default='', help='')
    parser.add_argument('--degree', type=int, default=3, help='')
    parser.add_argument('--model_type', type=str, default="word2vec", help='embedding_model_type(word2vec or fasttext)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    RBP_dict = [
        'WTAP','RBPMS','FXR1','FOX2','QKI','TNRC6','ALKBH5','TAF15','C17ORF85',
        'AUF1','PUM2','TIA1','AGO3','HNRNPC','TDP43','METTL3','EWSR1','FXR2','MOV10',
        'TIAL1','CAPRIN1','C22ORF28','LIN28B','U2AF65','SFRS1','ZC3H7B','AGO1','LIN28A',
        'DGCR8','FUS','IGF2BP2',]

    embedding_model = ['embedding/circRNA2vec_3mer_hg19_60',]
    degree = ['3']
    vec_dim = ['60']
    n_epoch = ['30']
    batch = ['64']

    import sys
    for i in range(len(n_epoch)):
        print(embedding_model[i])
        print(degree[i])
        sys.argv.extend(['--embed_model_path', str(embedding_model[i])])
        sys.argv.extend(['--degree', str(degree[i])])
        sys.argv.extend(['--embed_vector_dim', str(vec_dim[i])])
        sys.argv.extend(['--n_epochs', str(n_epoch[i])])
        sys.argv.extend(['--batch_size', str(batch[i])])
        aucs = []
        accs = []
        precisions = []
        recalls = []
        fscores = []
        MCCs = []
        for j in range(len(RBP_dict)):
            print(str(RBP_dict[j]))
            sys.argv.extend(['--protein', str(RBP_dict[j])+"-crip"])
            parser = argparse.ArgumentParser()
            args = parse_arguments(parser)
            auc,acc,precision,recall,fscore,MCC = CircEM(args)
            aucs.append([str(RBP_dict[j]),auc])
            accs.append([str(RBP_dict[j]),acc])
            precisions.append([str(RBP_dict[j]), precision])
            recalls.append([str(RBP_dict[j]), recall])
            fscores.append([str(RBP_dict[j]), fscore])
            MCCs.append([str(RBP_dict[j]), MCC])

        aucs_df = pd.DataFrame(aucs)
        accs_df = pd.DataFrame(accs)
        precision_df = pd.DataFrame(precisions)
        recall_df = pd.DataFrame(recalls)
        fscore_df = pd.DataFrame(fscores)
        MCC_df = pd.DataFrame(MCCs)
        excel_path = "embedding-results.xlsx"
        with pd.ExcelWriter(excel_path, mode="a") as writer:  # doctest: +SKIP
            aucs_df.to_excel(writer, sheet_name="aucs_"+str(degree[i])+"_"+str(vec_dim[i]),index=False)
            accs_df.to_excel(writer, sheet_name="accs_"+str(degree[i])+"_"+str(vec_dim[i]),index=False)
            precision_df.to_excel(writer, sheet_name='precisions_'+str(degree[i])+"_"+str(vec_dim[i]),index=False)
            recall_df.to_excel(writer, sheet_name='recalls_'+str(degree[i])+"_"+str(vec_dim[i]),index=False)
            fscore_df.to_excel(writer, sheet_name='fscores_'+str(degree[i])+"_"+str(vec_dim[i]),index=False)
            MCC_df.to_excel(writer, sheet_name='MCCs_'+str(degree[i])+"_"+str(vec_dim[i]),index=False)