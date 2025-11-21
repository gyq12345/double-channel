import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Dense, BatchNormalization, Dropout, Concatenate)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detected, available GPUs:", len(gpus))
    except RuntimeError as e:
        print("GPU configuration error:", e)
else:
    print("No GPU detected, using CPU.")

gesture_dir = r'E:/postgraduate/help/shoushi/shuju1'
temp_dir    = r'E:/postgraduate/help/shoushi/gesture_recognition_3200_database'
IMG_SIZE = (128, 128)
EXPECTED_ROIS = 5
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-5
SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

def combine_rois_to_channels(rois, target_size=IMG_SIZE):
    gray_rois = [cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) for roi in rois]
    gray_rois = [cv2.resize(roi, target_size) for roi in gray_rois]
    combined = np.stack(gray_rois, axis=-1)
    return combined

def extract_waveform_rois(image_path):
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Failed to read: {image_path}")
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois, boxes_y = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(cnt)
            roi = img[y:y+h, x:x+w]
            if roi.size == 0: continue
            roi = cv2.resize(roi, IMG_SIZE)
            rois.append(roi)
            boxes_y.append(y)
    if len(rois) > 0:
        rois = [roi for _,roi in sorted(zip(boxes_y, rois), key=lambda x:x[0])]
        print(f"[INFO] Detected {len(rois)} ROIs: {image_path}")
    else:
        print(f"[INFO] No ROI detected, generating black image for padding: {image_path}")
    return rois

def pad_rois(rois, target_size=IMG_SIZE, expected=EXPECTED_ROIS):
    while len(rois) < expected:
        rois.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
    return rois[:expected]

def load_and_pair_datasets(gesture_root, temp_root):
    gesture_classes = sorted([d for d in os.listdir(gesture_root) if os.path.isdir(os.path.join(gesture_root,d))])
    temp_classes = sorted([d for d in os.listdir(temp_root) if os.path.isdir(os.path.join(temp_root,d))])
    classes = sorted(list(set(gesture_classes) & set(temp_classes)))
    print("Detected classes:", classes)

    temp_mapping = {
        "-20℃": ["A","B","C","Comma","D","Delete","E","F","G","H","I","J"],
        "0℃": ["K","L","M","Mark","N","O","P","Period","Q","R","S","Space","T"],
        "20℃": ["Togglecase","U","V","W","X","Y","Z"]
    }
    def get_temperature(label):
        for temp, group in temp_mapping.items():
            if label in group:
                return temp
        return "unknown"

    X_g, X_t, Y = [], [], []

    for cls in classes:
        g_dir = os.path.join(gesture_root, cls)
        t_dir = os.path.join(temp_root, cls)
        g_files = sorted([f for f in os.listdir(g_dir) if f.lower().endswith(('.png','.jpg'))])
        t_files = sorted([f for f in os.listdir(t_dir) if f.lower().endswith(('.png','.jpg'))])
        for g_file, t_file in zip(g_files, t_files):
            g_path = os.path.join(g_dir, g_file)
            t_path = os.path.join(t_dir, t_file)
            rois_g = pad_rois(extract_waveform_rois(g_path))
            rois_t = pad_rois(extract_waveform_rois(t_path))
            X_g.append(combine_rois_to_channels(rois_g))
            X_t.append(combine_rois_to_channels(rois_t))
            temp_label = get_temperature(cls)
            joint_label = f"{cls}_{temp_label}"
            Y.append(joint_label)

    X_g = np.array(X_g, dtype='float32')/255.0
    X_t = np.array(X_t, dtype='float32')/255.0
    Y = np.array(Y)
    print(f"[INFO] Loading completed: samples={len(Y)}, X_g shape={X_g.shape}, X_t shape={X_t.shape}")
    return X_g, X_t, Y

def build_branch(input_shape=(128,128,5), name='branch'):
    inp = Input(shape=input_shape, name=f'{name}_input')
    x = Conv2D(32,(3,3),activation='relu',padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.2)(x)
    branch_out = Dense(64, activation='relu', name=f'{name}_feature')(x)
    return inp, branch_out

if __name__=='__main__':
    X_gesture, X_temp, y_raw = load_and_pair_datasets(gesture_dir, temp_dir)
    if len(y_raw)==0:
        raise SystemExit("[ERROR] No samples loaded, please check paths and ROI extraction.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)
    y_cat = to_categorical(y_encoded, num_classes=num_classes)
    print("Joint category list:", label_encoder.classes_)

    Xg_train, Xg_temp, Xt_train, Xt_temp, y_train, y_temp = train_test_split(
        X_gesture, X_temp, y_cat, test_size=0.4, random_state=SEED, stratify=y_encoded)
    Xg_val, Xg_test, Xt_val, Xt_test, y_val, y_test = train_test_split(
        Xg_temp, Xt_temp, y_temp, test_size=0.5, random_state=SEED,
        stratify=np.argmax(y_temp, axis=1))
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    gesture_input, gesture_feat = build_branch(input_shape=(IMG_SIZE[0],IMG_SIZE[1],EXPECTED_ROIS),
                                               name='gesture')
    temp_input, temp_feat = build_branch(input_shape=(IMG_SIZE[0],IMG_SIZE[1],EXPECTED_ROIS),
                                         name='temp')

    merged = Concatenate()([gesture_feat, temp_feat])
    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.5)(x)
    final_output = Dense(num_classes, activation='softmax', name='final_output')(x)

    model = Model(inputs=[gesture_input,temp_input], outputs=final_output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint('best_multimodal_model.h5', monitor='val_accuracy',
                                 save_best_only=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    history = model.fit([Xg_train,Xt_train], y_train,
                        validation_data=([Xg_val,Xt_val], y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[checkpoint, reduce_lr], verbose=1)

    best_model = load_model('best_multimodal_model.h5')
    loss, acc = best_model.evaluate([Xg_test,Xt_test], y_test, verbose=1)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

    y_pred_probs = best_model.predict([Xg_test,Xt_test])
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    labels = label_encoder.classes_
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    history_df = pd.DataFrame({
        'epoch': list(range(1,len(history.history['accuracy'])+1)),
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    })
    history_df.to_excel('training_log.xlsx', index=False, engine='openpyxl')
    print("Training log saved to training_log.xlsx")
