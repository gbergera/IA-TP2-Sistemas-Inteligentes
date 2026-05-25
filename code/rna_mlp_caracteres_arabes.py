"""
RNA Multi-Perceptrón Backpropagation para procesar imágenes y clasificar caracteres árabes.

Adaptado del notebook original de Google Colab.
Requiere que las imágenes estén organizadas en subdirectorios por clase dentro de:
  - data/train/   (imágenes de entrenamiento)
  - data/test/    (imágenes de prueba)
"""

# ─────────────────────────────────────────────
# CONFIGURACIÓN — ajustá estos parámetros
# ─────────────────────────────────────────────

# Rutas de datos (relativas al script o absolutas)
PATH_DATOS         = './data'
PATH_ENTRENAMIENTO = '/train'
PATH_PRUEBA        = '/test'

# Parámetros de imagen
IMAGEN_ANCHO  = 32
IMAGEN_ALTO   = 32
IMAGEN_COLOR  = False     # escala de grises es suficiente para caracteres y reduce el input

# Tipo de problema
CONSIDERAR_ATRIBUTO_CLASE = "discreto"   # "discreto" = clasificación | "continuo" = estimación

# Límite de imágenes a procesar (para evitar colapso de RAM)
LIMITE_PROCESAMIENTO = 3000   # ↑ usa todo el dataset (era 500)

# Capas ocultas lineales: neuronas separadas por coma, "D" para Dropout, "BN" para BatchNorm
LINEAL_CANT_NEURONAS = '1024, BN, D, 512, BN, D, 256, BN, D, 128'
LINEAL_TIPO_FUNCION  = 'relu'
LINEAL_PORC_DROPOUT  = 0.5    # dropout moderado
LINEAL_L2            = 0.0    # ← L2 desactivado, estaba aplastando el aprendizaje
RNA_TIPO_CAPA_SALIDA = 'softmax'

# Optimizador
OPT_TIPO          = 'Adam'
OPT_LEARNING_RATE = 0.0005   # vuelve al valor que dio 51% accuracy

# Entrenamiento
CANT_EPOCAS               = 400
PORC_VALIDACION           = 15.0

# Early Stopping — monitorea val_accuracy en vez de val_loss
EARLY_STOPPING_ACTIVO              = True
EARLY_STOPPING_EPOCA_INICIO        = 300
EARLY_STOPPING_MIN_DELTA           = 0.001
EARLY_STOPPING_PACIENCIA           = 60
EARLY_STOPPING_RESTAURAR_MEJORES   = True

# Data Augmentation — valores conservadores
APLICAR_DATA_AUGMENTATION = False
DA_FLIP_HORIZONTAL        = False
DA_FLIP_VERTICAL          = False
DA_TRANSLATION_H          = 0.1
DA_TRANSLATION_V          = 0.1
DA_ROTATION               = 0.05
DA_ZOOM                   = 0.1
DA_CONTRAST               = 0.2
DA_BRIGHTNESS             = 0.2


# ─────────────────────────────────────────────
# LIBRERÍAS
# ─────────────────────────────────────────────

import os
import time
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras # type: ignore
from keras.models import Model
import keras.backend as K
from keras.utils import to_categorical

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print("Librerías cargadas.")


# ─────────────────────────────────────────────
# CARGA DE IMÁGENES
# ─────────────────────────────────────────────

IMAGE_SHAPE = (
    max(IMAGEN_ANCHO, 10),
    max(IMAGEN_ALTO, 10),
    3 if IMAGEN_COLOR else 1
)


def extraer_label_del_nombre(nombre_archivo):
    """
    Extrae la clase del nombre de archivo.
    Formato esperado: id_XXXXX_label_YY.png → devuelve YY como string
    """
    nombre = os.path.splitext(nombre_archivo)[0]  # saca extensión
    partes = nombre.split('_label_')
    if len(partes) == 2:
        return partes[1]
    # fallback: usa el nombre completo si no tiene el formato esperado
    return nombre


def cargar_imagen(image_path_fn):
    """Carga una imagen, la convierte al modo correcto y la redimensiona."""
    imag = Image.open(image_path_fn)
    modo = 'RGB' if IMAGE_SHAPE[2] == 3 else 'L'
    imag = imag.convert(modo)
    imag = imag.resize((IMAGE_SHAPE[0], IMAGE_SHAPE[1]), Image.LANCZOS)
    return np.array(imag)


def cargar_imagenes_path(imag_path, mostrar_img=False, recursive=True,
                          limitar_cant_dir=None, class_name=None):
    """Carga imágenes desde una carpeta plana o árbol de directorios.
    Si los archivos tienen formato id_XXXXX_label_YY.png extrae la clase del nombre.
    """
    classes_ori = []
    images_ori  = []

    all_els = sorted(os.listdir(imag_path))
    for each_el in all_els:
        auxi_path = os.path.join(imag_path, each_el)

        if each_el.lower().endswith(('.png', '.jpg', '.jpeg')):
            imag = cargar_imagen(auxi_path)
            images_ori.append(imag)
            # determina la clase
            if class_name is not None:
                clase = class_name
            else:
                clase = extraer_label_del_nombre(each_el)
            classes_ori.append(clase)

            if mostrar_img:
                print("\n", clase)
                plt.imshow(imag)
                plt.axis('off')
                plt.show()

            if limitar_cant_dir is not None and len(classes_ori) >= limitar_cant_dir:
                break

        elif recursive and os.path.isdir(auxi_path):
            sub_imgs, sub_classes = cargar_imagenes_path(
                imag_path=auxi_path,
                mostrar_img=mostrar_img,
                recursive=recursive,
                limitar_cant_dir=limitar_cant_dir,
                class_name=each_el
            )
            images_ori.extend(sub_imgs)
            classes_ori.extend(sub_classes)

    return np.array(images_ori) if images_ori else np.array([]), classes_ori


imagPath_train = PATH_DATOS + PATH_ENTRENAMIENTO
imagPath_test  = PATH_DATOS + PATH_PRUEBA

images_train, classes_train = cargar_imagenes_path(imagPath_train)
print("> Para Entrenamiento:")
print("  - Clases únicas:", len(np.unique(classes_train)))
print("  - Imágenes:", len(classes_train))

images_test, classes_test = cargar_imagenes_path(imagPath_test)
print("\n> Para Prueba:")
print("  - Clases únicas:", len(np.unique(classes_test)))
print("  - Imágenes:", len(images_test))


# ─────────────────────────────────────────────
# PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────

def prepare_image_list(imag_list):
    arr = np.array(imag_list)
    return arr.reshape((len(arr), IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))


def plot_image(imag):
    if IMAGE_SHAPE[2] == 1:
        plt.imshow(imag.reshape(IMAGE_SHAPE[0], IMAGE_SHAPE[1]).astype(np.uint8))
        plt.gray()
    else:
        plt.imshow(imag.reshape(IMAGE_SHAPE).astype(np.uint8))
    plt.axis('off')


def limpiar_label(label):
    """Extrae solo los dígitos de un label, ej: '23(1)' → '23'"""
    import re
    solo_digitos = re.match(r'(\d+)', str(label))
    return solo_digitos.group(1) if solo_digitos else label


es_problema_clasificacion = CONSIDERAR_ATRIBUTO_CLASE.strip().lower().startswith('d')

# Limpia labels con caracteres extraños tipo '23(1)' → '23'
classes_train = [limpiar_label(c) for c in classes_train]
classes_test  = [limpiar_label(c) for c in classes_test]

x_train = prepare_image_list(images_train[:LIMITE_PROCESAMIENTO])
x_test  = prepare_image_list(images_test[:LIMITE_PROCESAMIENTO])

classes_train_lim = classes_train[:LIMITE_PROCESAMIENTO]
classes_test_lim  = classes_test[:LIMITE_PROCESAMIENTO]

CLASES = []
y_train, y_test = [], []
y_trainEnc, y_testEnc = [], []

if es_problema_clasificacion:
    print(f"\n> Problema de CLASIFICACIÓN (limitado a {LIMITE_PROCESAMIENTO})")
    # Usa siempre el enfoque por nombre para manejar cualquier formato de clase
    # Toma las clases de AMBOS conjuntos para que los índices sean consistentes
    CLASES = sorted(set(list(classes_train_lim) + list(classes_test_lim)),
                    key=lambda x: int(x) if x.isnumeric() else x)
    y_train = [CLASES.index(y) for y in classes_train_lim]
    y_test  = [CLASES.index(y) for y in classes_test_lim]
    y_trainEnc = to_categorical(y_train, num_classes=len(CLASES))
    y_testEnc  = to_categorical(y_test,  num_classes=len(CLASES))
    print(f"> {len(CLASES)} clases: {CLASES}")
else:
    print(f"\n> Problema de ESTIMACIÓN (limitado a {LIMITE_PROCESAMIENTO})")
    if str(classes_train_lim[0]).isnumeric():
        y_train = [int(y) for y in classes_train_lim]
        y_test  = [int(y) for y in classes_test_lim]
    else:
        CLASES = sorted(set(list(classes_train_lim) + list(classes_test_lim)))
        y_train = [CLASES.index(y) for y in classes_train_lim]
        y_test  = [CLASES.index(y) for y in classes_test_lim]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test  = np.array(x_test)
y_test  = np.array(y_test)

# ── SHUFFLE ──────────────────────────────────────────────────────────────────
# Las imágenes se cargan ordenadas por clase, entonces validation_split (que
# toma el final del array) vería solo las últimas clases → val_accuracy = 0%.
# Mezclar antes garantiza que la validación tenga todas las clases.
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]
if len(y_trainEnc) > 0:
    y_trainEnc = np.array(y_trainEnc)[idx]
print("  ✅ Datos mezclados (shuffle aplicado)")
# ─────────────────────────────────────────────────────────────────────────────

print("\n> Datos preparados:")
print("  x_train:", x_train.shape)
print("  x_test: ", x_test.shape)


# ─────────────────────────────────────────────
# CONSTRUCCIÓN DEL MODELO
# ─────────────────────────────────────────────

# --- Data Augmentation ---
daLayers_modelo = []

if APLICAR_DATA_AUGMENTATION:
    if DA_FLIP_HORIZONTAL or DA_FLIP_VERTICAL:
        modo = ("horizontal_and_vertical" if DA_FLIP_HORIZONTAL and DA_FLIP_VERTICAL
                else ("horizontal" if DA_FLIP_HORIZONTAL else "vertical"))
        daLayers_modelo.append(tf.keras.layers.RandomFlip(mode=modo, name=f"da_flip_{modo}"))
    if DA_TRANSLATION_H != 0.0 or DA_TRANSLATION_V != 0.0:
        daLayers_modelo.append(tf.keras.layers.RandomTranslation(
            height_factor=DA_TRANSLATION_V / 100, width_factor=DA_TRANSLATION_H,
            fill_mode='nearest', name='da_translation'))
    if DA_ROTATION != 0.0:
        daLayers_modelo.append(tf.keras.layers.RandomRotation(
            factor=DA_ROTATION, fill_mode='nearest', name='da_rotation'))
    if DA_ZOOM != 0.0:
        daLayers_modelo.append(tf.keras.layers.RandomZoom(
            height_factor=DA_ZOOM, fill_mode='nearest', name='da_zoom'))
    if DA_CONTRAST != 0.0:
        daLayers_modelo.append(tf.keras.layers.RandomContrast(
            factor=DA_CONTRAST, dtype='float32', name='da_contrast'))
    if DA_BRIGHTNESS != 0.0:
        daLayers_modelo.append(tf.keras.layers.RandomBrightness(
            factor=DA_BRIGHTNESS, name='da_brightness'))


# --- Funciones auxiliares para construir el modelo ---

def agregar_capas_lineales(prev_lay, config_capas_list, tipo_funcion='relu', porc_dropout=0.1, l2_reg=0.0):
    porc_dropout = min(max(0.1, porc_dropout), 0.9)
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    idx = 1
    for val in config_capas_list:
        val = val.strip()
        if val.upper() == 'D':
            prev_lay = tf.keras.layers.Dropout(porc_dropout, name=f"d_{idx}")(prev_lay)
        elif val.upper() == 'BN':
            prev_lay = tf.keras.layers.BatchNormalization(name=f"bn_{idx}")(prev_lay)
        elif val.isnumeric():
            prev_lay = tf.keras.layers.Dense(
                int(val), activation=tipo_funcion,
                kernel_regularizer=regularizer,
                name=f"hidd_{idx}")(prev_lay)
        else:
            print(f"Tipo de capa '{val}' no reconocido, ignorado.")
        idx += 1
    return prev_lay


def agregar_capa_salida(prev_lay, es_clasificacion, usar_softmax=False, cant_clases=1):
    if es_clasificacion:
        metrics_type = ['accuracy']
        if usar_softmax:
            output_lay = tf.keras.layers.Dense(
                cant_clases, activation='softmax', name='output')(prev_lay)
            loss_type = 'categorical_crossentropy'
        else:
            output_lay = tf.keras.layers.Dense(1, activation=None, name='output')(prev_lay)
            loss_type = 'mse'
    else:
        output_lay = tf.keras.layers.Dense(1, activation=None, name='output')(prev_lay)
        metrics_type = ['RootMeanSquaredError']
        loss_type = 'mse'
    return output_lay, metrics_type, loss_type


def definir_optimizador(opt_tipo, lr=0.01):
    lr = min(max(0.00001, lr), 10)
    opts = {
        'Gradiente Decreciente': keras.optimizers.SGD(learning_rate=lr),
        'Adam':                  keras.optimizers.Adam(learning_rate=lr),
        'Adadelta':              keras.optimizers.Adadelta(learning_rate=lr),
        'Adagrad':               keras.optimizers.Adagrad(learning_rate=lr),
        'Adamax':                keras.optimizers.Adamax(learning_rate=lr),
        'Nadam':                 keras.optimizers.Nadam(learning_rate=lr),
        'RMSprop':               keras.optimizers.RMSprop(learning_rate=lr),
        'Momentum':              keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
        'NAG':                   keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True),
    }
    return opts.get(opt_tipo, keras.optimizers.Adam(learning_rate=lr))


# --- Construcción ---
tipo_output_softmax = (RNA_TIPO_CAPA_SALIDA.lower().startswith('softmax')
                       if es_problema_clasificacion else False)

input_lay = tf.keras.layers.Input(shape=IMAGE_SHAPE, name='input_img')
each_lay  = input_lay

for da_lay in daLayers_modelo:
    each_lay = da_lay(each_lay)

each_lay = tf.keras.layers.Rescaling(1. / 255, name='rescaling')(each_lay)
each_lay = tf.keras.layers.Flatten(name='flat')(each_lay)
each_lay = agregar_capas_lineales(
    each_lay,
    config_capas_list=LINEAL_CANT_NEURONAS.split(','),
    tipo_funcion=LINEAL_TIPO_FUNCION,
    porc_dropout=LINEAL_PORC_DROPOUT,
    l2_reg=LINEAL_L2
)

output_lay, metrics_type, loss_type = agregar_capa_salida(
    each_lay,
    es_clasificacion=es_problema_clasificacion,
    usar_softmax=tipo_output_softmax,
    cant_clases=len(CLASES)
)

opt_model = definir_optimizador(OPT_TIPO, OPT_LEARNING_RATE)
model = Model(input_lay, output_lay, name='RNA_MLP')
model.compile(optimizer=opt_model, loss=loss_type, metrics=metrics_type)

print(f"\nModelo creado con {len(model.layers)} capas:")
model.summary()


# ─────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────

cant_epocas      = max(1, CANT_EPOCAS)
porc_validacion  = min(49.9, max(0.5, PORC_VALIDACION))

print(f"\n> De {len(x_train)} ejemplos: "
      f"{round(100 - porc_validacion, 1)}% entrenamiento / "
      f"{round(porc_validacion, 1)}% validación.")

callbacks_list = []
if EARLY_STOPPING_ACTIVO:
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',        # ← monitorea accuracy real, no loss
        min_delta=max(0, EARLY_STOPPING_MIN_DELTA),
        patience=max(1, EARLY_STOPPING_PACIENCIA),
        verbose=1,
        mode='max',                    # ← busca maximizar (era 'min' para loss)
        restore_best_weights=EARLY_STOPPING_RESTAURAR_MEJORES,
        start_from_epoch=max(2, EARLY_STOPPING_EPOCA_INICIO)
    )
    callbacks_list.append(early_stopping)
    print("> Early Stopping activado.")


def get_early_stopping_epoch(cbs):
    for cb in cbs:
        if isinstance(cb, keras.callbacks.EarlyStopping):
            if cb.restore_best_weights and cb.best_epoch > 0:
                return cb.best_epoch + 1
            if not cb.restore_best_weights and cb.stopped_epoch > 0:
                return cb.stopped_epoch + 1
    return None


start_time = time.process_time()
print("\n> Comienza el entrenamiento...\n")

history = model.fit(
    x=x_train,
    y=y_trainEnc if tipo_output_softmax else y_train,
    epochs=cant_epocas,
    validation_split=porc_validacion / 100.0,
    callbacks=callbacks_list
)

duration = time.process_time() - start_time
if duration > 60:
    print(f"\n> Entrenamiento finalizado: {duration / 60:.3f} minutos.")
else:
    print(f"\n> Entrenamiento finalizado: {duration:.3f} segundos.")

epoch_early_stopping = get_early_stopping_epoch(callbacks_list)


# ─────────────────────────────────────────────
# GRÁFICOS DE ENTRENAMIENTO
# ─────────────────────────────────────────────

def graficar_entrenamiento(history, epoch_es=None, es_clasificacion=True):
    legends = ['entrenamiento', 'validación']

    # Loss
    plt.figure(figsize=(15, 8))
    plt.plot([None] + history.history['loss'])
    plt.plot([None] + history.history['val_loss'])
    plt.title('Error del Entrenamiento')
    plt.xlabel('época')
    if epoch_es:
        label_es = f"EarlyStopping({epoch_es})"
        plt.axvline(x=epoch_es, color='g', linestyle=':', label=label_es)
        legends.append(label_es)
    plt.legend(legends, loc='upper left')
    plt.margins(x=0)
    plt.show()

    # Métrica
    plt.figure(figsize=(15, 8))
    legends = ['entrenamiento', 'validación']
    if es_clasificacion:
        plt.plot([None] + history.history['accuracy'])
        plt.plot([None] + history.history['val_accuracy'])
        plt.title('Exactitud del Entrenamiento')
    else:
        plt.plot([None] + history.history['RootMeanSquaredError'])
        plt.plot([None] + history.history['val_RootMeanSquaredError'])
        plt.title('RMSE del Entrenamiento')
    plt.xlabel('época')
    if epoch_es:
        plt.axvline(x=epoch_es, color='g', linestyle=':')
        legends.append(f"EarlyStopping({epoch_es})")
    plt.legend(legends, loc='upper left')
    plt.margins(x=0)
    plt.show()


graficar_entrenamiento(history, epoch_early_stopping, es_problema_clasificacion)


# ─────────────────────────────────────────────
# EVALUACIÓN DEL MODELO
# ─────────────────────────────────────────────

def interpretar_pred(pred_y, es_clasificacion, usar_softmax=False, umbral=0.5):
    if es_clasificacion:
        if usar_softmax:
            return int(np.argmax(pred_y, axis=0))
        else:
            aux = pred_y[0]
            pid = int(aux)
            return pid + 1 if abs(aux - pid) > umbral else pid
    return pred_y[0]


def asignar_class_name(class_id, dict_map):
    if class_id in dict_map:
        return dict_map[class_id], dict_map
    nombre = f"CLASE {class_id} INVÁLIDA"
    dict_map[class_id] = nombre
    return nombre, dict_map


def resumen_clasificacion(class_real, class_preds, clases_labels):
    print("\nReporte de Clasificación:")
    print(classification_report(class_real, class_preds, zero_division=0))
    print("Matriz de Confusión (real / modelo):")
    cm = confusion_matrix(class_real, class_preds, labels=clases_labels)
    cmtx = pd.DataFrame(
        cm,
        index=[f"r:{x}" for x in clases_labels],
        columns=[f"m:{x}" for x in clases_labels]
    )
    print(cmtx)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases_labels)
    fig, ax = plt.subplots(figsize=(15, 5))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='g', colorbar=False)
    disp.ax_.set_title('Gráfico de Confusión')
    disp.ax_.set(xlabel='Modelo', ylabel='Real')
    plt.tight_layout()
    plt.show()


def resumen_estimacion(ar, titulo, bins=10, color=None):
    print(f"\nEstadísticas para {titulo}:")
    print(f"  Mínimo:   {np.min(ar):.4f}")
    print(f"  Promedio: {np.mean(ar):.4f} ± {np.std(ar):.4f}")
    print(f"  Máximo:   {np.max(ar):.4f}")
    plt.figure(figsize=(15, 5))
    plt.hist(ar, bins=bins, color=color)
    plt.grid(color='lightgrey', linestyle='solid', linewidth=0.3)
    plt.title(f"Distribución de {titulo}")
    plt.show()


def evaluar_modelo(modelo, es_clasificacion, datos_x, datos_y,
                   clases_map=None, usar_softmax=False, umbral=0.5):
    preds_y = modelo.predict(datos_x, verbose=0)
    dict_map = dict(enumerate(clases_map)) if clases_map else {}

    if es_clasificacion:
        class_real, class_preds = [], []
        for r_id, p_v in zip(datos_y, preds_y):
            cl_real, dict_map = asignar_class_name(r_id, dict_map)
            p_id = interpretar_pred(p_v, True, usar_softmax, umbral)
            cl_pred, dict_map = asignar_class_name(p_id, dict_map)
            class_real.append(cl_real)
            class_preds.append(cl_pred)
        clases_labels = list(dict_map.values())
        resumen_clasificacion(class_real, class_preds, clases_labels)
    else:
        arAbs, arRel = [], []
        for r, p_v in zip(datos_y, preds_y):
            p = interpretar_pred(p_v, False)
            if not (math.isnan(r) or math.isnan(p)):
                e_abs = abs(r - p)
                e_rel = (e_abs / (r if r != 0 else 1)) * 100.0
                arAbs.append(e_abs)
                arRel.append(e_rel)
        resumen_estimacion(arAbs, "Error Absoluto", 20, "red")
        resumen_estimacion(arRel, "Error Relativo", 10, "magenta")


# Evaluación con datos de entrenamiento
LIMITE_EVAL = 1000
print(f"\n{'='*60}")
print(f"Resultados con datos de ENTRENAMIENTO (limitado a {LIMITE_EVAL}):")
print('='*60)
evaluar_modelo(
    model, es_problema_clasificacion,
    datos_x=x_train[:LIMITE_EVAL],
    datos_y=y_train[:LIMITE_EVAL],
    clases_map=CLASES,
    usar_softmax=tipo_output_softmax
)

# Evaluación con datos de prueba
print(f"\n{'='*60}")
print(f"Resultados con datos de PRUEBA (limitado a {LIMITE_EVAL}):")
print('='*60)
evaluar_modelo(
    model, es_problema_clasificacion,
    datos_x=x_test[:LIMITE_EVAL],
    datos_y=y_test[:LIMITE_EVAL],
    clases_map=CLASES,
    usar_softmax=tipo_output_softmax
)
