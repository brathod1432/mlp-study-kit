#!/usr/bin/env python3.11
__author__ = "brijesh_ganpatbhai.rathod.stud@pw.edu.pl"
# Album No.: 309169
import sys
import os
import datetime
import math
import numpy as np
# from modules.GeneralUtils import ObjLogger as ObjLogger2
# from modules.NeuralNetwork import Activation_fcn, NeuralNetCore, Exercise10Data, Homework2Demos, Homework1Utils, Homework3VGG16

import sys as _sys, os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from nn_core.logger import ObjLogger, title_message
import tensorflow as tf

logger = ObjLogger("Task_set_12")

class Homework3VGG16:
    """
    Homework 3:
        Task 1: Manual VGG16 architecture (Keras) and (Torch optional earlier)
        Task 2: Train VGG16 (Keras) on Cats vs Dogs (TFDS), metrics + plots + qualitative results
    """

    # ============================================================
    # TASK 1 (Keras): Manual VGG16 builder (no pre-built loaders)
    # ============================================================

    @staticmethod
    def _vgg_conv_block_keras(
        x,
        filters: int,
        conv_count: int,
        block_name: str,
        kernel_size: tuple = (3, 3),
        activation: str = "relu",
        padding: str = "same",
        kernel_initializer: str = "glorot_uniform",
    ):
        """
        Creates VGG-style block:
            (Conv2D + ReLU) repeated conv_count times, then MaxPool2D.
        """
        try:
            from tensorflow.keras import layers

            for i in range(conv_count):
                x = layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    bias_initializer="zeros",
                    name=f"{block_name}_conv{i+1}",
                )(x)

            x = layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name=f"{block_name}_pool",
            )(x)

            return x

        except Exception as e:
            logger(f"VGG block creation failed\tblock={block_name}\t{e}", color="red")
            raise

    @staticmethod
    def build_vgg16_keras(
        input_shape: tuple = (224, 224, 3),
        num_classes: int = 1000,
        include_top: bool = True,
        dropout_rate: float = 0.5,
        kernel_initializer: str = "glorot_uniform",
        model_name: str = "VGG16_Manual",
    ):
        """
        Manual VGG16 (Keras) per original design:
            Block1: 64,64 + pool
            Block2: 128,128 + pool
            Block3: 256,256,256 + pool
            Block4: 512,512,512 + pool
            Block5: 512,512,512 + pool
            Top: Flatten -> 4096 -> 4096 -> num_classes(softmax)
        """
        title_message("HW3\tTask1\tBuilding VGG16 (Keras) Manually", color="blue")

        try:
            from tensorflow.keras import layers, models

            if not (isinstance(input_shape, tuple) and len(input_shape) == 3):
                raise ValueError(f"Invalid input_shape\tExpected (H,W,C)\tGot={input_shape}")
            if not isinstance(num_classes, int) or num_classes <= 1:
                raise ValueError(f"Invalid num_classes\tExpected int > 1\tGot={num_classes}")
            if not (0.0 <= float(dropout_rate) <= 1.0):
                raise ValueError(f"Invalid dropout_rate\tExpected [0,1]\tGot={dropout_rate}")

            inputs = layers.Input(shape=input_shape, name="input")
            logger(f"Input shape\t{input_shape}", color="cyan")

            x = Homework3VGG16._vgg_conv_block_keras(
                inputs, filters=64, conv_count=2, block_name="block1", kernel_initializer=kernel_initializer
            )
            x = Homework3VGG16._vgg_conv_block_keras(
                x, filters=128, conv_count=2, block_name="block2", kernel_initializer=kernel_initializer
            )
            x = Homework3VGG16._vgg_conv_block_keras(
                x, filters=256, conv_count=3, block_name="block3", kernel_initializer=kernel_initializer
            )
            x = Homework3VGG16._vgg_conv_block_keras(
                x, filters=512, conv_count=3, block_name="block4", kernel_initializer=kernel_initializer
            )
            x = Homework3VGG16._vgg_conv_block_keras(
                x, filters=512, conv_count=3, block_name="block5", kernel_initializer=kernel_initializer
            )

            if include_top:
                x = layers.Flatten(name="flatten")(x)
                x = layers.Dense(
                    4096,
                    activation="relu",
                    kernel_initializer=kernel_initializer,
                    bias_initializer="zeros",
                    name="fc1",
                )(x)
                x = layers.Dropout(rate=dropout_rate, name="dropout1")(x)
                x = layers.Dense(
                    4096,
                    activation="relu",
                    kernel_initializer=kernel_initializer,
                    bias_initializer="zeros",
                    name="fc2",
                )(x)
                x = layers.Dropout(rate=dropout_rate, name="dropout2")(x)
                outputs = layers.Dense(
                    num_classes,
                    activation="softmax",
                    kernel_initializer=kernel_initializer,
                    bias_initializer="zeros",
                    name="predictions",
                )(x)
            else:
                outputs = x  # feature extractor output

            model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
            logger(f"VGG16 model created\tname={model_name}", color="green")
            logger(f"Model parameters\t{model.count_params():,}", color="cyan")

            return model

        except Exception as e:
            logger(f"VGG16 build failed\t{e}", color="red")
            raise

    @staticmethod
    def log_vgg16_summary(model) -> None:
        """
        Prints the Keras model summary.
        """
        title_message("HW3\tTask1\tVGG16 Summary", color="magenta")

        try:
            if model is None:
                raise ValueError("Model is None\tCannot print summary")

            logger("Printing model.summary() ...", color="yellow")
            model.summary()
            logger("Summary printed.", color="green")

        except Exception as e:
            logger(f"Model summary failed\t{e}", color="red")
            raise

    # ============================================================
    # TASK 2 (Keras): TFDS Cats vs Dogs training pipeline
    # ============================================================

    @staticmethod
    def _prepare_tfds_cats_vs_dogs(
            image_size: tuple = (224, 224),
            batch_size: int = 32,
            shuffle_buffer: int = 2048,
            seed: int = 42,
            data_dir: str = None,
    ):
        """
        TFDS Cats vs Dogs:
            - 80/20 split
            - VGG preprocessing (RGB->BGR + mean subtraction)
        """
        title_message("HW3\tTask2\tTFDS Load + VGG Preprocess", color="blue")

        try:
            # import tensorflow as tf
            import tensorflow_datasets as tfds
            from tensorflow.keras.applications.vgg16 import preprocess_input

            if batch_size <= 0:
                raise ValueError(f"batch_size must be > 0\tGot={batch_size}")
            if not (isinstance(image_size, tuple) and len(image_size) == 2):
                raise ValueError(f"image_size must be (H,W)\tGot={image_size}")

            (train_raw, val_raw), ds_info = tfds.load(
                "cats_vs_dogs",
                split=["train[:80%]", "train[80%:]"],
                as_supervised=True,
                with_info=True,
                data_dir=data_dir,
            )

            class_names = ds_info.features["label"].names
            logger(f"Classes\t{class_names}", color="cyan")

            def _preprocess(img, label):
                img = tf.image.resize(img, image_size)
                img = tf.cast(img, tf.float32)
                img = preprocess_input(img)
                return img, label

            train_ds = train_raw.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
            train_ds = train_ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
            train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            val_ds = val_raw.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
            val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            logger("TFDS datasets ready.", color="green")
            return train_ds, val_ds, class_names

        except Exception as e:
            logger(f"TFDS load failed\t{e}", color="red")
            raise

    @staticmethod
    def _prepare_local_cats_vs_dogs_dir(
            dataset_dir: str,
            image_size: tuple = (224, 224),
            batch_size: int = 32,
            seed: int = 42,
            cache: bool = False,
    ):
        """
        Local PetImages folder:
            dataset_dir/
                Cat/
                Dog/
        """
        title_message("HW3\tTask2\tLocal Dir Load + VGG Preprocess", color="blue")

        try:
            import os
            # import tensorflow as tf
            from tensorflow.keras.applications.vgg16 import preprocess_input

            if not isinstance(dataset_dir, str) or len(dataset_dir.strip()) == 0:
                raise ValueError("dataset_dir must be a non-empty string")
            if not os.path.isdir(dataset_dir):
                raise ValueError(f"dataset_dir not found\t{dataset_dir}")

            train_ds = tf.keras.utils.image_dataset_from_directory(
                dataset_dir,
                labels="inferred",
                label_mode="int",
                validation_split=0.2,
                subset="training",
                seed=seed,
                image_size=image_size,
                batch_size=batch_size,
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                dataset_dir,
                labels="inferred",
                label_mode="int",
                validation_split=0.2,
                subset="validation",
                seed=seed,
                image_size=image_size,
                batch_size=batch_size,
            )

            class_names = list(train_ds.class_names)
            logger(f"Classes\t{class_names}", color="cyan")

            def _vgg_map(x, y):
                x = tf.cast(x, tf.float32)
                x = preprocess_input(x)
                return x, y

            options = tf.data.Options()
            options.experimental_deterministic = False  # faster input pipeline

            train_ds = train_ds.map(_vgg_map, num_parallel_calls=tf.data.AUTOTUNE)
            train_ds = train_ds.ignore_errors()
            train_ds = train_ds.with_options(options)
            if cache:
                train_ds = train_ds.cache()
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

            val_ds = val_ds.map(_vgg_map, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.ignore_errors()
            val_ds = val_ds.with_options(options)
            if cache:
                val_ds = val_ds.cache()
            val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

            logger("Local datasets ready.", color="green")
            return train_ds, val_ds, class_names

        except Exception as e:
            logger(f"Local dataset prep failed\t{e}", color="red")
            raise

    @staticmethod
    def clean_corrupted_petimages_files(
            petimages_dir: str,
            max_delete: int = 10_000,
    ):
        """
        Deletes corrupted image files from:
            petimages_dir/Cat
            petimages_dir/Dog
        """
        title_message("HW3\tTask2\tClean Corrupted Images", color="magenta")

        try:
            import os
            from PIL import Image

            if not os.path.isdir(petimages_dir):
                raise ValueError(f"petimages_dir not found\t{petimages_dir}")

            folders = ["Cat", "Dog"]
            deleted = 0
            scanned = 0

            for folder in folders:
                class_dir = os.path.join(petimages_dir, folder)
                if not os.path.isdir(class_dir):
                    raise ValueError(f"Missing class folder\t{class_dir}")

                for fname in os.listdir(class_dir):
                    fpath = os.path.join(class_dir, fname)
                    if not os.path.isfile(fpath):
                        continue
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    scanned += 1
                    try:
                        with Image.open(fpath) as img:
                            img.verify()  # verifies file integrity
                    except Exception:
                        try:
                            os.remove(fpath)
                            deleted += 1
                            if deleted >= max_delete:
                                logger(f"Reached max_delete={max_delete}", color="yellow")
                                logger(f"Scanned\t{scanned}\tDeleted\t{deleted}", color="cyan")
                                return deleted
                        except Exception:
                            pass

            logger(f"Scanned\t{scanned}\tDeleted\t{deleted}", color="green")
            return deleted

        except Exception as e:
            logger(f"Corrupt clean failed\t{e}", color="red")
            raise

    @staticmethod
    def _deprocess_vgg16_for_display(x):
        """
        Undo VGG preprocessing for visualization:
            - add ImageNet means (BGR order)
            - convert to RGB
            - scale to [0,1]
        """
        try:
            # import tensorflow as tf

            means_bgr = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
            x = x + means_bgr
            x = x[..., ::-1]  # BGR -> RGB
            x = tf.clip_by_value(x / 255.0, 0.0, 1.0)
            return x

        except Exception as e:
            logger(f"Deprocess failed\t{e}", color="red")
            raise

    @staticmethod
    def _compute_val_metrics_keras(model, val_ds, class_names: list):
        """
        Metrics:
            - overall accuracy
            - per-class precision
        """
        title_message("HW3\tTask2\tValidation Metrics", color="blue")

        try:
            # import tensorflow as tf
            from sklearn.metrics import accuracy_score, precision_score

            y_true = []
            y_pred = []

            for xb, yb in val_ds:
                probs = model(xb, training=False)
                preds = tf.argmax(probs, axis=1)
                y_true.extend(yb.numpy().tolist())
                y_pred.extend(preds.numpy().tolist())

            acc = float(accuracy_score(y_true, y_pred))
            prec = precision_score(
                y_true,
                y_pred,
                average=None,
                labels=list(range(len(class_names))),
                zero_division=0,
            )

            metrics_dict = {"val_accuracy": acc, "class_names": class_names}
            for i, cname in enumerate(class_names):
                metrics_dict[f"precision_{cname}"] = float(prec[i])

            logger(f"Val Accuracy\t{acc:.4f}", color="green")
            for cname in class_names:
                logger(f"Precision\t{cname}\t{metrics_dict[f'precision_{cname}']:.4f}", color="green")

            return metrics_dict

        except Exception as e:
            logger(f"Metric computation failed\t{e}", color="red")
            raise

    @staticmethod
    def _plot_train_val_loss(train_loss: list, val_loss: list, title: str):
        """
        Plot train/val loss over epochs.
        """
        title_message("HW3\tTask2\tPlot Loss Curves", color="blue")

        try:
            import matplotlib
            import os as _mpl_os
            _mpl_backend = _mpl_os.environ.get("MPLBACKEND", "")
            if _mpl_backend:
                matplotlib.use(_mpl_backend)
            import matplotlib.pyplot as plt

            if len(train_loss) == 0:
                logger("No loss data found to plot.", color="red")
                return

            epochs = list(range(1, len(train_loss) + 1))
            plt.figure()
            plt.plot(epochs, train_loss, label="train_loss")
            if len(val_loss) == len(train_loss):
                plt.plot(epochs, val_loss, label="val_loss")
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

            logger("Loss curves plotted.", color="green")

        except Exception as e:
            logger(f"Loss plot failed\t{e}", color="red")
            raise

    @staticmethod
    def _show_qualitative_keras(model, val_ds, class_names: list, count: int = 12):
        """
        Display images with predicted vs true labels (validation batch).
        """
        title_message("HW3\tTask2\tQualitative Results", color="blue")

        try:
            # import tensorflow as tf
            # import numpy as np
            import matplotlib
            import os as _mpl_os
            _mpl_backend = _mpl_os.environ.get("MPLBACKEND", "")
            if _mpl_backend:
                matplotlib.use(_mpl_backend)
            import matplotlib.pyplot as plt

            xb, yb = next(iter(val_ds))
            probs = model(xb, training=False)
            preds = tf.argmax(probs, axis=1).numpy()

            images_disp = Homework3VGG16._deprocess_vgg16_for_display(xb).numpy()
            labels = yb.numpy()

            n = min(count, images_disp.shape[0])
            cols = 4
            rows = int(np.ceil(n / cols))

            plt.figure(figsize=(12, 3 * rows))
            for i in range(n):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(images_disp[i])
                t = class_names[int(labels[i])]
                p = class_names[int(preds[i])]
                plt.title(f"T:{t}  P:{p}")
                plt.axis("off")
            plt.tight_layout()
            plt.show()

            logger("Qualitative samples displayed.", color="green")

        except Exception as e:
            logger(f"Qualitative display failed\t{e}", color="red")
            raise

    @staticmethod
    def train_vgg16_cats_vs_dogs_keras(
            input_shape: tuple = (224, 224, 3),
            batch_size: int = 16,
            epochs: int = 5,
            learning_rate: float = 1e-3,
            momentum: float = 0.9,
            dropout_rate: float = 0.5,
            seed: int = 42,
            use_early_stopping: bool = True,
            early_stop_patience: int = 4,
            show_qualitative: bool = True,
            qualitative_count: int = 12,
            dataset_source: str = "local",
            data_dir: str = None,
            local_dataset_dir: str = None,
            auto_download_on_tfds_fail: bool = True,
            auto_download_root_dir: str = ".",
            force_redownload: bool = False,
            steps_per_epoch: int = None,
            validation_steps: int = None,
            log_every_n_batches: int = 50,
            verbose: int = 1,
    ):
        """
        Task 2:
            - Supports TFDS or local dataset
            - Optional steps_per_epoch/validation_steps for faster runs
            - Batch logging so epoch doesn't look "stuck"
        """
        title_message("HW3\tTask2\tTrain VGG16 (Keras) Cats vs Dogs", color="magenta")

        try:
            # import tensorflow as tf
            # import numpy as np

            if not (isinstance(input_shape, tuple) and len(input_shape) == 3):
                raise ValueError(f"input_shape must be (H,W,C)\tGot={input_shape}")
            if epochs <= 0:
                raise ValueError(f"epochs must be > 0\tGot={epochs}")

            tf.random.set_seed(seed)
            np.random.seed(seed)

            # --- Data ---
            if dataset_source.lower() == "tfds":
                try:
                    train_ds, val_ds, class_names = Homework3VGG16._prepare_tfds_cats_vs_dogs(
                        image_size=(input_shape[0], input_shape[1]),
                        batch_size=batch_size,
                        seed=seed,
                        data_dir=data_dir,
                    )
                except Exception as e_tfds:
                    if not auto_download_on_tfds_fail:
                        raise
                    logger(f"TFDS failed -> switching to PetImages\t{e_tfds}", color="yellow")
                    petimages_dir = Homework3VGG16.download_cats_vs_dogs_petimages_to_dir(
                        target_root_dir=auto_download_root_dir,
                        force_redownload=force_redownload,
                    )
                    train_ds, val_ds, class_names = Homework3VGG16._prepare_local_cats_vs_dogs_dir(
                        dataset_dir=petimages_dir,
                        image_size=(input_shape[0], input_shape[1]),
                        batch_size=batch_size,
                        seed=seed,
                    )
            elif dataset_source.lower() == "local":
                if local_dataset_dir is None:
                    raise ValueError("local_dataset_dir is required when dataset_source='local'")
                train_ds, val_ds, class_names = Homework3VGG16._prepare_local_cats_vs_dogs_dir(
                    dataset_dir=local_dataset_dir,
                    image_size=(input_shape[0], input_shape[1]),
                    batch_size=batch_size,
                    seed=seed,
                )
            else:
                raise ValueError("dataset_source must be 'tfds' or 'local'")

            # Repeat only when step limits are used (prevents iterator exhaustion)
            if steps_per_epoch is not None:
                train_ds = train_ds.repeat()
                logger(f"Using steps_per_epoch={steps_per_epoch}", color="yellow")
            if validation_steps is not None:
                val_ds = val_ds.repeat()
                logger(f"Using validation_steps={validation_steps}", color="yellow")

            # --- Model ---
            model = Homework3VGG16.build_vgg16_keras(
                input_shape=input_shape,
                num_classes=2,
                include_top=True,
                dropout_rate=dropout_rate,
                model_name="VGG16_Manual_CatsDogs_Keras",
            )

            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            logger(f"Params\tbatch={batch_size}\tepochs={epochs}\tlr={learning_rate}\tmomentum={momentum}\tdropout={dropout_rate}\tsource={dataset_source}", color="cyan")
            callbacks = []

            if use_early_stopping:
                callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=early_stop_patience,
                        restore_best_weights=True,
                    )
                )
                logger(f"EarlyStopping\tpatience={early_stop_patience}", color="yellow")

            class BatchLogger(tf.keras.callbacks.Callback):
                def on_train_batch_end(self, batch, logs=None):
                    if log_every_n_batches and (batch + 1) % log_every_n_batches == 0:
                        loss_v = float(logs.get("loss", 0.0)) if logs else 0.0
                        acc_v = float(logs.get("accuracy", 0.0)) if logs else 0.0
                        logger(f"Batch {batch + 1}\tloss={loss_v:.4f}\tacc={acc_v:.4f}", color="cyan")

            callbacks.append(BatchLogger())

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                verbose=verbose,
            )

            metrics_dict = Homework3VGG16._compute_val_metrics_keras(
                model=model,
                val_ds=val_ds.take(validation_steps) if validation_steps else val_ds,
                class_names=class_names,
            )

            Homework3VGG16._plot_train_val_loss(
                train_loss=history.history.get("loss", []),
                val_loss=history.history.get("val_loss", []),
                title="VGG16 (Keras) Cats vs Dogs - Loss Curves",
            )

            if show_qualitative:
                Homework3VGG16._show_qualitative_keras(
                    model=model,
                    val_ds=val_ds.take(1),
                    class_names=class_names,
                    count=qualitative_count,
                )

            logger("HW3 Task2 completed successfully.", color="green")
            return model, history, metrics_dict

        except Exception as e:
            logger(f"HW3 Task2 failed\t{e}", color="red")
            raise

    @staticmethod
    def download_cats_vs_dogs_petimages_to_dir(
            target_root_dir: str = ".",
            force_redownload: bool = False,
            url: str = None,
    ) -> str:
        """
        Downloads Cats vs Dogs zip and extracts it into:
            {target_root_dir}/cats_vs_dogs_data/extracted/PetImages

        Skip rules:
            - If PetImages exists -> skip download + extraction
            - Else if zip exists -> skip download, do extraction
            - force_redownload=True -> wipe work_dir and redo everything
        """
        title_message("HW3\tTask2\tDownload Cats vs Dogs (PetImages)", color="magenta")

        try:
            import os
            import zipfile
            import shutil
            import urllib.request

            root_dir = os.path.abspath(target_root_dir)
            work_dir = os.path.join(root_dir, "cats_vs_dogs_data")
            zip_path = os.path.join(work_dir, "kagglecatsanddogs_5340.zip")
            extract_dir = os.path.join(work_dir, "extracted")
            petimages_dir = os.path.join(extract_dir, "PetImages")

            if url is None:
                url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

            # Force redownload resets everything
            if force_redownload and os.path.exists(work_dir):
                logger("force_redownload=True -> removing existing dataset folder", color="yellow")
                shutil.rmtree(work_dir)

            os.makedirs(work_dir, exist_ok=True)

            # If already extracted and valid, skip everything
            cat_dir = os.path.join(petimages_dir, "Cat")
            dog_dir = os.path.join(petimages_dir, "Dog")
            if os.path.isdir(cat_dir) and os.path.isdir(dog_dir):
                logger(f"PetImages already present -> skipping\t{petimages_dir}", color="green")
                return petimages_dir

            import ssl
            import urllib.request

            # Disable SSL certificate verification (for this session)
            ssl._create_default_https_context = ssl._create_unverified_context

            # Download only if zip doesn't exist
            if not os.path.exists(zip_path):
                logger(f"Downloading\t{url}", color="yellow")
                urllib.request.urlretrieve(url, zip_path)
                logger(f"Downloaded\t{zip_path}", color="green")
            else:
                logger(f"Zip already exists -> skipping download\t{zip_path}", color="cyan")

            # Extract only if PetImages is missing
            if not os.path.exists(petimages_dir):
                os.makedirs(extract_dir, exist_ok=True)
                logger("Extracting zip...", color="yellow")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)
                logger(f"Extracted\t{extract_dir}", color="green")
            else:
                logger(f"Extract folder exists -> skipping extraction\t{petimages_dir}", color="cyan")

            # Validate structure
            if not (os.path.isdir(cat_dir) and os.path.isdir(dog_dir)):
                raise ValueError(f"Invalid structure\tExpected Cat/ and Dog/\tGot={petimages_dir}")

            logger(f"PetImages ready\t{petimages_dir}", color="green")
            return petimages_dir

        except Exception as e:
            logger(f"Download/extract failed\t{e}", color="red")
            raise

    @staticmethod
    def run_task2_with_auto_download(
            input_shape: tuple = (224, 224, 3),
            batch_size: int = 32,
            epochs: int = 5,
            learning_rate: float = 1e-3,
            momentum: float = 0.9,
            dropout_rate: float = 0.5,
            seed: int = 42,
            use_early_stopping: bool = True,
            early_stop_patience: int = 3,
            show_qualitative: bool = True,
            qualitative_count: int = 12,
            target_root_dir: str = ".",
            force_redownload: bool = False,
            clean_corrupt: bool = True,
    ):
        """
        - Downloads + extracts PetImages into project folder
        - Optionally deletes corrupted images once
        - Trains using local loader
        """
        title_message("HW3\tTask2\tAuto-Download + Train", color="magenta")

        try:
            petimages_dir = Homework3VGG16.download_cats_vs_dogs_petimages_to_dir(
                target_root_dir=target_root_dir,
                force_redownload=force_redownload,
            )

            if clean_corrupt:
                deleted = Homework3VGG16.clean_corrupted_petimages_files(petimages_dir)
                logger(f"Corrupt files deleted\t{deleted}", color="yellow")

            return Homework3VGG16.train_vgg16_cats_vs_dogs_keras(
                input_shape=input_shape,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                momentum=momentum,
                dropout_rate=dropout_rate,
                seed=seed,
                use_early_stopping=use_early_stopping,
                early_stop_patience=early_stop_patience,
                show_qualitative=show_qualitative,
                qualitative_count=qualitative_count,
                dataset_source="local",
                local_dataset_dir=petimages_dir,
            )

        except Exception as e:
            logger(f"Auto-download train failed\t{e}", color="red")
            raise


if __name__ == "__main__":
    title_message("HW3\tMAIN\tDemo: Task1 + Task2", color="magenta")

    # model = Homework3VGG16.build_vgg16_keras(input_shape=(224,224,3), num_classes=1000, include_top=True)
    # Homework3VGG16.log_vgg16_summary(model)
    #
    # model_k, hist_k, met_k = Homework3VGG16.train_vgg16_cats_vs_dogs_keras(
    #     dataset_source="tfds",
    #     auto_download_on_tfds_fail=True,
    #     auto_download_root_dir=".",
    #     force_redownload=False,
    #     epochs=5,
    # )

    # model_k, hist_k, met_k = Homework3VGG16.run_task2_with_auto_download(
    #     batch_size=8,
    #     epochs=3,
    #     force_redownload=False,
    #     clean_corrupt=True,
    # )

    # model_k, hist_k, met_k = Homework3VGG16.train_vgg16_cats_vs_dogs_keras(
    #     dataset_source="local",
    #     local_dataset_dir=r".\cats_vs_dogs_data\extracted\PetImages",
    #     batch_size=8,
    #     epochs=3,
    #     steps_per_epoch=100,
    #     validation_steps=50,
    #     log_every_n_batches=10,
    #     verbose=1,
    # )

    # ----------------------------
    # Task 1: Manual VGG16 sanity tests
    # ----------------------------
    title_message("HW3\tMAIN\tTask1 Tests", color="blue")



    vgg_model = Homework3VGG16.build_vgg16_keras(
        input_shape=(224, 224, 3),
        num_classes=2,
        include_top=True,
        dropout_rate=0.5,
        model_name="VGG16_Manual_Task1_Test",
    )

    Homework3VGG16.log_vgg16_summary(vgg_model)

    x_dummy = np.random.rand(2, 224, 224, 3).astype(np.float32)
    y_dummy = vgg_model(x_dummy, training=False)
    logger(f"Task1 forward pass OK\tinput={x_dummy.shape}\toutput={tuple(y_dummy.shape)}", color="green")

    # ----------------------------
    # Task 2: Dataset quick test + training quick run
    # ----------------------------
    title_message("HW3\tMAIN\tTask2 Tests", color="blue")

    petimages_dir = Homework3VGG16.download_cats_vs_dogs_petimages_to_dir(
        target_root_dir=".",
        force_redownload=False,
    )

    deleted = Homework3VGG16.clean_corrupted_petimages_files(
        petimages_dir=petimages_dir,
        max_delete=10000,
    )
    logger(f"Corrupt cleanup done\tdeleted={deleted}", color="yellow")

    model_k, hist_k, met_k = Homework3VGG16.train_vgg16_cats_vs_dogs_keras(
        input_shape=(160, 160, 3), # reducing for faster processing
        dataset_source="local",
        local_dataset_dir=petimages_dir,

        batch_size=16,
        epochs=15,
        # steps_per_epoch=100,
        # validation_steps=50,

        use_early_stopping=True,
        early_stop_patience=4,

        log_every_n_batches=50,
        verbose=1,
        show_qualitative=True,
        qualitative_count=12,
    )

    logger(f"Task2 metrics\t{met_k}", color="green")

    title_message("HW3\tMAIN\tCompleted", color="green")


"""
Output from Colab:
2026-01-19 00:44:00		[Task_set_12]	######################################
2026-01-19 00:44:00		[Task_set_12]	#	HW3	MAIN	Demo: Task1 + Task2	#
2026-01-19 00:44:00		[Task_set_12]	######################################
2026-01-19 00:44:00		[Task_set_12]	##############################
2026-01-19 00:44:00		[Task_set_12]	#	HW3	MAIN	Task1 Tests	#
2026-01-19 00:44:00		[Task_set_12]	##############################
2026-01-19 00:44:18		[Task_set_12]	###################################################
2026-01-19 00:44:18		[Task_set_12]	#	HW3	Task1	Building VGG16 (Keras) Manually	#
2026-01-19 00:44:18		[Task_set_12]	###################################################
2026-01-19 00:44:18		[Task_set_12]	Input shape	(224, 224, 3)
2026-01-19 00:44:23		[Task_set_12]	VGG16 model created	name=VGG16_Manual_Task1_Test
2026-01-19 00:44:23		[Task_set_12]	Model parameters	134,268,738
2026-01-19 00:44:23		[Task_set_12]	#################################
2026-01-19 00:44:23		[Task_set_12]	#	HW3	Task1	VGG16 Summary	#
2026-01-19 00:44:23		[Task_set_12]	#################################
2026-01-19 00:44:23		[Task_set_12]	Printing model.summary() ...
Model: "VGG16_Manual_Task1_Test"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input (InputLayer)              │ (None, 224, 224, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block1_conv1 (Conv2D)           │ (None, 224, 224, 64)   │         1,792 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block1_conv2 (Conv2D)           │ (None, 224, 224, 64)   │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block1_pool (MaxPooling2D)      │ (None, 112, 112, 64)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block2_conv1 (Conv2D)           │ (None, 112, 112, 128)  │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block2_conv2 (Conv2D)           │ (None, 112, 112, 128)  │       147,584 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block2_pool (MaxPooling2D)      │ (None, 56, 56, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block3_conv1 (Conv2D)           │ (None, 56, 56, 256)    │       295,168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block3_conv2 (Conv2D)           │ (None, 56, 56, 256)    │       590,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block3_conv3 (Conv2D)           │ (None, 56, 56, 256)    │       590,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block3_pool (MaxPooling2D)      │ (None, 28, 28, 256)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block4_conv1 (Conv2D)           │ (None, 28, 28, 512)    │     1,180,160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block4_conv2 (Conv2D)           │ (None, 28, 28, 512)    │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block4_conv3 (Conv2D)           │ (None, 28, 28, 512)    │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block4_pool (MaxPooling2D)      │ (None, 14, 14, 512)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block5_conv1 (Conv2D)           │ (None, 14, 14, 512)    │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block5_conv2 (Conv2D)           │ (None, 14, 14, 512)    │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block5_conv3 (Conv2D)           │ (None, 14, 14, 512)    │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block5_pool (MaxPooling2D)      │ (None, 7, 7, 512)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 25088)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ fc1 (Dense)                     │ (None, 4096)           │   102,764,544 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout1 (Dropout)              │ (None, 4096)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ fc2 (Dense)                     │ (None, 4096)           │    16,781,312 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout2 (Dropout)              │ (None, 4096)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ predictions (Dense)             │ (None, 2)              │         8,194 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 134,268,738 (512.19 MB)
 Trainable params: 134,268,738 (512.19 MB)
 Non-trainable params: 0 (0.00 B)
2026-01-19 00:44:23		[Task_set_12]	Summary printed.
2026-01-19 00:44:25		[Task_set_12]	Task1 forward pass OK	input=(2, 224, 224, 3)	output=(2, 2)
2026-01-19 00:44:25		[Task_set_12]	##############################
2026-01-19 00:44:25		[Task_set_12]	#	HW3	MAIN	Task2 Tests	#
2026-01-19 00:44:25		[Task_set_12]	##############################
2026-01-19 00:44:25		[Task_set_12]	#####################################################
2026-01-19 00:44:25		[Task_set_12]	#	HW3	Task2	Download Cats vs Dogs (PetImages)	#
2026-01-19 00:44:25		[Task_set_12]	#####################################################
2026-01-19 00:44:25		[Task_set_12]	Downloading	https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
2026-01-19 00:44:28		[Task_set_12]	Downloaded	/content/cats_vs_dogs_data/kagglecatsanddogs_5340.zip
2026-01-19 00:44:28		[Task_set_12]	Extracting zip...
2026-01-19 00:44:35		[Task_set_12]	Extracted	/content/cats_vs_dogs_data/extracted
2026-01-19 00:44:35		[Task_set_12]	PetImages ready	/content/cats_vs_dogs_data/extracted/PetImages
2026-01-19 00:44:35		[Task_set_12]	##########################################
2026-01-19 00:44:35		[Task_set_12]	#	HW3	Task2	Clean Corrupted Images	#
2026-01-19 00:44:35		[Task_set_12]	##########################################
/usr/local/lib/python3.12/dist-packages/PIL/TiffImagePlugin.py:950: UserWarning: Truncated File Read
  warnings.warn(str(msg))
2026-01-19 00:44:37		[Task_set_12]	Scanned	25000	Deleted	2
2026-01-19 00:44:37		[Task_set_12]	Corrupt cleanup done	deleted=2
2026-01-19 00:44:37		[Task_set_12]	####################################################
2026-01-19 00:44:37		[Task_set_12]	#	HW3	Task2	Train VGG16 (Keras) Cats vs Dogs	#
2026-01-19 00:44:37		[Task_set_12]	####################################################
2026-01-19 00:44:37		[Task_set_12]	###################################################
2026-01-19 00:44:37		[Task_set_12]	#	HW3	Task2	Local Dir Load + VGG Preprocess	#
2026-01-19 00:44:37		[Task_set_12]	###################################################
Found 24998 files belonging to 2 classes.
Using 19999 files for training.
Found 24998 files belonging to 2 classes.
Using 4999 files for validation.
2026-01-19 00:44:40		[Task_set_12]	Classes	['Cat', 'Dog']
2026-01-19 00:44:40		[Task_set_12]	Local datasets ready.
2026-01-19 00:44:40		[Task_set_12]	###################################################
2026-01-19 00:44:40		[Task_set_12]	#	HW3	Task1	Building VGG16 (Keras) Manually	#
2026-01-19 00:44:40		[Task_set_12]	###################################################
2026-01-19 00:44:40		[Task_set_12]	Input shape	(160, 160, 3)
2026-01-19 00:44:40		[Task_set_12]	VGG16 model created	name=VGG16_Manual_CatsDogs_Keras
2026-01-19 00:44:40		[Task_set_12]	Model parameters	83,937,090
2026-01-19 00:44:40		[Task_set_12]	Params	batch=16	epochs=15	lr=0.001	momentum=0.9	dropout=0.5	source=local
2026-01-19 00:44:40		[Task_set_12]	EarlyStopping	patience=4
Epoch 1/15
     49/Unknown 27s 153ms/step - accuracy: 0.4957 - loss: 0.73352026-01-19 00:45:07		[Task_set_12]	Batch 50	loss=0.7163	acc=0.5000
     99/Unknown 35s 154ms/step - accuracy: 0.4971 - loss: 0.72222026-01-19 00:45:15		[Task_set_12]	Batch 100	loss=0.7075	acc=0.4981
    149/Unknown 43s 154ms/step - accuracy: 0.5017 - loss: 0.71612026-01-19 00:45:23		[Task_set_12]	Batch 150	loss=0.7013	acc=0.5192
    199/Unknown 51s 155ms/step - accuracy: 0.5052 - loss: 0.71212026-01-19 00:45:31		[Task_set_12]	Batch 200	loss=0.6995	acc=0.5153
    249/Unknown 58s 156ms/step - accuracy: 0.5085 - loss: 0.70932026-01-19 00:45:39		[Task_set_12]	Batch 250	loss=0.6970	acc=0.5238
    299/Unknown 66s 156ms/step - accuracy: 0.5110 - loss: 0.70712026-01-19 00:45:46		[Task_set_12]	Batch 300	loss=0.6966	acc=0.5208
    349/Unknown 74s 156ms/step - accuracy: 0.5126 - loss: 0.70562026-01-19 00:45:54		[Task_set_12]	Batch 350	loss=0.6955	acc=0.5282
    399/Unknown 82s 156ms/step - accuracy: 0.5147 - loss: 0.70422026-01-19 00:46:02		[Task_set_12]	Batch 400	loss=0.6936	acc=0.5320
    449/Unknown 90s 156ms/step - accuracy: 0.5167 - loss: 0.70302026-01-19 00:46:10		[Task_set_12]	Batch 450	loss=0.6926	acc=0.5350
    499/Unknown 98s 156ms/step - accuracy: 0.5188 - loss: 0.70192026-01-19 00:46:18		[Task_set_12]	Batch 500	loss=0.6905	acc=0.5399
    549/Unknown 105s 156ms/step - accuracy: 0.5209 - loss: 0.70082026-01-19 00:46:26		[Task_set_12]	Batch 550	loss=0.6892	acc=0.5443
    599/Unknown 113s 156ms/step - accuracy: 0.5232 - loss: 0.69972026-01-19 00:46:33		[Task_set_12]	Batch 600	loss=0.6857	acc=0.5520
    649/Unknown 121s 156ms/step - accuracy: 0.5253 - loss: 0.69872026-01-19 00:46:41		[Task_set_12]	Batch 650	loss=0.6867	acc=0.5499
    699/Unknown 129s 156ms/step - accuracy: 0.5272 - loss: 0.69782026-01-19 00:46:49		[Task_set_12]	Batch 700	loss=0.6860	acc=0.5527
    749/Unknown 137s 156ms/step - accuracy: 0.5290 - loss: 0.69692026-01-19 00:46:57		[Task_set_12]	Batch 750	loss=0.6849	acc=0.5548
    799/Unknown 144s 156ms/step - accuracy: 0.5307 - loss: 0.69622026-01-19 00:47:05		[Task_set_12]	Batch 800	loss=0.6842	acc=0.5566
    849/Unknown 152s 156ms/step - accuracy: 0.5323 - loss: 0.69542026-01-19 00:47:12		[Task_set_12]	Batch 850	loss=0.6827	acc=0.5593
    899/Unknown 160s 156ms/step - accuracy: 0.5338 - loss: 0.69472026-01-19 00:47:20		[Task_set_12]	Batch 900	loss=0.6822	acc=0.5596
    949/Unknown 168s 156ms/step - accuracy: 0.5353 - loss: 0.69402026-01-19 00:47:28		[Task_set_12]	Batch 950	loss=0.6808	acc=0.5622
    999/Unknown 176s 156ms/step - accuracy: 0.5367 - loss: 0.69332026-01-19 00:47:36		[Task_set_12]	Batch 1000	loss=0.6783	acc=0.5656
   1049/Unknown 184s 156ms/step - accuracy: 0.5381 - loss: 0.69252026-01-19 00:47:44		[Task_set_12]	Batch 1050	loss=0.6768	acc=0.5681
   1099/Unknown 191s 156ms/step - accuracy: 0.5396 - loss: 0.69172026-01-19 00:47:52		[Task_set_12]	Batch 1100	loss=0.6739	acc=0.5719
   1149/Unknown 199s 156ms/step - accuracy: 0.5411 - loss: 0.69092026-01-19 00:47:59		[Task_set_12]	Batch 1150	loss=0.6720	acc=0.5745
   1199/Unknown 207s 156ms/step - accuracy: 0.5425 - loss: 0.69012026-01-19 00:48:07		[Task_set_12]	Batch 1200	loss=0.6710	acc=0.5773
   1244/Unknown 230s 169ms/step - accuracy: 0.5439 - loss: 0.6894
/usr/local/lib/python3.12/dist-packages/keras/src/trainers/epoch_iterator.py:160: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 256s 190ms/step - accuracy: 0.5439 - loss: 0.6894 - val_accuracy: 0.6398 - val_loss: 0.6495
Epoch 2/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:05 156ms/step - accuracy: 0.6603 - loss: 0.64732026-01-19 00:49:04		[Task_set_12]	Batch 50	loss=0.6404	acc=0.6450
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:58 156ms/step - accuracy: 0.6558 - loss: 0.64062026-01-19 00:49:12		[Task_set_12]	Batch 100	loss=0.6305	acc=0.6594
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:50 156ms/step - accuracy: 0.6558 - loss: 0.63702026-01-19 00:49:20		[Task_set_12]	Batch 150	loss=0.6260	acc=0.6596
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 156ms/step - accuracy: 0.6575 - loss: 0.63382026-01-19 00:49:27		[Task_set_12]	Batch 200	loss=0.6224	acc=0.6653
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:35 156ms/step - accuracy: 0.6593 - loss: 0.63142026-01-19 00:49:35		[Task_set_12]	Batch 250	loss=0.6214	acc=0.6680
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:27 156ms/step - accuracy: 0.6609 - loss: 0.62942026-01-19 00:49:43		[Task_set_12]	Batch 300	loss=0.6173	acc=0.6708
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:19 156ms/step - accuracy: 0.6629 - loss: 0.62732026-01-19 00:49:51		[Task_set_12]	Batch 350	loss=0.6111	acc=0.6786
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 156ms/step - accuracy: 0.6646 - loss: 0.62532026-01-19 00:49:59		[Task_set_12]	Batch 400	loss=0.6111	acc=0.6755
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 156ms/step - accuracy: 0.6657 - loss: 0.62372026-01-19 00:50:06		[Task_set_12]	Batch 450	loss=0.6101	acc=0.6729
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 156ms/step - accuracy: 0.6664 - loss: 0.62212026-01-19 00:50:14		[Task_set_12]	Batch 500	loss=0.6075	acc=0.6731
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:48 156ms/step - accuracy: 0.6671 - loss: 0.62072026-01-19 00:50:22		[Task_set_12]	Batch 550	loss=0.6060	acc=0.6743
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:40 156ms/step - accuracy: 0.6677 - loss: 0.61952026-01-19 00:50:30		[Task_set_12]	Batch 600	loss=0.6044	acc=0.6759
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:32 156ms/step - accuracy: 0.6683 - loss: 0.61832026-01-19 00:50:38		[Task_set_12]	Batch 650	loss=0.6041	acc=0.6759
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 156ms/step - accuracy: 0.6691 - loss: 0.61712026-01-19 00:50:45		[Task_set_12]	Batch 700	loss=0.6009	acc=0.6793
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 156ms/step - accuracy: 0.6698 - loss: 0.61602026-01-19 00:50:53		[Task_set_12]	Batch 750	loss=0.5998	acc=0.6809
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 156ms/step - accuracy: 0.6705 - loss: 0.61502026-01-19 00:51:01		[Task_set_12]	Batch 800	loss=0.5990	acc=0.6813
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 156ms/step - accuracy: 0.6711 - loss: 0.61402026-01-19 00:51:09		[Task_set_12]	Batch 850	loss=0.5970	acc=0.6825
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 53s 156ms/step - accuracy: 0.6718 - loss: 0.61302026-01-19 00:51:17		[Task_set_12]	Batch 900	loss=0.5959	acc=0.6835
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 156ms/step - accuracy: 0.6725 - loss: 0.61212026-01-19 00:51:25		[Task_set_12]	Batch 950	loss=0.5946	acc=0.6849
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 156ms/step - accuracy: 0.6731 - loss: 0.61122026-01-19 00:51:32		[Task_set_12]	Batch 1000	loss=0.5927	acc=0.6859
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 156ms/step - accuracy: 0.6737 - loss: 0.61022026-01-19 00:51:40		[Task_set_12]	Batch 1050	loss=0.5910	acc=0.6868
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 156ms/step - accuracy: 0.6744 - loss: 0.60932026-01-19 00:51:48		[Task_set_12]	Batch 1100	loss=0.5889	acc=0.6883
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 156ms/step - accuracy: 0.6750 - loss: 0.60842026-01-19 00:51:56		[Task_set_12]	Batch 1150	loss=0.5856	acc=0.6907
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 156ms/step - accuracy: 0.6757 - loss: 0.60742026-01-19 00:52:04		[Task_set_12]	Batch 1200	loss=0.5838	acc=0.6927
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 212s 170ms/step - accuracy: 0.6764 - loss: 0.6065 - val_accuracy: 0.7265 - val_loss: 0.5390
Epoch 3/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:07 157ms/step - accuracy: 0.7587 - loss: 0.50332026-01-19 00:52:36		[Task_set_12]	Batch 50	loss=0.5037	acc=0.7512
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:59 157ms/step - accuracy: 0.7574 - loss: 0.50362026-01-19 00:52:44		[Task_set_12]	Batch 100	loss=0.5123	acc=0.7537
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:51 157ms/step - accuracy: 0.7540 - loss: 0.50622026-01-19 00:52:52		[Task_set_12]	Batch 150	loss=0.5127	acc=0.7442
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 157ms/step - accuracy: 0.7520 - loss: 0.50802026-01-19 00:53:00		[Task_set_12]	Batch 200	loss=0.5142	acc=0.7459
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:35 157ms/step - accuracy: 0.7504 - loss: 0.50952026-01-19 00:53:07		[Task_set_12]	Batch 250	loss=0.5141	acc=0.7453
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:27 157ms/step - accuracy: 0.7501 - loss: 0.51012026-01-19 00:53:15		[Task_set_12]	Batch 300	loss=0.5123	acc=0.7506
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:20 157ms/step - accuracy: 0.7507 - loss: 0.50992026-01-19 00:53:23		[Task_set_12]	Batch 350	loss=0.5039	acc=0.7577
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 156ms/step - accuracy: 0.7511 - loss: 0.50982026-01-19 00:53:31		[Task_set_12]	Batch 400	loss=0.5088	acc=0.7533
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 156ms/step - accuracy: 0.7513 - loss: 0.50972026-01-19 00:53:39		[Task_set_12]	Batch 450	loss=0.5086	acc=0.7539
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 157ms/step - accuracy: 0.7516 - loss: 0.50952026-01-19 00:53:47		[Task_set_12]	Batch 500	loss=0.5086	acc=0.7530
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:48 157ms/step - accuracy: 0.7518 - loss: 0.50922026-01-19 00:53:54		[Task_set_12]	Batch 550	loss=0.5036	acc=0.7549
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:41 157ms/step - accuracy: 0.7521 - loss: 0.50882026-01-19 00:54:02		[Task_set_12]	Batch 600	loss=0.5022	acc=0.7572
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 157ms/step - accuracy: 0.7525 - loss: 0.50822026-01-19 00:54:10		[Task_set_12]	Batch 650	loss=0.5009	acc=0.7577
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 157ms/step - accuracy: 0.7530 - loss: 0.50752026-01-19 00:54:18		[Task_set_12]	Batch 700	loss=0.4952	acc=0.7613
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 157ms/step - accuracy: 0.7536 - loss: 0.50662026-01-19 00:54:26		[Task_set_12]	Batch 750	loss=0.4950	acc=0.7607
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 157ms/step - accuracy: 0.7541 - loss: 0.50582026-01-19 00:54:34		[Task_set_12]	Batch 800	loss=0.4920	acc=0.7630
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 157ms/step - accuracy: 0.7546 - loss: 0.50502026-01-19 00:54:41		[Task_set_12]	Batch 850	loss=0.4893	acc=0.7650
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 54s 157ms/step - accuracy: 0.7552 - loss: 0.50412026-01-19 00:54:49		[Task_set_12]	Batch 900	loss=0.4884	acc=0.7644
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 157ms/step - accuracy: 0.7557 - loss: 0.50332026-01-19 00:54:57		[Task_set_12]	Batch 950	loss=0.4883	acc=0.7649
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 157ms/step - accuracy: 0.7562 - loss: 0.50252026-01-19 00:55:05		[Task_set_12]	Batch 1000	loss=0.4871	acc=0.7659
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 157ms/step - accuracy: 0.7567 - loss: 0.50172026-01-19 00:55:13		[Task_set_12]	Batch 1050	loss=0.4856	acc=0.7673
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.7572 - loss: 0.50092026-01-19 00:55:21		[Task_set_12]	Batch 1100	loss=0.4826	acc=0.7690
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.7577 - loss: 0.50012026-01-19 00:55:28		[Task_set_12]	Batch 1150	loss=0.4815	acc=0.7697
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.7582 - loss: 0.49932026-01-19 00:55:36		[Task_set_12]	Batch 1200	loss=0.4788	acc=0.7716
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 213s 171ms/step - accuracy: 0.7588 - loss: 0.4985 - val_accuracy: 0.8061 - val_loss: 0.4327
Epoch 4/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:06 156ms/step - accuracy: 0.8597 - loss: 0.39142026-01-19 00:56:09		[Task_set_12]	Batch 50	loss=0.4115	acc=0.8200
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:59 156ms/step - accuracy: 0.8362 - loss: 0.40322026-01-19 00:56:17		[Task_set_12]	Batch 100	loss=0.4151	acc=0.8106
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:51 157ms/step - accuracy: 0.8295 - loss: 0.40492026-01-19 00:56:24		[Task_set_12]	Batch 150	loss=0.4046	acc=0.8187
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 157ms/step - accuracy: 0.8270 - loss: 0.40452026-01-19 00:56:32		[Task_set_12]	Batch 200	loss=0.4074	acc=0.8169
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:35 157ms/step - accuracy: 0.8253 - loss: 0.40452026-01-19 00:56:40		[Task_set_12]	Batch 250	loss=0.4021	acc=0.8203
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:28 157ms/step - accuracy: 0.8238 - loss: 0.40492026-01-19 00:56:48		[Task_set_12]	Batch 300	loss=0.4103	acc=0.8133
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:20 156ms/step - accuracy: 0.8228 - loss: 0.40502026-01-19 00:56:56		[Task_set_12]	Batch 350	loss=0.4019	acc=0.8186
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 156ms/step - accuracy: 0.8222 - loss: 0.40472026-01-19 00:57:03		[Task_set_12]	Batch 400	loss=0.4009	acc=0.8192
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 157ms/step - accuracy: 0.8219 - loss: 0.40432026-01-19 00:57:11		[Task_set_12]	Batch 450	loss=0.4008	acc=0.8197
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 156ms/step - accuracy: 0.8217 - loss: 0.40382026-01-19 00:57:19		[Task_set_12]	Batch 500	loss=0.3997	acc=0.8208
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:48 157ms/step - accuracy: 0.8217 - loss: 0.40342026-01-19 00:57:27		[Task_set_12]	Batch 550	loss=0.3994	acc=0.8206
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:40 157ms/step - accuracy: 0.8216 - loss: 0.40302026-01-19 00:57:35		[Task_set_12]	Batch 600	loss=0.3997	acc=0.8212
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 156ms/step - accuracy: 0.8216 - loss: 0.40282026-01-19 00:57:43		[Task_set_12]	Batch 650	loss=0.3992	acc=0.8224
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 156ms/step - accuracy: 0.8217 - loss: 0.40242026-01-19 00:57:50		[Task_set_12]	Batch 700	loss=0.3965	acc=0.8238
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 156ms/step - accuracy: 0.8219 - loss: 0.40202026-01-19 00:57:58		[Task_set_12]	Batch 750	loss=0.3947	acc=0.8252
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 156ms/step - accuracy: 0.8221 - loss: 0.40152026-01-19 00:58:06		[Task_set_12]	Batch 800	loss=0.3937	acc=0.8262
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 156ms/step - accuracy: 0.8224 - loss: 0.40102026-01-19 00:58:14		[Task_set_12]	Batch 850	loss=0.3927	acc=0.8276
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 53s 156ms/step - accuracy: 0.8227 - loss: 0.40052026-01-19 00:58:22		[Task_set_12]	Batch 900	loss=0.3917	acc=0.8278
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 156ms/step - accuracy: 0.8230 - loss: 0.40002026-01-19 00:58:30		[Task_set_12]	Batch 950	loss=0.3906	acc=0.8284
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 156ms/step - accuracy: 0.8233 - loss: 0.39942026-01-19 00:58:37		[Task_set_12]	Batch 1000	loss=0.3887	acc=0.8293
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 156ms/step - accuracy: 0.8236 - loss: 0.39892026-01-19 00:58:45		[Task_set_12]	Batch 1050	loss=0.3872	acc=0.8297
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.8238 - loss: 0.39842026-01-19 00:58:53		[Task_set_12]	Batch 1100	loss=0.3878	acc=0.8289
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.8241 - loss: 0.39792026-01-19 00:59:01		[Task_set_12]	Batch 1150	loss=0.3849	acc=0.8307
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.8244 - loss: 0.39732026-01-19 00:59:09		[Task_set_12]	Batch 1200	loss=0.3836	acc=0.8314
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 212s 171ms/step - accuracy: 0.8246 - loss: 0.3968 - val_accuracy: 0.8453 - val_loss: 0.3578
Epoch 5/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:07 157ms/step - accuracy: 0.8647 - loss: 0.31672026-01-19 00:59:41		[Task_set_12]	Batch 50	loss=0.3135	acc=0.8675
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:59 157ms/step - accuracy: 0.8690 - loss: 0.31592026-01-19 00:59:49		[Task_set_12]	Batch 100	loss=0.3149	acc=0.8719
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:51 157ms/step - accuracy: 0.8691 - loss: 0.31512026-01-19 00:59:57		[Task_set_12]	Batch 150	loss=0.3156	acc=0.8683
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 157ms/step - accuracy: 0.8695 - loss: 0.31432026-01-19 01:00:05		[Task_set_12]	Batch 200	loss=0.3088	acc=0.8728
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:35 157ms/step - accuracy: 0.8691 - loss: 0.31452026-01-19 01:00:12		[Task_set_12]	Batch 250	loss=0.3176	acc=0.8640
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:27 156ms/step - accuracy: 0.8681 - loss: 0.31552026-01-19 01:00:20		[Task_set_12]	Batch 300	loss=0.3229	acc=0.8629
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:19 156ms/step - accuracy: 0.8675 - loss: 0.31622026-01-19 01:00:28		[Task_set_12]	Batch 350	loss=0.3174	acc=0.8650
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 156ms/step - accuracy: 0.8671 - loss: 0.31632026-01-19 01:00:36		[Task_set_12]	Batch 400	loss=0.3144	acc=0.8648
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 156ms/step - accuracy: 0.8669 - loss: 0.31612026-01-19 01:00:44		[Task_set_12]	Batch 450	loss=0.3144	acc=0.8647
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 156ms/step - accuracy: 0.8668 - loss: 0.31572026-01-19 01:00:52		[Task_set_12]	Batch 500	loss=0.3109	acc=0.8666
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:48 156ms/step - accuracy: 0.8668 - loss: 0.31532026-01-19 01:00:59		[Task_set_12]	Batch 550	loss=0.3128	acc=0.8661
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:40 157ms/step - accuracy: 0.8668 - loss: 0.31502026-01-19 01:01:07		[Task_set_12]	Batch 600	loss=0.3106	acc=0.8675
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 157ms/step - accuracy: 0.8668 - loss: 0.31482026-01-19 01:01:15		[Task_set_12]	Batch 650	loss=0.3107	acc=0.8669
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 157ms/step - accuracy: 0.8668 - loss: 0.31442026-01-19 01:01:23		[Task_set_12]	Batch 700	loss=0.3082	acc=0.8686
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 157ms/step - accuracy: 0.8669 - loss: 0.31392026-01-19 01:01:31		[Task_set_12]	Batch 750	loss=0.3074	acc=0.8687
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 157ms/step - accuracy: 0.8670 - loss: 0.31352026-01-19 01:01:39		[Task_set_12]	Batch 800	loss=0.3058	acc=0.8693
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 157ms/step - accuracy: 0.8671 - loss: 0.31312026-01-19 01:01:46		[Task_set_12]	Batch 850	loss=0.3062	acc=0.8687
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 54s 157ms/step - accuracy: 0.8672 - loss: 0.31272026-01-19 01:01:54		[Task_set_12]	Batch 900	loss=0.3071	acc=0.8681
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 157ms/step - accuracy: 0.8672 - loss: 0.31252026-01-19 01:02:02		[Task_set_12]	Batch 950	loss=0.3080	acc=0.8673
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 157ms/step - accuracy: 0.8673 - loss: 0.31222026-01-19 01:02:10		[Task_set_12]	Batch 1000	loss=0.3069	acc=0.8680
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 157ms/step - accuracy: 0.8673 - loss: 0.31192026-01-19 01:02:18		[Task_set_12]	Batch 1050	loss=0.3065	acc=0.8680
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.8674 - loss: 0.31172026-01-19 01:02:26		[Task_set_12]	Batch 1100	loss=0.3047	acc=0.8693
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.8674 - loss: 0.31132026-01-19 01:02:33		[Task_set_12]	Batch 1150	loss=0.3033	acc=0.8697
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.8675 - loss: 0.31102026-01-19 01:02:41		[Task_set_12]	Batch 1200	loss=0.3036	acc=0.8697
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 213s 171ms/step - accuracy: 0.8676 - loss: 0.3107 - val_accuracy: 0.8547 - val_loss: 0.3200
Epoch 6/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:07 157ms/step - accuracy: 0.9082 - loss: 0.25052026-01-19 01:03:14		[Task_set_12]	Batch 50	loss=0.2529	acc=0.9038
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:59 156ms/step - accuracy: 0.9076 - loss: 0.24622026-01-19 01:03:21		[Task_set_12]	Batch 100	loss=0.2335	acc=0.9131
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:51 156ms/step - accuracy: 0.9079 - loss: 0.24332026-01-19 01:03:29		[Task_set_12]	Batch 150	loss=0.2436	acc=0.9025
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 156ms/step - accuracy: 0.9066 - loss: 0.24242026-01-19 01:03:37		[Task_set_12]	Batch 200	loss=0.2369	acc=0.9038
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:35 156ms/step - accuracy: 0.9056 - loss: 0.24192026-01-19 01:03:45		[Task_set_12]	Batch 250	loss=0.2401	acc=0.9008
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:27 156ms/step - accuracy: 0.9045 - loss: 0.24222026-01-19 01:03:53		[Task_set_12]	Batch 300	loss=0.2442	acc=0.8996
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:20 156ms/step - accuracy: 0.9037 - loss: 0.24242026-01-19 01:04:01		[Task_set_12]	Batch 350	loss=0.2437	acc=0.8995
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 156ms/step - accuracy: 0.9032 - loss: 0.24252026-01-19 01:04:08		[Task_set_12]	Batch 400	loss=0.2442	acc=0.8991
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 157ms/step - accuracy: 0.9028 - loss: 0.24272026-01-19 01:04:16		[Task_set_12]	Batch 450	loss=0.2426	acc=0.8996
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 157ms/step - accuracy: 0.9024 - loss: 0.24262026-01-19 01:04:24		[Task_set_12]	Batch 500	loss=0.2426	acc=0.8990
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:48 157ms/step - accuracy: 0.9021 - loss: 0.24262026-01-19 01:04:32		[Task_set_12]	Batch 550	loss=0.2423	acc=0.8986
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:41 157ms/step - accuracy: 0.9018 - loss: 0.24262026-01-19 01:04:40		[Task_set_12]	Batch 600	loss=0.2414	acc=0.8994
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 157ms/step - accuracy: 0.9017 - loss: 0.24252026-01-19 01:04:48		[Task_set_12]	Batch 650	loss=0.2421	acc=0.8990
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 157ms/step - accuracy: 0.9015 - loss: 0.24232026-01-19 01:04:55		[Task_set_12]	Batch 700	loss=0.2401	acc=0.8994
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 157ms/step - accuracy: 0.9014 - loss: 0.24212026-01-19 01:05:03		[Task_set_12]	Batch 750	loss=0.2390	acc=0.9006
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 157ms/step - accuracy: 0.9014 - loss: 0.24192026-01-19 01:05:11		[Task_set_12]	Batch 800	loss=0.2384	acc=0.9006
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 157ms/step - accuracy: 0.9013 - loss: 0.24172026-01-19 01:05:19		[Task_set_12]	Batch 850	loss=0.2372	acc=0.9010
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 54s 157ms/step - accuracy: 0.9013 - loss: 0.24142026-01-19 01:05:27		[Task_set_12]	Batch 900	loss=0.2361	acc=0.9017
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 157ms/step - accuracy: 0.9014 - loss: 0.24112026-01-19 01:05:35		[Task_set_12]	Batch 950	loss=0.2370	acc=0.9013
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 157ms/step - accuracy: 0.9014 - loss: 0.24092026-01-19 01:05:43		[Task_set_12]	Batch 1000	loss=0.2365	acc=0.9011
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 157ms/step - accuracy: 0.9013 - loss: 0.24072026-01-19 01:05:50		[Task_set_12]	Batch 1050	loss=0.2383	acc=0.9007
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.9013 - loss: 0.24052026-01-19 01:05:58		[Task_set_12]	Batch 1100	loss=0.2372	acc=0.9012
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.9013 - loss: 0.24032026-01-19 01:06:06		[Task_set_12]	Batch 1150	loss=0.2358	acc=0.9015
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.9013 - loss: 0.24022026-01-19 01:06:14		[Task_set_12]	Batch 1200	loss=0.2356	acc=0.9014
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 213s 171ms/step - accuracy: 0.9013 - loss: 0.2400 - val_accuracy: 0.8997 - val_loss: 0.2516
Epoch 7/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:06 156ms/step - accuracy: 0.9173 - loss: 0.19762026-01-19 01:06:46		[Task_set_12]	Batch 50	loss=0.1864	acc=0.9300
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:58 156ms/step - accuracy: 0.9235 - loss: 0.18962026-01-19 01:06:54		[Task_set_12]	Batch 100	loss=0.1818	acc=0.9281
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:51 156ms/step - accuracy: 0.9247 - loss: 0.18712026-01-19 01:07:02		[Task_set_12]	Batch 150	loss=0.1873	acc=0.9246
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 156ms/step - accuracy: 0.9252 - loss: 0.18622026-01-19 01:07:10		[Task_set_12]	Batch 200	loss=0.1800	acc=0.9275
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:35 156ms/step - accuracy: 0.9256 - loss: 0.18552026-01-19 01:07:18		[Task_set_12]	Batch 250	loss=0.1901	acc=0.9243
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:27 156ms/step - accuracy: 0.9252 - loss: 0.18662026-01-19 01:07:26		[Task_set_12]	Batch 300	loss=0.1938	acc=0.9229
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:19 156ms/step - accuracy: 0.9250 - loss: 0.18742026-01-19 01:07:33		[Task_set_12]	Batch 350	loss=0.1907	acc=0.9245
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 156ms/step - accuracy: 0.9248 - loss: 0.18792026-01-19 01:07:41		[Task_set_12]	Batch 400	loss=0.1921	acc=0.9223
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 156ms/step - accuracy: 0.9245 - loss: 0.18832026-01-19 01:07:49		[Task_set_12]	Batch 450	loss=0.1916	acc=0.9222
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 156ms/step - accuracy: 0.9242 - loss: 0.18872026-01-19 01:07:57		[Task_set_12]	Batch 500	loss=0.1923	acc=0.9208
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:48 156ms/step - accuracy: 0.9240 - loss: 0.18882026-01-19 01:08:05		[Task_set_12]	Batch 550	loss=0.1894	acc=0.9228
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:40 156ms/step - accuracy: 0.9239 - loss: 0.18892026-01-19 01:08:13		[Task_set_12]	Batch 600	loss=0.1895	acc=0.9231
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 156ms/step - accuracy: 0.9238 - loss: 0.18912026-01-19 01:08:20		[Task_set_12]	Batch 650	loss=0.1920	acc=0.9215
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 157ms/step - accuracy: 0.9236 - loss: 0.18922026-01-19 01:08:28		[Task_set_12]	Batch 700	loss=0.1898	acc=0.9221
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 157ms/step - accuracy: 0.9235 - loss: 0.18922026-01-19 01:08:36		[Task_set_12]	Batch 750	loss=0.1902	acc=0.9218
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 157ms/step - accuracy: 0.9234 - loss: 0.18932026-01-19 01:08:44		[Task_set_12]	Batch 800	loss=0.1909	acc=0.9221
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 157ms/step - accuracy: 0.9234 - loss: 0.18942026-01-19 01:08:52		[Task_set_12]	Batch 850	loss=0.1902	acc=0.9221
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 54s 157ms/step - accuracy: 0.9233 - loss: 0.18942026-01-19 01:09:00		[Task_set_12]	Batch 900	loss=0.1888	acc=0.9233
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 157ms/step - accuracy: 0.9233 - loss: 0.18932026-01-19 01:09:07		[Task_set_12]	Batch 950	loss=0.1897	acc=0.9229
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 157ms/step - accuracy: 0.9233 - loss: 0.18932026-01-19 01:09:15		[Task_set_12]	Batch 1000	loss=0.1893	acc=0.9234
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 157ms/step - accuracy: 0.9233 - loss: 0.18932026-01-19 01:09:23		[Task_set_12]	Batch 1050	loss=0.1883	acc=0.9238
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.9234 - loss: 0.18922026-01-19 01:09:31		[Task_set_12]	Batch 1100	loss=0.1870	acc=0.9243
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.9234 - loss: 0.18912026-01-19 01:09:39		[Task_set_12]	Batch 1150	loss=0.1870	acc=0.9246
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.9235 - loss: 0.18912026-01-19 01:09:47		[Task_set_12]	Batch 1200	loss=0.1866	acc=0.9247
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 213s 171ms/step - accuracy: 0.9235 - loss: 0.1890 - val_accuracy: 0.9131 - val_loss: 0.2281
Epoch 8/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:07 157ms/step - accuracy: 0.9344 - loss: 0.14282026-01-19 01:10:19		[Task_set_12]	Batch 50	loss=0.1460	acc=0.9400
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:59 156ms/step - accuracy: 0.9360 - loss: 0.14632026-01-19 01:10:27		[Task_set_12]	Batch 100	loss=0.1458	acc=0.9387
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:51 157ms/step - accuracy: 0.9364 - loss: 0.14792026-01-19 01:10:35		[Task_set_12]	Batch 150	loss=0.1578	acc=0.9346
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 157ms/step - accuracy: 0.9362 - loss: 0.15052026-01-19 01:10:43		[Task_set_12]	Batch 200	loss=0.1590	acc=0.9366
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:35 157ms/step - accuracy: 0.9362 - loss: 0.15162026-01-19 01:10:50		[Task_set_12]	Batch 250	loss=0.1529	acc=0.9373
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:27 157ms/step - accuracy: 0.9362 - loss: 0.15222026-01-19 01:10:58		[Task_set_12]	Batch 300	loss=0.1568	acc=0.9344
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:20 157ms/step - accuracy: 0.9358 - loss: 0.15312026-01-19 01:11:06		[Task_set_12]	Batch 350	loss=0.1564	acc=0.9350
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 157ms/step - accuracy: 0.9357 - loss: 0.15362026-01-19 01:11:14		[Task_set_12]	Batch 400	loss=0.1576	acc=0.9345
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 157ms/step - accuracy: 0.9355 - loss: 0.15412026-01-19 01:11:22		[Task_set_12]	Batch 450	loss=0.1579	acc=0.9342
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 157ms/step - accuracy: 0.9355 - loss: 0.15442026-01-19 01:11:29		[Task_set_12]	Batch 500	loss=0.1572	acc=0.9353
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:48 156ms/step - accuracy: 0.9355 - loss: 0.15452026-01-19 01:11:37		[Task_set_12]	Batch 550	loss=0.1561	acc=0.9357
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:40 157ms/step - accuracy: 0.9356 - loss: 0.15452026-01-19 01:11:45		[Task_set_12]	Batch 600	loss=0.1539	acc=0.9367
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 157ms/step - accuracy: 0.9356 - loss: 0.15462026-01-19 01:11:53		[Task_set_12]	Batch 650	loss=0.1573	acc=0.9353
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 156ms/step - accuracy: 0.9356 - loss: 0.15472026-01-19 01:12:01		[Task_set_12]	Batch 700	loss=0.1548	acc=0.9364
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 157ms/step - accuracy: 0.9357 - loss: 0.15472026-01-19 01:12:09		[Task_set_12]	Batch 750	loss=0.1548	acc=0.9367
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 157ms/step - accuracy: 0.9357 - loss: 0.15472026-01-19 01:12:16		[Task_set_12]	Batch 800	loss=0.1562	acc=0.9362
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 157ms/step - accuracy: 0.9358 - loss: 0.15482026-01-19 01:12:24		[Task_set_12]	Batch 850	loss=0.1554	acc=0.9365
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 54s 157ms/step - accuracy: 0.9358 - loss: 0.15482026-01-19 01:12:32		[Task_set_12]	Batch 900	loss=0.1563	acc=0.9360
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 157ms/step - accuracy: 0.9358 - loss: 0.15502026-01-19 01:12:40		[Task_set_12]	Batch 950	loss=0.1571	acc=0.9354
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 157ms/step - accuracy: 0.9358 - loss: 0.15502026-01-19 01:12:48		[Task_set_12]	Batch 1000	loss=0.1560	acc=0.9360
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 157ms/step - accuracy: 0.9358 - loss: 0.15512026-01-19 01:12:56		[Task_set_12]	Batch 1050	loss=0.1557	acc=0.9363
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.9358 - loss: 0.15512026-01-19 01:13:03		[Task_set_12]	Batch 1100	loss=0.1561	acc=0.9360
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.9358 - loss: 0.15512026-01-19 01:13:11		[Task_set_12]	Batch 1150	loss=0.1549	acc=0.9365
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.9358 - loss: 0.15512026-01-19 01:13:19		[Task_set_12]	Batch 1200	loss=0.1556	acc=0.9366
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 212s 171ms/step - accuracy: 0.9359 - loss: 0.1551 - val_accuracy: 0.9211 - val_loss: 0.2252
Epoch 9/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:06 156ms/step - accuracy: 0.9455 - loss: 0.15462026-01-19 01:13:51		[Task_set_12]	Batch 50	loss=0.1264	acc=0.9475
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:59 157ms/step - accuracy: 0.9476 - loss: 0.13872026-01-19 01:13:59		[Task_set_12]	Batch 100	loss=0.1200	acc=0.9506
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:51 156ms/step - accuracy: 0.9479 - loss: 0.13332026-01-19 01:14:07		[Task_set_12]	Batch 150	loss=0.1258	acc=0.9471
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 156ms/step - accuracy: 0.9473 - loss: 0.13212026-01-19 01:14:15		[Task_set_12]	Batch 200	loss=0.1314	acc=0.9441
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:35 157ms/step - accuracy: 0.9467 - loss: 0.13152026-01-19 01:14:23		[Task_set_12]	Batch 250	loss=0.1266	acc=0.9453
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:27 157ms/step - accuracy: 0.9461 - loss: 0.13142026-01-19 01:14:31		[Task_set_12]	Batch 300	loss=0.1343	acc=0.9417
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:20 157ms/step - accuracy: 0.9455 - loss: 0.13182026-01-19 01:14:38		[Task_set_12]	Batch 350	loss=0.1342	acc=0.9421
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 157ms/step - accuracy: 0.9451 - loss: 0.13212026-01-19 01:14:46		[Task_set_12]	Batch 400	loss=0.1335	acc=0.9420
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 157ms/step - accuracy: 0.9447 - loss: 0.13222026-01-19 01:14:54		[Task_set_12]	Batch 450	loss=0.1324	acc=0.9425
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 157ms/step - accuracy: 0.9445 - loss: 0.13212026-01-19 01:15:02		[Task_set_12]	Batch 500	loss=0.1322	acc=0.9423
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:48 157ms/step - accuracy: 0.9444 - loss: 0.13212026-01-19 01:15:10		[Task_set_12]	Batch 550	loss=0.1302	acc=0.9440
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:40 157ms/step - accuracy: 0.9444 - loss: 0.13192026-01-19 01:15:18		[Task_set_12]	Batch 600	loss=0.1296	acc=0.9447
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 157ms/step - accuracy: 0.9444 - loss: 0.13182026-01-19 01:15:26		[Task_set_12]	Batch 650	loss=0.1313	acc=0.9440
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 157ms/step - accuracy: 0.9444 - loss: 0.13172026-01-19 01:15:33		[Task_set_12]	Batch 700	loss=0.1296	acc=0.9450
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 157ms/step - accuracy: 0.9444 - loss: 0.13162026-01-19 01:15:41		[Task_set_12]	Batch 750	loss=0.1321	acc=0.9443
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 157ms/step - accuracy: 0.9444 - loss: 0.13162026-01-19 01:15:49		[Task_set_12]	Batch 800	loss=0.1316	acc=0.9447
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 157ms/step - accuracy: 0.9445 - loss: 0.13162026-01-19 01:15:57		[Task_set_12]	Batch 850	loss=0.1292	acc=0.9459
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 54s 157ms/step - accuracy: 0.9446 - loss: 0.13142026-01-19 01:16:05		[Task_set_12]	Batch 900	loss=0.1284	acc=0.9464
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 157ms/step - accuracy: 0.9447 - loss: 0.13132026-01-19 01:16:13		[Task_set_12]	Batch 950	loss=0.1286	acc=0.9466
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 157ms/step - accuracy: 0.9448 - loss: 0.13112026-01-19 01:16:20		[Task_set_12]	Batch 1000	loss=0.1294	acc=0.9468
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 157ms/step - accuracy: 0.9449 - loss: 0.13102026-01-19 01:16:28		[Task_set_12]	Batch 1050	loss=0.1288	acc=0.9475
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.9450 - loss: 0.13092026-01-19 01:16:36		[Task_set_12]	Batch 1100	loss=0.1266	acc=0.9484
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.9452 - loss: 0.13072026-01-19 01:16:44		[Task_set_12]	Batch 1150	loss=0.1265	acc=0.9485
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.9453 - loss: 0.13052026-01-19 01:16:52		[Task_set_12]	Batch 1200	loss=0.1269	acc=0.9484
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 213s 171ms/step - accuracy: 0.9454 - loss: 0.1304 - val_accuracy: 0.9243 - val_loss: 0.2057
Epoch 10/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:07 157ms/step - accuracy: 0.9667 - loss: 0.09922026-01-19 01:17:24		[Task_set_12]	Batch 50	loss=0.0933	acc=0.9638
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:59 157ms/step - accuracy: 0.9674 - loss: 0.09342026-01-19 01:17:32		[Task_set_12]	Batch 100	loss=0.0837	acc=0.9706
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:51 157ms/step - accuracy: 0.9676 - loss: 0.09232026-01-19 01:17:40		[Task_set_12]	Batch 150	loss=0.0944	acc=0.9654
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 157ms/step - accuracy: 0.9670 - loss: 0.09362026-01-19 01:17:48		[Task_set_12]	Batch 200	loss=0.0985	acc=0.9641
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:35 157ms/step - accuracy: 0.9664 - loss: 0.09452026-01-19 01:17:56		[Task_set_12]	Batch 250	loss=0.0966	acc=0.9645
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:28 157ms/step - accuracy: 0.9657 - loss: 0.09572026-01-19 01:18:04		[Task_set_12]	Batch 300	loss=0.1054	acc=0.9604
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:20 157ms/step - accuracy: 0.9651 - loss: 0.09692026-01-19 01:18:11		[Task_set_12]	Batch 350	loss=0.1023	acc=0.9618
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 157ms/step - accuracy: 0.9646 - loss: 0.09762026-01-19 01:18:19		[Task_set_12]	Batch 400	loss=0.1032	acc=0.9605
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 157ms/step - accuracy: 0.9639 - loss: 0.09842026-01-19 01:18:27		[Task_set_12]	Batch 450	loss=0.1043	acc=0.9589
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 157ms/step - accuracy: 0.9634 - loss: 0.09912026-01-19 01:18:35		[Task_set_12]	Batch 500	loss=0.1056	acc=0.9580
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:48 157ms/step - accuracy: 0.9629 - loss: 0.09962026-01-19 01:18:43		[Task_set_12]	Batch 550	loss=0.1051	acc=0.9573
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:41 157ms/step - accuracy: 0.9624 - loss: 0.10012026-01-19 01:18:51		[Task_set_12]	Batch 600	loss=0.1058	acc=0.9571
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 157ms/step - accuracy: 0.9620 - loss: 0.10052026-01-19 01:18:58		[Task_set_12]	Batch 650	loss=0.1060	acc=0.9572
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 157ms/step - accuracy: 0.9617 - loss: 0.10092026-01-19 01:19:06		[Task_set_12]	Batch 700	loss=0.1045	acc=0.9580
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 157ms/step - accuracy: 0.9614 - loss: 0.10112026-01-19 01:19:14		[Task_set_12]	Batch 750	loss=0.1049	acc=0.9579
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 157ms/step - accuracy: 0.9612 - loss: 0.10142026-01-19 01:19:22		[Task_set_12]	Batch 800	loss=0.1056	acc=0.9570
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 157ms/step - accuracy: 0.9610 - loss: 0.10162026-01-19 01:19:30		[Task_set_12]	Batch 850	loss=0.1057	acc=0.9569
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 54s 157ms/step - accuracy: 0.9607 - loss: 0.10182026-01-19 01:19:38		[Task_set_12]	Batch 900	loss=0.1047	acc=0.9571
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 157ms/step - accuracy: 0.9606 - loss: 0.10192026-01-19 01:19:45		[Task_set_12]	Batch 950	loss=0.1046	acc=0.9576
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 157ms/step - accuracy: 0.9604 - loss: 0.10212026-01-19 01:19:53		[Task_set_12]	Batch 1000	loss=0.1059	acc=0.9569
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 157ms/step - accuracy: 0.9602 - loss: 0.10232026-01-19 01:20:01		[Task_set_12]	Batch 1050	loss=0.1071	acc=0.9564
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.9601 - loss: 0.10252026-01-19 01:20:09		[Task_set_12]	Batch 1100	loss=0.1054	acc=0.9572
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.9600 - loss: 0.10262026-01-19 01:20:17		[Task_set_12]	Batch 1150	loss=0.1042	acc=0.9578
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.9599 - loss: 0.10272026-01-19 01:20:25		[Task_set_12]	Batch 1200	loss=0.1046	acc=0.9575
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 212s 171ms/step - accuracy: 0.9598 - loss: 0.1028 - val_accuracy: 0.9249 - val_loss: 0.2448
Epoch 11/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:06 156ms/step - accuracy: 0.9757 - loss: 0.07222026-01-19 01:21:35		[Task_set_12]	Batch 50	loss=0.0855	acc=0.9737
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:59 156ms/step - accuracy: 0.9737 - loss: 0.07782026-01-19 01:21:43		[Task_set_12]	Batch 100	loss=0.0824	acc=0.9719
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:52 157ms/step - accuracy: 0.9720 - loss: 0.08072026-01-19 01:21:51		[Task_set_12]	Batch 150	loss=0.0945	acc=0.9633
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:45 158ms/step - accuracy: 0.9702 - loss: 0.08402026-01-19 01:21:59		[Task_set_12]	Batch 200	loss=0.0911	acc=0.9666
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:37 159ms/step - accuracy: 0.9698 - loss: 0.08462026-01-19 01:22:07		[Task_set_12]	Batch 250	loss=0.0848	acc=0.9690
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:29 158ms/step - accuracy: 0.9695 - loss: 0.08492026-01-19 01:22:15		[Task_set_12]	Batch 300	loss=0.0888	acc=0.9665
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:21 158ms/step - accuracy: 0.9690 - loss: 0.08572026-01-19 01:22:23		[Task_set_12]	Batch 350	loss=0.0897	acc=0.9663
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:13 158ms/step - accuracy: 0.9687 - loss: 0.08602026-01-19 01:22:31		[Task_set_12]	Batch 400	loss=0.0869	acc=0.9666
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:05 158ms/step - accuracy: 0.9685 - loss: 0.08612026-01-19 01:22:39		[Task_set_12]	Batch 450	loss=0.0870	acc=0.9672
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:57 158ms/step - accuracy: 0.9684 - loss: 0.08612026-01-19 01:22:46		[Task_set_12]	Batch 500	loss=0.0864	acc=0.9679
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:49 157ms/step - accuracy: 0.9684 - loss: 0.08612026-01-19 01:22:54		[Task_set_12]	Batch 550	loss=0.0852	acc=0.9683
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:41 157ms/step - accuracy: 0.9684 - loss: 0.08592026-01-19 01:23:02		[Task_set_12]	Batch 600	loss=0.0826	acc=0.9687
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 157ms/step - accuracy: 0.9684 - loss: 0.08572026-01-19 01:23:10		[Task_set_12]	Batch 650	loss=0.0841	acc=0.9684
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 157ms/step - accuracy: 0.9684 - loss: 0.08572026-01-19 01:23:18		[Task_set_12]	Batch 700	loss=0.0855	acc=0.9679
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 157ms/step - accuracy: 0.9683 - loss: 0.08572026-01-19 01:23:26		[Task_set_12]	Batch 750	loss=0.0859	acc=0.9681
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:10 157ms/step - accuracy: 0.9683 - loss: 0.08572026-01-19 01:23:33		[Task_set_12]	Batch 800	loss=0.0863	acc=0.9677
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:02 157ms/step - accuracy: 0.9683 - loss: 0.08582026-01-19 01:23:41		[Task_set_12]	Batch 850	loss=0.0860	acc=0.9679
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 54s 157ms/step - accuracy: 0.9682 - loss: 0.08582026-01-19 01:23:49		[Task_set_12]	Batch 900	loss=0.0862	acc=0.9675
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 157ms/step - accuracy: 0.9682 - loss: 0.08582026-01-19 01:23:57		[Task_set_12]	Batch 950	loss=0.0867	acc=0.9674
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 157ms/step - accuracy: 0.9682 - loss: 0.08592026-01-19 01:24:05		[Task_set_12]	Batch 1000	loss=0.0882	acc=0.9666
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 157ms/step - accuracy: 0.9681 - loss: 0.08602026-01-19 01:24:13		[Task_set_12]	Batch 1050	loss=0.0884	acc=0.9663
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.9680 - loss: 0.08612026-01-19 01:24:20		[Task_set_12]	Batch 1100	loss=0.0885	acc=0.9664
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.9679 - loss: 0.08622026-01-19 01:24:28		[Task_set_12]	Batch 1150	loss=0.0883	acc=0.9665
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.9679 - loss: 0.08622026-01-19 01:24:36		[Task_set_12]	Batch 1200	loss=0.0881	acc=0.9665
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 252s 171ms/step - accuracy: 0.9678 - loss: 0.0863 - val_accuracy: 0.9326 - val_loss: 0.2008
Epoch 12/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:06 156ms/step - accuracy: 0.9762 - loss: 0.06152026-01-19 01:25:08		[Task_set_12]	Batch 50	loss=0.0591	acc=0.9800
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:59 156ms/step - accuracy: 0.9791 - loss: 0.05602026-01-19 01:25:16		[Task_set_12]	Batch 100	loss=0.0449	acc=0.9831
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:51 156ms/step - accuracy: 0.9790 - loss: 0.05582026-01-19 01:25:24		[Task_set_12]	Batch 150	loss=0.0624	acc=0.9762
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 156ms/step - accuracy: 0.9782 - loss: 0.05792026-01-19 01:25:32		[Task_set_12]	Batch 200	loss=0.0653	acc=0.9750
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:35 156ms/step - accuracy: 0.9776 - loss: 0.05912026-01-19 01:25:40		[Task_set_12]	Batch 250	loss=0.0619	acc=0.9758
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:27 156ms/step - accuracy: 0.9772 - loss: 0.05982026-01-19 01:25:48		[Task_set_12]	Batch 300	loss=0.0705	acc=0.9719
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:20 156ms/step - accuracy: 0.9762 - loss: 0.06182026-01-19 01:25:55		[Task_set_12]	Batch 350	loss=0.0755	acc=0.9696
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 156ms/step - accuracy: 0.9754 - loss: 0.06352026-01-19 01:26:03		[Task_set_12]	Batch 400	loss=0.0739	acc=0.9706
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 156ms/step - accuracy: 0.9749 - loss: 0.06462026-01-19 01:26:11		[Task_set_12]	Batch 450	loss=0.0732	acc=0.9708
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 156ms/step - accuracy: 0.9745 - loss: 0.06542026-01-19 01:26:19		[Task_set_12]	Batch 500	loss=0.0729	acc=0.9714
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:48 157ms/step - accuracy: 0.9742 - loss: 0.06612026-01-19 01:26:27		[Task_set_12]	Batch 550	loss=0.0718	acc=0.9719
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:40 156ms/step - accuracy: 0.9741 - loss: 0.06652026-01-19 01:26:35		[Task_set_12]	Batch 600	loss=0.0736	acc=0.9718
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 157ms/step - accuracy: 0.9739 - loss: 0.06712026-01-19 01:26:42		[Task_set_12]	Batch 650	loss=0.0733	acc=0.9721
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 157ms/step - accuracy: 0.9738 - loss: 0.06752026-01-19 01:26:50		[Task_set_12]	Batch 700	loss=0.0723	acc=0.9724
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 157ms/step - accuracy: 0.9737 - loss: 0.06782026-01-19 01:26:58		[Task_set_12]	Batch 750	loss=0.0734	acc=0.9721
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 157ms/step - accuracy: 0.9736 - loss: 0.06822026-01-19 01:27:06		[Task_set_12]	Batch 800	loss=0.0740	acc=0.9718
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 157ms/step - accuracy: 0.9735 - loss: 0.06852026-01-19 01:27:14		[Task_set_12]	Batch 850	loss=0.0732	acc=0.9721
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 54s 157ms/step - accuracy: 0.9734 - loss: 0.06882026-01-19 01:27:22		[Task_set_12]	Batch 900	loss=0.0733	acc=0.9719
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 157ms/step - accuracy: 0.9733 - loss: 0.06902026-01-19 01:27:30		[Task_set_12]	Batch 950	loss=0.0730	acc=0.9718
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 157ms/step - accuracy: 0.9732 - loss: 0.06922026-01-19 01:27:37		[Task_set_12]	Batch 1000	loss=0.0751	acc=0.9709
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 157ms/step - accuracy: 0.9731 - loss: 0.06952026-01-19 01:27:45		[Task_set_12]	Batch 1050	loss=0.0755	acc=0.9706
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.9730 - loss: 0.06982026-01-19 01:27:53		[Task_set_12]	Batch 1100	loss=0.0753	acc=0.9706
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.9729 - loss: 0.07002026-01-19 01:28:01		[Task_set_12]	Batch 1150	loss=0.0752	acc=0.9708
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.9728 - loss: 0.07032026-01-19 01:28:09		[Task_set_12]	Batch 1200	loss=0.0751	acc=0.9710
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 213s 171ms/step - accuracy: 0.9727 - loss: 0.0705 - val_accuracy: 0.9380 - val_loss: 0.1960
Epoch 13/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:07 157ms/step - accuracy: 0.9867 - loss: 0.03002026-01-19 01:28:41		[Task_set_12]	Batch 50	loss=0.0269	acc=0.9900
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:58 156ms/step - accuracy: 0.9855 - loss: 0.03452026-01-19 01:28:49		[Task_set_12]	Batch 100	loss=0.0476	acc=0.9806
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:51 156ms/step - accuracy: 0.9836 - loss: 0.03922026-01-19 01:28:57		[Task_set_12]	Batch 150	loss=0.0499	acc=0.9792
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 156ms/step - accuracy: 0.9824 - loss: 0.04222026-01-19 01:29:05		[Task_set_12]	Batch 200	loss=0.0504	acc=0.9787
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:35 156ms/step - accuracy: 0.9818 - loss: 0.04372026-01-19 01:29:12		[Task_set_12]	Batch 250	loss=0.0504	acc=0.9795
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:27 156ms/step - accuracy: 0.9813 - loss: 0.04502026-01-19 01:29:20		[Task_set_12]	Batch 300	loss=0.0540	acc=0.9783
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:19 156ms/step - accuracy: 0.9807 - loss: 0.04682026-01-19 01:29:28		[Task_set_12]	Batch 350	loss=0.0587	acc=0.9768
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 156ms/step - accuracy: 0.9802 - loss: 0.04842026-01-19 01:29:36		[Task_set_12]	Batch 400	loss=0.0610	acc=0.9762
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 156ms/step - accuracy: 0.9797 - loss: 0.04992026-01-19 01:29:44		[Task_set_12]	Batch 450	loss=0.0606	acc=0.9764
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 156ms/step - accuracy: 0.9794 - loss: 0.05102026-01-19 01:29:52		[Task_set_12]	Batch 500	loss=0.0623	acc=0.9763
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:48 157ms/step - accuracy: 0.9791 - loss: 0.05202026-01-19 01:30:00		[Task_set_12]	Batch 550	loss=0.0616	acc=0.9769
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:41 157ms/step - accuracy: 0.9790 - loss: 0.05282026-01-19 01:30:07		[Task_set_12]	Batch 600	loss=0.0618	acc=0.9770
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 157ms/step - accuracy: 0.9788 - loss: 0.05362026-01-19 01:30:15		[Task_set_12]	Batch 650	loss=0.0641	acc=0.9757
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 157ms/step - accuracy: 0.9785 - loss: 0.05442026-01-19 01:30:23		[Task_set_12]	Batch 700	loss=0.0657	acc=0.9748
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 157ms/step - accuracy: 0.9783 - loss: 0.05522026-01-19 01:30:31		[Task_set_12]	Batch 750	loss=0.0662	acc=0.9743
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 157ms/step - accuracy: 0.9780 - loss: 0.05592026-01-19 01:30:39		[Task_set_12]	Batch 800	loss=0.0655	acc=0.9750
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 157ms/step - accuracy: 0.9779 - loss: 0.05642026-01-19 01:30:47		[Task_set_12]	Batch 850	loss=0.0637	acc=0.9760
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 54s 157ms/step - accuracy: 0.9778 - loss: 0.05672026-01-19 01:30:54		[Task_set_12]	Batch 900	loss=0.0629	acc=0.9766
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 157ms/step - accuracy: 0.9777 - loss: 0.05712026-01-19 01:31:02		[Task_set_12]	Batch 950	loss=0.0633	acc=0.9766
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 157ms/step - accuracy: 0.9777 - loss: 0.05742026-01-19 01:31:10		[Task_set_12]	Batch 1000	loss=0.0631	acc=0.9765
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 157ms/step - accuracy: 0.9776 - loss: 0.05772026-01-19 01:31:18		[Task_set_12]	Batch 1050	loss=0.0643	acc=0.9758
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.9775 - loss: 0.05802026-01-19 01:31:26		[Task_set_12]	Batch 1100	loss=0.0632	acc=0.9763
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.9775 - loss: 0.05822026-01-19 01:31:34		[Task_set_12]	Batch 1150	loss=0.0625	acc=0.9765
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.9774 - loss: 0.05842026-01-19 01:31:41		[Task_set_12]	Batch 1200	loss=0.0633	acc=0.9765
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 212s 171ms/step - accuracy: 0.9774 - loss: 0.0585 - val_accuracy: 0.9296 - val_loss: 0.2357
Epoch 14/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:08 157ms/step - accuracy: 0.9890 - loss: 0.04552026-01-19 01:32:14		[Task_set_12]	Batch 50	loss=0.0502	acc=0.9850
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:59 157ms/step - accuracy: 0.9864 - loss: 0.04962026-01-19 01:32:21		[Task_set_12]	Batch 100	loss=0.0498	acc=0.9856
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:51 157ms/step - accuracy: 0.9854 - loss: 0.05022026-01-19 01:32:29		[Task_set_12]	Batch 150	loss=0.0530	acc=0.9821
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 157ms/step - accuracy: 0.9846 - loss: 0.05092026-01-19 01:32:37		[Task_set_12]	Batch 200	loss=0.0517	acc=0.9822
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:35 157ms/step - accuracy: 0.9843 - loss: 0.05082026-01-19 01:32:45		[Task_set_12]	Batch 250	loss=0.0494	acc=0.9833
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:27 157ms/step - accuracy: 0.9841 - loss: 0.05042026-01-19 01:32:53		[Task_set_12]	Batch 300	loss=0.0477	acc=0.9831
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:20 157ms/step - accuracy: 0.9839 - loss: 0.05042026-01-19 01:33:01		[Task_set_12]	Batch 350	loss=0.0516	acc=0.9823
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 157ms/step - accuracy: 0.9836 - loss: 0.05072026-01-19 01:33:08		[Task_set_12]	Batch 400	loss=0.0532	acc=0.9806
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 157ms/step - accuracy: 0.9832 - loss: 0.05102026-01-19 01:33:16		[Task_set_12]	Batch 450	loss=0.0522	acc=0.9810
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 157ms/step - accuracy: 0.9830 - loss: 0.05112026-01-19 01:33:24		[Task_set_12]	Batch 500	loss=0.0526	acc=0.9810
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:48 157ms/step - accuracy: 0.9828 - loss: 0.05132026-01-19 01:33:32		[Task_set_12]	Batch 550	loss=0.0541	acc=0.9800
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:41 157ms/step - accuracy: 0.9825 - loss: 0.05162026-01-19 01:33:40		[Task_set_12]	Batch 600	loss=0.0548	acc=0.9798
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 157ms/step - accuracy: 0.9823 - loss: 0.05192026-01-19 01:33:48		[Task_set_12]	Batch 650	loss=0.0560	acc=0.9789
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 157ms/step - accuracy: 0.9820 - loss: 0.05222026-01-19 01:33:56		[Task_set_12]	Batch 700	loss=0.0564	acc=0.9785
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 157ms/step - accuracy: 0.9818 - loss: 0.05252026-01-19 01:34:03		[Task_set_12]	Batch 750	loss=0.0581	acc=0.9778
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 157ms/step - accuracy: 0.9816 - loss: 0.05282026-01-19 01:34:11		[Task_set_12]	Batch 800	loss=0.0572	acc=0.9784
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 157ms/step - accuracy: 0.9814 - loss: 0.05312026-01-19 01:34:19		[Task_set_12]	Batch 850	loss=0.0562	acc=0.9789
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 54s 157ms/step - accuracy: 0.9813 - loss: 0.05322026-01-19 01:34:27		[Task_set_12]	Batch 900	loss=0.0558	acc=0.9792
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 157ms/step - accuracy: 0.9812 - loss: 0.05342026-01-19 01:34:35		[Task_set_12]	Batch 950	loss=0.0561	acc=0.9791
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 157ms/step - accuracy: 0.9811 - loss: 0.05352026-01-19 01:34:43		[Task_set_12]	Batch 1000	loss=0.0557	acc=0.9794
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 157ms/step - accuracy: 0.9810 - loss: 0.05362026-01-19 01:34:50		[Task_set_12]	Batch 1050	loss=0.0548	acc=0.9799
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.9809 - loss: 0.05372026-01-19 01:34:58		[Task_set_12]	Batch 1100	loss=0.0554	acc=0.9795
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.9809 - loss: 0.05382026-01-19 01:35:06		[Task_set_12]	Batch 1150	loss=0.0554	acc=0.9795
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.9808 - loss: 0.05382026-01-19 01:35:14		[Task_set_12]	Batch 1200	loss=0.0553	acc=0.9797
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 213s 171ms/step - accuracy: 0.9808 - loss: 0.0539 - val_accuracy: 0.9314 - val_loss: 0.2280
Epoch 15/15
  49/1244 ━━━━━━━━━━━━━━━━━━━━ 3:06 156ms/step - accuracy: 0.9813 - loss: 0.05942026-01-19 01:35:46		[Task_set_12]	Batch 50	loss=0.0571	acc=0.9800
  99/1244 ━━━━━━━━━━━━━━━━━━━━ 2:59 157ms/step - accuracy: 0.9821 - loss: 0.05312026-01-19 01:35:54		[Task_set_12]	Batch 100	loss=0.0388	acc=0.9862
 149/1244 ━━━━━━━━━━━━━━━━━━━━ 2:51 157ms/step - accuracy: 0.9835 - loss: 0.04892026-01-19 01:36:02		[Task_set_12]	Batch 150	loss=0.0457	acc=0.9846
 199/1244 ━━━━━━━━━━━━━━━━━━━━ 2:43 157ms/step - accuracy: 0.9837 - loss: 0.04852026-01-19 01:36:10		[Task_set_12]	Batch 200	loss=0.0498	acc=0.9841
 249/1244 ━━━━━━━━━━━━━━━━━━━━ 2:36 157ms/step - accuracy: 0.9838 - loss: 0.04832026-01-19 01:36:17		[Task_set_12]	Batch 250	loss=0.0451	acc=0.9848
 299/1244 ━━━━━━━━━━━━━━━━━━━━ 2:28 157ms/step - accuracy: 0.9841 - loss: 0.04762026-01-19 01:36:25		[Task_set_12]	Batch 300	loss=0.0440	acc=0.9858
 349/1244 ━━━━━━━━━━━━━━━━━━━━ 2:20 157ms/step - accuracy: 0.9844 - loss: 0.04702026-01-19 01:36:33		[Task_set_12]	Batch 350	loss=0.0430	acc=0.9864
 399/1244 ━━━━━━━━━━━━━━━━━━━━ 2:12 157ms/step - accuracy: 0.9846 - loss: 0.04652026-01-19 01:36:41		[Task_set_12]	Batch 400	loss=0.0428	acc=0.9866
 449/1244 ━━━━━━━━━━━━━━━━━━━━ 2:04 157ms/step - accuracy: 0.9849 - loss: 0.04612026-01-19 01:36:49		[Task_set_12]	Batch 450	loss=0.0423	acc=0.9867
 499/1244 ━━━━━━━━━━━━━━━━━━━━ 1:56 157ms/step - accuracy: 0.9850 - loss: 0.04582026-01-19 01:36:57		[Task_set_12]	Batch 500	loss=0.0435	acc=0.9860
 549/1244 ━━━━━━━━━━━━━━━━━━━━ 1:49 157ms/step - accuracy: 0.9850 - loss: 0.04572026-01-19 01:37:05		[Task_set_12]	Batch 550	loss=0.0451	acc=0.9848
 599/1244 ━━━━━━━━━━━━━━━━━━━━ 1:41 157ms/step - accuracy: 0.9850 - loss: 0.04572026-01-19 01:37:12		[Task_set_12]	Batch 600	loss=0.0457	acc=0.9844
 649/1244 ━━━━━━━━━━━━━━━━━━━━ 1:33 157ms/step - accuracy: 0.9849 - loss: 0.04572026-01-19 01:37:20		[Task_set_12]	Batch 650	loss=0.0480	acc=0.9839
 699/1244 ━━━━━━━━━━━━━━━━━━━━ 1:25 157ms/step - accuracy: 0.9849 - loss: 0.04592026-01-19 01:37:28		[Task_set_12]	Batch 700	loss=0.0480	acc=0.9839
 749/1244 ━━━━━━━━━━━━━━━━━━━━ 1:17 157ms/step - accuracy: 0.9848 - loss: 0.04602026-01-19 01:37:36		[Task_set_12]	Batch 750	loss=0.0484	acc=0.9837
 799/1244 ━━━━━━━━━━━━━━━━━━━━ 1:09 157ms/step - accuracy: 0.9847 - loss: 0.04612026-01-19 01:37:44		[Task_set_12]	Batch 800	loss=0.0474	acc=0.9839
 849/1244 ━━━━━━━━━━━━━━━━━━━━ 1:01 157ms/step - accuracy: 0.9847 - loss: 0.04622026-01-19 01:37:52		[Task_set_12]	Batch 850	loss=0.0469	acc=0.9837
 899/1244 ━━━━━━━━━━━━━━━━━━━━ 54s 157ms/step - accuracy: 0.9846 - loss: 0.04622026-01-19 01:37:59		[Task_set_12]	Batch 900	loss=0.0464	acc=0.9839
 949/1244 ━━━━━━━━━━━━━━━━━━━━ 46s 157ms/step - accuracy: 0.9846 - loss: 0.04622026-01-19 01:38:07		[Task_set_12]	Batch 950	loss=0.0469	acc=0.9837
 999/1244 ━━━━━━━━━━━━━━━━━━━━ 38s 157ms/step - accuracy: 0.9846 - loss: 0.04632026-01-19 01:38:15		[Task_set_12]	Batch 1000	loss=0.0468	acc=0.9836
1049/1244 ━━━━━━━━━━━━━━━━━━━━ 30s 157ms/step - accuracy: 0.9845 - loss: 0.04632026-01-19 01:38:23		[Task_set_12]	Batch 1050	loss=0.0466	acc=0.9835
1099/1244 ━━━━━━━━━━━━━━━━━━━━ 22s 157ms/step - accuracy: 0.9845 - loss: 0.04632026-01-19 01:38:31		[Task_set_12]	Batch 1100	loss=0.0455	acc=0.9838
1149/1244 ━━━━━━━━━━━━━━━━━━━━ 14s 157ms/step - accuracy: 0.9844 - loss: 0.04622026-01-19 01:38:39		[Task_set_12]	Batch 1150	loss=0.0466	acc=0.9833
1199/1244 ━━━━━━━━━━━━━━━━━━━━ 7s 157ms/step - accuracy: 0.9844 - loss: 0.04632026-01-19 01:38:46		[Task_set_12]	Batch 1200	loss=0.0488	acc=0.9825
1244/1244 ━━━━━━━━━━━━━━━━━━━━ 213s 171ms/step - accuracy: 0.9843 - loss: 0.0464 - val_accuracy: 0.9412 - val_loss: 0.2109
2026-01-19 01:39:11		[Task_set_12]	######################################
2026-01-19 01:39:11		[Task_set_12]	#	HW3	Task2	Validation Metrics	#
2026-01-19 01:39:11		[Task_set_12]	######################################
2026-01-19 01:39:32		[Task_set_12]	Val Accuracy	0.9382
2026-01-19 01:39:32		[Task_set_12]	Precision	Cat	0.9376
2026-01-19 01:39:32		[Task_set_12]	Precision	Dog	0.9387
2026-01-19 01:39:32		[Task_set_12]	####################################
2026-01-19 01:39:32		[Task_set_12]	#	HW3	Task2	Plot Loss Curves	#
2026-01-19 01:39:32		[Task_set_12]	####################################
 
2026-01-19 01:39:32		[Task_set_12]	Loss curves plotted.
2026-01-19 01:39:32		[Task_set_12]	#######################################
2026-01-19 01:39:32		[Task_set_12]	#	HW3	Task2	Qualitative Results	#
2026-01-19 01:39:32		[Task_set_12]	#######################################
 
2026-01-19 01:39:34		[Task_set_12]	Qualitative samples displayed.
2026-01-19 01:39:34		[Task_set_12]	HW3 Task2 completed successfully.
2026-01-19 01:39:34		[Task_set_12]	Task2 metrics	{'val_accuracy': 0.9381898454746137, 'class_names': ['Cat', 'Dog'], 'precision_Cat': 0.9376273950264982, 'precision_Dog': 0.9387351778656127}
2026-01-19 01:39:34		[Task_set_12]	############################
2026-01-19 01:39:34		[Task_set_12]	#	HW3	MAIN	Completed	#
2026-01-19 01:39:34		[Task_set_12]	############################



"""
