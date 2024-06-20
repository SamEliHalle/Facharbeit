import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from skimage import transform
from skimage.color import gray2rgb
from keras.callbacks import Callback
from matplotlib import pyplot as plt
import os

class LogCallback(Callback):
    """
    Callback Klasse, die die Genauigkeit und den Verlust während des Trainings speichert und visualisiert.
    """
    accuracy = []
    loss = []
    
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.accuracy.append(logs["accuracy"])
        self.loss.append(logs["loss"])
        #print(f"Epoch end; acc: {logs['accuracy']}, loss: {logs['loss']}")
    
    def visualise(self, save_path=None):
        """
        Visualisiert die Genauigkeit und den Verlust über die Epochen.
        
        Args:
            save_path (string, optional): Pfad, an dem das Diagramm gespeichert werden soll. Defaults to None.
        """
        epochs = list(range(1, len(self.accuracy) + 1))
        plt.plot(epochs, self.accuracy, label="Accuracy")
        plt.plot(epochs, self.loss, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy/Loss")
        
        if save_path is not None:
            plt.savefig(save_path)
        
        plt.show()


def adjustData(img, mask):
    """Normalisiert die Bilddaten und die Masken für den Trainingsprozess.

    Args:
        img (np Array): Bild im numpy Array Format
        mask (np Array): Maske im numpy Array Format

    Returns:
        (np Array, np Array): Normalisiertes Bild und Maske
    """
    if (np.max(img) > 1): # normalisierung der Bilddaten, sollte es sich um RGB Daten handeln
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, target_dir = None, image_save_prefix="image", mask_save_prefix="mask", target_size=(256, 256), seed=1):
    """
    Gibt Bilder und Masken im passenden Format zurück, die in den gegebenen Pfaden liegen.
    Ist aug_dict nicht leer, werden die Bilder und Masken entsprechend der Werte in aug_dict augmentiert.

    Args:
        batch_size (int): Batch Größe
        train_path (string): Stammordner der Dateien
        image_folder (string): Ordnerpfad der Bilder, vom Stammordner aus
        mask_folder (string): Ordnerpfad der Masken, vom Stammordner aus
        aug_dict (dict): Dictionary mit den Augmentierungswerten
        target_dir (string, optional): Zielordner für die augmentierten Bilder und Masken; None, wenn die Bilder nicht gespeichert werden sollen. Defaults to None.
        image_save_prefix (str, optional): Präfix für das Abspeichern der Bilder. Defaults to "image".
        mask_save_prefix (str, optional): Präfix für das Abspeichern der Masken. Defaults to "mask".
        target_size (tuple, optional): Dimensionen der generierten Bilder (Breite, Höhe). Defaults to (256, 256).
        seed (int, optional): Seed. Defaults to 1.

    Yields:
        (np Array, np Array): Bild und dazugehörende Maske
    """
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=target_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=target_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    training_generator = zip(image_generator, mask_generator)
    for (img, mask) in training_generator:
        img, mask = adjustData(img, mask)
        yield (img, mask)


def normaliseImageNames(folder_path):
    for i, filename in enumerate(os.listdir(folder_path)):
        filename = filename.split(".")[0]
        filename = filename.split("_", 1)[1]
        os.rename(os.path.join(folder_path, f"image_{filename}.png"), os.path.join(folder_path, f"{i}.png"))
        os.rename(os.path.join(folder_path, f"mask_{filename}.png"), os.path.join(folder_path, f"{i}_mask.png"))

def clearFolder(folder_path):
    for filename in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, filename))


def normaliseImage(image, target_size=(256, 256)):
    img = image
    if (np.max(img) > 1):
        img = img / 255

    width, height = img.shape
    if (width < target_size[0] or height < target_size[1]):
        # sollte das bild zu klein sein, hochskalieren
        if (height < width):
            scale = target_size[1] / height
        else:
            scale = target_size[0] / width
        img = transform.rescale(img, scale)

    img = img[0:target_size[0], 0:target_size[1]]  # zuschneiden des Bildes
    # umformen der Bildarray in eine 4D Array von der Form (1, breite, höhe, 1)
    # die erste 1 steht für die Anzahl der Bilder, die zweite 1 steht für die Anzahl der Farbkanäle, die analysiert werden
    return img

def testGenerator(test_path, image_num=30, target_size=(256, 256), startIndex=0):
    """Generiert die Testdaten für das Modell anhand bestehender Bilder im angegeben Ordner.

    Args:
        test_path (string): Ordner, der die Testbilder enthält.
        image_num (int, optional): Anzahl der benötigten Testbilder. Defaults to 30.
        target_size (tuple, optional): Größe der Bilder. Sollten die Bilder kleiner sein, werden sie hochskaliert, sind sie größer, wird die jeweilige Region ausgeschnitten. Defaults to (256, 256).
        startIndex (int, optional): Startindex der Dateinamen (z.B. 0.png). Defaults to 0.

    Yields:
        np Array: Bild im passenden Format für das Modell
    """
    for j in range(image_num + startIndex):
        i = j + startIndex
        img = io.imread(os.path.join(test_path, f"{i}.png"), as_gray=True)
        
        img = normaliseImage(img, target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)

        yield img

def saveResult(save_path, npyfile, merge=True):
    """Speichert die generierten Masken als PNG Dateien ab.

    Args:
        save_path (string): Ordnerpfad, in dem die Masken gespeichert werden sollen.
        npyfile (np file): Array mit den generierten Masken.
        merge (bool, optional): Ob zusätzlich die Masken mit den ursprünglichen Bildern gemischt werden sollen. Defaults to True.
    """
    print("Saving results...")
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]

        io.imsave(os.path.join(save_path, f"{i}_predict.png"), (img * 255).astype(np.uint8))
    print("Results saved.")
    
    if merge:
        print("Merging images...")
        mergeImageMask(save_path=save_path, mask_suffix="_predict", merge_suffix="_overlay")
        print("Merging done.")


def alpha_blend(rgba, rgb):  # von Copilot
    """Überlagert eine RGBA Farbe auf eine RGB Farbe.
    Generiert mithilfe von GitHub Copilot.

    Args:
        rgba (float, float, float, float): Farbe im RGBA Format
        rgb (float, float, float): Farbe im RGB Format

    Returns:
        (float, float, float): Überlagerte Farbe im RGB Format
    """
    alpha = rgba[3] / 255.0
    blended = [alpha * rgba[i] + (1 - alpha) * rgb[i] for i in range(3)]
    return list(map(int, blended))


def mergeImageMask(image_suffix="", mask_suffix="_predict", merge_suffix="_overlay", save_path="data/cells/test"):
    """Blendet die generierten Masken in grüner Farbe in die Bilder ein und speichert das Ergebnis ab.

    Args:
        image_suffix (str, optional): Suffix der Bilddateien (ohne .png). Defaults to "".
        mask_suffix (str, optional): Suffix der Maskendateien (ohne .png). Defaults to "_predict".
        merge_suffix (str, optional): Suffix der Überlagerten Dateien (ohne .png). Defaults to "_overlay".
        save_path (str, optional): Ordnerpfad, in dem die Masken und Bilder liegen. Defaults to "data/cells/test".
    """
    for filename in os.listdir(save_path):
        if filename.endswith(mask_suffix + ".png") or filename.endswith(merge_suffix + ".png") or not filename.endswith(".png"):
            continue
    
        img = io.imread(os.path.join(save_path, filename), as_gray=False)
        img = gray2rgb(img)

        maskname = ""
        merged = ""
        if image_suffix != "":
            maskname = filename.replace(image_suffix, mask_suffix)
            merged = filename.replace(image_suffix, merge_suffix)
        else:
            maskname = filename.split(".")[0] + mask_suffix + ".png"
            merged = filename.split(".")[0] + merge_suffix + ".png"

        mask = io.imread(os.path.join(save_path, maskname))

        for i in range(0, len(mask)):
            for j in range(0, len(mask[i])):
                if mask[i][j] < 130:
                    alpha = 255 - mask[i][j]
                    green = (34, 139, 34, alpha)  # grün
                    img[i][j] = alpha_blend(green, img[i][j])

        io.imsave(os.path.join(save_path, merged), img)
