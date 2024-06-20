# Bildsegmentierung mittles U-Net

Dieses Repository beinhaltet den Code zur Facharbeit.

Das Experiment wird aus Jupyter-Notebooks heraus gesteuert und die
Dependencies sind im Anaconda-Environment in der Datei conda_env.yaml enthalten
und können so importiert werden.

## Trainieren

Zum Trainieren muss lediglich die Step-Anzahl in `training.ipynb` angepasst werden.
Das Modell wird dann unter `./experiment/<step-anzahl>.hdf5` abgespeichert.

**Die für die Facharbeit trainierten Modelle sind leider nicht in diesem Repository enthalten, da die Dateien das Limit von 25mb überschreiten.**

**Die Modelldateien können unter dem folgenden [OneDrive-Link](https://elghalle-my.sharepoint.com/:f:/g/personal/samuel_moravec_ess-elisabeth_de/Ei7DWbRmoRBOgb5d8YrZRvoB4iXCtqEia-oSQDD53vjxLA?e=jhTRY1) heruntergeladen und in `./experiments` eingefügt werden um diese auszuwerten**
## Auswerten

Um ein trainiertes Modell auszuwerten, muss man die Code-Zellen von
`experiment.ipynb` ausführen. Die letzte ist optional und nicht notwendig
für die Auswertung, da sie nur die Overlays erstellt.

## Dependencies

Die hauptsächlich verwendeten Bibliotheken sind:
- Keras (wrapper für TensorFlow und andere Bibliotheken)
- numpy
- matplotlib
- scikit-image

## Beispiele

Nachfolgend ein originales Bild mit seiner vorhergesagten Maske sowie 
ein Kompositbild, in dem die Maske über die Mikroskopierung gelegt wurde.

![Mikroskopierung](./data/cells/test/0.png)
![Maske](./data/cells/test/0_predict.png)
![Mikroskopierung mit Overlay](./data/cells/test/0_overlay.png)

### Danksagung

Einige Code-Snippets wurden inspiriert vom folgenden Repository: 
[zhixuhao/unet](https://github.com/zhixuhao/unet)