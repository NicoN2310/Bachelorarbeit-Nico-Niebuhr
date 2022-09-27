# Bachelorarbeit-Nico-Niebuhr
Optimierung und Training eines Machine Learning Modells zur Vorhersage von Transportersubstraten anhand von Aminosäuresequenzen

Der Inhalt des Ordners "/create_train_test/" stammt von meinem Betreuer Alexander Kroll.
<br>
<br>
<br>
## Übersicht Skripte für Datenbank und Modell:

| **Datei**                                                                                  | **Inhalt**                                                                                               |
|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| /create_data/create_data.R                                                                 | Skript zur Erstellung einer Datenbank durch TCDB                                                         |
| /create_train_test/Preprocessing data-V3.ipynb                                             | Jupyter-Notebook von Alexander Kroll zur Erstellung <br>der Trainings- und Testdaten                     |
| /hyperparameter_und_training/hyperopt_ba.py<br>/hyperparameter_und_training/hyperopt_ba.sh | Python-Skript für die Hyperparameter-Optimierung, <br>gemeinsam mit einem Skript für das HILBERT der HHU |
| /hyperparameter_und_training/test.ipynb                                                    | Jupyter-Notebook zur Analyse der Leistung des Modells                                                    |
<br>

## Übersicht Dateien für Datenbank:

| **Datei**                                                                      | **Inhalt**                                                                                                |
|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| /create_data/original_data/                                                    | Original Dateien der TCDB                                                                                 |
| /create_data/component_data/                                                   | Dateien mit Teilinformationen, die für die einzelnen <br>Schritte der Datenbankerstellung erzeugt wurden  |
| /create_data/database.csv                                                      | Datenbank, bestehend aus TCDB und weiteren<br>Informationen; erstellt von "create_data.R"                 |
| /create_train_test/database_TCDB_and_GOA.csv                                   | Datenbank, nachdem TCDB "database.csv" mit GOA <br>zusammengefügt wurde (Alexander Kroll)                |
| /create_train_test/training_data_V3.pkl<br>/create_train_test/test_data_V3.pkl | Trainings- und Testdatensatz für XGBoost; erstellt<br>von "Preprocessing data-V3.ipynb" (Alexander Kroll) |
