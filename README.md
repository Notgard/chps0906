# Rendu TP CHPS0906
Le travail des TP 1 et 2 se trouve dans le notebook `TP.ipynb`. La partie du TP2 sur les modèles *bottleneck* et *inverted bottleneck* se trouve dans les notebooks `bottleneck.ipynb` et `inverted_bottleneck.ipynb` respectivement.
  
Des dépots séparés pour ces derniers ont été faits et se trouvent [ici](https://github.com/Notgard/CNN_Bottleneck_Stacking) et [ici](https://github.com/Notgard/CNN_Inverted_Bottleneck_Stacking).
  
Les logs de l'entraînement des modèles sont enregistrés via Tensorboard et sont disponibles dans le répertoire `runs/`. De plus, durant l'étape de validation du modèle, une sauvegarde de ce modèle est enregistrée dans le répertoire `saved_models/`. Actuellement, un modèle de test est disponible pour l'architecture *bottleneck* et *inverted bottleneck* dans les répertoire `saved_models/bt` et `saved_models/ibt` respectivement.
# Installation

```bash
pip install -r requirements.txt
```
