
# Détection et classification de dommages automobiles — INF1022

Projet de synthèse en apprentissage profond : pipeline complet de détection de pièces automobiles endommagées par YOLO11, avec localisation de la zone d'impact, estimation de la sévérité et filtrage spatial multi-véhicules.

## Vue d'ensemble du pipeline

- **Constitution du dataset :** 800 images sélectionnées manuellement (500 accidentées, 300 intactes) en 3 types de véhicules (VUS, berline, citadine) et 3 niveaux de sévérité (léger, modéré, sévère)
- **Annotation :** 17 classes sur Roboflow (8 pièces × sain/endommagé + classe voiture), polygones convertis en bounding boxes via script custom
- **Augmentation asymétrique :** taux d'augmentation adapté par classe selon le nombre d'instances (x1 à x4) avec stratification multi-label train/val/test (60/20/20)
- **Modèle principal :** YOLO11s V5 — meilleur compromis de 5 versions entraînées (mAP@0.5 = 0.769, F1 = 0.749, précision = 0.778, rappel = 0.722)
- **Localisation de zone d'impact :** système de vote pondéré par position spatiale des bounding boxes (avant, arrière, gauche, droite) avec gestion des zones combinées
- **Estimation de sévérité :** score composite basé sur le nombre de pièces endommagées, la surface totale des zones et le type de pièces affectées (léger / modéré / sévère)
- **Filtrage multi-véhicules :** algorithme de filtrage spatial pour limiter la détection aux pièces du véhicule principal et éviter les faux positifs sur d'autres voitures
- **Rapport automatique :** génération automatique d'un rapport Markdown par image avec pièces endommagées, zone d'impact et niveau de sévérité

## Résultats du modèle V5 (final)

| Métrique | Score |
|---|---|
| mAP@0.5 | 0.769 |
| mAP@0.5-0.95 | 0.576 |
| Précision | 0.778 |
| Rappel | 0.722 |
| F1-score | 0.749 |
| Seuil optimal | 0.353 (courbe F1) |

## Évolution des versions

| Version | Architecture | Note |
|---|---|---|
| V1 | YOLO11s | Baseline — bug fliplr (confusion gauche/droite) |
| V2 | YOLO11m | fliplr=0 → meilleur mAP mais surdimensionné |
| V3 | YOLO11s + SGD | Convergence instable |
| V4 | YOLO11s (précision) | cls=1.0, box=10.0 → moins de rappel |
| **V5** ✅ | **YOLO11s** | **Hyperparamètres V1 + correction fliplr → meilleur compromis** |

## Technologies

- Python
- YOLO11 (Ultralytics)
- Roboflow (annotation + augmentation)
- OpenCV
- Pandas / NumPy / Matplotlib
- iterative-stratification (split multi-label)
- Google Colab (GPU)

## Prérequis

```bash
pip install ultralytics iterative-stratification opencv-python matplotlib pandas numpy
```

Développé sous Google Colab avec GPU. Le notebook se connecte à Google Drive pour accéder aux images, annotations et poids du modèle.

## Ressources

- Dataset et annotations : Google Drive (lien dans le notebook)
- Annotations Roboflow V5 : `roboflow.com/yan-jrjsi/projet-synthese`
- Rapport complet, présentation et affiche scientifique inclus dans le repo

## Structure

```
INF1022.ipynb            — notebook principal (8 sections)
Rapport.docx             — rapport académique détaillé
Presentation.pptx        — diapositives de soutenance
AfficheScientifique.pptx — affiche scientifique
```

---

Projet de synthèse universitaire solo — cours INF1022, UQTR.
