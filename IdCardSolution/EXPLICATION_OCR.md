# Explication Complète du Système OCR — PaddleOCR

## Table des matières
1. [Architecture globale](#1-architecture-globale)
2. [PaddleOCR — Théorie](#2-paddleocr--théorie)
3. [Migration EasyOCR → PaddleOCR](#3-migration-easyocr--paddleocr)
4. [Modèles utilisés](#4-modèles-utilisés)
5. [Pipeline d'extraction](#5-pipeline-dextraction)
6. [Double passe OCR (Arabe + Français)](#6-double-passe-ocr-arabe--français)
7. [Déduplication intelligente](#7-déduplication-intelligente)
8. [Détection du NIN](#8-détection-du-nin)
9. [Détection des champs identity](#9-détection-des-champs-identity)
10. [Séparation label:valeur](#10-séparation-labelvaleur)
11. [Affichage Arabic BiDi](#11-affichage-arabic-bidi)
12. [ROI Refinement (Allowlist)](#12-roi-refinement-allowlist)
13. [Intégration smart_scanner.py](#13-intégration-smart_scannerpy)
14. [Problèmes rencontrés et solutions](#14-problèmes-rencontrés-et-solutions)

---

## 1. Architecture globale

```
┌─────────────────────────────────────────────────────┐
│                  smart_scanner.py                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ Détection    │  │ Capture      │  │ Affichage │ │
│  │ carte (Canny │  │ Perspective  │  │ OpenCV    │ │
│  │ + contour)   │  │ Transform    │  │ overlay   │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┘ │
│         │                  │                         │
│         └────────┬─────────┘                         │
│                  ▼                                   │
│  ┌─────────────────────────────┐                     │
│  │     extract_fields()        │                     │
│  │     ocr_extractor.py        │                     │
│  └─────────────────────────────┘                     │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│               ocr_extractor.py                       │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐                   │
│  │ PaddleOCR   │  │ PaddleOCR   │  ← 2 passes       │
│  │ ARABIC      │  │ FRENCH/EN   │                   │
│  │ PP-OCRv5    │  │ PP-OCRv5    │                   │
│  └──────┬──────┘  └──────┬──────┘                   │
│         └───────┬─────────┘                          │
│                 ▼                                    │
│  ┌──────────────────────┐                            │
│  │ Fusion + Dédup       │                            │
│  │ par bbox proximity   │                            │
│  └──────────┬───────────┘                            │
│             ▼                                        │
│  ┌──────────────────────┐                            │
│  │ Détection NIN        │                            │
│  │ (regex + blocs)      │                            │
│  └──────────┬───────────┘                            │
│             ▼                                        │
│  ┌──────────────────────┐                            │
│  │ Détection champs     │                            │
│  │ (labels + position)  │                            │
│  └──────────┬───────────┘                            │
│             ▼                                        │
│  ┌──────────────────────┐                            │
│  │ ROI Refinement       │                            │
│  │ (re-read + allowlist)│                            │
│  └──────────┬───────────┘                            │
│             ▼                                        │
│  ExtractedFields                                     │
│  {nin, nom, prenom, date, lieu}                      │
└─────────────────────────────────────────────────────┘
```

---

## 2. PaddleOCR — Théorie

### Qu'est-ce que PaddleOCR ?

PaddleOCR est un toolkit OCR open-source développé par **Baidu** basé sur leur framework de deep learning **PaddlePaddle** (similaire à TensorFlow/PyTorch). Il excelle dans la reconnaissance de texte asiatique et arabe.

### Architecture Pipeline OCR

PaddleOCR utilise un pipeline en 3 étapes :

```
Image → [1. Text Detection] → [2. Text Direction] → [3. Text Recognition] → Texte
```

#### Étape 1 : Text Detection (Détection de texte)
- **Modèle** : DB-Net++ (Differentiable Binarization)   
- **Rôle** : Trouve les zones contenant du texte dans l'image
- **Sortie** : Polygones (coordonnées x,y des coins du texte)
- **Fonctionnement** :
  - L'image passe dans un CNN backbone (ResNet)
  - Un module de segmentation produit une carte de probabilité
  - Binarisation différentiable → contours nets des zones texte
  - Post-traitement → polygones

```
Image → ResNet → Feature Map → DB Head → Probability Map → Binarization → Polygons
```

#### Étape 2 : Text Direction Classification
- **Modèle** : Classificateur binaire
- **Rôle** : Déterminer si le texte est horizontal (0°) ou vertical (90°)
- **Pourquoi** : Le reconnaisseur ne traite que du texte horizontal
- **Note** : Désactivé dans notre cas (`use_textline_orientation=False`) car nos cartes sont toujours horizontales

#### Étape 3 : Text Recognition (Reconnaissance de texte)
- **Modèle** : SVTR (Single Visual model for Scene Text Recognition)
- **Rôle** : Convertir les images de zones texte en chaînes de caractères
- **Architecture** :
  - Encoder : CNN + Transformer
  - Décodeur : Attention-based seq2seq
  - CTC (Connectionist Temporal Classification) pour l'alignement

```
Zone texte image → CNN Encoder → Transformer → CTC Decoder → "اللقب براهيمي"
```

### PP-OCRv5 vs PP-OCRv6

| Caractéristique | PP-OCRv5 | PP-OCRv6 |
|---|---|---|
| Performance CPU | ✅ Rapide | ❌ Crash sur CPU (PIR bug) |
| Précision arabe | ~91% | ~93% |
| Modèles | mobile_det + mobile_rec | medium_det + medium_rec |
| Taille | Plus léger | Plus lourd |
| Compatibilité CPU | ✅ PaddlePaddle 3.2.1 | ❌ NotImplementedError |

**Notre choix** : PP-OCRv5 car PP-OCRv6 plante sur CPU avec l'erreur :
```
NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
```

---

## 3. Migration EasyOCR → PaddleOCR

### Pourquoi changer ?

| Critère | EasyOCR | PaddleOCR |
|---|---|---|
| Précision arabe | ~67% | ~91% |
| Dépendances | torch + torchvision (2 Go) | paddlepaddle (500 Mo) |
| Vitesse CPU | Lent | Rapide |
| Arabic natif | Non | Oui (modèle dédié) |
| Support GPU | CUDA obligatoire | CPU ou GPU |

### Dépendances installées

```bash
# Python 3.13 requis (PaddlePaddle ne supporte pas 3.14)
py -3.13 -m venv .venv

# PaddlePaddle CPU (pas de GPU disponible)
pip install paddlepaddle==3.2.1

# PaddleOCR avec support document
pip install paddleocr[doc-parser]>=3.6.0

# Pour l'affichage Arabic BiDi
pip install arabic-reshaper python-bidi
```

### Pourquoi Python 3.13 et pas 3.14 ?

PaddlePaddle compile des extensions C++ native. Les wheels PyPI ne sont publiées que pour Python 3.9-3.13. Python 3.14 n'est pas encore supporté.

### Pourquoi PaddlePaddle 3.2.1 et pas 3.3.0 ?

PaddlePaddle 3.3.0 a un bug PIR (Paddle Intermediate Representation) sur CPU :
```
RuntimeError: PIR is not supported on CPU with PaddlePaddle 3.3.0
```

---

## 4. Modèles utilisés

### Configuration des moteurs OCR

```python
# Moteur Arabe
ocr_ar = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",    # Détection universelle
    text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",  # Reconnaissance arabe
)

# Moteur Français/Anglais
ocr_fr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",    # Même détection
    text_recognition_model_name="PP-OCRv5_mobile_rec",   # Reconnaissance FR/EN
)

# Moteur de raffinement ROI (re-read zones crops)
rec_ar = TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec")
rec_fr = TextRecognition(model_name="PP-OCRv5_mobile_rec")
```

### Pourquoi 4 modèles ?

1. **Détection** (1 modèle partagé) : Trouver les zones de texte → `PP-OCRv5_mobile_det`
2. **Reconnaissance arabe** (1 modèle) : Lire le texte arabe → `arabic_PP-OCRv5_mobile_rec`
3. **Reconnaissance FR/EN** (1 modèle) : Lire le texte latin → `PP-OCRv5_mobile_rec`
4. **Re-read ROI** (2 modèles) : Relire une zone cropée avec allowlist → `TextRecognition`

### Le format des modèles

```
PP-OCRv5_mobile_det/
├── inference.pdiparams      # Poids du modèle (params)
├── inference.pdmodel        # Graph du modèle (structure)
└── inference.pdiparams.info # Métadonnées

arabic_PP-OCRv5_mobile_rec/
├── inference.pdiparams
├── inference.pdmodel
└── inference.pdiparams.info
```

Les modèles sont téléchargés automatiquement depuis les serveurs Baidu au premier lancement, puis mis en cache dans `~/.paddlex/official_models/`.

---

## 5. Pipeline d'extraction

### Flux complet

```
Image JPEG (BGR)
     │
     ▼
cv2.cvtColor(BGR → RGB)
     │
     ├──→ _run_ocr(ocr_ar, rgb)  →  results_ar  [(bbox, text, conf), ...]
     │
     ├──→ _run_ocr(ocr_fr, rgb)  →  results_fr  [(bbox, text, conf), ...]
     │
     ▼
Fusion AR + FR (dedup par bbox proximity)
     │
     ▼
blocks[] = [{text, confidence, bbox}, ...]
     │
     ├──→ _find_nin()          →  extracted.nin
     ├──→ _find_identity_fields() → extracted.nom, prenom, date, lieu
     │
     ▼
ROI Refinement (re-read chaque champ trouvé)
     │
     ▼
ExtractedFields {nin, nom, prenom, date_naissance, lieu_naissance}
```

### Adaptateur _run_ocr()

PaddleOCR retourne un format spécifique :
```python
# Sortie PaddleOCR
{
    'dt_polys': [array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]]), ...],
    'rec_texts': ['اللقب', 'براهيمي', ...],
    'rec_scores': [0.95, 0.92, ...]
}
```

Notre adaptateur convertit en format EasyOCR :
```python
# Sortie convertie
[(bbox, text, confidence), ...]
# bbox = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
```

---

## 6. Double passe OCR (Arabe + Français)

### Pourquoi deux passes ?

La CNI algérienne contient du texte dans **deux langues** :
- **Arabe** : labels (اللقب, الاسم, تاريخ الميلاد...) et valeurs (prénoms arabes, lieu)
- **Français** : NIN, dates, groupes sanguins, codes

Un seul modèle ne peut pas bien reconnaître les deux langues. PaddleOCR a des modèles spécialisés par langue.

### Logique de fusion

```python
for (bbox, text, confidence) in results_ar + results_fr:
    cx = centre_x(bbox)
    cy = centre_y(bbox)
    
    is_dup = False
    for idx, (sx, sy) in enumerate(seen_centers):
        if abs(cx - sx) < 30 and abs(cy - sy) < 30:  # Même zone
            is_dup = True
            # garder le bloc avec PLUS de chiffres (utile pour NIN)
            if new_digits > existing_digits:
                all_results[idx] = (bbox, text, confidence)
            break
    
    if not is_dup:
        all_results.append((bbox, text, confidence))
```

### Seuil de proximité : 30px

Deux blocs sont considérés comme dupliqués si leurs centres sont à moins de 30 pixels. Ce seuil a été choisi car :
- Les deux passes OCR détectent les mêmes zones texte
- Les bbox ne sont jamais exactement identiques (légère variance)
- 30px est assez large pour capturer cette variance, assez petit pour ne pas fusionner des blocs différents

---

## 7. Déduplication intelligente

### Problème : NIN écrasé par texte arabe

Quand le bloc arabe `رقم التعريف الوطني` (label) et le bloc français `100060581038770008` (NIN) sont au même endroit, lequel garder ?

### Solution : préférer le bloc avec plus de chiffres

```python
existing_digits = sum(c.isdigit() for c in existing_text)
new_digits = sum(c.isdigit() for c in new_text)

if new_digits > existing_digits:
    # Remplacer : le nouveau a plus de chiffres
    all_results[idx] = new_bbox, new_text, new_conf
```

**Résultat** :
- `رقم التعريف الوطني` → 0 chiffres → rejeté
- `100060581038770008` → 18 chiffres → gardé

---

## 8. Détection du NIN

Le NIN (Numéro d'Identification National) est un code de **18 chiffres** sur la CNI algérienne.

### Algorithme à 4 méthodes

```
Méthode 1 : regex \d{18} dans le texte brut (après nettoyage espaces)
    ↓ si échec
Méthode 2 : bloc OCR contenant exactement 18 chiffres
    ↓ si échec
Méthode 3 : concaténer tous les blocs numériques, chercher 18 chiffres
    ↓ si échec
Méthode 4 : tolérance 16-19 chiffres, tronquer à 18
```

### Pourquoi 4 méthodes ?

L'OCR peut parfois :
- Séparer le NIN en 2 blocs (`1000605` + `581038770008`)
- Ajouter des espaces (`1000605810 38770008`)
- Confondre des caractères proches (`0/O`, `1/l`)

---

## 9. Détection des champs identity

### Ordre de détection

```
1. Date de naissance (par label + position)
2. Nom (label detection → position fallback → fallback latin/arabe)
3. Prénom (pareil)
4. Lieu de naissance (label detection)
```

### Labels supportés

| Champ | Arabe | Français | OCR misreads |
|---|---|---|---|
| Nom | اللقب, اسم العائلة | nom, surname | للقب |
| Prénom | الاسم, الإسم | prenom, prénom | للاسم |
| Date | تاريخ المildad | date de naissance | - |
| Lieu | مكان الميلاد | lieu de naissance | - |

### Position-based fallback

Quand les labels ne sont pas détectés (texte trop petit ou flou), on utilise la **position** :

Sur une CNI algérienne, la disposition est toujours :
```
┌──────────────────────────────┐
│  ┌─────┐  ┌───────────────┐  │
│  │Photo│  │رقم التعريف    │  │  ← NIN (y=100)
│  │     │  │الوطني         │  │
│  │     │  │10006058103877 │  │
│  └─────┘  │     0008      │  │
│           ├───────────────┤  │
│           │اللقب          │  │  ← y > NIN_bottom (y=160)
│           │براهيمي        │  │  ← Nom
│           ├───────────────┤  │
│           │الاسم          │  │  ← Prénom
│           │طارق           │  │
│           ├───────────────┤  │
│           │تاريخ الميلاد  │  │
│           │2006.05.25     │  │  ← Date
│           ├───────────────┤  │
│           │مكان الميلاد   │  │
│           │القبة           │  │  ← Lieu
│           └───────────────┘  │
└──────────────────────────────┘
```

L'algorithme :
1. Trouver le bloc du NIN (contient >10 chiffres)
2. Calculer `nin_bottom` (coordonnée y inférieure)
3. Chercher tous les blocs **sous** le NIN (`block_top > nin_bottom`)
4. Filtrer : texte arabe, pas de chiffres, 2-25 caractères, pas un label
5. Les 2 premiers blocs = nom + prénom

---

## 10. Séparation label:valeur

### Problème

L'OCR fusionne parfois le label et la valeur en un seul bloc :
```
OCR lit : "اللقب براهمي"    (un seul bloc)
Attendu : "اللقب" + "براهمي" (deux blocs séparés)
```

### Solution : regex de séparation

```python
label_value_patterns = [
    r'(اللقب)\s*[:\u061A\u061B]?\s*(.+)',   # اللقب + valeur
    r'(للقب)\s*[:\u061A\u061B]?\s*(.+)',     # للقب (OCR misread)
    r'(الاسم)\s*[:\u061A\u061B]?\s*(.+)',    # الاسم + valeur
    r'(للاسم)\s*[:\u061A\u061B]?\s*(.+)',    # للاسم (OCR misread)
    r'(الإسم)\s*[:\u061A\u061B]?\s*(.+)',    # الإسم (hamza)
    r'(مكان الميلاد)\s*[:\u061A\u061B]?\s*(.+)',
    # ... patterns français aussi
    r'(NOM)\s*:?\s*(.+)',
    r'(PRENOM[S]?)\s*:?\s*(.+)',
]
```

### Syntaxe regex expliquée

```
(اللقب)        ← Groupe 1 : capture le label
\s*            ← Zéro ou plusieurs espaces
[:\u061A\u061B]?  ← Optionnel : ':' ou ':' arabe (U+061A) ou ';' arabe (U+061B)
\s*            ← Zéro ou plusieurs espaces
(.+)           ← Groupe 2 : capture tout le reste (la valeur)
```

### Résultat

```
Input:  "اللقب براهمي"
Output: ["اللقب", "براهمي"]  → 2 blocs dans expanded_blocks
```

---

## 11. Affichage Arabic BiDi

### Problème

PaddleOCR retourne le texte arabe dans l'ordre **visuel** (LTR), mais l'arabe se lit de **droite à gauche** (RTL). Dans le terminal, le texte apparaît inversé :

```
PaddleOCR retourne : "يميهارب"   ← à l'envers
Attendu :           "براهيمي"   ← correct
```

### Théorie : Unicode Bidirectional Algorithm (BiDi)

Le standard Unicode définit un algorithme pour gérer le mélange texte RTL (arabe, hébreu) et LTR (latin) :

1. **arabic-reshaper** : Reconnecte les lettres arabes concaténées
   - L'arabe a des formes de lettres liées (début, milieu, fin, isolé)
   - `reshape("براهيمي")` → caractères avec formes correctes

2. **python-bidi** : Applique l'algorithme BiDi
   - Détermine la direction de chaque segment
   - Réordonne les caractères pour l'affichage LTR
   - Gère le mélange arabe+latin

### Implémentation

```python
@staticmethod
def fix_arabic_display(text):
    # Vérifier si le texte contient des caractères arabes
    if not any('\u0600' <= c <= '\u06FF' for c in text):
        return text  # Pas d'arabe, retourner tel quel
    
    from bidi.algorithm import get_display
    import arabic_reshaper
    
    # Étape 1 : Reconnecter les lettres
    reshaped = arabic_reshaper.reshape(text)
    
    # Étape 2 : Réordonner pour affichage LTR
    return get_display(reshaped)
```

### Important : BiDi uniquement pour l'affichage

Le BiDi est appliqué **uniquement** au moment de l'affichage terminal, **jamais** pendant l'extraction. Pourquoi ?

Si on appliquait BiDi aux blocs OCR, les regex de détection de labels ne marcheraient plus :
```
Sans BiDi : "اللقب براهمي"  → regex détecte "الköp" ✅
Avec BiDi : "يميهارب البق"  → regex ne trouve rien ❌
```

---

## 12. ROI Refinement (Allowlist)

### Problème

L'OCR peut confondre des caractères visuellement similaires :
- `0` (zéro) vs `O` (lettre O)
- `1` (un) vs `l` (L minuscule)
- `8` vs `B`
- Chiffres arabes vs latins

### Solution : Re-read avec allowlist

Pour chaque champ déjà trouvé, on recadre la zone (ROI = Region of Interest) et on relit avec un **allowlist** qui restreint les caractères autorisés.

```python
# Pour le NIN : uniquement des chiffres
cleaned = extraire_texte_roi(rec_engine, roi, allowlist='0123456789')

# Pour la date : chiffres + / + .
cleaned = extraire_texte_roi(rec_engine, roi, allowlist='0123456789/.')

# Pour nom/prenom : latin + arabe + espace
cleaned = extraire_texte_roi(rec_engine, roi, is_num=False)
```

### Fonctionnement de TextRecognition.predict()

```python
def extraire_texte_roi(paddle_engine, roi_image, allowlist=None):
    result = paddle_engine.predict(roi_image)
    text = result[0]['rec_text']
    
    # Filtrer avec allowlist
    text = ''.join(c for c in text if c in allowlist)
    return text
```

### Pourquoi un seul résultat `result[0]` ?

`TextRecognition.predict()` retourne une liste de résultats (un par ligne de texte détectée). Pour un petit crop ROI, il n'y a qu'une seule ligne → `result[0]`.

### Post-traitement des labels

L'OCR peut parfois re-détecter le label dans le ROI :
```
ROI crop = zone "براهيمي"
Re-read = "اللقب براهمي"  ← label réapparaît
```

Solution : strip automatique des prefixes labels après re-read :
```python
_label_prefixes = ["اللقب", "للقب", "الاسم", "الإسم", ...]
for prefix in _label_prefixes:
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix):].lstrip(' :')
```

---

## 13. Intégration smart_scanner.py

### Auto-switch venv

Windows n'a pas de `#!/usr/bin/env python`. Le script vérifie s'il tourne avec le bon Python :

```python
_VENV_PYTHON = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
if sys.executable.lower() != _VENV_PYTHON.lower():
    # Relancer avec le venv
    subprocess.call([_VENV_PYTHON] + sys.argv)
    sys.exit(ret)
```

### Suppression du bruit PaddlePaddle

PaddlePaddle utilise GLOG (Google Logging) qui écrit directement sur stdout/stderr au niveau C++. `contextlib.redirect_stderr()` ne suffit pas car c'est au niveau OS :

```python
# Rediriger les file descriptors au niveau OS
devnull = os.open(os.devnull, os.O_WRONLY)
old_stdout = os.dup(1)
old_stderr = os.dup(2)
os.dup2(devnull, 1)  # stdout → /dev/null
os.dup2(devnull, 2)  # stderr → /dev/null

# ... charger les modèles ...

os.dup2(old_stdout, 1)  # restaurer stdout
os.dup2(old_stderr, 2)  # restaurer stderr
```

---

## 14. Problèmes rencontrés et solutions

| # | Problème | Cause | Solution |
|---|---|---|---|
| 1 | PP-OCRv6 crash CPU | `NotImplementedError: ConvertPirAttribute2RuntimeAttribute` | Forcer PP-OCRv5 |
| 2 | PaddlePaddle 3.3.0 crash | PIR bug sur CPU | Downgrader à 3.2.1 |
| 3 | NIN écrasé par texte arabe | Dédup gardait le premier bloc | Préférer bloc avec plus de chiffres |
| 4 | `الإسم` non détecté | Hamza (إ) différent de alef (ا) | Ajouter pattern `الإسم` séparé |
| 5 | `للقب` non détecté | OCR perd l'alef | Ajouter pattern `للقب` |
| 6 | Date mauvaise (issue vs birth) | Premier regex DD.MM.YYYY trouvait la date d'issue | Chercher par label `تاريخ الميلاد` d'abord |
| 7 | Date sans points | Allowlist `'0123456789/'` manquait `.` | Ajouter `.` à l'allowlist |
| 8 | Lieu fusionné avec label | Re-read restaurait `مكان الميلاد القبة` | Strip des prefixes labels après re-read |
| 9 | `INCONNU` au lieu de `CNI` | `stable_label` ("DETECTING") passé à print | Passer `card_type` résolu |
| 10 | Date/Lieu manquants dans le terminal | `print_capture_result` ne les affichait pas | Ajouter les champs au print |
| 11 | Bruit PaddlePaddle au lancement | GLOG écrit sur stdout/stderr C++ | Rediriger les file descriptors OS |
| 12 | `_ensure_package` crash Python 3.14 | PaddlePaddle pas compatible 3.14 | Supprimer auto-install, auto-switch venv |
| 13 | Texte arabe inversé | PaddleOCR retourne ordre visuel | `arabic-reshaper` + `python-bidi` pour affichage |
| 14 | Permis: labels FR non détectés | Patterns uniquement arabes | Ajouter patterns français (NOM, PRENOM...) |

---

## Résumé technique final

```
Stack:          PaddlePaddle 3.2.1 + PaddleOCR 3.7.0
Python:         3.13.14 (venv)
Modèles:        PP-OCRv5_mobile_det + arabic_PP-OCRv5_mobile_rec + PP-OCRv5_mobile_rec
Pipeline:       2 passes (AR + FR) → fusion → détection → raffinement
Champs:         NIN (18 chiffres), Nom, Prénom, Date, Lieu
Performance:    ~5s par image (CPU), 91% confiance moyenne
Précision:      NIN 6/6, Date 5/6, Nom/Prénom variable selon qualité capture
```
