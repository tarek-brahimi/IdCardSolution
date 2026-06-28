# Explication du Code — ocr_extractor.py

## Table des matières
1. [Vue d'ensemble du fichier](#1-vue-densemble-du-fichier)
2. [Import et configuration initiale](#2-import-et-configuration-initiale)
3. [Suppression du bruit PaddlePaddle](#3-suppression-du-bruit-paddlepaddle)
4. [Structure de données ExtractedFields](#4-structure-de-données-extractedfields)
5. [Fonction extraire_texte_roi](#5-fonction-extraire_texte_roi)
6. [Dictionnaires de labels](#6-dictionnaires-de-labels)
7. [Classe CardOCR — Initialisation](#7-classe-cardocr--initialisation)
8. [CardOCR._run_ocr — Adaptateur PaddleOCR](#8-cardocr_run_ocr--adaptateur-paddleocr)
9. [CardOCR.fix_arabic_display — BiDi](#9-cardocr_fix_arabic_display--bidi)
10. [CardOCR.extract — Pipeline principal](#10-cardocr_extract--pipeline-principal)
11. [Déduplication par bbox](#11-déduplication-par-bbox)
12. [CardOCR._find_nin — Détection NIN](#12-cardocr_find_nin--détection-nin)
13. [CardOCR._find_identity_fields — Champs identity](#13-cardocr_find_identity_fields--champs-identity)
14. [Séparation label:valeur](#14-séparation-labelvaleur)
15. [Détection de la date](#15-détection-de-la-date)
16. [Détection nom/prénom/lieu](#16-détection-nomprénomlieu)
17. [Fallback par position](#17-fallback-par-position)
18. [Fallback par texte propre](#18-fallback-par-texte-propre)
19. [ROI Refinement (re-read)](#19-roi-refinement-re-read)
20. [Fonction extract_fields](#20-fonction-extract_fields)
21. [Bloc __main__](#21-bloc-__main__)

---

## 1. Vue d'ensemble du fichier

```
ocr_extractor.py (677 lignes)
├── Lignes 1-11      : Docstring
├── Lignes 12-46     : Imports, auto-switch venv, suppression bruit
├── Lignes 48-51     : Imports Python (re, numpy, dataclass)
├── Lignes 54-65     : class ExtractedFields (dataclass)
├── Lignes 68-81     : extraire_texte_roi() (re-read ROI)
├── Lignes 84-98     : LABELS_FR + LABELS_AR (dictionnaires)
├── Lignes 102-155   : class CardOCR.__init__ + _init_reader
├── Lignes 157-182   : _run_ocr() + fix_arabic_display()
├── Lignes 184-319   : extract() (pipeline principal)
├── Lignes 321-360   : _find_nin()
├── Lignes 362-628   : _find_identity_fields()
├── Lignes 631-645   : extract_fields() (fonction raccourci)
└── Lignes 649-677   : __main__ (CLI)
```

**Responsabilités :**
1. Initialiser les moteurs OCR (Arabe + Français)
2. Extraire tout le texte de la carte
3. Identifier les champs : NIN, nom, prénom, date de naissance, lieu
4. Retourner une structure `ExtractedFields`

---

## 2. Import et configuration initiale

```python
import sys
import os

# Auto-switch to .venv Python when run directly
if __name__ == "__main__":
    _VENV_PYTHON = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".venv", "Scripts", "python.exe"
    )
    if (sys.platform == "win32"
        and os.path.exists(_VENV_PYTHON)
        and sys.executable.lower() != _VENV_PYTHON.lower()):
        import subprocess
        ret = subprocess.call([_VENV_PYTHON] + sys.argv)
        sys.exit(ret)
```

**Ce que ça fait :**
- `os.path.abspath(__file__)` → chemin complet du fichier ocr_extractor.py
- `os.path.dirname(...)` → le dossier parent (IdCardSolution/)
- On construit le chemin vers `.venv/Scripts/python.exe`
- Si on n'est PAS déjà dans le venv, on **relance le script** avec le bon Python
- `sys.argv` = les arguments du script (ex: `["ocr_extractor.py", "image.jpg"]`)
- `subprocess.call()` lance le processus et attend qu'il finisse
- `sys.exit(ret)` sort avec le même code de retour

**Pourquoi :** Le venv contient Python 3.13 + PaddlePaddle. Sans ça, le script utilise le Python système 3.14 qui crash.

---

## 3. Suppression du bruit PaddlePaddle

```python
_devnull_fd = os.open(os.devnull, os.O_WRONLY)   # Ouvre /dev/null
_old_stdout = os.dup(1)                           # Sauvegarde stdout
_old_stderr = os.dup(2)                           # Sauvegarde stderr
os.dup2(_devnull_fd, 1)                           # stdout → /dev/null
os.dup2(_devnull_fd, 2)                           # stderr → /dev/null
os.close(_devnull_fd)                             # Ferme /dev/null (copie gardée)
try:
    import paddle
    import paddleocr
except ImportError:
    raise ImportError("Missing dependencies...")
finally:
    os.dup2(_old_stdout, 1)                       # Restaure stdout
    os.dup2(_old_stderr, 2)                       # Restaure stderr
    os.close(_old_stdout)                         # Ferme les sauvegardes
    os.close(_old_stderr)
```

**Comment ça marche :**

`os.dup2(fd1, fd2)` copie le file descriptor `fd1` sur `fd2`. En Python :
- FD 1 = stdout (sortie standard)
- FD 2 = stderr (erreur standard)

```
Avant :  FD 1 → terminal (console)
         FD 2 → terminal (console)

Pendant : FD 1 → /dev/null (jeté)
          FD 2 → /dev/null (jeté)

Après :  FD 1 → terminal (restauré)
         FD 2 → terminal (restauré)
```

**Pourquoi au niveau OS et pas Python :**
PaddlePaddle écrit ses logs depuis du **code C++** (GLOG). `contextlib.redirect_stderr()` ne capture que Python. Il faut rediriger les file descriptors au niveau OS.

**Que supprime-t-on :**
```
INFO: Could not find files for the given pattern(s).
[32mCreating model: ('PP-OCRv5_mobile_det', None, None)[0m
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0625 21:01:28 onednn_context.cc:81 oneDNN v3.6.2
```

---

## 4. Structure de données ExtractedFields

```python
@dataclass
class ExtractedFields:
    """Fields extracted from the ID card."""
    raw_text: str                           # Tout le texte brut concaténé
    raw_blocks: List[Dict] = field(default_factory=list)  # Blocs OCR individuels
    nin: Optional[str] = None               # Numéro Identification National (18 chiffres)
    nom: Optional[str] = None               # Nom de famille (اللقب)
    prenom: Optional[str] = None            # Prénom (الاسم)
    date_naissance: Optional[str] = None    # Date de naissance
    lieu_naissance: Optional[str] = None    # Lieu de naissance
    all_numbers: List[str] = field(default_factory=list)  # Tous les nombres trouvés
    confidence_moyenne: float = 0.0         # Confiance moyenne OCR
```

**@dataclass** génère automatiquement :
- `__init__(self, raw_text, raw_blocks=[], nin=None, ...)`
- `__repr__()` pour l'affichage
- `__eq__()` pour la comparaison

**Structure de `raw_blocks` :**
```python
[
    {
        "text": "100060581038770008",     # Texte du bloc
        "confidence": 0.97,               # Score de confiance (0-1)
        "bbox": [[x1,y1], [x2,y2],       # 4 coins du polygone
                 [x3,y3], [x4,y4]]
    },
    ...
]
```

---

## 5. Fonction extraire_texte_roi

```python
def extraire_texte_roi(paddle_engine, roi_image, is_num=False, allowlist=None):
    if allowlist is None:
        if is_num:
            allowlist = '0123456789'
        else:
            arabic = ''.join(chr(c) for c in range(0x0600, 0x0700))
            allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ' + arabic
    result = paddle_engine.predict(roi_image)
    text = ""
    for res in result:
        text = res['rec_text']
        break
    text = ''.join(c for c in text if c in allowlist)
    return text
```

**Ce que ça fait :**
1. Définit la liste des caractères autorisés (allowlist)
2. Lance la reconnaissance OCR sur une zone cropée (ROI)
3. Filtre le résultat pour garder uniquement les caractères autorisés

**Paramètres :**
- `paddle_engine` : `TextRecognition` (moteur de reconnaissance seul, pas de détection)
- `roi_image` : image cropée (numpy array BGR)
- `is_num` : si True, allowlist = chiffres uniquement
- `allowlist` : liste custom de caractères

**Pourquoi un allowlist :**

L'OCR peut confondre :
```
O (lettre) → 0 (chiffre)    avec allowlist='0123456789', O est rejeté
l (L min)  → 1 (chiffre)    avec allowlist='0123456789', l est rejeté
B (maj)    → 8 (chiffre)    avec allowlist='0123456789', B est rejeté
```

**Comment marche `predict()` :**

```python
result = paddle_engine.predict(roi_image)
# result = [
#     {
#         'rec_text': '100060581038770008',  # Texte reconnu
#         'rec_score': 0.97,                  # Confiance
#         'rec_polys': array(...)             # Position (inutile ici)
#     }
# ]
```

On prend `result[0]['rec_text']` car pour un petit crop, il n'y a qu'une ligne de texte.

---

## 6. Dictionnaires de labels

```python
LABELS_FR = {
    "nom": ["nom", "surname", "nom de famille", "family name"],
    "prenom": ["prénom", "prenom", "first name", "prénoms"],
    "date_naissance": ["date de naissance", "date naissance", "born", "birth date", "né(e) le"],
    "lieu_naissance": ["lieu de naissance", "lieu naissance", "born in", "birth place", "né à"],
    "nin": ["nin", "n° identification", "numéro identification", "national id"],
}

LABELS_AR = {
    "nom": ["اللقب", "الاسم العائلي"],
    "prenom": ["الاسم", "الاسم الشخصي"],
    "date_naissance": ["تاريخ الميلاد", "تاريخ الازدياد"],
    "lieu_naissance": ["مكان الميلاد", "محل الميلاد", "بلدية الميلاد"],
    "nin": ["رقم التعريف الوطني", "رقم التعريف"],
}
```

**Utilisation :** Ces dictionnaires ne sont pas utilisés directement dans le code actuel (ils sont en backup). La logique principale utilise des listes inline dans `_find_identity_fields()`.

**Structure :** `{nom_champ: [liste_de_variants]}`

---

## 7. Classe CardOCR — Initialisation

```python
class CardOCR:
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.languages = languages or ['ar', 'en']
        self.gpu = gpu
        self.ocr_ar = None    # Moteur OCR complet (détection + reconnaissance) arabe
        self.ocr_fr = None    # Moteur OCR complet FR/EN
        self.rec_ar = None    # Moteur de reconnaissance seul (pour ROI re-read) arabe
        self.rec_fr = None    # Moteur de reconnaissance seul FR/EN
```

**Lazy initialization** : Les moteurs ne sont pas chargés au `__init__`, mais au premier appel de `extract()`. C'est le pattern **lazy loading**.

**Pourquoi 4 moteurs :**

| Moteur | Type | Usage |
|---|---|---|
| `ocr_ar` | PaddleOCR complet | Passe 1 : détecte + lit le texte arabe |
| `ocr_fr` | PaddleOCR complet | Passe 2 : détecte + lit le texte FR/EN |
| `rec_ar` | TextRecognition seul | ROI refinement : relit un crop arabe |
| `rec_fr` | TextRecognition seul | ROI refinement : relit un crop FR/EN |

La différence :
- `PaddleOCR` = **détection** (trouve les zones) + **reconnaissance** (lit le texte)
- `TextRecognition` = **reconnaissance seulement** (on lui donne déjà le crop)

```python
def _init_reader(self):
    if self.ocr_ar is None:          # Lazy : seulement si pas encore chargé
        print("[OCR] Loading models...")
        from paddleocr import PaddleOCR, TextRecognition

        # [Suppression bruit — même principe que l'import]
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            # Moteur Arabe : détection mobile + reconnaissance arabe
            self.ocr_ar = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,   # Pas de classification orientation
                use_doc_unwarping=False,              # Pas de déformation document
                use_textline_orientation=False,       # Pas de détection orientation ligne
            )
            self.rec_ar = TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec")

            # Moteur Français : même détection + reconnaissance FR/EN
            self.ocr_fr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            self.rec_fr = TextRecognition(model_name="PP-OCRv5_mobile_rec")
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
        print("[OCR] Models loaded.")
```

**Paramètres PaddleOCR :**
- `text_detection_model_name` : quel modèle utiliser pour la détection
- `text_recognition_model_name` : quel modèle pour la reconnaissance
- `use_doc_orientation_classify=False` : on sait que la carte est horizontale
- `use_doc_unwarping=False` : on a déjà fait la transform perspective
- `use_textline_orientation=False` : on ne traite pas de texte vertical

---

## 8. CardOCR._run_ocr — Adaptateur PaddleOCR

```python
def _run_ocr(self, ocr_engine, image):
    results = ocr_engine.predict(image)
    output = []
    for res in results:
        polys = res['dt_polys']       # Polygones de détection (numpy arrays)
        texts = res['rec_texts']      # Textes reconnus (liste de strings)
        scores = res['rec_scores']    # Scores de confiance (liste de floats)
        for i in range(len(texts)):
            bbox = polys[i].tolist()   # Convertit numpy array → liste Python
            output.append((bbox, texts[i], float(scores[i])))
    return output
```

**Format de sortie PaddleOCR :**
```python
# Un seul résultat de predict() contient :
{
    'dt_polys': [                     # Liste de polygones (un par zone détectée)
        array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]]),  # Zone 1
        array([[x1,y1], [x2,y2], [x3,y3], [x4,y5]]),  # Zone 2
    ],
    'rec_texts': ['100060581038770008', 'اللقب'],      # Textes
    'rec_scores': [0.97, 0.93]                         # Confiances
}
```

**Format de sortie adapté (compatible EasyOCR) :**
```python
[
    (bbox, text, confidence),    # Zone 1
    (bbox, text, confidence),    # Zone 2
    ...
]
# bbox = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
```

**Pourquoi un adaptateur :** Le code était initialement écrit pour EasyOCR. Plutôt que de tout réécrire, on convertit la sortie PaddleOCR au même format.

---

## 9. CardOCR.fix_arabic_display — BiDi

```python
@staticmethod
def fix_arabic_display(text):
    # Vérifier si le texte contient des caractères arabes
    if not any('\u0600' <= c <= '\u06FF' or    # Arabe principal (U+0600-U+06FF)
               '\u0750' <= c <= '\u077F' or    # Arabe supplement (U+0750-U+077F)
               '\uFB50' <= c <= '\uFDFF' or    # Formes initiales (U+FB50-U+FDFF)
               '\uFE70' <= c <= '\uFEFF' for c in text):  # Formes finales (U+FE70-U+FEFF)
        return text    # Pas d'arabe → retourner tel quel

    try:
        from bidi.algorithm import get_display
        import arabic_reshaper

        reshaped = arabic_reshaper.reshape(text)   # Reconnecter les lettres liées
        return get_display(reshaped)               # Réordonner pour affichage LTR
    except ImportError:
        return text    # Si les libs ne sont pas installées
```

**Étapes :**

1. **Détection** : Le texte contient-il des caractères arabes ?
   - `\u0600-\u06FF` : bloc arabe principal (lettres, chiffres, signes)
   - `\uFB50-\uFDFF` : formes initiales de lettres arabes
   - `\uFE70-\uFEFF` : formes finales de lettres arabes

2. **arabic_reshaper** : Les lettres arabes ont 4 formes selon leur position
   ```
   ب (isolé) → ـب (début) → ـبـ (milieu) → ب (fin)
   
   "براهيمي" sans reshape :
   [ب] [ر] [ا] [ه] [ي] [م] [ي]    ← lettres séparées
   
   "براهيمي" avec reshape :
   [بـ] [ـر] [ـا] [ـه] [ـي] [ـم] [ي]  ← lettres liées
   ```

3. **python-bidi** : Algorithme Unicode Bidirectional
   ```
   Input (visual, LTR) :  "يميهارب"     ← inversé
   Output (logical, RTL): "براهيمي"     ← correct
   ```

---

## 10. CardOCR.extract — Pipeline principal

```python
def extract(self, image: np.ndarray) -> ExtractedFields:
    self._init_reader()

    from cv2 import cvtColor, COLOR_BGR2RGB
    rgb_image = cvtColor(image, COLOR_BGR2RGB)
    # OpenCV charge en BGR par défaut, PaddleOCR attend RGB

    print("[OCR] Extracting text (Arabic pass)...")
    results_ar = self._run_ocr(self.ocr_ar, rgb_image)
    print("[OCR] Extracting text (French pass)...")
    results_fr = self._run_ocr(self.ocr_fr, rgb_image)
```

**Flux :**
```
Image BGR (OpenCV) → cvtColor → RGB → _run_ocr(ocr_ar) → blocs arabes
                                     → _run_ocr(ocr_fr) → blocs français
                                     → Fusion → Extraction champs
```

---

## 11. Déduplication par bbox

```python
    all_results = []
    seen_centers = []

    for (bbox, text, confidence) in results_ar + results_fr:
        cx = sum(p[0] for p in bbox) / 4     # Centre X du polygone (moyenne des 4 x)
        cy = sum(p[1] for p in bbox) / 4     # Centre Y du polygone (moyenne des 4 y)

        is_dup = False
        for idx, (sx, sy) in enumerate(seen_centers):
            if abs(cx - sx) < 30 and abs(cy - sy) < 30:  # Même zone (30px tolerance)
                is_dup = True
                existing = all_results[idx]
                existing_digits = sum(c.isdigit() for c in existing[1])  # Compte chiffres
                new_digits = sum(c.isdigit() for c in text)

                # Garder le bloc avec PLUS de chiffres
                if new_digits > existing_digits or \
                   (new_digits == existing_digits and confidence > existing[2]):
                    all_results[idx] = (bbox, text, confidence)
                break

        if not is_dup:
            all_results.append((bbox, text, confidence))
            seen_centers.append((cx, cy))
```

**Exemple concret :**
```
Bloc arabe : bbox centre=(500, 130), text="رقم التعريف الوطني", digits=0
Bloc français: bbox centre=(505, 132), text="100060581038770008", digits=18

→ distance = sqrt((505-500)² + (132-130)²) = sqrt(25+4) = 5.4px < 30px
→ C'est un doublon !
→ new_digits (18) > existing_digits (0) → garder le bloc français
```

**Pourquoi 30px :** Les deux passes OCR détectent la même zone texte mais avec une légère variance de position. 30px est le compromis entre :
- Assez large pour capturer la variance (< 30px = doublon)
- Assez petit pour ne pas fusionner des blocs différents (> 30px = séparés)

---

## 12. CardOCR._find_nin — Détection NIN

```python
def _find_nin(self, text: str, blocks: List[Dict]) -> Optional[str]:
```

Le NIN est un code de **18 chiffres** sur la CNI algérienne. L'algorithme utilise 4 méthodes par ordre de fiabilité :

### Méthode 1 : Regex directe
```python
    cleaned = re.sub(r'\s+', '', text)           # Supprime tous les espaces
    match_18 = re.search(r'\b\d{18}\b', cleaned) # Cherche exactement 18 chiffres
    if match_18:
        return match_18.group()                   # "100060581038770008"
```
**Cas idéal :** Le NIN est lu en un seul bloc sans espaces.

### Méthode 2 : Bloc exact
```python
    for block in blocks:
        block_cleaned = re.sub(r'\s+', '', block["text"])
        if re.match(r'^\d{18}$', block_cleaned):  # Un bloc = exactement 18 chiffres
            return block_cleaned
```
**Cas :** Un seul bloc OCR contient exactement 18 chiffres.

### Méthode 3 : Concaténation
```python
    number_blocks = []
    for block in blocks:
        nums = re.findall(r'\d+', block["text"])  # Extrait les suites de chiffres
        number_blocks.extend(nums)                 # ex: ["1000605", "81038770008"]

    all_digits = ''.join(number_blocks)            # "100060581038770008"
    match = re.search(r'\d{18}', all_digits)
    if match:
        return match.group()
```
**Cas :** L'OCR a séparé le NIN en plusieurs blocs (`"1000605"` + `"81038770008"`).

### Méthode 4 : Tolérance
```python
    match_long = re.search(r'\d{16,19}', all_digits)  # 16 à 19 chiffres
    if match_long:
        candidate = match_long.group()
        if len(candidate) >= 18:
            return candidate[:18]                      # Tronquer à 18
```
**Cas :** L'OCR a ajouté des caractères parasites (ex: `1000605810387700087` = 19 chiffres → tronquer).

---

## 13. CardOCR._find_identity_fields — Champs identity

C'est la méthode la plus complexe (260 lignes). Elle utilise **3 stratégies** par ordre de fiabilité :

### Étape 1 : Séparation label:valeur (voir section 14)
### Étape 2 : Détection de la date (voir section 15)
### Étape 3 : Détection nom/prénom/lieu (voir section 16)
### Étape 4 : Fallback par position (voir section 17)
### Étape 5 : Fallback par texte propre (voir section 18)

---

## 14. Séparation label:valeur

```python
    # Patterns pour séparer les blocs fusionnés
    label_value_patterns = [
        r'(اللقب)\s*[:\u061A\u061B]?\s*(.+)',      # اللقب + valeur
        r'(للقب)\s*[:\u061A\u061B]?\s*(.+)',        # للقب (OCR misread)
        r'(الاسم)\s*[:\u061A\u061B]?\s*(.+)',       # الاسم + valeur
        r'(للاسم)\s*[:\u061A\u061B]?\s*(.+)',       # للاسم (OCR misread)
        r'(الإسم)\s*[:\u061A\u061B]?\s*(.+)',       # الإسم (hamza)
        r'(مكان الميلاد)\s*[:\u061A\u061B]?\s*(.+)',
        r'(تاريخ الميلاد)\s*[:\u061A\u061B]?\s*(.+)',
        r'(NOM)\s*:?\s*(.+)',                        # Patterns français
        r'(PRENOM[S]?)\s*:?\s*(.+)',
        # ...
    ]

    expanded_blocks = []
    for block in blocks:
        text = block["text"]
        matched = False
        for pattern in label_value_patterns:
            m = re.search(pattern, text)
            if m:
                # Créer 2 blocs : label + valeur
                expanded_blocks.append({
                    "text": m.group(1),              # Le label
                    "confidence": block["confidence"],
                    "bbox": block["bbox"]            # Même bbox
                })
                expanded_blocks.append({
                    "text": m.group(2),              # La valeur
                    "confidence": block["confidence"],
                    "bbox": block["bbox"]
                })
                matched = True
                break
        if not matched:
            expanded_blocks.append(block)            # Pas de match → garder tel quel
```

**Exemple :**
```
Input:  block = {"text": "اللقب براهمي", "bbox": ...}
Pattern: r'(اللقب)\s*[:\u061A\u061B]?\s*(.+)'
Match:   group(1) = "اللقب", group(2) = "براهمي"

Output: [
    {"text": "اللقب", ...},     # Bloc label
    {"text": "براهمي", ...}      # Bloc valeur
]
```

**Syntaxe regex :**
```
(اللقب)        ← Groupe 1 : capture le label
\s*            ← Zéro ou plus espaces
[:\u061A\u061B]?  ← Optionnel : ':' ou ':' arabe ou ';' arabe
\s*            ← Zéro ou plus espaces
(.+)           ← Groupe 2 : capture tout le reste
```

---

## 15. Détection de la date

```python
    date_naissance_labels = ["تاريخ المildad", "date de naissance", "born on"]

    # 3 formats de date supportés
    date_pattern_dmy = re.compile(r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b')     # DD/MM/YYYY
    date_pattern_dot_dmy = re.compile(r'\b(\d{2}\.\d{2}\.\d{4})\b')     # DD.MM.YYYY
    date_pattern_dot_ymd = re.compile(r'\b(\d{4}\.\d{2}\.\d{2})\b')     # YYYY.MM.DD
```

### Stratégie 1 : Par label

```python
    _sorted = sorted(_expanded, key=lambda b: b["bbox"][0][1])  # Trier par Y

    for i, block in enumerate(_sorted):
        text_lower = block["text"].lower().strip()
        if any(lab in text_lower for lab in date_naissance_labels):
            # Trouver le label "تاريخ الميلad" → chercher la date juste en dessous
            for j in range(i + 1, len(_sorted)):
                next_text = _sorted[j]["text"].strip()
                for dp in all_date_patterns:
                    dm = dp.search(next_text)
                    if dm:
                        extracted.date_naissance = dm.group(1)
                        date_found = True
                        break
                if date_found:
                    break
                # Arrêter si on atteint un autre label "تاريخ" ou "date"
                if any(lab in next_text.lower() for lab in ["تاريخ", "date"]):
                    break
            break
```

**Logique :**
```
Bloc triés par Y :
  y=100: "رقم التعريف الوطني"     ← label NIN
  y=130: "100060581038770008"     ← NIN
  y=200: "تاريخ الميلاد"          ← ← on trouve ce label
  y=230: "2006.05.25"             ← ← on prend cette date
  y=260: "تاريخ الاستخراج"        ← ← on s'arrête ici (autre label "تاريخ")
```

### Stratégie 2 : Fallback par date la plus ancienne

```python
    if not date_found:
        ymd_dates = []
        for block in blocks:
            dm = date_pattern_dot_ymd.search(block["text"])
            if dm:
                ymd_dates.append(dm.group(1))
        if ymd_dates:
            ymd_dates.sort()              # Trier chronologiquement
            extracted.date_naissance = ymd_dates[0]  # La plus ancienne = date de naissance
```

**Logique :** Sur une CNI, il y a 3 dates :
- `2006.05.25` → date de naissance (la plus ancienne)
- `2021.05.31` → date d'issue
- `2026.05.30` → date d'expiration (la plus récente)

La date de naissance est toujours la plus ancienne → `sort()` + `dates[0]`.

### Stratégie 3 : Premier match dans le texte brut

```python
        else:
            for dp in all_date_patterns:
                dm = dp.search(extracted.raw_text)
                if dm:
                    extracted.date_naissance = dm.group(1)
                    break
```

---

## 16. Détection nom/prénom/lieu

### Listes de labels

```python
    nom_labels = [
        "اللقب", "للقب",              # Arabe + OCR misread
        "اسم العائلة", "الاسم العائli",
        "nom", "surname",             # Français
        "nom de famille", "family name",
    ]
    prenom_labels = [
        "الاسم", "الإسم", "للاسم",    # Arabe + variants
        "الاسم الشخصي",
        "prenom", "prénom",            # Français
        "first name", "prénoms",
    ]
    lieu_labels = [
        "مكان الميلاد", "محل الميلاد", "بلدية الميلاد",
        "lieu de naissance", "born in", "birth place", "né à",
    ]
```

### Détection par label

```python
    for i, block in enumerate(sorted_blocks):
        text = block["text"].strip()
        text_lower = text.lower()

        is_nom_label = any(kw.lower() in text_lower for kw in nom_labels)
        is_prenom_label = any(kw.lower() in text_lower for kw in prenom_labels)
        is_lieu_label = any(kw.lower() in text_lower for kw in lieu_labels)

        if is_nom_label or is_prenom_label or is_lieu_label:
            # Chercher le bloc suivant comme valeur
            for j in range(i + 1, len(sorted_blocks)):
                next_text = sorted_blocks[j]["text"].strip()

                # Sauter si le bloc suivant est aussi un label
                is_next_label = False
                for kw_list in [nom_labels, prenom_labels, lieu_labels]:
                    if any(kw.lower() in next_text.lower() for kw in kw_list):
                        is_next_label = True
                        break

                if is_next_label or len(next_text) < 2:
                    continue    # Sauter ce bloc, essayer le suivant

                # Assigner au bon champ
                if is_nom_label and extracted.nom is None:
                    extracted.nom = next_text
                    break
                elif is_prenom_label and extracted.prenom is None:
                    extracted.prenom = next_text
                    break
                elif is_lieu_label and extracted.lieu_naissance is None:
                    extracted.lieu_naissance = next_text
                    break
```

**Logique :**
```
Bloc: "اللقب" → is_nom_label = True
  Suivant: "براهيمي" → pas un label, len > 2 → nom = "براهيمي"
```

---

## 17. Fallback par position

Si les labels ne sont pas détectés, on utilise la **position physique** sur la carte :

```python
    if extracted.nom is None and extracted.prenom is None:
        # Trouver le NIN (bloc avec >10 chiffres)
        nin_bbox = None
        for block in blocks:
            cleaned = re.sub(r'\s+', '', block["text"])
            if re.search(r'\d{10,}', cleaned):
                nin_bbox = block["bbox"]
                break

        if nin_bbox is not None:
            nin_bottom = max(p[1] for p in nin_bbox)  # Y inférieur du NIN
            nin_left = min(p[0] for p in nin_bbox)    # X gauche du NIN

            below_blocks = []
            for block in expanded_blocks:
                text = block["text"].strip()
                bbox = block["bbox"]
                block_top = min(p[1] for p in bbox)   # Y supérieur du bloc

                # Filtres : sous le NIN, 2-25 caractères, arabe, pas de chiffres
                if (block_top > nin_bottom                    # Sous le NIN
                    and block_top - nin_bottom < 200          # Pas trop loin (< 200px)
                    and len(text) >= 2 and len(text) <= 25    # Taille raisonnable
                    and any('\u0600' <= c <= '\u06FF' for c in text)  # Contient arabe
                    and not re.search(r'\d', text)):          # Pas de chiffres

                    # Vérifier que ce n'est pas un label
                    is_label = any(kw in text for kw in [
                        "اللقب", "الاسم", "رقم", "تاريخ", "مكان",
                        "التعريف", "الوطني", "الوطنية", "الرقم",
                        "بطاقة", "الميلاد", "الاستخراج", "الانتهاء",
                    ])
                    if not is_label:
                        below_blocks.append(block)

            if len(below_blocks) >= 2:
                extracted.nom = below_blocks[0]["text"].strip()     # 1er = nom
                extracted.prenom = below_blocks[1]["text"].strip()  # 2ème = prénom
```

**Disposition CNI :**
```
y=100: NIN (100060581038770008)
       ↓ nin_bottom = 160
y=180: "براهimi"  ← below_blocks[0] = nom
y=220: "طارق"     ← below_blocks[1] = prénom
```

---

## 18. Fallback par texte propre

Dernier recours si les 2 méthodes précédentes échouent :

```python
    if extracted.nom is None and extracted.prenom is None:
        skip_words = [
            "الجمهورية", "الجزائرية", "الشعبية",  # Mots-clés à ignorer
            "رخصة", "السباقة", "république",
            "permis", "conduire", "card", "identity",
            # ...
        ]

        def is_clean_name(text):
            t = text.strip()
            if len(t) < 2 or len(t) > 30:           # Trop court ou long
                return False
            if any(c.isdigit() for c in t):          # Contient des chiffres
                return False
            if any(kw.lower() in t.lower() for kw in skip_words):  # Mot-clé
                return False
            has_garbled = any(c in t for c in '!@#$%^&*()...')     # Caractères bizarres
            if has_garbled:
                return False

            has_latin = any('a' <= c.lower() <= 'z' for c in t)
            has_arabic = any('\u0600' <= c <= '\u06FF' for c in t)

            if has_latin and not has_arabic:    # Texte latin pur
                words = t.split()
                for w in words:
                    if len(w) > 2:
                        if not (w == w.upper() or w == w.lower()):  # Majuscules/mixte
                            return False
                return True
            if has_arabic and not has_latin:    # Texte arabe pur
                return True
            return False

        # Chercher d'abord des noms latins (plus fiables)
        latin_blocks = [b for b in sorted_blocks
                       if is_clean_name(b["text"])
                       and any('a' <= c.lower() <= 'z' for c in b["text"])
                       and not any('\u0600' <= c <= '\u06FF' for c in b["text"])]

        if len(latin_blocks) >= 2:
            extracted.nom = latin_blocks[0]["text"]      # 1er bloc latin = nom
            extracted.prenom = latin_blocks[1]["text"]   # 2ème = prénom
        else:
            text_blocks = [b for b in sorted_blocks if is_clean_name(b["text"])]
            if len(text_blocks) >= 2:
                extracted.nom = text_blocks[0]["text"]
                extracted.prenom = text_blocks[1]["text"]
```

**Filtres de `is_clean_name` :**
- Pas de chiffres
- Pas de mots-clés (république, carte, identité...)
- Pas de caractères bizarres (!@#$%...)
- Latin : doit être TOUT MAJUSCULES ou tout minuscules (ex: "BRAHIMI" ou "brahimi")
- Arabe : accepté tel quel

---

## 19. ROI Refinement (re-read)

Après avoir trouvé tous les champs, on **relit** chaque zone avec un allowlist pour corriger les erreurs OCR :

### NIN refinement
```python
    if extracted.nin:
        for block in blocks:
            block_cleaned = re.sub(r'\s+', '', block["text"])
            if extracted.nin in block_cleaned or re.search(r'\d{10,}', block_cleaned):
                # Extraire la zone (ROI)
                bbox = block["bbox"]
                x1 = int(min(p[0] for p in bbox))
                y1 = int(min(p[1] for p in bbox))
                x2 = int(max(p[0] for p in bbox))
                y2 = int(max(p[1] for p in bbox))
                roi = rgb_image[y1:y2, x1:x2]

                if roi.size > 0:    # Vérifier que le crop n'est pas vide
                    cleaned = extraire_texte_roi(self.rec_fr, roi, allowlist='0123456789')
                    if cleaned and len(cleaned) >= 16:
                        extracted.nin = cleaned[:18]   # Tronquer à 18 chiffres
                break
```

### Date refinement
```python
    if extracted.date_naissance:
        for block in blocks:
            if extracted.date_naissance in block["text"]:
                # Même logique : extraire ROI → re-read avec allowlist='0123456789/.'
                # Le '.' est crucial pour garder le séparateur YYYY.MM.DD
```

### Nom/prénom/lieu refinement
```python
    _label_prefixes = [
        "اللقب", "للقب", "الاسم", "الإسم", "للاسم",
        "مكان الميلاد", "تاريخ الميلad", "تاريخ الاستخراج", "تاريخ الانتهاء",
    ]

    for field_name in ["nom", "prenom", "lieu_naissance"]:
        field_value = getattr(extracted, field_name)
        if field_value:
            for block in blocks:
                if field_value.lower() in block["text"].lower():
                    # Extraire ROI
                    roi = rgb_image[y1:y2, x1:x2]
                    if roi.size > 0:
                        has_arabic = any('\u0600' <= c <= '\u06FF' for c in field_value)
                        rec_engine = self.rec_ar if has_arabic else self.rec_fr
                        cleaned = extraire_texte_roi(rec_engine, roi, is_num=False)

                        if cleaned:
                            # Strip le label si l'OCR l'a réintroduit
                            for prefix in _label_prefixes:
                                if cleaned.startswith(prefix) and len(cleaned) > len(prefix):
                                    candidate = cleaned[len(prefix):].lstrip(' :')
                                    if candidate:
                                        cleaned = candidate
                                        break
                            setattr(extracted, field_name, cleaned)
                    break
```

**Problème résolu :** Quand on re-lit la zone "براهيمي", l'OCR peut parfois retourner "اللقب براهمi" (le label est réapparu). Le strip enlève automatiquement le prefixe.

---

## 20. Fonction extract_fields

```python
_default_reader: Optional[CardOCR] = None    # Instance singleton

def extract_fields(image: np.ndarray, gpu: bool = False) -> ExtractedFields:
    global _default_reader
    if _default_reader is None:
        _default_reader = CardOCR(gpu=gpu)   # Créer une seule fois
    return _default_reader.extract(image)    # Réutiliser à chaque appel
```

**Pattern Singleton :**
- `_default_reader` est une variable globale
- Au premier appel : créer `CardOCR()` (charge les modèles ~5s)
- Aux appels suivants : réutiliser la même instance (instantané)
- Les modèles restent en mémoire RAM

---

## 21. Bloc __main__

```python
if __name__ == "__main__":
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])          # Charger l'image
        if img is not None:
            fields = extract_fields(img)        # Extraire les champs

            # Afficher le texte brut (AVANT refinement)
            raw_results = _default_reader._run_ocr(reader, rgb)
            for (bbox, text, conf) in raw_results:
                display_text = CardOCR.fix_arabic_display(text)
                print(f"  [{conf:.2f}] {display_text}")

            # Afficher les champs (APRÈS refinement)
            print(f"  NIN              : {fields.nin}")
            print(f"  Nom              : {CardOCR.fix_arabic_display(fields.nom)}")
            # ...
```

**Usage :**
```bash
python ocr_extractor.py "captures/national_id/card.jpg"
```

**Sortie :**
```
=== AVANT allowlist (raw predict) ===
  [1.00] 100060581038770008
  [0.90] 2021.05.31:7
  [0.97] 100060581038770008山
  [0.49] ull :
  [0.95] 2006.05.25:

=== APRÈS allowlist (refined fields) ===
  NIN              : 100060581038770008
  Nom              : براهيمي
  Prénom           : طارق
  Date naissance   : 2006.05.25
  Lieu naissance   : القبة
  Confiance        : 91.0%
```

---

## Résumé du flux complet

```
Image JPEG
  │
  ├─ [1] cvtColor(BGR→RGB)
  │
  ├─ [2] _run_ocr(ocr_ar)  →  blocs arabes [(bbox, text, conf), ...]
  │
  ├─ [3] _run_ocr(ocr_fr)  →  blocs français [(bbox, text, conf), ...]
  │
  ├─ [4] Fusion AR+FR (dedup par bbox < 30px, préférer chiffres)
  │       →  all_results = [(bbox, text, conf), ...]
  │
  ├─ [5] Créer blocks[] = [{text, confidence, bbox}, ...]
  │
  ├─ [6] _find_nin()  →  extracted.nin
  │       Méthode 1: regex \d{18} dans texte brut
  │       Méthode 2: bloc exact 18 chiffres
  │       Méthode 3: concat tous chiffres
  │       Méthode 4: tolérance 16-19
  │
  ├─ [7] _find_identity_fields()
  │       ├─ [7a] Séparation label:valeur (regex)
  │       │       "اللقب براهمimi" → "اللقب" + "براهمimi"
  │       │
  │       ├─ [7b] Détection date
  │       │       Stratégie 1: par label "تاريخ الميلاد"
  │       │       Stratégie 2: plus ancienne YYYY.MM.DD
  │       │       Stratégie 3: premier match dans texte
  │       │
  │       ├─ [7c] Détection nom/prénom/lieu
  │       │       Par label: "اللقب" → bloc suivant = nom
  │       │       Par position: blocs sous le NIN
  │       │       Par texte propre: latin/arabe clean
  │       │
  │       └─ [7d] Extraire expanded_blocks + sorted_blocks
  │
  ├─ [8] ROI Refinement
  │       NIN: re-read avec allowlist='0123456789'
  │       Date: re-read avec allowlist='0123456789/.'
  │       Nom/Prenom: re-read + strip label prefix
  │
  └─ [9] Return ExtractedFields
          {nin, nom, prenom, date_naissance, lieu_naissance}
```
