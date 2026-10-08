# TODO — repositionner le site

Objectif : que le site présente un **chercheur senior** (et, à terme, un futur manager),
pas seulement un carnet de notes. Un recruteur doit comprendre en 30 secondes qui tu es,
sur quoi tu travailles et ce que tu as produit.

Ordre de priorité : 1 → 5.

## 1. Page Research / Publications (`research.qmd`)

Le manque le plus important : aucune trace des publications sur le site.

- [x] Créer `research.qmd` et l'ajouter à la navbar (`_quarto.yml`).
- [x] Lister les articles avec lien arXiv/journal + code :
  - The Tamed Unadjusted Langevin Algorithm → repo `TULA`
  - Normalizing constants of log-concave densities → repo `normalizingconstant`
  - Diffusion approximations and control variates for MCMC → repo `controlvariates`
  - SGLD vs SGD → repo `sgld-sgd`
  - Uncertainties in deep neural networks → repo `uncertainties`
  - Thèse de doctorat (DOI theses.fr)
- [x] Venues vérifiées : SPA 2019, EJS 2018, NeurIPS 2018, COLT 2017,
  Comput. Math. Math. Phys. 2024 (control variates, version publiée avec Samsonov),
  chapitre Wiley 2024 (Iktos). Last-layer reste en preprint (aucune version publiée trouvée).
- [ ] Ajouter d'éventuels travaux GenBio / MBZUAI (rien trouvé à ton nom sur bioRxiv).
- [x] Section « Preprints » : boundary-layer asymptotics (arXiv:2607.04514).
- [x] EDM error propagation (arXiv:2610.10190) : entrée Research, post publié, dépôt de code
  public et lié depuis Projects.
- [ ] Vérifier la règle AISTATS sur la publicité pendant la relecture (post EDM en ligne).
- [x] Section « Human–AI Mathematics » : framework, template et manuscrit KLS (preuves
  vérifiées par des agents, pas encore relues par des humains).
- [x] Courte intro (2–3 lignes) sur les axes : modèles génératifs, diffusion / flow
  matching, méthodes MCMC et leur analyse.
- [ ] Option : générer la liste depuis un `.bib` (Quarto le gère) plutôt qu'à la main.

## 2. Page d'accueil repositionnée (`index.qmd`)

Aujourd'hui : un tableau de posts triés par date, sujets mélangés.

- [x] En tête : 3–4 lignes de positionnement (qui, quoi, où), photo, liens.
- [x] Bloc « Selected work » : 3 éléments maximum
  (ex. série flow matching, boundary layer, un article).
- [x] Puis « Latest posts » (listing limité à ~5, type `default` ou `grid` plutôt que `table`).
- [x] Déplacer le listing complet sur une page `blog.qmd`.
- [x] Renommer le titre du site : « Nicolas' Notebook » → nom propre + rôle
  (ex. « Nicolas Brosse — Generative modeling »). Le blog peut garder « Notebook ».

## 3. Page About (`about.qmd`)

- [x] Mettre **Short Bio** en premier, « Why I Write » en dernier (ou le supprimer).
- [x] Supprimer « If you find my work interesting, please consider citing it ».
- [x] Reformuler l'épisode 2025 (venture + freelance) comme expérience de terrain :
  livraison de bout en bout, relation client, cadrage de projets.
- [ ] Ajouter une section **Mentoring, teaching & talks** (encadrement de stagiaires
  ou de doctorants, enseignements, exposés, reviewing). C'est le principal signal
  « management » qui manque.
- [x] Ajouter un lien vers un **CV PDF** (`cv.pdf` à la racine, lien dans la navbar).
  À chaque mise à jour du CV : recopier `personal-notes/cv/build/cv-2026-clean-public.pdf`
  (jamais la version `nophoto`, qui contient adresse et téléphone).
- [x] Ajouter un moyen de contact (email ou formulaire).

## 4. Page Projects / Software (`projects.qmd`)

- [x] Créer la page et l'ajouter à la navbar.
- [x] Section « AI for mathematics » : Human–AI Mathematics, kls-conjecture-search, latexfmt.
- [x] Une entrée par projet (liste simple plutôt que des cartes) : nom, une phrase, ce que ça montre, lien repo / post.
  - **latexfmt** : formateur LaTeX sémantique (outillage de recherche, tests golden,
    idempotence, rollback). Voir `postdoc/latexfmt/TODO.md`.
  - **haiku-shunt** : délégation de lectures à un petit modèle, évaluée par A/B.
    Voir `haiku-shunt/TODO.md`. **Pas encore sur la page : le repo est privé.**
  - **flow-matching-notes** : notebooks des posts flow matching.
  - pdf-parsing (7 ⭐), pdf-rag, mhcpred, raman-spectra, llm-observability,
    llm-slide-deck, plus les démos HF Spaces (`pdf-parsing-demo`,
    `pdf-rag-metadata-demo`).
- [ ] Option : publier une partie de `postdoc/llm-lab` (synthèse sur l'entraînement
  des LLM « écrite pour des postdocs ») : c'est du contenu de transmission, donc un bon
  signal de leadership.

## 5. Blog : structure et posts

- [x] Séparer en deux rubriques : **Research notes** (flow matching, boundary layer,
  posterior concentration, Uni-Mol) et **Engineering notes** (PDF parsing, RAG,
  observability, slides, Raman, MHC).
- [x] Harmoniser les catégories (ex. `pdf-parsing` est en « deep learning » ;
  « machine learning » et « deep learning » coexistent).
- [x] Uniformiser le format des dates dans le front-matter (`llm-slides` n'a pas de
  guillemets, les autres si).
- [x] Nouveau post : haiku-shunt, présenté comme un retour sur la méthode d'évaluation
  des agents de code (voir `haiku-shunt/TODO.md`).
- [x] Nouveau post : Human–AI Mathematics et la recherche sur KLS.
- [ ] Nouveau post (optionnel) : latexfmt (voir `postdoc/latexfmt/TODO.md`).

## Divers

- [x] Ajouter une image `og:image` / Twitter card par défaut (`website.open-graph`,
  `website.twitter-card` dans `_quarto.yml`) pour les partages via `nbrosse-social`.
- [x] Mettre à jour `README.md` (actuellement générique).
- [x] Vérifier le rendu mobile de la page d'accueil après refonte.
