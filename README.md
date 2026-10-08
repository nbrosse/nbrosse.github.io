# nbrosse.github.io

Source of [nbrosse.github.io](https://nbrosse.github.io), my personal research site, built
with [Quarto](https://quarto.org).

| page | file |
|---|---|
| Home | `index.qmd` — positioning, latest posts; carries the RSS feed (`/index.xml`) |
| Research | `research.qmd` — Human–AI Mathematics, publications and preprints |
| Projects | `projects.qmd` — code, tools and demos |
| Blog | `blog.qmd` — every post, filterable by category |
| About | `about.qmd` — bio, contact |
| CV | `cv.pdf` — public version of the CV, copied from the LaTeX source (declared in `project.resources`) |

Posts live in `posts/<slug>/`.

## Build

```bash
quarto preview            # local preview on port 1234
quarto render             # build into _site/
quarto publish gh-pages   # deploy
```
