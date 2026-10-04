# nbrosse.github.io

Source of [nbrosse.github.io](https://nbrosse.github.io), my personal research site, built
with [Quarto](https://quarto.org).

| page | file |
|---|---|
| Home | `index.qmd` — positioning, selected work, latest posts; carries the RSS feed (`/index.xml`) |
| Research | `research.qmd` — publications and preprints |
| Projects | `projects.qmd` — code, tools and demos |
| Notebook | `blog.qmd` — every post, split into research and engineering notes |
| About | `about.qmd` |

Posts live in `posts/<slug>/`. Each post sets `post-type: research` or
`post-type: engineering` in its front-matter, which decides where it appears on the
Notebook page.

## Build

```bash
quarto preview            # local preview on port 1234
quarto render             # build into _site/
quarto publish gh-pages   # deploy
```
