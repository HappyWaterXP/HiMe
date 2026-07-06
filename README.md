# HiMe Demo

This website uses the Nerfies project page template:

https://github.com/nerfies/nerfies.github.io

We thank the Nerfies authors for releasing the template.

## Files

- `index.html`
- `styles.css`
- `script.js`
- `serve.py`
- `HiMe/example_paper.pdf`
- `HiMe/imgs/intro.png`
- `HiMe/imgs/hime.png`
- `HiMe/imgs/hime_exp3.png`
- `HiMe/imgs/setting.png`
- `HiMe/videos/search-0706.mp4`
- `HiMe/videos/counting-0706.mp4`
- `HiMe/videos/rearrangement-0706.mp4`

## Local Preview

Use the included server for local preview because it supports HTTP Range
requests required by video seeking:

```bash
python3 serve.py --port 8000
```

Then open:

```text
http://localhost:8000/index.html
```
