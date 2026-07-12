# Submodules

This project depends on several third-party repositories. Most of them ship their own
`setup.py` or `pyproject.toml` and are installed directly via pip extras declared in
the root `pyproject.toml` (e.g., `pip install ".[lm_eval]"`).

However, the three repositories below are **research codebases that lack Python packaging
metadata**. We maintain custom `setup.py` wrappers in `_packaging/` so they can be
pip-installed after a one-time setup.

## Custom-Packaged Submodules

| Submodule | Package | Purpose | Upstream |
|-----------|---------|---------|----------|
| CharXiv | `charxiv` | Chart understanding evaluation for multimodal LLMs | https://github.com/princeton-nlp/CharXiv |
| Tar | `ta_tok` | Text-aligned visual tokenizer (TaTok, NeurIPS 2025) | https://github.com/csuhan/Tar |
| VoiceBench | `voice_bench` | Voice assistant evaluation benchmark | https://github.com/MatthewCYM/VoiceBench |

### Installing

Each module follows the same three steps:

```bash
git submodule update --init submodules/CharXiv
cp -r submodules/_packaging/CharXiv/* submodules/CharXiv/
pip install submodules/CharXiv
```

Repeat for `Tar`, `VoiceBench`.

### Updating

```bash
cd submodules/CharXiv
git fetch origin && git checkout origin/main
cd ../..
cp -r submodules/_packaging/CharXiv/* submodules/CharXiv/
pip install submodules/CharXiv
git add submodules/CharXiv
git commit -m "chore: update CharXiv submodule"
```

Repeat for `Tar`, `VoiceBench`.

### What's in `_packaging/`?

Minimal `setup.py` files that map each repository's source directory to a pip-installable
package name (e.g., `src/` → `charxiv`).
