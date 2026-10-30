# Troubleshooting

## Red underlines in VS Code: `'tensor.h' file not found`

**Date:** 2026-10-29

### Symptom
`src/tensor.cpp` is covered in red. The first error is `'tensor.h' file not found`,
followed by dozens of `Use of undeclared identifier 'Tensor'`.

### How to tell it's the editor, not the compiler
The errors come from **clangd** (VS Code language server), not from `clang++`.
Check with the real compiler:

```bash
clang++ -std=c++17 -Iinclude -fsyntax-only src/tensor.cpp
```

If this finds `tensor.h`, the compiler is fine and only clangd is misconfigured.

### Cause
clangd learns include paths (`-Iinclude`) from `build/compile_commands.json`.
That file didn't exist because:

1. `pybind11` was not installed in `.venv`, so `find_package(pybind11)` in
   `CMakeLists.txt` failed and CMake never finished configuring
   (`pybind11_DIR-NOTFOUND` in `build/CMakeCache.txt`).
2. `CMAKE_EXPORT_COMPILE_COMMANDS` was not turned on.

### Fix
Run these from the project root (one command per line, **no inline `#` comments**,
see the gotcha below):

```bash
uv pip install --python .venv/bin/python pybind11
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
```

Then restart clangd: **Cmd+Shift+P → "clangd: Restart language server"**.
clangd won't pick up a new `compile_commands.json` for files that are already
open until it restarts.

Quick alternative if you don't want to run CMake: create `.clangd` in the repo root:

```yaml
CompileFlags:
  Add: [-Iinclude, -std=c++17]
```

### Gotcha: `#` comments in zsh
Interactive zsh does **not** treat `#` as a comment by default. Running
`uv pip install pybind11   # some note` passes `#`, `some`, and `note` to `uv`,
which fails with `Failed to parse: '#'`. Remove the comment, or enable comments
with `setopt interactive_comments` in `~/.zshrc`.

### Known leftover: CMake picks the wrong Python
CMake found `/usr/local/bin/python3` (3.13) instead of `.venv` (3.12). This doesn't
affect the editor, but a module built from `build/` won't import in the 3.12 venv.
Fix: add `set(PYBIND11_FINDPYTHON ON)` before `find_package(pybind11 ...)` in
`CMakeLists.txt`, or configure with `-DPython_EXECUTABLE=.venv/bin/python`.

### After the fix
Some red will remain, and those are **real** code errors from the in-progress
`TensorImpl` refactor:
- `tensor.cpp` still uses `shape`, `data`, `requires_grad`, `grad_fn` directly.
  These now live in `TensorImpl` and must be accessed as `impl->shape`, etc.
  Constructors must create it first: `impl = std::make_shared<TensorImpl>();`
- `autograd.h`: subclasses declare `backward(Tensor&) override` but the base
  declares `backward(TensorImpl&)`. Make the types match.
- `GradFn::inputs` is now `vector<shared_ptr<TensorImpl>>`, so
  `fn->inputs = {this, &n}` must become `{impl, n.impl}`.
