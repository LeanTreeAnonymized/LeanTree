## 🚀 Installation

Clone the repository including submodules:

```bash
git clone --recurse-submodules https://github.com/LeanTreeAnonymized/LeanTree.git
```

Ensure [Lean 4.19](https://docs.lean-lang.org/lean4/doc/setup.html) is installed via elan.

```bash
lake --version
```

At the moment, LeanTree only supports Lean 4.19.

Make sure `pip` is installed.
Then, run:

```bash
make install
```

Alternatively, use Poetry explicitly:

```bash
pip install poetry
poetry install
```

For running tests or experiments, refer to the [Development](#development) section.

## 📘 Basic Usage

> **Note:** You must create a Lean project to use Lean.
> See [Lean project guide](https://leanprover-community.github.io/install/project.html) for more information.
 
Start by creating or loading a Lean project:

```python
from leantree import LeanProject

project = LeanProject.create("path/to/project")
# or load an existing one:
project = LeanProject("path/to/project")
```

The created project includes Mathlib by default.
If no path is provided in `LeanProject.create`, the project will be created in or loaded from the `leantree_project` subdirectory of the current directory.

### 🔍 Starting a Proof

Using the environment, you can initialize a proof search by supplying a theorem with one or more `sorry` keywords.

```python
with project.environment() as env:
    env.send_command("import Mathlib\nopen BigOperators Real Nat Topology Rat")
    branch = env.proof_from_sorry("theorem succ_less_double_succ (n : Nat) : n > 0 → n < 2 * n := by sorry")
    zero, succ = branch.apply_tactic("cases n")
    print("Factorized proof states after `cases n`:")
    print(zero.state)
    print(succ.state)
    assert not zero.is_solved
    ...
```

### ⚡ Async API

```python
async with project.environment() as env:
    await env.send_command_async("import Mathlib\nopen BigOperators Real Nat Topology Rat")
    branch = await env.proof_from_sorry_async("theorem succ_less_double_succ (n : Nat) : n > 0 → n < 2 * n := by sorry")
    zero, succ = await branch.apply_tactic_async("cases n")
    ...
```

### 📂 Data Extraction

Using the project, you can parse a Lean file and build all proof trees.
Then, you can use the environment to start a proof for each tactic block in the file.

```python
file = project.load_file("Example.lean")

for thm in file.theorems:
    print(thm.load_source() + "\n")
    for by_block in thm.by_blocks:
        print(by_block.tree.pretty_print())
    print("-" * 100)

for thm, branch in env.file_proofs(file):
    if isinstance(branch, Exception):
        print(f"Could not start theorem '{thm}' due to exception: {branch}")
    ...
```

### 📊 Dataset Generation

You can easily generate a dataset containing all Lean files in a directory.
For production-ready dataset generation, see the [Datasets section](#datasets)

```python
import glob
import json

with open("dataset.json", "w") as f:
    for path in glob.glob('**/*.lean', recursive=True):
        file = project.load_file(path)
        f.write(json.dumps(file.serialize()) + "\n")
```

### 💾 Save/Restore Environment State

The environment state can be saved to disk and restored later.

```python

with project.environment() as env:
    env.send_command("import Mathlib\nopen BigOperators Real Nat Topology Rat")
    env.pickle("env.pkl")

with project.environment() as env:
    env.unpickle("env.pkl")
    branch = env.proof_from_sorry("theorem succ_less_double_succ (n : Nat) : n > 0 → n < 2 * n := by sorry")
    zero, succ = branch.apply_tactic("cases n")
    ...
```

## 📁 Datasets

Assuming you have already create a Lean project in `leantree_project`, you can recreate the whole Mathlib dataset by running:

```bash
python dataset/tree_dataset.py generate --project_path leantree_project --source_files mathlib/Mathlib
```

## 🛠️ Development

Install all development and experiments dependencies:

```bash
make install-dev
```

### ✅ Running Tests

To run tests, first create a Lean 4.19 project called `leantree_project` in the LeanTree directory.
You can use LeanTree for that - in the leantree directory, run the following Python code:

```python
from leantree import LeanProject

project = LeanProject.create()
```

After the project is created, run:

```bash
make test
```

### 🐞 Debugging Tips

When working with a Lean environment, you can use `env.take_control()` to debug the underlying Lean REPL.
This method connects your stdin/stdout to the REPL's stdin/stdout.

---

## 🧰 Related Tools

* **[LeanDojo](https://github.com/lean-dojo)**
* **[Pantograph](https://github.com/stanford-centaur/PyPantograph)**

> For a detailed comparison, refer to the LeanTree paper.
