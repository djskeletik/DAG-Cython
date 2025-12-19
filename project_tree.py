from pathlib import Path

def print_tree(path: Path, prefix=""):

    if not path.exists():
        print(f"{path} не существует")
        return

    children = list(path.iterdir())
    pointers = ["├── "] * (len(children) - 1) + ["└── "] if children else []

    for pointer, child in zip(pointers, children):
        print(prefix + pointer + child.name)
        if child.is_dir():
            extension = "│   " if pointer == "├── " else "    "
            print_tree(child, prefix + extension)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    print(project_root.name)
    print_tree(project_root)
