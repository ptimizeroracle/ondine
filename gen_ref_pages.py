"""Generate API reference pages from Python modules."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Modules to include in API reference
root = Path(__file__).parent.parent
src = root / "ondine"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    # Skip __pycache__ and test files
    if "__pycache__" in parts or "test_" in path.name:
        continue

    # Handle __init__.py files
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    # Skip __main__.py files
    elif parts[-1] == "__main__":
        continue

    # Add to navigation
    nav[parts] = doc_path.as_posix()

    # Generate the API reference page
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# {parts[-1] if parts else 'ondine'}\n\n")
        fd.write(f"::: {ident}\n")

    # Set up edit path
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write the navigation structure
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
