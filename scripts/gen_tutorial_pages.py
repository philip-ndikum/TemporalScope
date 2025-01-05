# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Generate tutorial pages from Jupyter notebooks."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Get root directory and tutorials directory
root = Path(__file__).parent.parent
tutorials_dir = root / "tutorial_notebooks"

if not tutorials_dir.exists():
    raise Warning(f"Directory {tutorials_dir} does not exist. Please run this script from the root of the repository.")

for path in sorted(tutorials_dir.rglob("*.ipynb")):
    # Skip checkpoint files
    if ".ipynb_checkpoints" in str(path):
        continue

    # Create the doc path relative to the tutorials directory - keep .ipynb extension
    doc_path = path.relative_to(tutorials_dir)
    full_doc_path = Path("tutorials", doc_path)

    # Create parts for navigation - this will create the hierarchy in mkdocs
    parts = tuple(["Walkthroughs"] + list(path.relative_to(tutorials_dir).with_suffix("").parts))

    # Add to navigation - use .ipynb extension
    nav[parts] = doc_path.as_posix()

    # Generate the notebook file in the docs directory
    with mkdocs_gen_files.open(full_doc_path, "wb") as fd, open(path, "rb") as source_file:
        # Write the content of the source file
        fd.write(source_file.read())

    # Set edit path
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write navigation file
with mkdocs_gen_files.open("tutorials/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
