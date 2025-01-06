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
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied.  See the License for
# the specific language governing permissions and limitations
# under the License.

"""Generate tutorial pages from Jupyter notebooks."""

from pathlib import Path

import mkdocs_gen_files
import nbformat

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

    # Read the notebook
    with open(path, encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Create a new markdown cell with the source note using HTML
    # Create the full URL for the notebook
    repo_url = "https://github.com/philip-ndikum/TemporalScope/blob/main"
    notebook_relative_path = path.relative_to(root).as_posix()  # Convert path to POSIX style for URLs
    notebook_name = path.name  # Extracts just the notebook file name
    source_url = f"{repo_url}/{notebook_relative_path}"

    # Create the source note
    source_note = (
        '<div class="admonition info">\n'
        '    <p class="admonition-title">Info</p>\n'
        "    <p>This tutorial was auto-generated from the TemporalScope repository.</p>\n"
        "    <p>If you would like to suggest enhancements or report issues, please submit a Pull Request following the contribution guidelines.</p>\n"
        f'    <p>Source notebook: <a href="{source_url}" target="_blank">{notebook_name}</a></p>\n'
        "</div>\n"
    )

    # Create a copyright notice and warning
    disclaimer_notice = (
        '<div class="admonition danger">\n'
        '    <p class="admonition-title">Disclaimer & Copyright</p>\n'
        '    <p>THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE '
        "WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR "
        "COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, "
        "ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.</p>\n"
        "    <p>THIS SOFTWARE IS INTENDED FOR ACADEMIC AND INFORMATIONAL PURPOSES ONLY. IT SHOULD NOT BE USED IN PRODUCTION ENVIRONMENTS "
        "OR FOR CRITICAL DECISION-MAKING WITHOUT PROPER VALIDATION. ANY USE OF THIS SOFTWARE IS AT THE USER'S OWN RISK.</p>\n"
        "<hr/>\n"
        "<p>&copy; 2024 Philip Ndikum</p>\n"
        "</div>\n"
    )

    # Add source note and disclaimer cells
    new_source_cell = nbformat.v4.new_markdown_cell(source_note)
    new_disclaimer_cell = nbformat.v4.new_markdown_cell(disclaimer_notice)
    notebook.cells.append(new_source_cell)
    notebook.cells.append(new_disclaimer_cell)

    # Generate the modified notebook file in the docs directory
    with mkdocs_gen_files.open(full_doc_path, "w", encoding="utf-8") as fd:
        nbformat.write(notebook, fd)

    # Set edit path
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write navigation file
with mkdocs_gen_files.open("tutorials/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
