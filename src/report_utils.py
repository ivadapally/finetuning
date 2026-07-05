"""Small helper shared by the three notebooks to auto-fill the markdown
evaluation reports in reports/. Kept dependency-free (stdlib only) so it
works inside a fresh Colab runtime without extra installs.
"""
import re
from pathlib import Path


def fill_report_section(md_path, marker, content):
    """Replace everything between the AUTO:<marker> START/END comments in
    md_path with `content`. Creates the markers at the end of the file if
    they don't exist yet, so this is safe to call even on a hand-edited file.
    """
    md_path = Path(md_path)
    text = md_path.read_text(encoding="utf-8")
    start = f"<!-- AUTO:{marker}:START -->"
    end = f"<!-- AUTO:{marker}:END -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    replacement = f"{start}\n{content}\n{end}"
    if pattern.search(text):
        new_text = pattern.sub(replacement, text)
    else:
        new_text = text.rstrip() + "\n\n" + replacement + "\n"
    md_path.write_text(new_text, encoding="utf-8")


def make_markdown_table(headers, rows):
    """Build a GitHub-flavored markdown table from a header list and rows
    (list of lists/tuples). Cell values are stringified and pipe characters
    / newlines are escaped so a long model answer can't break the table.
    """
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [str(c).replace("\n", " ").replace("|", "\\|") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
