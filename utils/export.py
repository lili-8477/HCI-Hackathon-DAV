"""
Export Utility
Packages the full analysis session (conversation text + figures) into a
downloadable ZIP containing a Markdown report and all plot images.
"""

import io
import zipfile
from datetime import datetime


def build_export_zip(messages: list) -> bytes:
    """
    Build a ZIP file from the conversation history.

    The ZIP contains:
      - report.md   – full conversation formatted as Markdown
      - figures/     – every plot image as a numbered PNG

    Args:
        messages: list of message dicts with keys 'role', 'content', 'plots'

    Returns:
        ZIP file as bytes, ready for st.download_button
    """
    buf = io.BytesIO()
    plot_counter = 0

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # --- Build Markdown report ---
        lines = []
        lines.append("# Data Analysis Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            plots = msg.get("plots", [])

            if role == "user":
                lines.append(f"## User\n\n{content}\n")
            else:
                lines.append(f"## Assistant\n\n{content}\n")

            # Reference each plot image in the markdown
            for img_bytes in plots:
                plot_counter += 1
                fname = f"figure_{plot_counter}.png"
                lines.append(f"\n![Figure {plot_counter}](figures/{fname})\n")
                # Write image into ZIP
                zf.writestr(f"figures/{fname}", img_bytes)

            lines.append("---\n")

        zf.writestr("report.md", "\n".join(lines))

    buf.seek(0)
    return buf.getvalue()
