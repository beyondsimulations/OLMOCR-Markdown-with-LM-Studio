#!/usr/bin/env python3
"""
olmOCR 2 with LM Studio for macOS

Convert PDFs to markdown using olmOCR 2 running locally via LM Studio.
See README.md for full documentation and setup instructions.

Quick start:
    1. Install LM Studio and download an olmOCR GGUF model
    2. Start the LM Studio server (Developer tab)
    3. uv pip install olmocr openai pypdf tqdm python-dotenv
    4. cp .env.example .env  # Edit with your model name
    5. python olmocr_lmstudio.py input.pdf -o output.md
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

# Check all dependencies upfront with helpful messages
_missing = []
try:
    from dotenv import load_dotenv
except ImportError:
    _missing.append("python-dotenv")
try:
    from openai import OpenAI
except ImportError:
    _missing.append("openai")
try:
    from pypdf import PdfReader
except ImportError:
    _missing.append("pypdf")
try:
    from tqdm import tqdm
except ImportError:
    _missing.append("tqdm")
try:
    from olmocr.pipeline import build_page_query
except ImportError:
    _missing.append("olmocr")

if _missing:
    print(f"Error: Missing dependencies. Run:\n  uv pip install {' '.join(_missing)}")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()


# Default configuration (can be overridden by .env or command line)
DEFAULT_SERVER = os.getenv("OLMOCR_SERVER", "http://localhost:1234/v1")
DEFAULT_MODEL = os.getenv("OLMOCR_MODEL", "allenai_olmocr-2-7b-1025")
DEFAULT_TIMEOUT = int(os.getenv("OLMOCR_TIMEOUT", "120"))


def create_client(
    base_url: str | None = None,
    api_key: str = "lm-studio",
    timeout: int | None = None,
) -> OpenAI:
    """Create OpenAI client configured for LM Studio."""
    return OpenAI(
        base_url=base_url or DEFAULT_SERVER,
        api_key=api_key,
        timeout=timeout or DEFAULT_TIMEOUT,
    )


def get_loaded_model(client: OpenAI) -> str:
    """Get the currently loaded model from LM Studio."""
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception:
        pass
    return "local-model"


def convert_latex_delimiters(text: str) -> str:
    r"""Convert LaTeX delimiters to common markdown format.

    \(...\) → $...$  (inline)
    \[...\] → $$...$$ (block)
    """
    # Block math first (to avoid nested replacements)
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    # Inline math
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    return text


def convert_html_tables_to_markdown(text: str) -> str:
    """Convert HTML tables to markdown tables.

    Note: colspan/rowspan are simplified (content duplicated or merged).
    """
    def parse_table(table_html: str) -> str:
        rows = []
        # Extract rows
        for tr_match in re.finditer(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL):
            row_html = tr_match.group(1)
            cells = []
            is_header = False
            # Extract cells (th or td)
            for cell_match in re.finditer(r'<(th|td)[^>]*>(.*?)</\1>', row_html, re.DOTALL):
                tag, content = cell_match.groups()
                if tag == 'th':
                    is_header = True
                # Clean content: strip tags, normalize whitespace
                content = re.sub(r'<[^>]+>', '', content)
                content = ' '.join(content.split())
                cells.append(content)
            if cells:
                rows.append((cells, is_header))

        if not rows:
            return table_html  # Return original if parsing failed

        # Build markdown table
        md_lines = []
        max_cols = max(len(r[0]) for r in rows)

        for i, (cells, is_header) in enumerate(rows):
            # Pad cells if needed
            cells = cells + [''] * (max_cols - len(cells))
            md_lines.append('| ' + ' | '.join(cells) + ' |')
            # Add separator after header row
            if is_header or i == 0:
                md_lines.append('|' + '|'.join(['---'] * max_cols) + '|')

        return '\n'.join(md_lines)

    # Find and replace all tables
    return re.sub(
        r'<table[^>]*>(.*?)</table>',
        lambda m: parse_table(m.group(0)),
        text,
        flags=re.DOTALL | re.IGNORECASE
    )


def convert_footnotes(text: str) -> str:
    """Convert Unicode superscript footnotes to markdown footnote format.

    In text: ¹ → [^1]
    Definitions: ¹URL → [^1]: URL
    """
    # Map Unicode superscripts to digits
    superscript_map = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
    }
    superscript_pattern = '[' + ''.join(superscript_map.keys()) + ']+'

    def superscript_to_num(s: str) -> str:
        return ''.join(superscript_map[c] for c in s)

    # Convert footnote definitions (at start of line): ¹URL → [^1]: URL
    text = re.sub(
        rf'^({superscript_pattern})(.+)$',
        lambda m: f'[^{superscript_to_num(m.group(1))}]: {m.group(2)}',
        text,
        flags=re.MULTILINE
    )

    # Convert inline references: word¹ → word[^1]
    text = re.sub(
        rf'({superscript_pattern})',
        lambda m: f'[^{superscript_to_num(m.group(0))}]',
        text
    )

    return text


def postprocess_text(text: str) -> str:
    """Apply all post-processing to olmOCR output."""
    text = convert_latex_delimiters(text)
    text = convert_html_tables_to_markdown(text)
    text = convert_footnotes(text)
    return text


def parse_olmocr_response(content: str) -> tuple[str, dict]:
    """Parse olmOCR response, extracting text and metadata from YAML front matter.

    olmOCR returns markdown with YAML front matter like:
        ---
        primary_language: en
        is_rotation_valid: True
        rotation_correction: 0
        is_table: False
        is_diagram: False
        ---
        # Actual content here...

    This function extracts the clean text and parses the metadata.

    Args:
        content: Raw response from olmOCR model

    Returns:
        Tuple of (clean_text, metadata_dict)
    """
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            yaml_text = parts[1].strip()
            natural_text = parts[2].strip()
            # Parse YAML metadata
            metadata = {}
            for line in yaml_text.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    # Convert string booleans to actual booleans
                    if value.lower() == "true":
                        metadata[key] = True
                    elif value.lower() == "false":
                        metadata[key] = False
                    else:
                        metadata[key] = value
            return postprocess_text(natural_text), metadata
    return postprocess_text(content), {}


async def process_page(
    client: OpenAI,
    filename: str,
    page_num: int,
    model_name: str,
    target_longest_image_dim: int = 1288,  # olmOCR 2 expects longest dim = 1288px
) -> dict:
    """Process a single page and return the result.

    build_page_query handles:
    - Rendering the page image at the correct resolution
    - Extracting document metadata (page dimensions, text boxes, etc.)
    - Constructing the prompt with the required format
    """

    query = await build_page_query(
        filename,
        page_num,
        target_longest_image_dim,
    )

    # Override model name to match LM Studio loaded model
    query["model"] = model_name

    try:
        response = client.chat.completions.create(**query)
        content = response.choices[0].message.content

        # Parse olmOCR response (handles YAML front matter)
        text, metadata = parse_olmocr_response(content)

        return {
            "page": page_num,
            "success": True,
            "text": text,
            "metadata": {
                "language": metadata.get("primary_language", "unknown"),
                "is_table": metadata.get("is_table", False),
                "is_diagram": metadata.get("is_diagram", False),
            },
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    except Exception as e:
        return {
            "page": page_num,
            "success": False,
            "text": "",
            "error": str(e),
        }


def parse_page_range(page_range: str | None, max_pages: int) -> list[int]:
    """Parse page range string like '1-5' or '1,3,5' or '1-3,7,9-11'."""
    if not page_range:
        return list(range(1, max_pages + 1))

    pages = set()
    for part in page_range.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = int(start) if start else 1
            end = int(end) if end else max_pages
            pages.update(range(start, min(end, max_pages) + 1))
        else:
            page = int(part)
            if 1 <= page <= max_pages:
                pages.add(page)

    return sorted(pages)


async def process_pdf(
    pdf_path: str,
    client: OpenAI,
    model_name: str,
    pages: str | None = None,
    verbose: bool = True,
) -> tuple[str, list[dict]]:
    """Process all pages of a PDF and return combined text and results."""

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    page_numbers = parse_page_range(pages, num_pages)

    if verbose:
        print(f"Processing {pdf_path}: {len(page_numbers)} pages")

    results = []

    # Use tqdm for progress bar
    iterator = tqdm(page_numbers, desc="Pages", disable=not verbose)

    for page_num in iterator:
        result = await process_page(client, pdf_path, page_num, model_name)
        results.append(result)

        if not result["success"]:
            iterator.set_postfix_str(
                f"Page {page_num} failed: {result.get('error', 'unknown')}"
            )

    # Combine text from all successful pages (basic join)
    texts = []
    for result in results:
        if result["success"] and result["text"]:
            texts.append(result["text"].strip())

    combined_text = "\n\n".join(texts)

    return combined_text, results


def format_markdown(
    results: list[dict],
    filename: str,
    include_title: bool = True,
    include_page_markers: bool = False,
    include_metadata: bool = False,
) -> str:
    """Format results into clean markdown.

    Args:
        results: List of page results
        filename: Original PDF filename
        include_title: Add document title at top
        include_page_markers: Add page number comments between pages
        include_metadata: Add processing metadata at top
    """
    parts = []

    # Document title
    if include_title:
        title = Path(filename).stem.replace("_", " ").replace("-", " ")
        parts.append(f"# {title}\n")

    # Metadata block
    if include_metadata:
        successful = sum(1 for r in results if r["success"])
        total_pages = len(results)
        parts.append(
            f"> Converted from `{Path(filename).name}` ({successful}/{total_pages} pages)\n"
        )

    # Page contents
    for result in results:
        if not result["success"] or not result["text"]:
            continue

        text = result["text"].strip()

        if include_page_markers:
            parts.append(f"<!-- Page {result['page']} -->\n")

        parts.append(text)

    return "\n\n".join(parts)


def save_markdown(text: str, output_path: Path) -> None:
    """Save text to markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def print_stats(results: list[dict]) -> None:
    """Print processing statistics."""
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    total_input = sum(r.get("input_tokens", 0) for r in results)
    total_output = sum(r.get("output_tokens", 0) for r in results)

    print(f"\nStats:")
    print(f"  Pages processed: {successful}/{len(results)}")
    if failed > 0:
        print(f"  Failed pages: {failed}")
    print(f"  Total tokens: {total_input:,} input, {total_output:,} output")


async def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to markdown using olmOCR 2 with LM Studio. See README.md for full documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf                     Process and print to stdout
  %(prog)s document.pdf -o output.md        Save to markdown file
  %(prog)s document.pdf --pages 1-5         Process specific pages
  %(prog)s *.pdf --output-dir ./converted/  Batch process multiple PDFs
  %(prog)s doc.pdf --page-markers           Add page boundary comments
        """,
    )

    parser.add_argument("pdfs", nargs="+", help="PDF file(s) to process")
    parser.add_argument("-o", "--output", help="Output markdown file (single PDF only)")
    parser.add_argument("--output-dir", help="Output directory for multiple PDFs")
    parser.add_argument(
        "--pages", help="Page range to process (e.g., '1-5' or '1,3,5-7')"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name as loaded in LM Studio (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"LM Studio server URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show processing statistics"
    )
    parser.add_argument(
        "--no-title", action="store_true", help="Don't add document title to markdown"
    )
    parser.add_argument(
        "--page-markers",
        action="store_true",
        help="Add HTML comments marking page boundaries",
    )
    parser.add_argument(
        "--metadata", action="store_true", help="Add conversion metadata to markdown"
    )
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Automatically redact PII from the output",
    )
    parser.add_argument(
        "--redact-model",
        help="Model for PII redaction (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.output and len(args.pdfs) > 1:
        print(
            "Error: --output can only be used with a single PDF. Use --output-dir for multiple files."
        )
        sys.exit(1)

    # Create client
    client = create_client(base_url=args.server, timeout=args.timeout)

    # Test connection
    try:
        client.models.list()
    except Exception as e:
        print(f"Error: Cannot connect to LM Studio at {args.server}")
        print(f"Make sure LM Studio is running and the server is enabled.")
        print(f"Details: {e}")
        sys.exit(1)

    verbose = not args.quiet

    for pdf_path in args.pdfs:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            print(f"Warning: File not found: {pdf_path}")
            continue

        # Process the PDF
        text, results = await process_pdf(
            str(pdf_path),
            client,
            args.model,
            pages=args.pages,
            verbose=verbose,
        )

        # Format as proper markdown
        formatted_text = format_markdown(
            results,
            str(pdf_path),
            include_title=not args.no_title,
            include_page_markers=args.page_markers,
            include_metadata=args.metadata,
        )

        # Redact PII if requested
        if args.redact:
            from redact_pii import create_client as create_redact_client, redact_document

            redact_client = create_redact_client(base_url=args.server, timeout=args.timeout)
            redact_model = args.redact_model or get_loaded_model(redact_client)

            if verbose:
                print(f"Redacting PII using model: {redact_model}")

            formatted_text, pii_found, errors = redact_document(
                redact_client, formatted_text, redact_model, verbose=verbose
            )

            if verbose and pii_found:
                print(f"  Redacted {len(pii_found)} PII item(s)")

        # Determine output
        if args.output:
            output_path = Path(args.output)
        elif args.output_dir:
            output_path = Path(args.output_dir) / pdf_path.with_suffix(".md").name
        else:
            output_path = None

        # Save or print
        if output_path:
            save_markdown(formatted_text, output_path)
            if verbose:
                print(f"Saved to: {output_path}")
        else:
            print("\n" + "=" * 60)
            print(f"OUTPUT: {pdf_path}")
            print("=" * 60)
            print(formatted_text)

        if args.stats:
            print_stats(results)


if __name__ == "__main__":
    asyncio.run(main())
