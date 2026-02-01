#!/usr/bin/env python3
"""
PII Redactor for Markdown Files

Identifies and redacts Personally Identifiable Information (PII) from markdown
files using a local LLM via LM Studio.

See README.md for full documentation and setup instructions.

Quick start:
    1. Start LM Studio with any chat model loaded
    2. uv run redact_pii.py input.md -o output_redacted.md
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Check dependencies
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
    from tqdm import tqdm
except ImportError:
    _missing.append("tqdm")

if _missing:
    print(f"Error: Missing dependencies. Run:\n  uv pip install {' '.join(_missing)}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_SERVER = os.getenv(
    "REDACT_SERVER", os.getenv("OLMOCR_SERVER", "http://localhost:1234/v1")
)
DEFAULT_MODEL = os.getenv(
    "REDACT_MODEL", ""
)  # Empty = use whatever is loaded in LM Studio
DEFAULT_TIMEOUT = int(os.getenv("REDACT_TIMEOUT", "120"))

# PII categories and their placeholder tokens
PII_CATEGORIES = {
    "name": "[NAME]",
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "address": "[ADDRESS]",
    "credit_card": "[CREDIT_CARD]",
    "ssn": "[SSN]",
    "date_of_birth": "[DOB]",
    "passport": "[PASSPORT]",
    "drivers_license": "[DRIVERS_LICENSE]",
    "bank_account": "[BANK_ACCOUNT]",
    "iban": "[IBAN]",
    "ip_address": "[IP_ADDRESS]",
    "vehicle_registration": "[VEHICLE_REG]",
    "national_id": "[NATIONAL_ID]",
    "tax_id": "[TAX_ID]",
    "medical_record": "[MEDICAL_RECORD]",
    "username": "[USERNAME]",
    "password": "[PASSWORD]",
}

# Pattern to detect any placeholder format: [WORD], [WORD_WORD], etc.
PLACEHOLDER_PATTERN = re.compile(r'\[([A-Z][A-Z0-9_]*)\]')


def detect_placeholders(original: str, redacted: str) -> list[dict]:
    """Detect placeholders in redacted text that weren't in the original.

    Fallback for when LLM doesn't populate pii_found properly.
    """
    original_placeholders = set(PLACEHOLDER_PATTERN.findall(original))
    redacted_placeholders = PLACEHOLDER_PATTERN.findall(redacted)

    detected = []
    for placeholder in redacted_placeholders:
        if placeholder not in original_placeholders:
            detected.append({
                "category": placeholder.lower(),
                "text": f"[{placeholder}]",
                "context": "(detected from output)",
            })

    return detected


SYSTEM_PROMPT = """You are a PII (Personally Identifiable Information) detection and redaction assistant.

Your task is to identify ALL personally identifiable information in the provided text and return a JSON response with:
1. A list of all PII found with their category and exact text
2. The redacted version of the text with PII replaced by category placeholders

PII categories to detect:
- name: Full names, first names, last names of individuals
- email: Email addresses
- phone: Phone numbers (any format)
- address: Physical addresses, street names with numbers, postal codes
- credit_card: Credit/debit card numbers
- ssn: Social Security Numbers or equivalent national IDs
- date_of_birth: Dates of birth
- passport: Passport numbers
- drivers_license: Driver's license numbers
- bank_account: Bank account numbers
- iban: IBAN numbers
- ip_address: IP addresses
- vehicle_registration: Vehicle registration/license plate numbers
- national_id: National ID numbers (other than SSN)
- tax_id: Tax identification numbers
- medical_record: Medical record numbers
- username: Usernames that could identify someone
- password: Passwords or secrets

Respond ONLY with valid JSON in this exact format:
{
  "pii_found": [
    {"category": "name", "text": "John Smith", "context": "brief context where found"},
    {"category": "email", "text": "john@example.com", "context": "brief context"}
  ],
  "redacted_text": "The full text with all PII replaced by placeholders like [NAME], [EMAIL], etc.",
  "summary": "Brief summary of what was redacted"
}

Be thorough but avoid false positives. Common words, product names, or company names that aren't identifying individuals should NOT be redacted.

CRITICAL - NEVER redact any of the following (these are NOT PII):
- Technical parameters, specifications, or configuration values (numbers like 4096, 32, 128)
- Model names, version numbers, or software identifiers (e.g., Llama, GPT-4, Mistral 7B)
- Author names on published academic papers - these are PUBLIC, not private individuals
- URLs, repository links, or documentation references
- Company names and product names (e.g., Google, OpenAI, Hugging Face)
- Benchmark scores, statistical results, or performance metrics
- Table data containing technical specifications - NEVER redact table values
- License information (Apache, MIT, etc.)
- Section headings or structural text

IMPORTANT: Only redact information that could identify or harm a PRIVATE individual if exposed. Academic authors, company employees mentioned in official contexts, and public figures are NOT private individuals.

Do NOT redact entire paragraphs or sections. Only redact the specific PII text, not surrounding content."""


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


def chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    """Split text into chunks, trying to break at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If single paragraph is too long, split by sentences
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= max_chars:
                        current_chunk += sent + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent + " "
            else:
                current_chunk = para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def redact_chunk(client: OpenAI, text: str, model: str) -> dict:
    """Send a text chunk to the LLM for PII detection and redaction."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Analyze this text for PII and return the redacted version:\n\n{text}",
                },
            ],
            temperature=0.1,  # Low temperature for consistent detection
            max_tokens=len(text) + 2000,  # Allow for JSON overhead
        )

        content = response.choices[0].message.content

        # Try to parse JSON response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content.strip())
            pii_found = result.get("pii_found", [])
            redacted_text = result.get("redacted_text", text)

            # Fallback: detect placeholders if LLM didn't report them
            if not pii_found and redacted_text != text:
                pii_found = detect_placeholders(text, redacted_text)

            return {
                "success": True,
                "pii_found": pii_found,
                "redacted_text": redacted_text,
                "summary": result.get("summary", ""),
            }
        except json.JSONDecodeError:
            # If JSON parsing fails, return original text
            return {
                "success": False,
                "pii_found": [],
                "redacted_text": text,
                "error": "Failed to parse LLM response as JSON",
                "raw_response": content[:500],
            }

    except Exception as e:
        return {
            "success": False,
            "pii_found": [],
            "redacted_text": text,
            "error": str(e),
        }


def redact_document(
    client: OpenAI,
    text: str,
    model: str,
    verbose: bool = True,
) -> tuple[str, list[dict], list[str]]:
    """Redact PII from an entire document."""
    chunks = chunk_text(text)

    all_pii = []
    redacted_chunks = []
    errors = []

    iterator = tqdm(chunks, desc="Processing", disable=not verbose)

    for chunk in iterator:
        result = redact_chunk(client, chunk, model)

        if result["success"]:
            all_pii.extend(result["pii_found"])
            redacted_chunks.append(result["redacted_text"])
        else:
            errors.append(result.get("error", "Unknown error"))
            redacted_chunks.append(result["redacted_text"])
            iterator.set_postfix_str(f"Error: {result.get('error', '')[:30]}")

    redacted_text = "\n\n".join(redacted_chunks)

    return redacted_text, all_pii, errors


def generate_report(pii_found: list[dict], errors: list[str]) -> str:
    """Generate a summary report of found PII."""
    lines = ["# PII Redaction Report\n"]

    if pii_found:
        lines.append(f"## Found {len(pii_found)} PII item(s)\n")

        # Group by category
        by_category = {}
        for item in pii_found:
            cat = item.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)

        for category, items in sorted(by_category.items()):
            placeholder = PII_CATEGORIES.get(category, f"[{category.upper()}]")
            lines.append(
                f"### {category.title()} ({len(items)} found) → {placeholder}\n"
            )
            for item in items:
                text = item.get("text", "")
                context = item.get("context", "")
                lines.append(f"- `{text}`")
                if context:
                    lines.append(f"  - Context: {context}")
            lines.append("")
    else:
        lines.append("No PII detected.\n")

    if errors:
        lines.append(f"## Errors ({len(errors)})\n")
        for error in errors:
            lines.append(f"- {error}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Redact PII from markdown files using a local LLM via LM Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.md                      Redact and print to stdout
  %(prog)s document.md -o redacted.md       Save redacted version
  %(prog)s document.md --report report.md   Also save PII report
  %(prog)s *.md --output-dir ./redacted/    Batch process files
  %(prog)s document.md --model llama3       Use specific model
        """,
    )

    parser.add_argument("files", nargs="+", help="Markdown file(s) to process")
    parser.add_argument("-o", "--output", help="Output file (single input only)")
    parser.add_argument("--output-dir", help="Output directory for multiple files")
    parser.add_argument("--report", help="Save PII report to this file")
    parser.add_argument("--report-dir", help="Directory for PII reports (batch mode)")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name in LM Studio (default: auto-detect loaded model)",
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
        "-q", "--quiet", action="store_true", help="Suppress progress output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be redacted without saving",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.output and len(args.files) > 1:
        print(
            "Error: --output can only be used with a single file. Use --output-dir for multiple files."
        )
        sys.exit(1)

    # Create client
    client = create_client(base_url=args.server, timeout=args.timeout)

    # Test connection and get model
    try:
        model = args.model if args.model else get_loaded_model(client)
        if not args.quiet:
            print(f"Using model: {model}")
    except Exception as e:
        print(f"Error: Cannot connect to LM Studio at {args.server}")
        print(f"Make sure LM Studio is running and the server is enabled.")
        print(f"Details: {e}")
        sys.exit(1)

    verbose = not args.quiet
    all_reports = []

    for file_path in args.files:
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue

        if verbose:
            print(f"\nProcessing: {file_path}")

        # Read input
        text = file_path.read_text(encoding="utf-8")

        # Redact
        redacted_text, pii_found, errors = redact_document(
            client, text, model, verbose=verbose
        )

        # Generate report
        report = generate_report(pii_found, errors)
        all_reports.append((file_path.name, report, pii_found))

        if verbose:
            print(f"  Found {len(pii_found)} PII item(s)")
            if errors:
                print(f"  Encountered {len(errors)} error(s)")

        if args.dry_run:
            print("\n" + report)
            continue

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        elif args.output_dir:
            output_path = (
                Path(args.output_dir) / f"{file_path.stem}_redacted{file_path.suffix}"
            )
        else:
            output_path = None

        # Save or print
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(redacted_text, encoding="utf-8")
            if verbose:
                print(f"  Saved to: {output_path}")
        else:
            print("\n" + "=" * 60)
            print(f"REDACTED: {file_path}")
            print("=" * 60)
            print(redacted_text)

        # Save individual report
        if args.report and len(args.files) == 1:
            Path(args.report).write_text(report, encoding="utf-8")
            if verbose:
                print(f"  Report saved to: {args.report}")
        elif args.report_dir:
            report_path = Path(args.report_dir) / f"{file_path.stem}_pii_report.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report, encoding="utf-8")
            if verbose:
                print(f"  Report saved to: {report_path}")

    # Summary for batch processing
    if len(args.files) > 1 and verbose:
        total_pii = sum(len(r[2]) for r in all_reports)
        print(f"\n{'=' * 60}")
        print(f"Processed {len(all_reports)} files, found {total_pii} total PII items")


if __name__ == "__main__":
    main()
