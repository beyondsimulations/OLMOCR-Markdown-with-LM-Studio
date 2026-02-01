# olmOCR 2 for macOS with LM Studio

Convert PDFs to clean markdown using [olmOCR 2](https://github.com/allenai/olmocr) running locally on your Mac via [LM Studio](https://lmstudio.ai/) or the [LM Studio CLI](https://lmstudio.ai/docs/cli).

## About olmOCR 2

olmOCR 2 is a vision-language model fine-tuned from Qwen2.5-VL-7B-Instruct for high-quality document OCR. It handles:

- Academic papers and technical documentation
- Tables, equations, and complex formatting
- Handwriting recognition
- Multi-column layouts
- Headers and footers removal

The model expects document images with the longest dimension at 1288 pixels, along with metadata extracted from the PDF. This script handles all of that automatically using the `olmocr` toolkit.

## Prerequisites

1. **LM Studio** or **LM Studio CLI**
2. **Python 3.12+** with uv
3. **poppler-utils** — Required for PDF rendering

### Install poppler (macOS)

```bash
brew install poppler
```

### Install Python dependencies

```bash
uv sync
```

## Setup

1. Open **LM Studio**
2. Search for `olmocr gguf` and download a model (e.g., `olmOCR-2-7B-1025`)
3. Go to the **Developer** tab
4. Load the model and toggle the server **on**
5. Note the model name shown in LM Studio

### Configure .env

Copy the example config and edit with your model name:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# LM Studio server URL
OLMOCR_SERVER=http://localhost:1234/v1

# Model name as shown in LM Studio
OLMOCR_MODEL=allenai_olmocr-2-7b-1025

# Request timeout in seconds
OLMOCR_TIMEOUT=120
```

The model name must match exactly what's shown in LM Studio's Developer tab.

## Usage

### Basic conversion

```bash
# Process a PDF and print to stdout
uv run olmocr_lmstudio.py document.pdf

# Save to a markdown file
uv run olmocr_lmstudio.py document.pdf -o output.md
```

### Page selection

```bash
# Process only pages 1-5
uv run olmocr_lmstudio.py document.pdf --pages 1-5

# Process specific pages
uv run olmocr_lmstudio.py document.pdf --pages 1,3,5-7,10
```

### Batch processing

```bash
# Convert multiple PDFs to a directory
uv run olmocr_lmstudio.py *.pdf --output-dir ./converted/
```

### Formatting options

```bash
# Add page boundary markers
uv run olmocr_lmstudio.py document.pdf -o output.md --page-markers

# Add conversion metadata
uv run olmocr_lmstudio.py document.pdf -o output.md --metadata

# Skip the auto-generated title
uv run olmocr_lmstudio.py document.pdf -o output.md --no-title

# Combine options
uv run olmocr_lmstudio.py document.pdf -o output.md --page-markers --metadata
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `pdfs` | PDF file(s) to process (positional argument) |
| `-o, --output` | Output markdown file (single PDF only) |
| `--output-dir` | Output directory for multiple PDFs |
| `--pages` | Page range (e.g., `1-5` or `1,3,5-7`) |
| `--model` | Model name (default: from `.env` or `allenai_olmocr-2-7b-1025`) |
| `--server` | Server URL (default: from `.env` or `http://localhost:1234/v1`) |
| `--timeout` | Request timeout in seconds (default: from `.env` or `120`) |
| `-q, --quiet` | Suppress progress output |
| `--stats` | Show processing statistics |
| `--no-title` | Don't add document title to markdown |
| `--page-markers` | Add HTML comments marking page boundaries |
| `--metadata` | Add conversion metadata to markdown |

Command line arguments override `.env` values.

## Output Format

### Default

```markdown
# Document Name

First page content here...

Second page content...
```

### With `--page-markers`

```markdown
# Document Name

<!-- Page 1 -->

First page content...

<!-- Page 2 -->

Second page content...
```

### With `--metadata`

```markdown
# Document Name

> Converted from `document.pdf` (10/10 pages)

Content here...
```

## PII Redaction

The `redact_pii.py` script identifies and redacts Personally Identifiable Information from markdown files using any chat LLM via LM Studio.

### Basic usage

```bash
# Redact PII and print to stdout
uv run redact_pii.py document.md

# Save redacted version
uv run redact_pii.py document.md -o redacted.md

# Also save a PII report
uv run redact_pii.py document.md -o redacted.md --report pii_report.md
```

### Batch processing

```bash
uv run redact_pii.py *.md --output-dir ./redacted/
```

### PII Redaction Options

| Option | Description |
|--------|-------------|
| `files` | Markdown file(s) to process |
| `-o, --output` | Output file (single input only) |
| `--output-dir` | Output directory for multiple files |
| `--report` | Save PII report to file |
| `--report-dir` | Directory for PII reports (batch mode) |
| `--model` | Model name (default: auto-detect) |
| `--server` | LM Studio server URL |
| `--timeout` | Request timeout in seconds (default: 120) |
| `-q, --quiet` | Suppress progress output |
| `--dry-run` | Preview what would be redacted |

### Detected PII Categories

Names, emails, phone numbers, addresses, credit cards, SSNs, dates of birth, passport numbers, driver's licenses, bank accounts, IBANs, IP addresses, vehicle registrations, national IDs, tax IDs, medical records, usernames, passwords.

## Troubleshooting

### "Cannot connect to LM Studio"

- Make sure LM Studio is running
- Check that the server is enabled in the Developer tab
- Verify `OLMOCR_SERVER` in `.env` matches the LM Studio server URL

### "Model not found"

- Check the exact model name in LM Studio's Developer tab
- Update `OLMOCR_MODEL` in `.env` to match exactly
- Or use `--model "exact-model-name"` on the command line
- You will need an olmocr model for OCR to work
- You will need a general chat model for redaction to work

### Slow performance

- This runs on CPU/Apple Silicon, expect up to 60 seconds per page
- Use `--pages` to process only the pages you need
- Consider using a smaller quantized model (Q4 vs Q8)

### Timeout errors

- Increase timeout in `.env`: `OLMOCR_TIMEOUT=300`
- Or use `--timeout 300` for complex pages
- Some pages with dense content may take longer

## How it works

1. The script uses `olmocr.pipeline.build_page_query()` to:
   - Render each PDF page as an image (1288px longest side)
   - Extract metadata (page dimensions, text boxes, etc.)
   - Construct the prompt in the format olmOCR 2 expects

2. Sends the request to LM Studio's OpenAI-compatible API

3. Parses the JSON response containing the OCR'd text

4. Combines all pages into formatted markdown

## Testing the CLI

To see an example of a conversion, this repo has an example PDF file (`document.pdf`) that you can use with the CLI:

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. de las, Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.-A., Stock, P., Scao, T. L., Lavril, T., Wang, T., Lacroix, T., & Sayed, W. E. (2023). Mistral 7B (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2310.06825

## Credits

- [olmOCR](https://github.com/allenai/olmocr) by Allen Institute for AI
- [LM Studio](https://lmstudio.ai/) for local model inference
- Original macOS guide by [Jonathan Soma](https://jonathansoma.com/words/olmocr-on-macos-with-lm-studio.html)
- 
## Licensing

This project is licensed under the **MIT License** - see the [LICENSE.md](LICENSE) file for details.

### Dependencies

This project uses the following third-party components:

- **[OLMOCR](https://github.com/allenai/olmocr)** (Apache License 2.0)
- **[Devstral 2](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)** (Apache License 2.0)
- **[LM Studio CLI](https://lmstudio.ai/docs/cli)** (MIT License)
- **[Poppler](https://github.com/innodatalabs/poppler#GPL-2.0-1-ov-file)** (GPL-2.0 License)
