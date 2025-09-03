from pathlib import Path
from pypdf import PdfReader, PdfWriter

def combine_pdfs(pdf_list, output_file):
    """
    Combine multiple PDFs into a single PDF.

    Parameters
    ----------
    pdf_list : list of str/Path
        List of PDF file paths in the order they should be merged.
    output_file : str/Path
        Path for the combined output PDF.
    """

    # --- sanity checks ---
    if not isinstance(pdf_list, (list, tuple)) or len(pdf_list) == 0:
        raise ValueError("pdf_list must be a non-empty list of PDF file names")

    pdf_paths = [Path(p) for p in pdf_list]

    for p in pdf_paths:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {p}")

    output_file = Path(output_file)
    if output_file.suffix.lower() != ".pdf":
        raise ValueError("Output file must end with .pdf")

    # --- combine PDFs ---
    writer = PdfWriter()

    for pdf_path in pdf_paths:
        try:
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    writer.add_page(page)
        except Exception as e:
            print(f"⚠️ Warning: Skipping {pdf_path} due to error: {e}")

    if len(writer.pages) == 0:
        raise RuntimeError("No pages were added. Output PDF will not be created.")

    try:
        with open(output_file, "wb") as out_f:
            writer.write(out_f)
        #print(f"✅ Combined PDF saved to {output_file}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save output PDF ({output_file}): {e}")
