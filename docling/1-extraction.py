from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from utils.sitemap import get_sitemap_urls

# Configure PDF processing with accurate table extraction
pipeline_options = PdfPipelineOptions(
    do_table_structure=True,
    table_structure_options={
        "mode": TableFormerMode.ACCURATE,
        "table_output_format": "markdown",
        "max_workers": 4,
        "crop_padding": 10
    },
    layout_analysis_engine="rtdetr"
)
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)
# --------------------------------------------------------------
# Basic PDF extraction
# --------------------------------------------------------------

result = converter.convert("C:/Users/Anarchy/Documents/Data_Science/CEMA/RCEMA/docling/Protocol on Alarm Fatigue May_26_2024.pdf")

document = result.document
markdown_output = document.export_to_markdown()
json_output = document.export_to_dict()

print(markdown_output)

# --------------------------------------------------------------
# Basic HTML extraction
# --------------------------------------------------------------

result = converter.convert("https://cema-africa.uonbi.ac.ke/")

document = result.document
markdown_output = document.export_to_markdown()
print(markdown_output)

# --------------------------------------------------------------
# Scrape multiple pages using the sitemap
# --------------------------------------------------------------

sitemap_urls = get_sitemap_urls("https://cema-africa.uonbi.ac.ke/")
conv_results_iter = converter.convert_all(sitemap_urls)

docs = []
for result in conv_results_iter:
    if result.document:
        document = result.document
        docs.append(document)
