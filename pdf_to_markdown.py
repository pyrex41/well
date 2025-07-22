#!/usr/bin/env python3
"""
PDF to Markdown Converter for Well Data Analysis
Extracts text, tables, and structure from PDF files and converts to markdown.
"""

import pdfplumber
import re
import sys
from pathlib import Path

def extract_pdf_content(pdf_path):
    """Extract structured content from PDF file."""
    content = {
        'title': '',
        'pages': [],
        'tables': [],
        'metadata': {}
    }
    
    with pdfplumber.open(pdf_path) as pdf:
        # Extract metadata
        content['metadata'] = pdf.metadata or {}
        
        for page_num, page in enumerate(pdf.pages, 1):
            page_content = {
                'page_number': page_num,
                'text': '',
                'tables': []
            }
            
            # Extract text
            text = page.extract_text()
            if text:
                page_content['text'] = text.strip()
            
            # Extract tables
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if table:  # Skip empty tables
                        page_content['tables'].append(table)
            
            content['pages'].append(page_content)
    
    return content

def format_table_as_markdown(table):
    """Convert table to markdown format."""
    if not table or not any(table):
        return ""
    
    markdown_table = []
    
    # Header row
    if table[0]:
        header = "| " + " | ".join(str(cell or "") for cell in table[0]) + " |"
        separator = "| " + " | ".join("---" for _ in table[0]) + " |"
        markdown_table.extend([header, separator])
    
    # Data rows
    for row in table[1:]:
        if row:
            row_text = "| " + " | ".join(str(cell or "") for cell in row) + " |"
            markdown_table.append(row_text)
    
    return "\n".join(markdown_table)

def convert_to_markdown(content, pdf_name):
    """Convert extracted content to markdown format."""
    markdown_lines = []
    
    # Title
    title = content['metadata'].get('title', pdf_name.replace('.pdf', ''))
    markdown_lines.append(f"# {title}\n")
    
    # Metadata section
    if content['metadata']:
        markdown_lines.append("## Document Metadata\n")
        for key, value in content['metadata'].items():
            if value:
                markdown_lines.append(f"- **{key}**: {value}")
        markdown_lines.append("")
    
    # Process each page
    for page in content['pages']:
        if page['text'] or page['tables']:
            markdown_lines.append(f"## Page {page['page_number']}\n")
            
            if page['text']:
                # Clean up text formatting
                text = page['text']
                # Split into paragraphs and clean up
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                for paragraph in paragraphs:
                    # Handle potential headers (lines that are short and in caps)
                    if len(paragraph) < 100 and paragraph.isupper():
                        markdown_lines.append(f"### {paragraph}\n")
                    else:
                        markdown_lines.append(f"{paragraph}\n")
            
            # Add tables
            if page['tables']:
                for i, table in enumerate(page['tables'], 1):
                    markdown_lines.append(f"### Table {i}\n")
                    markdown_lines.append(format_table_as_markdown(table))
                    markdown_lines.append("")
    
    return "\n".join(markdown_lines)

def main():
    if len(sys.argv) != 2:
        print("Usage: python pdf_to_markdown.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: File {pdf_path} not found")
        sys.exit(1)
    
    print(f"Processing {pdf_path.name}...")
    
    try:
        # Extract content
        content = extract_pdf_content(pdf_path)
        
        # Convert to markdown
        markdown = convert_to_markdown(content, pdf_path.name)
        
        # Write to output file
        output_path = pdf_path.with_suffix('.md')
        output_path.write_text(markdown)
        
        print(f"Successfully converted to {output_path}")
        print(f"Processed {len(content['pages'])} pages")
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()