import sys
import os

from PyPDF2 import PdfFileMerger

def pdf_merger(pdfs=None, indir=None, outfile=None):
    # how are inputs specified
    if pdfs is None:
        # check PDF names
        assert indir is not None, 'Must provide either `pdfs` (list) or `indir` (directory where PDFs are found) - not both.'
        pdfs = sorted([f'{indir}/{i}' for i in os.listdir(indir) if i.endswith('.pdf')])
    else:
        # check directory
        assert indir is None, 'Must provide either `pdfs` (list) or `indir` (directory where PDFs are found) - not both.'
        assert all([i.endswith('.pdf') for i in pdfs]), 'Not all files passed into `pdfs` are actual PDFs.'
        indir = '/'.join(pdfs[0].split('/')[:-1])

    # compile PDFs
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)

    # get output file (if already exists, delete before writing new file)
    if outfile is None:
        outfile = f'{indir}/_compiled.pdf'
    else:
        assert outfile.endswith('.pdf'), 'Provided output file is not a PDF.'
    if os.path.exists(outfile):
        os.remove(outfile)

    # write compiled PDF
    merger.write(outfile)
    merger.close()
