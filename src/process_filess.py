import os
import sys
from tika import parser
from sentences import split

def useful(s:str):
    """Filter out sentences that are not useful"""
    return len(s) > 2 and sum(c.isalpha() for c in s) > 2

class PdfProcessor():
    """
    Process a folder of PDFs, extracting text and splitting into sentences.
    """

    def __init__(self, folder:str):
        self.folder = folder


    def process_pdf(self, pdf:str):
        """Process a single PDF"""

        # extract text with tika
        raw = parser.from_file(pdf)

        # split article into sentences
        sents = split(raw['content'])

        # drop if too short
        drop_short = [p for p in sents if useful(p)]

        print(f"{len(drop_short)} in {pdf}")
        return drop_short

    def process_folder(self):
        """Process all PDFs in a folder"""

        lines = []
        for pdf in os.listdir(self.folder):
            if pdf.endswith(".pdf"):
                ls = self.process_pdf(os.path.join(self.folder, pdf)) # type: ignore
                lines.extend(ls)

        fn = f"./{self.folder}/lines.txt"
        with open(fn, 'w', encoding='utf-8') as the_file:
            the_file.writelines(f'{s}\n' for s in lines)
        print(f"wrote {len(lines)} total to {fn}")

if __name__ == "__main__":
    # get folder from command line
    if len(sys.argv) > 1:
        PdfProcessor(sys.argv[1]).process_folder()
    else:
        PdfProcessor('pdfs').process_folder()
