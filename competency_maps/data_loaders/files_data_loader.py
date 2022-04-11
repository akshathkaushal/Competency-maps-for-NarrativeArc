import os
from io import StringIO
from pathlib import Path

import pandas as pd
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from rich.console import Console
from rich.progress import track

from .data_loader import DataLoader


class FilesDataLoader(DataLoader):
    """ Data Loader Class to load data from a specified folder

    If the input learning resources is a file/folder, then use this class to scan the files in the folder and generate
    a dataframe

    Attributes:
          input_path: Folder containing the Learning Resources.
    """

    def __init__(self, path):
        DataLoader.__init__(self, path)
        self.console = Console()

    @staticmethod
    def _convert_pdf(path, outtype="txt", opts={}):
        """ Method used to extract textual content from a PDF File.

        Args:
            path (str): PDF File Path
            outtype (str): Output Type to be converted to
            opts (dict): Dictionary Object that contains any additional options to be used while conversion

        Returns:
            Text Extracted from the PDF File
        """
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = "utf-8"
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(path, "rb")
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()

        for page in PDFPage.get_pages(
            fp,
            pagenos,
            maxpages=maxpages,
            password=password,
            caching=caching,
            check_extractable=True,
        ):
            interpreter.process_page(page)

        text = retstr.getvalue()

        fp.close()
        device.close()
        retstr.close()
        return text

    def read_corpus(self):
        """ Method to read the corpus of documents from the input path

        Returns:
              A dataframe containing the filename and the contents of the file
        """
        self.console.log("Reading Files from folders")
        text_data = []
        converted_path = Path.joinpath(
            self.input_path.parent, "converted_files"
        )
        if not os.path.exists(converted_path):
            os.mkdir(converted_path)
        exts = [".pdf", ".txt"]
        files = list(self.input_path.rglob("*"))
        for idx in track(range(len(files)), description="Reading Files..."):
            file = files[idx]
            self.console.log(f"File: {file.name}")
            if file.suffix == ".pdf":
                self.console.log("Getting Text from PDF...")
                # print(file.as_posix())
                content = self._convert_pdf(file.as_posix())
                with open(
                    f'{converted_path}/{file.name.split(".")[0]}.txt',
                    "w",
                    encoding="utf-8",
                ) as f:
                    print(content, file=f)
            elif file.suffix == ".txt":
                with open(file, encoding="utf-8") as f:
                    content = f.read().replace("\n", " ")
            else:
                self.console.log(
                    f"[red]{file} is a Invalid File Type since its not a pdf or txt file.[/]"
                )
                content = None
            if content is not None:
                data = [f"{file.parent.name}_{file.name}", content]
                text_data.append(data)
        text_corpus = pd.DataFrame(
            text_data, columns=["resource_id", "description"]
        )
        return text_corpus
