'''
File to convert raw arxiv data stored in azure blob into transformed data that can be uploaded
to a graph and other downstream tasks for various learning tasks.
This is the main file that will call multiple other functions. 
arxiv_dl
    pdf
        1991
        1992
        1993
        ...
    src
        1991
        1992
        1993
        ...
    tmp
where src is the latex files stored in .tar files. The pdfs are also stored in .tar files
The etl process is as follows
Arxiv data in azure blob is stored as such (note the path where the tranformation takes place
needs to at least have 500GB of hard drive space):
1. Get list of all arxiv pdf .tar files for a given year
2. Run as parallel process for each .tar file
    2.1. unzip .tar to a new directory on local
    2.2 run pdfplumber on each pdf and save the full text as .txt files
    2.3 push back the .txt files to blob and delete the .tar file and the pdfs from local
'''

import os, uuid
from pathlib import Path
import shutil
import time
import glob
import pdfplumber
import pandas as pd
from pdf2image import convert_from_path
from pdf2image.exceptions import(
 PDFInfoNotInstalledError,
 PDFPageCountError,
 PDFSyntaxError
)
import ray
import tarfile
from retry import retry
import sys
import os
import logging

'''
Set logging
'''
# Gets or create a logger for this python module
logger = logging.getLogger(__file__)
# set log level
logger.setLevel(logging.DEBUG)
# define file handler and set formatter
file_handler = logging.FileHandler(
                os.path.join(
                    os.getcwd(),
                    'logs',
                    os.path.basename(__file__).split('.')[0]+'.log'
                    )
                )
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
# add file handler to logger
logger.addHandler(file_handler)
# add stream handler to logger so we can get the log outptus in the console as well
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
# set the time zone so logger is in the right time zone
os.environ['TZ'] = 'US/Eastern'
time.tzset()

'''
For ray to work all the files and modules have to be within the
original sys.path. for some reason you can't append another path and
import files from there. Therefore have to set the PYTHONPATH for
each new process. In this case the run_pdfplumber function is calling
src.fixunicode which is in another child directory
'''
os.environ['PYTHONPATH'] = os.path.dirname(os.getcwd())
sys.path.append(os.getcwd())
import config
import src.blob_data_transfer as blob_pull
import src.fixunicode as fixunicode
import src.get_citations as citations

# credentials for blob with our raw data
file_type = 'tar'
year_del = 2
prefix = 'arxiv_dl/pdf'
pdfplumber_path = 'arxiv_training_data/pdfplumber'


ray.init(include_webui=False, ignore_reinit_error=True)

def unzip_tar(tar_file, extract_path):
    
    while not os.path.exists(tar_file):
        time.sleep(0.5)
    if os.path.isfile(tar_file):
        # Can't open empty tar files becuase it produces a read error
        # so need to make sure tar file isn't empty
        try:
            tar = tarfile.open(tar_file)
            tar.extractall(path = extract_path)
            pdf_list = get_list(extract_path+'/**/*.pdf')
            tar.close()
            return pdf_list
        except Exception as e:
            logger.exception("{} for {}".format(e, tar_file))
            
@retry(tries=10, delay=1)
def open_pdf(filepath):
    try:
        with pdfplumber.open(filepath) as pdf:
            text = ""
            if pdf.pages is not None: 
                # ran into some random IndexError: list index out of range
                for page in range(len(pdf.pages)):
                    current_page = pdf.pages[page]
                    '''
                    if 'W' in obj and 'H' in obj:
                    TypeError: argument of type 'int' is not iterable
                    or
                    TypeError: a bytes-like object is required, not
                    'str'. Fix is this:
                    https://github.com/euske/pdfminer/issues/97
                    '''
                    t = current_page.extract_text(x_tolerance=1,
                                                  y_tolerance=0)
                    if t is not None:
                        text += t
        
        # run text through very basic unicode normalization routines
        # before sending back
        text = fixunicode.fix_unicode(text)
        
        return text
    except Exception as e:
        logger.info("Can't open file {}".format(filepath))
        logger.exception(e)
        
def run_pdfplumber(filepath, text_path):
    filename = filepath.split('/')[-1]
    
    while not os.path.exists(filepath):
        time.sleep(0.5)
    if os.stat(filepath).st_size != 0:  # file is empty
        try:
            text = open_pdf(filepath)
            if text is not None: 
            # in case pdf is empty and returns None
                if len(text) >= 45: 
                # if file is smaller than 45 words than don't convert it
                    text_file = text_path+'/'+filename.replace('.pdf','')+'.txt'
                    
                    f = open(text_file, 'w')
                    f.write(text)
                    f.close()
                    return text_file
        except Exception as e:
            logger.info("failed to open pdf {}".format(filepath))
            logger.exception(e)
    else:
        logger.info("PDF file is empty: {}".format(filepath))


def move_txt_to_blob(txt):
    blob_pull.send_to_blob(txt, 4)
            
def create_paths(year):
    extract_path = 'arxiv_pdf/'+year
    Path(extract_path).mkdir(parents=True, exist_ok=True)

    pdfplumber_path = 'arxiv_training_data/pdfplumber'

    text_path = pdfplumber_path+'/text/'+year
    Path(text_path).mkdir(parents=True, exist_ok=True)
    return extract_path, pdfplumber_path, text_path

def get_list(path):
    return glob.glob(path, recursive=True)


@ray.remote
def extract_txt(pdf, extract_path, text_path):
    text_file = run_pdfplumber(pdf, text_path)
    if text_file is not None:
        move_txt_to_blob(text_file)
        os.remove(text_file)
    os.remove(pdf)

@ray.remote
def get_citations(article):
    return citations.citation_list_inner(article)

# Main function
def create_txt_dataset():
    full_blob_list = blob_pull.get_blob_list(prefix)
    pdf_tar_list, year_list = blob_pull.get_blob_file_list(
                                                file_type,
                                                full_blob_list,
                                                year_del
                                                )
    year_list = ['1991', '1992']
    total_pdf = 0
    total_extraction_time = 0
    total_conversion_time = 0
    
    for year in year_list:
        extract_path, pdfplumber_path, text_path = create_paths(year)
        # get list of all the .tar files (blobs) for a given year and
        # run the end-to-end process through this list
        pdf_count = 0
        extraction_time = 0
        conversion_time = 0
        for blob in pdf_tar_list:
            if blob.split('/')[year_del]==year:
                t = time.time()
                tar_file = blob_pull.copy_blob(blob)
                pdf_list = unzip_tar(tar_file, extract_path)
                if pdf_list is not None: 
                # In case the .tar file is empty and returns None
                    pdf_count += len(pdf_list)
                    extraction_time += (time.time() - t)
                    total_extraction_time += extraction_time

                    t = time.time()
                    ray.get([extract_txt.remote(pdf,
                                                extract_path,
                                                text_path
                                               ) 
                             for pdf in pdf_list]
                           )
                    conversion_time += (time.time() - t)
                    total_conversion_time += conversion_time
                    os.remove(tar_file)
                
        logger.info("Completed conversion of {} pdfs for the year {}\nExtraction time: {}\nConversion Time: {}".format(
                            pdf_count,
                            year,
                            extraction_time,
                            conversion_time
                            )
                   )
        
        # Get citations for all the papers in a given year and then save
        # them to a file
        text_blob_list = blob_pull.get_blob_list(text_path)
        text_list = [blob.name for blob in text_blob_list]
        citation_list = ray.get([get_citations.remote(article) for article in text_list])
        citations.save_to_default_location(citation_list)
        
    logger.info("Completed creation of text dataset for all of arxiv")
    logger.info("Total pdf: {}\nTotal Extraction Time: {}\nTotal Conversion Time: {}".format(total_pdf, total_extraction_time, total_conversion_time))
    
if __name__ == "__main__":
    create_txt_dataset()