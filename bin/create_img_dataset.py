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
1. Pull arxiv pdf .tar files one year at a time
2. unzip .tars to a new directory on local
3. run pdfplumber on each pdf to get the word level and character level bounding boxes and save
   them as .csvs
4. run pdfplumber on each pdf and save the full text as .txt files
5. convert each pdf into a set of images
6. push the unzipped pdfs, images, and csvs to azure blob as training data
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
file_handler = logging.FileHandler(os.path.join(os.getcwd(),'logs',os.path.basename(__file__).split('.')[0]+'.log'))
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
# add file handler to logger
logger.addHandler(file_handler)
# add stream handler to logger so we can get the log outptus in the console as well
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

'''
For ray to work all the files and modules have to be within the original sys.path.
for some reason you can't append another path and import files from there. Therefore
have to set the PYTHONPATH for each new process. In this case the run_pdfplumber
function is calling src.fixunicode which is in another child directory
'''
sys.path.append(os.getcwd())
import config
import src.blob_data_transfer as blob_pull
import src.fixunicode as fixunicode

# credentials for blob with our raw data
file_type = 'tar'
year_del = 2
prefix = 'arxiv_dl/pdf'
pdfplumber_path = 'arxiv_training_data/pdfplumber'


ray.init(ignore_reinit_error=True)

@ray.remote
def unzip_tar(file, tar_path, extract_path):
    
    while not os.path.exists(os.path.join(tar_path, file)):
        time.sleep(0.5)
    if os.path.isfile(os.path.join(tar_path, file)):
        # Can't open empty tar files becuase it produces a read error so need to make sure
        # tar file isn't empty
        try:
            tar = tarfile.open(os.path.join(tar_path, file))
            tar.extractall(path = extract_path)
            tar.close()
        except Exception as e:
            logger.exception("{} for {}".format(e, file))
            
@retry(tries=10, delay=1)
def open_pdf(filepath):
    try:
        with pdfplumber.open(filepath) as pdf:
            chars = []
            words = []
            text = ""
            for page in range(len(pdf.pages)):
                current_page = pdf.pages[page]
                chars.extend(current_page.chars)
                words.extend(current_page.extract_words(x_tolerance=1, y_tolerance=0))
                t = current_page.extract_text(x_tolerance=1, y_tolerance=0)
                if t is not None:
                    text += t
        
        # run text through very basic unicode normalization routines before sending back
        text = fixunicode.fix_unicode(text)
        
        return chars, words, text
    except Exception as e:
        logger.info("Can't open file {}".format(filepath))
        logger.exception(e)
        
@ray.remote
def run_pdfplumber(filepath, char_path, word_path, text_path):
#     os.environ['PYTHONPATH'] = os.path.dirname(os.getcwd())
#     import src.fixunicode as fixunicode
    """
    Converting the output of pdfplumber to csv and txt files and then saving them is taking about 15 seconds per pdf
    """
    filename = filepath.split('/')[-1]
    
    while not os.path.exists(filepath):
        time.sleep(0.5)
    if os.stat(filepath).st_size != 0:  # file is empty
        try:
            chars, words, text = open_pdf(filepath)
            if len(text) >= 45: # if file is smaller than 45 words than don't convert it
                chars_df = pd.DataFrame(chars)
                chars_df.to_csv(char_path+'/'+filename.replace('.pdf','')+'_chars.csv', index=False)   
                words_df = pd.DataFrame(words)
                words_df.to_csv(word_path+'/'+filename.replace('.pdf','')+'_words.csv', index=False)

                # write full pdf text to csv
                f = open(text_path+'/'+filename.replace('.pdf','')+'.txt', 'w')
                f.write(text)
                f.close()
                return filepath
        except Exception as e:
            logger.info("failed to open pdf {}".format(filepath))
            logger.exception(e)
    else:
        logger.info("PDF file is empty: {}".format(filepath))

@ray.remote
def convert_to_image(filepath, image_path):
    if os.path.exists(filepath):
        year = filepath.split('/')[1]
        pdf_name = filepath.split('/')[-1]
        image_list = []
        try:
            images = convert_from_path(filepath)
            for i, image in enumerate(images):
                fname = image_path+"/"+pdf_name.replace('.pdf','')+"_"+str(i)+".png"
                image.save(fname, "PNG")
                image_list.extend(fname)
            return image_list
        except Exception as e:
            logger.info("Failed to convert {} to image".format(filepath))
            logger.exception(e)

@ray.remote
def move_csv_to_blob(csv):
    blob_pull.send_to_blob(csv, 4)
            
@ray.remote
def move_image_to_blob(img):
    blob_pull.send_to_blob(img, 3)

@ray.remote
def move_pdf_to_blob(filepath):
    blob_pull.send_to_blob(filepath, 1)

@ray.remote
def move_txt_to_blob(txt):
    blob_pull.send_to_blob(txt, 4)
            
def create_paths(year):
    extract_path = 'arxiv_pdf/'+year
    Path(extract_path).mkdir(parents=True, exist_ok=True)

    pdfplumber_path = 'arxiv_training_data/pdfplumber'

    char_path = pdfplumber_path+'/chars/'+year
    Path(char_path).mkdir(parents=True, exist_ok=True)

    word_path = pdfplumber_path+'/words/'+year
    Path(word_path).mkdir(parents=True, exist_ok=True)

    text_path = pdfplumber_path+'/text/'+year
    Path(text_path).mkdir(parents=True, exist_ok=True)

    image_path = "arxiv_training_data/pdf_images/"+year
    Path(image_path).mkdir(parents=True, exist_ok=True)
    return extract_path, pdfplumber_path, char_path, word_path, text_path, image_path

def get_list(path, file_type):
    return glob.glob(path+file_type, recursive=True)
            
# Main function
def convert_raw_arxiv_data():
    full_blob_list = blob_pull.get_blob_list(prefix)
    pdf_tar_list, year_list = blob_pull.get_blob_file_list(file_type, full_blob_list, year_del)
    #year_list = ['1991']
    pdf_count = 0
    total_pdf = 0
    pdf_time = 0
    pdfplumber_time = 0
    image_time = 0
    back_blob_time = 0
    
    
    for year in year_list:
        tar_path = blob_pull.copy_blob(year, pdf_tar_list, year_del)
        
        extract_path, pdfplumber_path, char_path, word_path, text_path, image_path = create_paths(year)

        # extract all the tar files for a given year
        t = time.time()
        ray.get([unzip_tar.remote(file, tar_path, extract_path) for file in os.listdir(tar_path)])
        logger.info("Finished unzipping {} tar files in {:0.2f} minutes".format(len(os.listdir(tar_path)), int((time.time() - t)/60)))

        # get list of all pdfs for the year that have been extracted
        sub_folders = os.listdir(extract_path)
        pdf_list = []
        for folder in sub_folders:
            pdf_list.extend(glob.glob(extract_path+'/'+folder+'/*.pdf'))
        total_pdf += len(pdf_list)
        # work on all the pdfs for a given year. Each work item is parallized through ray to
        # go through all the pdfs. Work items are run sequentally so we don't run into any io
        # issues
        t1 = time.time()
        plumbed_pdf_list = ray.get([run_pdfplumber.remote(pdf, char_path, word_path, text_path) for pdf in pdf_list])
        plumbed_pdf_list = list(filter(None, plumbed_pdf_list))  # remove None types
        t2 = (time.time() - t1)/60
        logger.info("Finished pdfplumber for {} pdf files in {:0.2f} minutes for year {}".format(len(pdf_list), t2, year))
        pdf_time += t2
        pdf_count += len(plumbed_pdf_list)
        
        t1 = time.time()
        ray.get([convert_to_image.remote(pdf, image_path) for pdf in plumbed_pdf_list])
        t2 = (time.time() - t1)/60
        logger.info("Finished converting {} files to images in {:0.2f} minutes for year {}".format(len(plumbed_pdf_list), t2, year))
        pdfplumber_time += t2
        
        t1 = time.time()
        ray.get([move_pdf_to_blob.remote(pdf) for pdf in plumbed_pdf_list])
        t2 = (time.time() - t1)/60
        logger.info("Moved {} pdfs to blob in {:0.2f} minutes for year {}".format(len(plumbed_pdf_list), t2, year))
        image_time += t2
        
        t1 = time.time()
        csv_list = get_list(pdfplumber_path, '**/**/*.csv')
        ray.get([move_csv_to_blob.remote(csv) for csv in csv_list])
        logger.info("Moved csvs to blob")
        
        text_list = get_list(text_path, '/*.txt')
        ray.get([move_txt_to_blob.remote(txt) for txt in text_list])
        logger.info("Moved text files to blob")
        
        image_list = get_list(image_path, '/*.png')
        ray.get([move_image_to_blob.remote(img) for img in image_list])
        logger.info("Moved images to blob")
        
        t2 = (time.time() - t1)/60
        back_blob_time += t2

        # delete all the paths and files we saved on local
        time.sleep(30)
        shutil.rmtree('arxiv_pdf')
        shutil.rmtree('arxiv_dl')
        shutil.rmtree('arxiv_training_data')
        
    logger.info("Total number of completed pdfs: {}/{}\nExtraction time: {:0.2f}\nPdfPlumber time: {:0.2f}\nImage converion time: {:0.2f}\nMoving back to blob time: {:0.2f}".format(pdf_count,total_pdf, pdf_time, pdfplumber_time, image_time, back_blob_time))
        
if __name__ == "__main__":
    convert_raw_arxiv_data()