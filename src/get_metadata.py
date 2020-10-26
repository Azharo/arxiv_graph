import os
import config
from dataclasses import dataclass
import gzip
import glob
import json
import hashlib
import requests
import json
import pandas as pd
import xml.etree.ElementTree as ET
import re
import ray
os.environ['PYTHONPATH'] = os.path.dirname(os.getcwd())
from src.authors import parse_authorline

@dataclass
class neo4j_import:
    '''
    Python dataclass to take the original metadata json from arxiv
    and convert it to neo4j import dataframes based on the graph
    data model for arxiv
    Each arxiv publication has the following dictionary fields in the
    metadata:
    {id, submitter, authors, title, comments, journal, doi, abstract,
    report-no, categories, versions}
    The publication data class will take in metadata at __init__ and
    then go through post-init processes to get all the other info
    
    Parameters
    ----------
        metadata: json metadata of articles
    
    objects
    -------
        df_authors: dataframe with each author node as a row
            properties: author_name
        df_articles: dataframe with each article node as a row
            properties: arxiv_id, title, pub_date, categories,
                        report_no, abstract, doi
        df_rels: dataframe with the (author)-(authored)->(article)
        relationships
            properties: author_name, arxiv_id
    '''
    ray.init()
    metadata: dict
        
    def __post_init__(self):
        self.metadata_df = pd.DataFrame.from_dict(metadata, orient='columns')
        authors_cols = ['authorId:ID(Author-ID)','author_name',':LABEL']
        articles_cols = ['articleId:ID(Article-ID)',
                         'title',
                         'pub_date',
                         'categories',
                         'report-no',
                         'abstract',
                         'doi']
        rels_cols = [':START_ID(Author-ID)',':END_ID(Article-ID)',':TYPE']
        self.df_authors = pd.DataFrame(columns=authors_cols)
        self.df_articles = pd.DataFrame(columns=articles_cols)
        self.df_rels = pd.DataFrame(columns=rels_cols)
        
        # create the author dataframe and save as csv in neo4j import folder
        self.get_author_df(list(self.metadata_df['authors']))
        '''
        CODE
        ----
        Save dataframe as csv and send to neo4j import folder
        '''
        # create the artciel df and save as csv in neo4j import folder
        self.get_article_df(self.metadata_df)
        '''
        CODE
        ----
        Save dataframe as csv and send to neo4j import folder
        '''
        
        '''
        CODE
        ----
        Create rel_df
        '''

# #         self.title_emb = self.get_embedding(config.embed_model_api,
# #                                        text_payload={'text':self.title})
# #         self.abstract_emb = self.get_embedding(config.embed_model_api,
# #                                          text_payload={
# #                                              'text':self.abstract
# #                                          }
# #                                         )
    @ray.remote
    def parse_author_line(authors):
        return parse_authorline(authors).split('; ')
    
    def parse_categories(cat_list):
        return cat_list[0].strip(' ')

    def parse_abstract(abstract):
        return abstract.replace('\n','')[2:]

    def get_author_df(authors):
        authors = ray.get([parse_author_line.remote(article)
                           for article in authors])
        authors = [author for article in authors
                      for author in article]
        authors = list(set(authors))
        self.df_authors['author_name'] = pd.Series(authors)
        self.df_authors['authorId:ID(Author-ID)'] = self.df_authors.reset_index()['index']
        self.df_authors[':LABEL'] = 'author'
    
    def get_article_df(articles):
        self.df_articles['title'] = articles['title']
        self.df_articles['pub_date'] = self.get_pub_date(articles['pub_id'])
        self.df_articles['categories'] = articles['categories'].apply(parse_categories)
        self.df_articles['report-no'] = articles['report-no']
        # all the abstracts have a space in front so the [2:] just
        # removes that space
        self.df_articles['abstract'] = articles['abstract'].apply(parse_abstract)
        self.df_articles['doi'] = articles['doi']
        self.df_articles['articleId:ID(Article-ID)'] = articles['id']
            
    def get_embedding(self,
                      model_api=config.embed_model_api,
                      text_payload=None):
        '''
        Function that takes in the text, either the title or the
        abstract for each paper, and runs them through a deployed
        embedding model to get the text embedding.

        Parameters
        ----------
            model_loc : url
                location of the model to make request post
            text_payload : dict
                dictionary that of the form {'text':text} where text is
                either the abstract or the title

        Returns
        -------
            embedding : pytorch.Tensor
                embedding representing the input text
        '''
        try:
            response = requests.post(model_api, json=text_payload)
            if response.status_code == 200:
                json_load = json.loads(response.content)
                embedding = torch.tensor(json_load['cls_embedding'])
            return embedding
        except Exception as e:
                print(e)
    
    def get_pub_date(self, pub_id):
        '''
        Get the arxiv year and month of publication from the arxiv id
        arxiv follows the naming pattern based on:
        https://arxiv.org/help/arxiv_identifier

        Parameters
        ----------
            pub_id: the arxiv id of the publication
        Return
        ------
            year: publication year
            month: publication date
        '''
        modern_pattern = re.compile('\d\d\d\d.\d\d\d\d')
        old_pattern = re.compile('/\d\d\d\d\d\d\d')

        if modern_pattern.match(pub_id):
            date = pub_id.split('.')[0]
            year = "20"+date[:2]
            month = date[2:]
        elif old_pattern.search(pub_id):
            date = pub_id.split('/')[-1]
            if date[0] == "0":
                year = "20"+date[:1]
            else:
                year = "19"+date[:1]
            day = date[2:3]
        return year+"_"+month