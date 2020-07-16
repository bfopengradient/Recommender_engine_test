  
import pandas as pd
import numpy as np
from numpy import count_nonzero
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

class artists:
    ''' Class that produces recommendations from the LastFM database ''' 
    
    #Dataframes to read in.
    df_listen= pd.read_csv('./hetrec2011-lastfm-2k/user_artists.dat',sep='\t',header=0, encoding='latin-1') 
    df_artists= pd.read_csv('./hetrec2011-lastfm-2k/artists.dat',sep='\t',header=0, encoding='latin-1')
    df_artists_lookup=pd.DataFrame(df_artists[['id','name']]).rename(columns={'id':'artistID','name':'artist_name'}).set_index('artistID')
    df_tags= pd.read_csv('/./hetrec2011-lastfm-2k/user_taggedartists.dat',sep='\t',header=0, encoding='latin-1')
    #user-item data. 
    df_user_artists= pd.read_csv('./hetrec2011-lastfm-2k/user_artists.dat',sep='\t',header=0, encoding='latin-1')    
    #For item utility matrix
    scraped_tags= pd.read_csv('./scrapedtags.csv').drop('Unnamed: 0',axis=1)    
    df_tagdata= pd.read_csv('./hetrec2011-lastfm-2k/tags.dat',sep='\t',header=0, encoding='latin-1')
 
 

    

    def scrape(i,j):
        #Scrape responsibly, only scrapping artists with no tags
        untagged= pd.concat([artists.df_artists['id'],artists.df_tags['artistID']]).drop_duplicates(keep=False) 
        df_untagged= artists.df_artists.loc[artists.df_artists['id'].isin(untagged.values)] 

        #Select urls from the untagged artists dataframe and produces the url and its corresponding first avaialable tag
        urls= np.array(df_untagged['url'][i:j])
               
        taggs=[]    
        for url in urls:
            req = requests.get(url)
            soup = BeautifulSoup(req.content, 'html.parser')    
            tags = soup.find(class_="tag")

            #Skip arists with no wikis
            if tags != None:
               for tag in tags:
                    tagged= ''.join(tag.find(text=True)) 
                    taggs.append(url)
                    taggs.append(tagged)
        #Save scraped tags to the scraped_tags dataframe            
        scraped_tags=pd.DataFrame(np.array(taggs).reshape(j,j),columns=(['url','tag']))
        
        #Commented out line below as it will override the scrapped tags dataframe 
        #scraped_tags.to_csv('...path')
        print(scraped_tags)   


 
 

    def create_ii_matrix():

        #Dataframes below help create an item-item utility matrix. Most data is extracted from existing user listening and tag behaviour plus at least one tag for almost every artist
        #feature space will be biased by how much an artist was listened to and tagged. 
        
        #Start with whether user has listened to artist and begin extracting item only data.
        df_item_a= artists.df_user_artists[['artistID','weight']].set_index('artistID') 
        df_item_a.index.names=['id']
        df_item_a['total_weighting'] = df_item_a.groupby(df_item_a.index).sum().astype(float) 
        df_item_a.drop(columns='weight',inplace=True)
        df_item_a.drop_duplicates(inplace=True)
        df_item_b= artists.df_artists   
        df_item_b=df_item_b.set_index('id')
        #Concatenated file    
        df_item_c= pd.concat([df_item_b,df_item_a],axis=1,sort=False)    


        #Add to item dataframe if an artist was tagged
        df_tags_a=artists.df_tags[['artistID','tagID']].set_index('artistID') 
        df_tags_a['total_tags'] = df_tags_a.groupby(df_tags_a.index).count().astype(float) 
        df_tags_a.drop(columns='tagID',inplace=True)
        df_tags_a.drop_duplicates(inplace=True)
        df_item_d=pd.concat([df_item_c,df_tags_a],axis=1,sort=False).fillna('0')  
        df_tagged= artists.df_tags.loc[:,['artistID','tagID']]
        df_tagged.rename(columns={'artistID':'id'},inplace=True)
        df_tagged.drop_duplicates('id',inplace=True)
        df_tagged.set_index('id',inplace=True)


        #Set up dataframe to match tags with tag ID, tags are set as the index
        #Set up dataframe to match tags with tagID        
        df_tagdata= artists.df_tagdata
        df_tagdata.rename(columns={'tagValue':'tag'},inplace=True)
        df_tagdata.set_index('tag')#,inplace=True)

        #Merge tagged and untagged files
        untagged= pd.concat([artists.df_artists['id'],artists.df_tags['artistID']]).drop_duplicates(keep=False) 
        df_untagged= artists.df_artists.loc[artists.df_artists['id'].isin(untagged.values)]
        df_untagged_tagged= pd.merge(df_untagged,artists.scraped_tags[['url','tag']],how='left').fillna('NA').set_index('tag')
        df_tagid_untagged= pd.concat([df_untagged_tagged,artists.df_tagdata],sort=False)

        #Merge the two dataframes to get a tagid for each untagged artist
        df_taggedid_untagged = df_untagged_tagged.merge(df_tagdata, left_index=True, right_index=True).set_index('id').drop(['name','url','pictureURL'],axis=1)

        #Then get tags for all/majority of  artists by combining tagged with untagged artists
        df_all_tagged=pd.concat([df_taggedid_untagged,df_tagged],sort=False) 
        df_all_tagged.sort_index(inplace=True)
        df_all_tagged.index.rename('id',inplace=True)

        #Concatenate files to produce the file with artist urls and tags
        df_item_e=pd.concat([artists.df_artists.set_index('id'),df_all_tagged],axis=1,sort=False) 
        df_item_e.dropna(inplace=True)
        df_item_e.drop(columns=['name','url','pictureURL'],inplace=True)

        #Concatenate df_item_d with df_item_e to produce latest item feature set and dataframe df_item_f. Assign zero to artist
        #where no tags are available

        #Add the listened to weightings to produce the final item-item dataframe
        df_item_f= pd.concat([df_item_d,df_item_e],axis=1).fillna(0)
        df_item_f['total_weighting']=df_item_f.total_weighting.astype(float)
        df_item_f['total_tags']=df_item_f.total_tags.astype(float)

        #Drop non-numeric features from final item dataset
        df_item_f.drop(columns=['name','url','pictureURL'],inplace=True)        
        return df_item_f
    
    
#Function to check sparsity of a utility matrix    
    def get_sparsity(self):              
            np.prod(self.shape)
            sparsity = 1.0 - ( count_nonzero(self) / float(self.size) )
            return sparsity 

             
#Function when called will make recommendations      
    def make_recommendations():
        
        #Create user-item utility matrix
        df2=pd.DataFrame(artists.df_listen.groupby(['userID','artistID']).size())
        df3= pd.pivot_table(df2, values=[0], index=['userID'], columns=['artistID']).fillna(0) 
        df4=df3[0]        
        #Filter database as needed to affect sparsity. Filter columns by number of times as artists was
        #listened to at least once and rows by numer of times a user listened to at least one artist in the
        #entire database        
        #Sum each column and drop any with count lower than number specified
        df5= df4.drop([col for col, val in df4.sum().iteritems() if val < 0], axis=1) 
        #Sum each row and drop any row with count lower that the number specified  
        df6= df5.drop([row for row, val in df5.sum(axis=1).iteritems() if val < 0], axis=0)        
        #Print user-item matrix sparsity
        print('user-item matrix sparsity is:',artists.get_sparsity(df6))  
                    
        #Create item-item utility matrices        
        df_item_f= artists.create_ii_matrix()       
        df_ui_similarity= pd.DataFrame(cosine_similarity(df6.T),index=df6.T.index,columns=df6.T.index)   
        df_ii_similarity= pd.DataFrame(cosine_similarity(df_item_f),index=df_item_f.index,columns=df_item_f.index)        
        #Print item-item sparsity 
        print('item-item matrix sparsity is ',artists.get_sparsity(df_ii_similarity))
        
        
        print('')
                           
        #Prompt box to check if artist is in the database and if so what is the artist ID for next prompt box
        #data is case sensitve so user has two goes before they are told artist unlikely in database.
        n = 4
        while n > 0:
            n -= 1   
            artist_name= input('Enter artist name to get ID:')
          
            result= artists.df_artists_lookup.loc[artists.df_artists_lookup['artist_name'].isin([artist_name])] 
           
            print('The artist ID is:',result.index.values) 
                       
            if result.index.size != 0:
                
                #Prompt box to enter artist ID.
                self= float(input('enter artistID:'))

                #Create two recommendation dataframes from the user-item and item-item utility matrices
                #User-item utility matrix will show top ten recommendations
                df_ui_most_similar=pd.DataFrame(df_ui_similarity.loc[self].sort_values(ascending=False, 
                        inplace=False, kind='quicksort', na_position='last')[1:10])
                df_ui_answer= pd.merge(df_ui_most_similar,artists.df_artists_lookup,left_index=True,right_index=True)

                #Item-item utility matrix will show top ten recommendations       
                df_ii_most_similar=pd.DataFrame(df_ii_similarity.loc[self].sort_values(ascending=False, 
                        inplace=False, kind='quicksort', na_position='last')[1:10])  
                df_ii_answer= pd.merge(df_ii_most_similar,artists.df_artists_lookup,left_index=True,right_index=True)  

                #Print recommendations from both utility matrices
                print('You selected', artists.df_artists_lookup.loc[self,:].values)
                print('')    
                print('If you like',artists.df_artists_lookup.loc[self,:].values, ' other users also liked:')    
                print('')
                print(df_ui_answer['artist_name'].values) 
                print('')
                print('You might also consider the following artists:',)
                print('')
                print(df_ii_answer['artist_name'].values) 
                
                break
                
            if artist_name != artists.df_artists_lookup['artist_name'].any():
                                         
                print('Search is case sensitive , cant find artist, please try again')           
            if n == 2:
                print('Sorry artist may not be in the database.')                                
                break  
                
         
        
  	
        

 
        
 
 
        
 
        